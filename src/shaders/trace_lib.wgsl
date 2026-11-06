struct RP {
  res: vec4u,       // 宽, 高, 采样序号起点, 本次采样数
  dims: vec4u,      // nx, ny, nz, 帧号
  camPos: vec4f,    // xyz, tan(半视场角)
  camFwd: vec4f,    // xyz, 宽高比
  camRight: vec4f,  // xyz, 光圈半径（米）
  camUp: vec4f,     // xyz, 对焦距离
  a: vec4f,         // dx, 步长（格）, 快门时间（秒）, 夜空亮度
  b: vec4f,         // 地面开关, 地面反照率, 反弹次数, 化学发光强度
  soot: vec4f,      // 每 ppm 碳烟的消光系数 rgb [1/m], 碳烟单次散射反照率
  c: vec4f,         // 白烟质量消光 [m²/g], 白烟反照率, HG 不对称因子 g, 环境温度
  alb: array<vec4f, 2>,
  gc: vec4f,        // 地面照度缓存区域 x0, z0, 边长, 模式（1 = 光照缓存，0 = 纯路径追踪）
}
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var volT: texture_3d<f32>;
@group(0) @binding(2) var solT: texture_3d<f32>;
@group(0) @binding(3) var velT: texture_3d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var<storage, read> bricks: array<f32>;
@group(0) @binding(6) var<storage, read> lut: array<vec4f>;
// 发光单元的累积分布（归一化到 1），最后一个元素之后存总功率
@group(0) @binding(8) var<storage, read> ecdf: array<f32>;

const PI: f32 = 3.14159265;
var<private> rs: u32;
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd() -> f32 { rs = pcg(rs); return f32(rs >> 8u) * (1.0 / 16777216.0); }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn dimsF() -> vec3f { return vec3f(f32(R.dims.x), f32(R.dims.y), f32(R.dims.z)); }
fn dimsV() -> vec3i { return vec3i(i32(R.dims.x), i32(R.dims.y), i32(R.dims.z)); }

// 查找表：[0,1024) 碳烟单位体积分数的发射（每米），[1024,2048) 黑体表面亮度；T ∈ [300, 3300] K
fn lutAt(off: u32, T: f32) -> vec3f {
  let x = clamp((T - 300.0) / 3000.0, 0.0, 1.0) * 1023.0;
  let i = u32(floor(x));
  let j = min(i + 1u, 1023u);
  return mix(lut[off + i].rgb, lut[off + j].rgb, x - f32(i));
}

// 运动模糊：快门时刻 τ 的场 = 当前场沿速度平移，位置 g 处的物质来自 g − v·τ
fn blurSrc(g: vec3f, tau: f32) -> vec3f {
  if (tau == 0.0) { return g; }
  let v = textureSampleLevel(velT, samp, g / dimsF(), 0.0).xyz;
  return g - v * tau / R.a.x;
}
struct Med { st: vec3f, ss: vec3f, em: vec3f }
fn medium(g: vec3f, tau: f32) -> Med {
  let gp = blurSrc(g, tau);
  let s = textureSampleLevel(volT, samp, gp / dimsF(), 0.0);
  let fv = max(s.x, 0.0);
  let aer = max(s.z, 0.0);
  var m: Med;
  let sootT = R.soot.rgb * fv;
  let aerT = vec3f(R.c.x * aer);
  m.st = sootT + aerT;
  m.ss = sootT * R.soot.w + aerT * R.c.y;
  // 碳烟热辐射（吸收系数 = 消光 ×（1 − 反照率），查找表已含 κ_λ 的波长依赖）
  m.em = lutAt(0u, s.y) * fv * 1e-6 + R.b.w * max(s.w, 0.0) * vec3f(0.12, 0.28, 1.0);
  return m;
}

struct Hit { t: f32, kind: u32, n: vec3f, Ts: f32, chr: f32, mat: u32 }
struct MR { L: vec3f, Tr: vec3f, W: f32, selP: vec3f, selThr: vec3f, hit: Hit }

fn boxHit(ro: vec3f, rd: vec3f, bmax: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let t0 = (vec3f(0.0) - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  return vec2f(max(max(tn.x, tn.y), max(tn.z, 0.0)), min(min(tf.x, tf.y), tf.z));
}

fn solidOcc(g: vec3f) -> f32 { return textureSampleLevel(solT, samp, g / dimsF(), 0.0).x; }

// 点 y 所在单元被 NEE 选中的体积概率密度
fn neePdf(g: vec3f) -> f32 {
  let dv = dimsV();
  let c = vec3i(floor(g));
  if (any(c < vec3i(0)) || any(c >= dv)) { return 0.0; }
  let i = u32(c.x + dv.x * (c.y + dv.y * c.z));
  let pb = ecdf[i] - select(0.0, ecdf[max(i, 1u) - 1u], i > 0u);
  return pb / (R.a.x * R.a.x * R.a.x);
}

// misPdf > 0：该射线是从散射/反射顶点按方向 pdf = misPdf 采样出来的，体发光按 MIS 权重累加
fn march(ro: vec3f, rd: vec3f, wantScatter: bool, tau: f32, misPdf: f32) -> MR {
  var r: MR;
  r.L = vec3f(0.0);
  r.Tr = vec3f(1.0);
  r.W = 0.0;
  r.hit.kind = 0u;
  r.hit.t = 1e30;
  let dx = R.a.x;
  let bmax = dimsF() * dx;
  // 地面
  if (R.b.x > 0.5 && rd.y < -1e-6 && ro.y > 0.0) {
    r.hit.t = -ro.y / rd.y;
    r.hit.kind = 2u;
    r.hit.n = vec3f(0.0, 1.0, 0.0);
  }
  let bh = boxHit(ro, rd, bmax);
  let t1 = min(bh.y, r.hit.t);
  if (bh.x >= t1) { return r; }
  let ds = R.a.y * dx;
  var t = bh.x + rnd() * ds;
  let bd = (dimsV() + vec3i(7)) / 8;
  let rdg = rd / dx;
  var guard = 0;
  loop {
    if (t >= t1 || guard > 4096) { break; }
    guard++;
    let p = ro + rd * t;
    let g = p / dx;
    let bc = clamp(vec3i(floor(g / 8.0)), vec3i(0), bd - vec3i(1));
    if (bricks[bc.x + bd.x * (bc.y + bd.y * bc.z)] < 1e-3) {
      // 空砖块：直接跳到出口
      let lo = vec3f(bc * 8);
      let hi = lo + vec3f(8.0);
      let te = (select(lo, hi, rdg > vec3f(0.0)) - g) / rdg;
      let tex = min(min(select(1e30, te.x, abs(rdg.x) > 1e-9), select(1e30, te.y, abs(rdg.y) > 1e-9)), select(1e30, te.z, abs(rdg.z) > 1e-9));
      t += max(tex, 0.01) * 1.0 + ds * rnd();
      continue;
    }
    // 固体表面
    if (solidOcc(g) > 0.5) {
      var ta = t - ds; var tb = t;
      for (var k = 0; k < 6; k++) {
        let tm = 0.5 * (ta + tb);
        if (solidOcc((ro + rd * tm) / dx) > 0.5) { tb = tm; } else { ta = tm; }
      }
      let gh = (ro + rd * tb) / dx;
      let e = 0.6;
      let n = -vec3f(
        solidOcc(gh + vec3f(e, 0.0, 0.0)) - solidOcc(gh - vec3f(e, 0.0, 0.0)),
        solidOcc(gh + vec3f(0.0, e, 0.0)) - solidOcc(gh - vec3f(0.0, e, 0.0)),
        solidOcc(gh + vec3f(0.0, 0.0, e)) - solidOcc(gh - vec3f(0.0, 0.0, e)));
      let sv = textureSampleLevel(solT, samp, gh / dimsF(), 0.0);
      r.hit.t = tb;
      r.hit.kind = 1u;
      r.hit.n = select(-rd, normalize(n), length(n) > 1e-5);
      r.hit.Ts = sv.y / max(sv.x, 0.5);
      r.hit.chr = clamp(sv.z / max(sv.x, 0.5), 0.0, 1.0);
      r.hit.mat = u32(round(sv.w / max(sv.x, 0.5)));
      break;
    }
    let h = min(ds, t1 - t);
    let m = medium(g, tau);
    let stl = max(m.st, vec3f(1e-6));
    let att = exp(-stl * h);
    // 段内解析积分：∫ Tr e^{−σs} ε ds = Tr ε (1 − e^{−σh}) / σ
    let seg = (vec3f(1.0) - att) / stl;
    var wm = 1.0;
    if (misPdf > 0.0) {
      let tm = t + 0.5 * h;
      let pp = misPdf / max(tm * tm, 1e-8);
      let pn = NEE_N * neePdf(blurSrc((ro + rd * tm) / dx, tau));
      wm = pp * pp / (pp * pp + pn * pn);
    }
    r.L += r.Tr * m.em * seg * wm;
    // 光照缓存模式：烟雾被火光照亮的单次散射直接查表（每个内核自己定义 inscatter）
    if (R.gc.w > 0.5) { r.L += r.Tr * m.ss * seg * inscatter(g, rd); }
    if (wantScatter) {
      let wv = r.Tr * m.ss * seg;
      let w = lum(wv);
      if (w > 0.0) {
        r.W += w;
        if (rnd() * r.W < w) {
          r.selP = ro + rd * (t + rnd() * h);
          r.selThr = wv / w;
        }
      }
    }
    r.Tr *= att;
    if (max(r.Tr.x, max(r.Tr.y, r.Tr.z)) < 1e-3) { r.Tr = vec3f(0.0); r.hit.kind = 3u; break; }
    t += h;
  }
  return r;
}

// 阴影射线透射率（步长 1 格，跳过空砖块，碰到固体为 0）
fn shadowTr(ro: vec3f, rd: vec3f, tmax: f32, tau: f32) -> vec3f {
  let dx = R.a.x;
  let bh = boxHit(ro, rd, dimsF() * dx);
  let t1 = min(bh.y, tmax);
  var tr = vec3f(1.0);
  if (bh.x >= t1) { return tr; }
  let ds = 1.0 * dx;
  var t = bh.x + rnd() * ds;
  let bd = (dimsV() + vec3i(7)) / 8;
  let rdg = rd / dx;
  var guard = 0;
  loop {
    if (t >= t1 || guard > 2048) { break; }
    guard++;
    let g = (ro + rd * t) / dx;
    let bc = clamp(vec3i(floor(g / 8.0)), vec3i(0), bd - vec3i(1));
    if (bricks[bc.x + bd.x * (bc.y + bd.y * bc.z)] < 1e-3) {
      let lo = vec3f(bc * 8);
      let te = (select(lo, lo + vec3f(8.0), rdg > vec3f(0.0)) - g) / rdg;
      let tex = min(min(select(1e30, te.x, abs(rdg.x) > 1e-9), select(1e30, te.y, abs(rdg.y) > 1e-9)), select(1e30, te.z, abs(rdg.z) > 1e-9));
      t += max(tex, 0.01) + ds * rnd();
      continue;
    }
    if (solidOcc(g) > 0.5) { return vec3f(0.0); }
    let h = min(ds, t1 - t);
    tr *= exp(-medium(g, tau).st * h);
    if (max(tr.x, max(tr.y, tr.z)) < 1e-3) { return vec3f(0.0); }
    t += h;
  }
  return tr;
}
// 每个顶点的 NEE 采样数（MIS 权重按 N·pdf 计）
const NEE_N: f32 = 2.0;
fn neeN(x: vec3f, wi: vec3f, n: vec3f, surf: bool, tau: f32) -> vec3f {
  var s = vec3f(0.0);
  for (var i = 0; i < 2; i++) { s += nee(x, wi, n, surf, tau); }
  return s;
}
fn hgPdf(ct: f32, g: f32) -> f32 {
  let d = 1.0 + g * g - 2.0 * g * ct;
  return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}
// 一次火光采样：c = T·ε·V/(pb·d²)（不含相函数/余弦），w = 指向发光点的方向
struct NS { c: vec3f, w: vec3f }
fn neeSample(x: vec3f, tau: f32) -> NS {
  var r: NS;
  r.c = vec3f(0.0);
  r.w = vec3f(0.0, 1.0, 0.0);
  let dv = dimsV();
  let nbk = u32(dv.x * dv.y * dv.z);
  if (ecdf[nbk] <= 0.0) { return r; }
  let u = rnd();
  var lo = 0u; var hi = nbk - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2u;
    if (ecdf[mid] < u) { lo = mid + 1u; } else { hi = mid; }
  }
  let pb = ecdf[lo] - select(0.0, ecdf[max(lo, 1u) - 1u], lo > 0u);
  if (pb <= 0.0) { return r; }
  let cc = vec3i(i32(lo) % dv.x, (i32(lo) / dv.x) % dv.y, i32(lo) / (dv.x * dv.y));
  let g = vec3f(cc) + vec3f(rnd(), rnd(), rnd());
  let dx = R.a.x;
  let em = medium(vec3f(cc) + 0.5, 0.0).em;
  if (max(em.x, max(em.y, em.z)) <= 0.0) { return r; }
  let d = g * dx - x;
  let dist = length(d);
  r.w = d / max(dist, 1e-6);
  // 距离² 设下限（缓存单元尺度），避免近处点光源奇点
  let d2 = max(dist * dist, 4.0 * dx * dx);
  r.c = shadowTr(x, r.w, dist, tau) * em * (dx * dx * dx) / (pb * d2);
  return r;
}

// 火光的下一事件估计：按每个单元的发光功率选单元，在单元内均匀取点，向它投阴影射线
// surf = true：漫反射表面（返回 cosθ/π 加权），否则介质（返回相函数加权）
fn nee(x: vec3f, wi: vec3f, n: vec3f, surf: bool, tau: f32) -> vec3f {
  let dv = dimsV();
  let nbk = u32(dv.x * dv.y * dv.z);
  let total = ecdf[nbk];
  if (total <= 0.0) { return vec3f(0.0); }
  // 二分查找
  let u = rnd();
  var lo = 0u; var hi = nbk - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2u;
    if (ecdf[mid] < u) { lo = mid + 1u; } else { hi = mid; }
  }
  let pb = ecdf[lo] - select(0.0, ecdf[max(lo, 1u) - 1u], lo > 0u);
  if (pb <= 0.0) { return vec3f(0.0); }
  let cc = vec3i(i32(lo) % dv.x, (i32(lo) / dv.x) % dv.y, i32(lo) / (dv.x * dv.y));
  let g0 = vec3f(cc) + vec3f(rnd(), rnd(), rnd());
  let dx = R.a.x;
  // 在未平移的场里选物质点，再按速度挪到本样本的快门时刻（与 march 里的 blurSrc 互逆）
  var g = g0;
  if (tau != 0.0) { g = g0 + textureSampleLevel(velT, samp, g0 / dimsF(), 0.0).xyz * tau / dx; }
  let y = g * dx;
  let vb = dx * dx * dx;
  // 发光取单元中心值（不平移），与选单元的概率一致（三线性插值值与 pdf 之比在火焰边缘会很大，产生萤火虫）
  let em = medium(vec3f(cc) + 0.5, 0.0).em;
  if (max(em.x, max(em.y, em.z)) <= 0.0) { return vec3f(0.0); }
  let d = y - x;
  let dist = length(d);
  let w = d / max(dist, 1e-6);
  var f = 0.0;
  if (surf) { f = max(dot(w, n), 0.0) / PI; } else { f = hgPdf(dot(wi, w), R.c.z); }
  if (f <= 0.0) { return vec3f(0.0); }
  let tr = shadowTr(x, w, dist, tau);
  // 体发光：L = ∫ ε(y) T / d² dV，pdf_V(y) = pb / V_cell；与方向采样（pdf_V = p_dir / d²）做平衡启发式 MIS
  let d2 = max(dist * dist, 1e-8);
  let pn = pb / vb;
  let pd = f / d2;
  let wn = (NEE_N * pn) * (NEE_N * pn) / ((NEE_N * pn) * (NEE_N * pn) + pd * pd);
  return tr * em * f * wn / (pn * d2 * NEE_N);
}

fn frame(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  return mat3x3f(tx, ty, n);
}
fn cosineDir(n: vec3f) -> vec3f {
  let u1 = rnd(); let u2 = rnd();
  let r = sqrt(u1);
  let ph = 2.0 * PI * u2;
  return frame(n) * vec3f(r * cos(ph), r * sin(ph), sqrt(max(1.0 - u1, 0.0)));
}
fn hgDir(wi: vec3f, g: f32) -> vec3f {
  let u1 = rnd(); let u2 = rnd();
  var ct: f32;
  if (abs(g) < 1e-3) { ct = 1.0 - 2.0 * u1; } else {
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    ct = (1.0 + g * g - s * s) / (2.0 * g);
  }
  let st = sqrt(max(1.0 - ct * ct, 0.0));
  let ph = 2.0 * PI * u2;
  return frame(wi) * vec3f(st * cos(ph), st * sin(ph), ct);
}
fn skyL(d: vec3f) -> vec3f {
  let h = clamp(d.y, -1.0, 1.0);
  let k = select(0.15, 0.6 + 0.4 * (1.0 - h), h > 0.0);
  return R.a.w * k * vec3f(0.55, 0.75, 1.25);
}
fn albedoOf(m: u32) -> f32 { return R.alb[m / 4u][m % 4u]; }
