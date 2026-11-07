// ===== 渲染：模拟场 → 渲染体纹理，体积路径追踪 =====

//#KERNEL pack
// 把模拟场打包成可硬件三线性过滤的 3D 纹理：
// vol = (碳烟体积分数 ppm, T [K], 白烟 g/m³, 放热率 kW/m³)；sol = (占据, Ts, 炭化比例, 材料槽)；vel = 单元中心速度 m/s
@group(0) @binding(1) var<storage, read> s0: array<vec4f>;
@group(0) @binding(2) var<storage, read> s1: array<vec4f>;
@group(0) @binding(3) var<storage, read> sol: array<vec4f>;
@group(0) @binding(4) var<storage, read> flags: array<u32>;
@group(0) @binding(5) var<storage, read> vu: array<f32>;
@group(0) @binding(6) var<storage, read> vv: array<f32>;
@group(0) @binding(7) var<storage, read> vw: array<f32>;
@group(0) @binding(8) var volT: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var solT: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var velT: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!inb(c)) { return; }
  let i = sidx(c);
  let a = s0[i];
  let b = s1[i];
  let rho = rhoOf(a.x);
  let solid = flags[i] == 1u;
  var v = vec4f(rho * a.w / RHO_SOOT * 1e6, a.x, rho * b.x * 1e3, b.y * 1e-3);
  if (solid) { v = vec4f(0.0, a.x, 0.0, 0.0); }
  textureStore(volT, c, v);
  let so = sol[i];
  let tot = so.y + so.z;
  // 除占据外各分量预乘占据，过滤后除以占据即得固体单元的加权平均
  let oc = select(0.0, 1.0, solid);
  textureStore(solT, c, vec4f(oc, so.w * oc, so.z / max(tot, 1e-6) * oc, so.x * oc));
  let X = vec3i(1, 0, 0); let Y = vec3i(0, 1, 0); let Z = vec3i(0, 0, 1);
  let u = 0.5 * vec3f(vu[vidx(c)] + vu[vidx(c + X)], vv[vidx(c)] + vv[vidx(c + Y)], vw[vidx(c)] + vw[vidx(c + Z)]);
  textureStore(velT, c, vec4f(u, 0.0));
}

//#KERNEL bricks
// 8³ 砖块的占据标记（含 1 格外扩，覆盖三线性过滤的影响范围），用于跳过空区域
@group(0) @binding(1) var volR: texture_3d<f32>;
@group(0) @binding(2) var solR: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> bricks: array<f32>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let bd = (dimsI() + vec3i(7)) / 8;
  let bc = vec3i(gid);
  if (any(bc >= bd)) { return; }
  var m = 0.0;
  for (var z = -1; z < 9; z++) {
    for (var y = -1; y < 9; y++) {
      for (var x = -1; x < 9; x++) {
        let c = bc * 8 + vec3i(x, y, z);
        if (!inb(c)) { continue; }
        let v = textureLoad(volR, c, 0);
        let s = textureLoad(solR, c, 0);
        m = max(m, max(max(v.x * 10.0, v.z * 2.0), max(v.w * 0.1, s.x * 10.0)));
      }
    }
  }
  bricks[bc.x + bd.x * (bc.y + bd.y * bc.z)] = m;
}

//#KERNEL ecells
// 每个单元的发光功率（亮度 × 体积），供火光重要性采样
@group(0) @binding(1) var volR: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> lut: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> ecell: array<f32>;
@group(0) @binding(4) var<uniform> chemi: vec4f;
fn lutAt(T: f32) -> vec3f {
  let x = clamp((T - 300.0) / 3000.0, 0.0, 1.0) * 1023.0;
  let i = u32(floor(x));
  return mix(lut[i].rgb, lut[min(i + 1u, 1023u)].rgb, x - f32(i));
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!inb(c)) { return; }
  let v = textureLoad(volR, c, 0);
  let em = lutAt(v.y) * max(v.x, 0.0) * 1e-6 + chemi.x * max(v.w, 0.0) * exp(-20.0 * max(v.x, 0.0)) * vec3f(0.12, 0.28, 1.0);
  var e = dot(em, vec3f(0.2126, 0.7152, 0.0722)) * P.dx * P.dx * P.dx;
  // 光学厚的火焰内部发出的光大多逃不出来：按 6 个轴向的平均逃逸透射率加权选取概率（只改变 pdf，不改变期望）
  if (e > 0.0) {
    var esc = 0.0;
    for (var k = 0; k < 6; k++) {
      var od = 0.0;
      var p = c;
      for (var s = 0; s < 48; s++) {
        p += DIRS[k];
        if (!inb(p) || od > 7.0) { break; }
        let w = textureLoad(volR, p, 0);
        od += (chemi.y * max(w.x, 0.0) + chemi.z * max(w.z, 0.0)) * P.dx;
      }
      esc += exp(-od);
    }
    e *= max(esc / 6.0, 0.02);
  }
  ecell[sidx(c)] = e;
}

//#KERNEL trace
//@TRACE
@group(0) @binding(7) var<storage, read_write> accum: array<vec4f>;
// 光照缓存：lvA = 火光通量 φ = ∫L dω（rgb），lvB = 平均入射方向（xyz）与集中度 a；gE = 地面照度（rgb）
@group(0) @binding(9) var lvA: texture_3d<f32>;
@group(0) @binding(10) var lvB: texture_3d<f32>;
@group(0) @binding(11) var gE: texture_3d<f32>;

// 入射光近似为"各向同性 + 沿平均方向的一束"：L_s(ω) = φ [ (1−a)/4π + a·p_HG(ω·d̄) ]
fn inscatter(g: vec3f, rd: vec3f) -> vec3f {
  let uv = g / (2.0 * vec3f(textureDimensions(lvA)));
  let A = textureSampleLevel(lvA, samp, uv, 0.0).rgb;
  let B = textureSampleLevel(lvB, samp, uv, 0.0);
  let a = clamp(length(B.xyz), 0.0, 1.0);
  let d = select(vec3f(0.0, 1.0, 0.0), B.xyz / max(length(B.xyz), 1e-6), a > 1e-4);
  return A * ((1.0 - a) / (4.0 * PI) + a * hgPdf(dot(rd, d), R.c.z));
}
// 表面照度：E = φ [ (1−a)/4 + a·max(n·d̄, 0) ]
fn surfIrr(g: vec3f, n: vec3f) -> vec3f {
  let uv = g / (2.0 * vec3f(textureDimensions(lvA)));
  let A = textureSampleLevel(lvA, samp, uv, 0.0).rgb;
  let B = textureSampleLevel(lvB, samp, uv, 0.0);
  let a = clamp(length(B.xyz), 0.0, 1.0);
  let d = select(vec3f(0.0, 1.0, 0.0), B.xyz / max(length(B.xyz), 1e-6), a > 1e-4);
  return A * ((1.0 - a) * 0.25 + a * max(dot(n, d), 0.0));
}
fn groundIrr(p: vec3f) -> vec3f {
  let uv = (p.xz - R.gc.xy) / R.gc.z;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec3f(0.0); }
  return textureSampleLevel(gE, samp, vec3f(uv, 0.5), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= R.res.x || gid.y >= R.res.y) { return; }
  let pix = gid.y * R.res.x + gid.x;
  var acc = vec3f(0.0);
  for (var sI = 0u; sI < R.res.w; sI++) {
    let sn = R.res.z + sI;
    rs = pcg(pix * 9781u + sn * 6271u + R.dims.w * 26699u + 17u);
    // 相机：像素抖动抗锯齿、快门时刻（运动模糊）、薄透镜景深
    let uv = (vec2f(f32(gid.x) + rnd(), f32(gid.y) + rnd()) / vec2f(f32(R.res.x), f32(R.res.y))) * 2.0 - 1.0;
    let tau = (rnd() - 0.5) * R.a.z;
    var dir = normalize(R.camFwd.xyz + uv.x * R.camPos.w * R.camFwd.w * R.camRight.xyz - uv.y * R.camPos.w * R.camUp.xyz);
    var ro = R.camPos.xyz;
    if (R.camRight.w > 0.0) {
      let fp = ro + dir * (R.camUp.w / dot(dir, R.camFwd.xyz));
      let a = 2.0 * PI * rnd();
      let rr = R.camRight.w * sqrt(rnd());
      ro += (R.camRight.xyz * cos(a) + R.camUp.xyz * sin(a)) * rr;
      dir = normalize(fp - ro);
    }
    var L = vec3f(0.0);
    if (R.gc.w > 0.5) {
      // 光照缓存模式：一条相机射线 = 自发光 + 缓存的火光单次散射 + 表面（缓存照度）
      let m = march(ro, dir, false, tau, 0.0);
      L = m.L;
      let hp = ro + dir * m.hit.t;
      if (m.hit.kind == 1u) {
        let salb = mix(albedoOf(m.hit.mat), 0.035, m.hit.chr);
        L += m.Tr * (lutAt(1024u, m.hit.Ts) * 0.9 + salb / PI * surfIrr(hp / R.a.x + m.hit.n * 1.5, m.hit.n));
      } else if (m.hit.kind == 2u) {
        L += m.Tr * R.b.y / PI * (groundIrr(hp) + PI * R.a.w * 0.6);
      } else if (m.hit.kind == 0u) {
        L += m.Tr * skyL(dir);
      }
    } else {
      var thr = vec3f(1.0);
      let nb = u32(R.b.z);
      var misPdf = 0.0;
      for (var bo = 0u; bo <= nb; bo++) {
        let m = march(ro, dir, bo < nb, tau, misPdf);
        L += thr * m.L;
        var salb = vec3f(0.0);
        if (m.hit.kind == 1u) {
          // 固体表面：自身热辐射（发红的炭/燃烧表面）
          L += thr * m.Tr * lutAt(1024u, m.hit.Ts) * 0.9;
          salb = vec3f(mix(albedoOf(m.hit.mat), 0.035, m.hit.chr));
        } else if (m.hit.kind == 2u) {
          salb = vec3f(R.b.y);
        } else if (m.hit.kind == 0u) {
          L += thr * m.Tr * skyL(dir);
        }
        if (bo == nb) { break; }
        let surfW = lum(m.Tr * salb);
        let tot = m.W + surfW;
        if (tot < 1e-6) { break; }
        if (rnd() * tot < m.W) {
          thr *= m.selThr * tot;
          ro = m.selP;
          L += thr * neeN(ro, dir, vec3f(0.0), false, tau);
          let nd = hgDir(dir, R.c.z);
          misPdf = hgPdf(dot(dir, nd), R.c.z);
          dir = nd;
        } else {
          thr *= m.Tr * salb * (tot / surfW);
          ro = ro + dir * m.hit.t + m.hit.n * (0.05 * R.a.x);
          L += thr * neeN(ro, dir, m.hit.n, true, tau);
          dir = cosineDir(m.hit.n);
          misPdf = max(dot(dir, m.hit.n), 1e-4) / PI;
        }
        // 俄罗斯轮盘
        if (bo >= 1u) {
          let pc = clamp(max(thr.x, max(thr.y, thr.z)) * 4.0, 0.05, 1.0);
          if (rnd() > pc) { break; }
          thr /= pc;
        }
      }
    }
    // 夹住极端的萤火虫噪点（只影响罕见的高方差路径）
    let ml = max(L.x, max(L.y, L.z));
    if (ml > 2e6) { L *= 2e6 / ml; }
    acc += L;
  }
  accum[pix] += vec4f(acc, f32(R.res.w));
}

//#KERNEL lightvol
// 火光照度缓存（半分辨率）：每个缓存单元向火焰投 N 条阴影射线，记录通量与平均入射方向
//@TRACE
@group(0) @binding(12) var lvAo: texture_storage_3d<rgba16float, write>;
@group(0) @binding(13) var lvBo: texture_storage_3d<rgba16float, write>;
fn inscatter(g: vec3f, rd: vec3f) -> vec3f { return vec3f(0.0); }
const LV_N: u32 = 96u;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let ld = vec3i(textureDimensions(lvAo));
  let c = vec3i(gid);
  if (any(c >= ld)) { return; }
  let g = (vec3f(c) + 0.5) * 2.0;
  // 缓存单元附近（含相邻砖块）没有烟、没有固体就跳过
  let bd = (dimsV() + vec3i(7)) / 8;
  var busy = false;
  for (var k = 0; k < 8; k++) {
    let o = vec3f(f32(k & 1), f32((k >> 1) & 1), f32((k >> 2) & 1)) * 2.0 - 1.0;
    let bc = clamp(vec3i(floor((g + o * 2.0) / 8.0)), vec3i(0), bd - vec3i(1));
    if (bricks[bc.x + bd.x * (bc.y + bd.y * bc.z)] >= 1e-3) { busy = true; }
  }
  if (!busy) {
    textureStore(lvAo, c, vec4f(0.0));
    textureStore(lvBo, c, vec4f(0.0));
    return;
  }
  rs = pcg(u32(c.x + ld.x * (c.y + ld.y * c.z)) * 7919u + R.dims.w * 104729u + 3u);
  let x = g * R.a.x;
  var F = vec3f(0.0);
  var D = vec3f(0.0);
  var S = 0.0;
  for (var i = 0u; i < LV_N; i++) {
    let s = neeSample(x, 0.0);
    F += s.c;
    let l = lum(s.c);
    D += l * s.w;
    S += l;
  }
  F /= f32(LV_N);
  textureStore(lvAo, c, vec4f(F, 0.0));
  textureStore(lvBo, c, vec4f(D / max(S, 1e-12), 0.0));
}

//#KERNEL lightground
// 地面照度缓存：E = ∫ L cosθ dω
//@TRACE
@group(0) @binding(14) var gEo: texture_storage_3d<rgba16float, write>;
fn inscatter(g: vec3f, rd: vec3f) -> vec3f { return vec3f(0.0); }
const GR_N: u32 = 128u;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let gd = textureDimensions(gEo).xy;
  if (gid.x >= gd.x || gid.y >= gd.y) { return; }
  rs = pcg((gid.x + gd.x * gid.y) * 7919u + R.dims.w * 104729u + 5u);
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(gd);
  let x = vec3f(R.gc.x + uv.x * R.gc.z, 0.002, R.gc.y + uv.y * R.gc.z);
  var E = vec3f(0.0);
  for (var i = 0u; i < GR_N; i++) {
    let s = neeSample(x, 0.0);
    E += s.c * max(s.w.y, 0.0);
  }
  textureStore(gEo, vec3u(gid.xy, 0u), vec4f(E / f32(GR_N), 0.0));
}

//#KERNEL blur
// 可分离高斯模糊（光照缓存去噪）：axis.x = 轴 0/1/2，axis.y = 半径
struct BA { axis: vec4u }
@group(0) @binding(0) var<uniform> BX: BA;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = vec3i(textureDimensions(src));
  let c = vec3i(gid);
  if (any(c >= d)) { return; }
  let ax = BX.axis.x;
  let e = vec3i(select(0, 1, ax == 0u), select(0, 1, ax == 1u), select(0, 1, ax == 2u));
  let r = i32(BX.axis.y);
  let s2 = 0.5 * f32(r * r) + 0.5;
  var acc = vec4f(0.0);
  var ws = 0.0;
  for (var k = -r; k <= r; k++) {
    let w = exp(-f32(k * k) / s2);
    acc += w * textureLoad(src, clamp(c + e * k, vec3i(0), d - vec3i(1)), 0);
    ws += w;
  }
  textureStore(dst, c, acc / ws);
}
