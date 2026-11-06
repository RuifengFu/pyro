// ===== 固体热解与燃烧化学 =====

//#KERNEL solid
@group(0) @binding(1) var<storage, read> solIn: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> solOut: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> pyr: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read> s0: array<vec4f>;
//@STATS 6
fn kindOf(s: vec4f) -> u32 { return u32(P.mats[u32(s.x)].a.w + 0.5); }
fn isSolidS(s: vec4f) -> bool {
  if (kindOf(s) == 2u) { return s.y > 0.5; }
  return s.y + s.z > P.burnout;
}
fn solidN(c: vec3i) -> bool { if (!inb(c)) { return false; } return isSolidS(solIn[sidx(c)]); }

// 沿一条射线做简化辐射传输（碳烟与燃烧产物按基尔霍夫定律既发射又吸收；碰到固体取其表面辐射）。返回入射辐照度 π·L
fn rayIrr(c0: vec3i, d: vec3i) -> f32 {
  var p = c0;
  var tr = 1.0;
  var acc = 0.0;
  let dl = length(vec3f(d)) * P.dx;
  var strides = array<i32, 10>(1, 1, 1, 2, 2, 3, 4, 5, 7, 9);
  for (var s = 0; s < 10; s++) {
    if (!inb(p)) { break; }
    let j = sidx(p);
    let so = solIn[j];
    if (isSolidS(so)) { acc += tr * pow4(so.w); tr = 0.0; break; }
    let g = s0[j];
    let T = g.x;
    let len = f32(strides[s]) * dl;
    let e = 1.0 - exp(-kapP(T, g.z, g.w) * len);
    acc += tr * e * pow4(T);
    tr *= 1.0 - e;
    p += d * strides[s];
  }
  acc += tr * pow4(P.t0);
  return SIGMA * acc;
}
// 半球辐照度：法向 + 4 条 45° 斜射线
fn faceIrr(c0: vec3i, k: i32) -> f32 {
  let d = DIRS[k];
  var t1 = vec3i(0, 1, 0);
  var t2 = vec3i(0, 0, 1);
  if (d.y != 0) { t1 = vec3i(1, 0, 0); }
  if (d.z != 0) { t2 = vec3i(1, 0, 0); }
  return 0.2 * (rayIrr(c0, d) + rayIrr(c0, d + t1) + rayIrr(c0, d - t1) + rayIrr(c0, d + t2) + rayIrr(c0, d - t2));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let c = vec3i(gid);
  var stPyro = 0.0; var stMass = 0.0; var stArea = 0.0;
  if (inb(c)) {
    let i = sidx(c);
    let s = solIn[i];
    var o = s;
    var po = vec4f(0.0);
    var fl = 0u;
    if (isSolidS(s)) {
      fl = 1u;
      let m = P.mats[u32(s.x)];
      let kind = kindOf(s);
      var fuel = s.y; var chr = s.z; var Ts = s.w;
      if (kind != 2u) {
        var nF = 0.0; var Tg = 0.0; var Yo = 0.0; var qin = 0.0; var qcond = 0.0;
        for (var k = 0; k < 6; k++) {
          let n = c + DIRS[k];
          if (n.y < 0) { continue; }
          if (solidN(n)) { qcond += m.b.w / P.dx * (solIn[sidx(n)].w - Ts); continue; }
          nF += 1.0;
          if (inb(n)) {
            let g = s0[sidx(n)];
            Tg += g.x; Yo += g.z;
            qin += faceIrr(n, k);
          } else {
            Tg += P.t0; Yo += P.yo2;
            qin += SIGMA * pow4(P.t0);
          }
        }
        // 液体只有表面约 1 mm 被加热（对流把热量带走得慢，表层很快到沸点）
        let layer = select(min(P.layer, P.dx), 0.001, kind == 1u);
        let cap = m.a.x * m.b.z * layer;
        if (nF > 0.0) {
          Tg /= nF; Yo /= nF; qin /= nF;
          var qnet = m.b.y * (qin - SIGMA * pow4(Ts)) + P.h_conv * (Tg - Ts) + qcond / nF;
          if (P.ign_on > 0.5) {
            let dI = length((vec3f(c) + 0.5) * P.dx - vec3f(P.ign_x, P.ign_y, P.ign_z));
            qnet += P.ign_flux * clamp(1.0 - dI / (1.5 * P.ign_r), 0.0, 1.0);
          }
          var mv = 0.0;
          var mc = 0.0;
          if (kind == 1u) {
            // 液池：升温到沸点后，净热流全部用于蒸发
            let Tb = m.a.z;
            // 易挥发液体在沸点以下就大量蒸发（庚烷闪点 −4 °C）：从沸点前 60 K 开始逐渐转为蒸发
            mv = smoothstep(Tb - 60.0, Tb, Ts) * max(qnet, 0.0) / m.a.y * m.c.y;
            Ts = min(Ts + P.dt * max(qnet - mv * m.a.y, min(qnet, 0.0)) / cap, Tb);
            fuel -= mv * nF / P.dx * P.dt;
          } else {
            // 固体：表面达到热解温度后，净热流按气化热 L_g 转为挥发分（Tewarson 稳态燃烧模型）
            if (fuel > 0.0) {
              mv = smoothstep(m.a.z - 20.0, m.a.z + 30.0, Ts) * max(qnet, 0.0) / m.a.y * m.c.y;
            }
            let chi = select(0.0, m.b.x, kind == 0u);
            // 焦炭氧化：动力学与氧扩散取小者；一半热量留在固体（发红的炭）
            let kin = P.char_a * exp(-P.char_e / (RGAS * Ts)) * chr * layer;
            let dif = P.h_mass * rhoOf(Tg) * Yo / 1.33;
            mc = min(kin, dif);
            Ts += P.dt * (qnet - mv * m.a.y + 0.5 * mc * P.dh_char) / cap;
            let virgin = min(mv / (1.0 - chi) * nF / P.dx * P.dt, fuel);
            fuel -= virgin;
            chr += virgin * chi - mc * nF / P.dx * P.dt;
          }
          stPyro = mv * nF * P.dx * P.dx;
          stArea = select(0.0, nF * P.dx * P.dx, mv > 1e-4);
          po = vec4f(mv, mc, nF, 0.0);
        } else {
          Ts += P.dt * qcond / (m.a.x * m.b.z * P.dx);
        }
        Ts = clamp(Ts, 250.0, 2000.0);
        fuel = max(fuel, 0.0);
        chr = max(chr, 0.0);
        o = vec4f(s.x, fuel, chr, Ts);
        if (fuel + chr <= P.burnout) { o = vec4f(0.0, 0.0, 0.0, P.t0); fl = 0u; po = vec4f(0.0); }
      }
      stMass = (o.y + o.z) * P.dx * P.dx * P.dx * select(1.0, 0.0, kind == 2u);
    }
    solOut[i] = o;
    pyr[i] = po;
    flags[i] = fl;
  }
  let a = wsum(li, stPyro);
  let b = wsum(li, stMass);
  let d = wsum(li, stArea);
  if (li == 0u) { sadd(2u, a); sadd(7u, b); sadd(9u, d); }
}

//#KERNEL react
@group(0) @binding(1) var<storage, read> s0in: array<vec4f>;
@group(0) @binding(2) var<storage, read> s1in: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> s0out: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> s1out: array<vec4f>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read> sol: array<vec4f>;
@group(0) @binding(7) var<storage, read> pyr: array<vec4f>;
@group(0) @binding(8) var<storage, read> curl: array<vec4f>;
//@STATS 9
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let c = vec3i(gid);
  var stH = 0.0; var stR = 0.0; var stS = 0.0; var stF = 0.0; var stM = 0.0; var stT = 0.0;
  let dV = P.dx * P.dx * P.dx;
  if (inb(c)) {
    let i = sidx(c);
    if (flags[i] == 1u) {
      // 固体内部：气相温度取表面温度，避免平流时把冷值吸进来
      s0out[i] = vec4f(sol[i].w, 0.0, P.yo2, 0.0);
      s1out[i] = vec4f(0.0);
    } else {
      var a = s0in[i];
      var b = s1in[i];
      // 亚格子湍流混合（Smagorinsky，|S| ≈ |ω|，Sc_t = 0.5）
      let nut = (P.cs * P.dx) * (P.cs * P.dx) * curl[i].w;
      if (P.cs > 0.0) {
        let kd = min(nut / 0.5 * P.dt / (P.dx * P.dx), 0.15);
        var lapA = vec4f(0.0);
        var lapB = 0.0;
        for (var k = 0; k < 6; k++) {
          let n = c + DIRS[k];
          if (inb(n)) {
            let j = sidx(n);
            if (flags[j] != 1u) { lapA += s0in[j] - a; lapB += s1in[j].x - b.x; }
          } else if (n.y >= 0) {
            lapA += ambient0() - a; lapB -= b.x;
          }
        }
        a += kd * lapA;
        b.x += kd * lapB;
      }
      var T = a.x;
      var yf = max(a.y, 0.0);
      var yo = clamp(a.z, 0.0, P.yo2);
      var ys = max(a.w, 0.0);
      var ym = max(b.x, 0.0);
      var S = 0.0;
      // 相邻固体面：热解气注入（按质量混合稀释其它组分）、对流换热、焦炭氧化耗氧与放热
      for (var k = 0; k < 6; k++) {
        let n = c + DIRS[k];
        if (!inb(n)) { continue; }
        let j = sidx(n);
        if (flags[j] != 1u) { continue; }
        let pr = pyr[j];
        let Ts = sol[j].w;
        let rho = rhoOf(T);
        if (pr.x > 0.0) {
          let dm = pr.x * P.dt / (rho * P.dx);
          let inv = 1.0 / (1.0 + dm);
          yf = (yf + dm) * inv; yo *= inv; ys *= inv; ym *= inv;
          T = (T + dm * Ts) * inv;
          S += pr.x / (rho * P.dx);
        }
        T += P.h_conv * (Ts - T) * P.dt / (rho * cpOf(T) * P.dx);
        if (pr.y > 0.0) {
          yo = max(yo - 1.33 * pr.y * P.dt / (rho * P.dx), 0.0);
          T += 0.5 * P.dh_char * pr.y * P.dt / (rho * cpOf(T) * P.dx);
        }
      }
      // 点火源：维持高温并注入少量燃料（引燃火焰）
      if (P.ign_on > 0.5) {
        let dI = length((vec3f(c) + 0.5) * P.dx - vec3f(P.ign_x, P.ign_y, P.ign_z));
        let w = clamp(1.0 - dI / P.ign_r, 0.0, 1.0);
        if (w > 0.0) {
          T = max(T, mix(P.t0, P.ign_temp, smoothstep(0.0, 0.5, w)));
          yf += P.ign_fuel * w * P.dt;
        }
      }
      // 涡耗散燃烧：混合时间取亚格子湍流扩散时间与浮力时间的小者（同 FDS），带点火温度门限
      var rho = rhoOf(T);
      // τ_u = C_u Δ / √(2k_sgs/3)，k_sgs 取 Deardorff 关系 (ν_t / (C_ν Δ))²，C_u = 0.4，C_ν = 0.1 → τ_u ≈ 0.049 Δ²/ν_t
      let tauU = 0.049 * P.dx * P.dx / (nut + 1e-6);
      let tauG = sqrt(2.0 * P.dx / P.g);
      let tau = max(min(tauU, tauG), P.tau_min);
      let gate = smoothstep(P.t_ign - 100.0, P.t_ign + 100.0, T);
      let lim = min(yf, yo / P.s_o2);
      let dY = lim * (1.0 - exp(-gate * P.dt / tau));
      yf -= dY;
      yo = max(yo - P.s_o2 * dY, 0.0);
      T += dY * P.dhc / cpOf(T + 0.5 * dY * P.dhc / cpOf(T));
      let q = rho * dY * P.dhc / P.dt;
      // 碳烟：在富燃区随燃烧生成，在高温富氧区被氧化；没烧掉的就成了黑烟
      let rich = smoothstep(0.0, 0.03, yf);
      ys += P.soot_form * dY * (0.25 + 0.75 * rich);
      let oxg = smoothstep(P.soot_ox_t, P.soot_ox_t + 300.0, T) * (yo / P.yo2);
      ys *= exp(-P.soot_ox * oxg * P.dt);
      // 没烧掉的热解气冷却后凝结成白色气溶胶
      let cold = 1.0 - smoothstep(450.0, 650.0, T);
      let cv = yf * min(P.smoke_conv * cold * P.dt, 1.0);
      yf -= cv;
      ym += cv;
      // 辐射损失：燃烧区按辐射分数 χr 扣除（与 FDS 默认做法一致，网格分辨不出火焰面时光学薄近似会高估）；
      // 燃烧区以外的热烟按碳烟/产物的光学薄辐射冷却（× 逃逸因子修正自吸收）
      rho = rhoOf(T);
      let kap = kapP(T, yo, ys);
      let tl = kap * P.rad_len;
      let esc = select(1.0, (1.0 - exp(-tl)) / tl, tl > 1e-3);
      // 注：曾试过燃烧区按 χr·q̇ 扣除（FDS 做法），但液池的热反馈随之失衡、点火源撤去后会熄灭；
      // 目前保留光学薄 × 逃逸因子（燃烧速率与火焰高度已对上实验），辐射分数偏高是已知局限
      let prad = 4.0 * kap * esc * SIGMA * (pow4(T) - pow4(P.t0));
      let dTr = min(prad * P.dt / (rho * cpOf(T)), 0.5 * max(T - P.t0, 0.0));
      T -= dTr;
      let pr = dTr * rho * cpOf(T) / P.dt;
      S += (q - pr) / (rho * cpOf(T) * T);
      // 开放边界吸收层
      let sp = max(spongeSide(c), spongeTop(c));
      let dec = 1.0 - exp(-P.dt * 4.0 * sp);
      T = mix(T, P.t0, dec); yf *= 1.0 - dec; yo = mix(yo, P.yo2, dec); ys *= 1.0 - dec; ym *= 1.0 - dec;
      T = clamp(T, 250.0, 2600.0);
      s0out[i] = vec4f(T, yf, yo, ys);
      s1out[i] = vec4f(ym, q, S, 0.0);
      rho = rhoOf(T);
      stH = q * dV; stR = pr * dV; stS = rho * ys * dV; stF = rho * yf * dV; stM = rho * ym * dV; stT = T;
      if (stH > 1.0) { sadd(16u + u32(c.y), stH); }
    }
  }
  let h = wsum(li, stH);
  let r = wsum(li, stR);
  let s = wsum(li, stS);
  let f = wsum(li, stF);
  let m = wsum(li, stM);
  let t = wmax(li, stT);
  if (li == 0u) { sadd(0u, h); sadd(1u, r); sadd(3u, s); sadd(6u, f); sadd(8u, m); smax(4u, t); }
}
