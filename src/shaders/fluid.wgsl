// ===== 涡量、外力、散度、投影 =====

//#KERNEL curl
@group(0) @binding(1) var<storage, read> vu: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vw: array<f32>;
@group(0) @binding(4) var<storage, read_write> curl: array<vec4f>;
//@STATS 5
fn uc(c0: vec3i) -> vec3f {
  let c = clampC(c0);
  let X = vec3i(1, 0, 0); let Y = vec3i(0, 1, 0); let Z = vec3i(0, 0, 1);
  return 0.5 * vec3f(vu[vidx(c)] + vu[vidx(c + X)], vv[vidx(c)] + vv[vidx(c + Y)], vw[vidx(c)] + vw[vidx(c + Z)]);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let c = vec3i(gid);
  var spd = 0.0;
  if (inb(c)) {
    let X = vec3i(1, 0, 0); let Y = vec3i(0, 1, 0); let Z = vec3i(0, 0, 1);
    let w = vec3f(
      (uc(c + Y).z - uc(c - Y).z) - (uc(c + Z).y - uc(c - Z).y),
      (uc(c + Z).x - uc(c - Z).x) - (uc(c + X).z - uc(c - X).z),
      (uc(c + X).y - uc(c - X).y) - (uc(c + Y).x - uc(c - Y).x)) / (2.0 * P.dx);
    curl[sidx(c)] = vec4f(w, length(w));
    spd = length(uc(c));
  }
  let m = wmax(li, spd);
  if (li == 0u) { smax(5u, m); }
}

//#KERNEL forces
@group(0) @binding(1) var<storage, read_write> vu: array<f32>;
@group(0) @binding(2) var<storage, read_write> vv: array<f32>;
@group(0) @binding(3) var<storage, read_write> vw: array<f32>;
@group(0) @binding(4) var<storage, read> s0: array<vec4f>;
@group(0) @binding(5) var<storage, read> s1: array<vec4f>;
@group(0) @binding(6) var<storage, read> curl: array<vec4f>;
@group(0) @binding(7) var<storage, read> flags: array<u32>;
fn solidAt(c: vec3i) -> bool { return inb(c) && flags[sidx(c)] == 1u; }
fn cm(c: vec3i) -> f32 { return curl[sidx(clampC(c))].w; }
// 单元中心的体积力：涡量约束（把数值耗散掉的小涡找回来，只在热气和烟里施加）
fn confine(c: vec3i) -> vec3f {
  let i = sidx(c);
  let T = s0[i].x;
  let smoke = s0[i].w + s1[i].x;
  let gate = max(smoothstep(P.vort_t, P.vort_t + 500.0, T), 0.35 * smoothstep(0.0, 2e-4, smoke));
  if (gate <= 0.0) { return vec3f(0.0); }
  let X = vec3i(1, 0, 0); let Y = vec3i(0, 1, 0); let Z = vec3i(0, 0, 1);
  let gr = vec3f(cm(c + X) - cm(c - X), cm(c + Y) - cm(c - Y), cm(c + Z) - cm(c - Z));
  let gl = length(gr);
  if (gl < 1e-6) { return vec3f(0.0); }
  var f = P.vort * P.dx * cross(gr / gl, curl[i].xyz) * gate;
  let fl = length(f);
  if (fl > 60.0) { f *= 60.0 / fl; }
  return f;
}
// 环境风（侧面吸收层强约束，内部弱约束）
fn windVel(k: u32) -> f32 {
  if (k == 0u) { return P.wind_x; }
  if (k == 2u) { return P.wind_z; }
  return 0.0;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  let i = vidx(c);
  for (var k = 0u; k < 3u; k++) {
    if (!compValid(k, c)) { continue; }
    let a = c - unitI(k);
    if ((k == 1u && c.y == 0) || solidAt(c) || solidAt(a)) { continue; }
    let ia = inb(a);
    let ib = inb(c);
    var f = 0.0;
    // 浮力：理想气体 (ρ0−ρ)/ρ = (T−T0)/T，减去烟颗粒自重
    if (k == 1u) {
      var Ta = P.t0; var Tb = P.t0; var ma = 0.0; var mb = 0.0;
      if (ia) { let s = s0[sidx(a)]; Ta = s.x; ma = s.w + s1[sidx(a)].x; }
      if (ib) { let s = s0[sidx(c)]; Tb = s.x; mb = s.w + s1[sidx(c)].x; }
      if (!ia) { Ta = Tb; ma = mb; }
      if (!ib) { Tb = Ta; mb = ma; }
      let Tf = 0.5 * (Ta + Tb);
      f += P.g * P.buoy * (Tf - P.t0) / Tf - P.g * P.smoke_w * 0.5 * (ma + mb);
    }
    var fc = vec3f(0.0);
    var n = 0.0;
    if (ia) { fc += confine(a); n += 1.0; }
    if (ib) { fc += confine(c); n += 1.0; }
    if (n > 0.0) { f += fc[k] / n; }
    var u = 0.0;
    if (k == 0u) { u = vu[i]; } else if (k == 1u) { u = vv[i]; } else { u = vw[i]; }
    u += f * P.dt;
    // 风与吸收层
    let cc = clampC(c);
    let ss = spongeSide(cc);
    if (k != 1u) {
      var Tl = P.t0;
      if (ib) { Tl = s0[sidx(c)].x; }
      let hot = smoothstep(350.0, 900.0, Tl);
      let kr = P.wind_relax * (1.0 - 0.85 * hot) + 8.0 * ss;
      u = mix(u, windVel(k), 1.0 - exp(-kr * P.dt));
    } else {
      // 顶部出流层：禁止向下回流
      let st = spongeTop(cc);
      if (st > 0.0) { u = mix(u, max(u, 0.0), 1.0 - exp(-8.0 * st * P.dt)); }
    }
    u = clamp(u, -40.0, 40.0);
    if (k == 0u) { vu[i] = u; } else if (k == 1u) { vv[i] = u; } else { vw[i] = u; }
  }
}

//#KERNEL divergence
@group(0) @binding(1) var<storage, read> vu: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vw: array<f32>;
@group(0) @binding(4) var<storage, read> s1: array<vec4f>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!inb(c)) { return; }
  let i = sidx(c);
  if (flags[i] == 1u) { b[i] = 0.0; return; }
  let X = vec3i(1, 0, 0); let Y = vec3i(0, 1, 0); let Z = vec3i(0, 0, 1);
  let d = (vu[vidx(c + X)] - vu[vidx(c)] + vv[vidx(c + Y)] - vv[vidx(c)] + vw[vidx(c + Z)] - vw[vidx(c)]) / P.dx;
  // 低马赫数：∇·u = S（燃烧放热膨胀 + 热解气注入）
  b[i] = d - s1[i].z;
}

//#KERNEL project
@group(0) @binding(1) var<storage, read_write> vu: array<f32>;
@group(0) @binding(2) var<storage, read_write> vv: array<f32>;
@group(0) @binding(3) var<storage, read_write> vw: array<f32>;
@group(0) @binding(4) var<storage, read> phi: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
fn solidAt(c: vec3i) -> bool { return inb(c) && flags[sidx(c)] == 1u; }
fn ph(c: vec3i) -> f32 { if (inb(c)) { return phi[sidx(c)]; } return 0.0; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  let i = vidx(c);
  for (var k = 0u; k < 3u; k++) {
    if (!compValid(k, c)) { continue; }
    let a = c - unitI(k);
    var u = 0.0;
    if (!((k == 1u && c.y == 0) || solidAt(c) || solidAt(a))) {
      if (k == 0u) { u = vu[i]; } else if (k == 1u) { u = vv[i]; } else { u = vw[i]; }
      u -= (ph(c) - ph(a)) / P.dx;
    }
    if (k == 0u) { vu[i] = u; } else if (k == 1u) { vv[i] = u; } else { vw[i] = u; }
  }
}
