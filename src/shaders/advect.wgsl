// ===== MacCormack 平流（二阶，带极值限制） =====
// 同一文件里有四个内核，用 //#KERNEL 名 分段，Rust 端按名切分

//#KERNEL adv_vel1
@group(0) @binding(1) var<storage, read> vu: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vw: array<f32>;
@group(0) @binding(4) var<storage, read_write> hu: array<f32>;
@group(0) @binding(5) var<storage, read_write> hv: array<f32>;
@group(0) @binding(6) var<storage, read_write> hw: array<f32>;
//@COMP vu vv vw
//@VEL
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  let i = vidx(c);
  if (compValid(0u, c)) { hu[i] = triC_vu(traceBack(vec3f(c) + compOff(0u)) - compOff(0u), compHi(0u)); }
  if (compValid(1u, c)) { hv[i] = triC_vv(traceBack(vec3f(c) + compOff(1u)) - compOff(1u), compHi(1u)); }
  if (compValid(2u, c)) { hw[i] = triC_vw(traceBack(vec3f(c) + compOff(2u)) - compOff(2u), compHi(2u)); }
}

//#KERNEL adv_vel2
@group(0) @binding(1) var<storage, read> vu: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vw: array<f32>;
@group(0) @binding(4) var<storage, read> hu: array<f32>;
@group(0) @binding(5) var<storage, read> hv: array<f32>;
@group(0) @binding(6) var<storage, read> hw: array<f32>;
@group(0) @binding(7) var<storage, read_write> ou: array<f32>;
@group(0) @binding(8) var<storage, read_write> ov: array<f32>;
@group(0) @binding(9) var<storage, read_write> ow: array<f32>;
@group(0) @binding(10) var<storage, read> flags: array<u32>;
//@COMP vu vv vw hu hv hw
//@VEL
fn solidAt(c: vec3i) -> bool { return inb(c) && flags[sidx(c)] == 1u; }
fn blocked(k: u32, c: vec3i) -> bool {
  if (k == 1u && c.y == 0) { return true; }
  return solidAt(c) || solidAt(c - unitI(k));
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  let i = vidx(c);
  for (var k = 0u; k < 3u; k++) {
    if (!compValid(k, c)) { continue; }
    if (blocked(k, c)) {
      if (k == 0u) { ou[i] = 0.0; } else if (k == 1u) { ov[i] = 0.0; } else { ow[i] = 0.0; }
      continue;
    }
    let off = compOff(k);
    let hi = compHi(k);
    let x = vec3f(c) + off;
    let b = traceBack(x) - off;
    let f = traceFwd(x) - off;
    var r = 0.0;
    if (k == 0u) {
      let mm = mmC_vu(b, hi);
      r = clamp(hu[i] + 0.5 * (vu[i] - triC_hu(f, hi)), mm.x, mm.y);
      ou[i] = r;
    } else if (k == 1u) {
      let mm = mmC_vv(b, hi);
      r = clamp(hv[i] + 0.5 * (vv[i] - triC_hv(f, hi)), mm.x, mm.y);
      ov[i] = r;
    } else {
      let mm = mmC_vw(b, hi);
      r = clamp(hw[i] + 0.5 * (vw[i] - triC_hw(f, hi)), mm.x, mm.y);
      ow[i] = r;
    }
  }
}

//#KERNEL adv_s1
@group(0) @binding(1) var<storage, read> vu: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vw: array<f32>;
@group(0) @binding(4) var<storage, read> s0: array<vec4f>;
@group(0) @binding(5) var<storage, read> s1: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> h0: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> h1: array<vec4f>;
//@COMP vu vv vw
//@SCAL s0 s1
//@VEL
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!inb(c)) { return; }
  let i = sidx(c);
  let b = traceBack(vec3f(c) + 0.5) - 0.5;
  h0[i] = advS_s0(b, ambient0());
  h1[i] = advS_s1(b, vec4f(0.0));
}

//#KERNEL adv_s2
@group(0) @binding(1) var<storage, read> vu: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vw: array<f32>;
@group(0) @binding(4) var<storage, read> s0: array<vec4f>;
@group(0) @binding(5) var<storage, read> s1: array<vec4f>;
@group(0) @binding(6) var<storage, read> h0: array<vec4f>;
@group(0) @binding(7) var<storage, read> h1: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> o0: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> o1: array<vec4f>;
//@COMP vu vv vw
//@SCAL s0 s1 h0 h1
//@VEL
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!inb(c)) { return; }
  let i = sidx(c);
  let x = vec3f(c) + 0.5;
  let b = traceBack(x) - 0.5;
  let f = traceFwd(x) - 0.5;
  var r0 = h0[i];
  var r1 = h1[i];
  // 回溯点在域外（流入）：直接用一阶结果
  if (outside(b) <= 0.0 && outside(f) <= 0.0) {
    let m0 = mmS_s0(b);
    let m1 = mmS_s1(b);
    r0 = clamp(r0 + 0.5 * (s0[i] - triS_h0(f)), m0.lo, m0.hi);
    r1 = clamp(r1 + 0.5 * (s1[i] - triS_h1(f)), m1.lo, m1.hi);
  }
  o0[i] = r0;
  o1[i] = r1;
}
