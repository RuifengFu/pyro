// ===== 多重网格（V-cycle）压力求解 =====
// 每层：phi、b（右端项）、f（1 = 固体/Neumann）。算子 (Σφ_n − N φ)/h²；地面 Neumann，侧面与顶部 φ = 0。
// 不含 common.wgsl：binding 0 是层参数

//#KERNEL mg_smooth
struct LV { nx: u32, ny: u32, nz: u32, h: f32 }
var<private> DIRS: array<vec3i, 6> = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
fn dimsOf(l: LV) -> vec3i { return vec3i(i32(l.nx), i32(l.ny), i32(l.nz)); }
fn inbOf(c: vec3i, d: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < d); }
fn idxOf(c: vec3i, d: vec3i) -> u32 { return u32(c.x + d.x * (c.y + d.y * c.z)); }
override COLOR: u32 = 0u;
override OMEGA: f32 = 1.0;
@group(0) @binding(0) var<uniform> L: LV;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> f: array<u32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = dimsOf(L);
  let y = i32(gid.y); let z = i32(gid.z);
  let c = vec3i(i32(gid.x) * 2 + ((y + z + i32(COLOR)) & 1), y, z);
  if (!inbOf(c, d)) { return; }
  let i = idxOf(c, d);
  if (f[i] == 1u) { return; }
  var sum = 0.0; var cnt = 0.0;
  for (var k = 0; k < 6; k++) {
    let n = c + DIRS[k];
    if (n.y < 0) { continue; }
    if (!inbOf(n, d)) { cnt += 1.0; continue; }
    let j = idxOf(n, d);
    if (f[j] == 1u) { continue; }
    sum += p[j]; cnt += 1.0;
  }
  if (cnt < 0.5) { return; }
  p[i] = mix(p[i], (sum - L.h * L.h * b[i]) / cnt, OMEGA);
}

//#KERNEL mg_restrict
struct LV { nx: u32, ny: u32, nz: u32, h: f32 }
var<private> DIRS: array<vec3i, 6> = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
fn dimsOf(l: LV) -> vec3i { return vec3i(i32(l.nx), i32(l.ny), i32(l.nz)); }
fn inbOf(c: vec3i, d: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < d); }
fn idxOf(c: vec3i, d: vec3i) -> u32 { return u32(c.x + d.x * (c.y + d.y * c.z)); }
@group(0) @binding(0) var<uniform> LF: LV;
@group(0) @binding(1) var<storage, read> pf: array<f32>;
@group(0) @binding(2) var<storage, read> bf: array<f32>;
@group(0) @binding(3) var<storage, read> ff: array<u32>;
@group(0) @binding(4) var<storage, read_write> bc: array<f32>;
@group(0) @binding(5) var<storage, read_write> pc: array<f32>;
@group(0) @binding(6) var<storage, read_write> fc: array<u32>;
@group(0) @binding(7) var<uniform> LC: LV;
fn resid(c: vec3i, d: vec3i) -> f32 {
  let i = idxOf(c, d);
  if (ff[i] == 1u) { return 0.0; }
  let pcen = pf[i];
  var sum = 0.0; var cnt = 0.0;
  for (var k = 0; k < 6; k++) {
    let n = c + DIRS[k];
    if (n.y < 0) { continue; }
    if (!inbOf(n, d)) { cnt += 1.0; continue; }
    let j = idxOf(n, d);
    if (ff[j] == 1u) { continue; }
    sum += pf[j]; cnt += 1.0;
  }
  return bf[i] - (sum - cnt * pcen) / (LF.h * LF.h);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dc = dimsOf(LC);
  let df = dimsOf(LF);
  let c = vec3i(gid);
  if (!inbOf(c, dc)) { return; }
  var r = 0.0; var nfl = 0.0;
  for (var k = 0; k < 8; k++) {
    let ch = c * 2 + vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    if (!inbOf(ch, df)) { continue; }
    if (ff[idxOf(ch, df)] != 1u) { nfl += 1.0; r += resid(ch, df); }
  }
  let i = idxOf(c, dc);
  bc[i] = select(0.0, r / max(nfl, 1.0), nfl > 0.0);
  pc[i] = 0.0;
  fc[i] = select(1u, 0u, nfl > 0.0);
}

//#KERNEL mg_prolong
struct LV { nx: u32, ny: u32, nz: u32, h: f32 }
fn dimsOf(l: LV) -> vec3i { return vec3i(i32(l.nx), i32(l.ny), i32(l.nz)); }
fn inbOf(c: vec3i, d: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < d); }
fn idxOf(c: vec3i, d: vec3i) -> u32 { return u32(c.x + d.x * (c.y + d.y * c.z)); }
@group(0) @binding(0) var<uniform> LF: LV;
@group(0) @binding(1) var<storage, read> pc: array<f32>;
@group(0) @binding(2) var<storage, read> ff: array<u32>;
@group(0) @binding(3) var<storage, read_write> pf: array<f32>;
@group(0) @binding(4) var<uniform> LC: LV;
@group(0) @binding(5) var<storage, read> fc: array<u32>;
// 三线性插值修正（分段常数插值 + 平均限制不满足 Poisson 方程的收敛阶条件）
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let df = dimsOf(LF);
  let dc = dimsOf(LC);
  let c = vec3i(gid);
  if (!inbOf(c, df)) { return; }
  let i = idxOf(c, df);
  if (ff[i] == 1u) { return; }
  let x = (vec3f(c) + 0.5) * 0.5 - 0.5;
  let i0 = vec3i(floor(x));
  let fr = x - vec3f(i0);
  var acc = 0.0; var ws = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let cc = clamp(i0 + o, vec3i(0), dc - vec3i(1));
    let j = idxOf(cc, dc);
    if (fc[j] == 1u) { continue; }
    let wv = select(vec3f(1.0) - fr, fr, o == vec3i(1));
    let w = wv.x * wv.y * wv.z;
    acc += w * pc[j];
    ws += w;
  }
  if (ws > 1e-4) { pf[i] += acc / ws; }
}

//#KERNEL mg_resid
// 细层残差平方和与右端项平方和（收敛诊断）
struct LV { nx: u32, ny: u32, nz: u32, h: f32 }
var<private> DIRS: array<vec3i, 6> = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
fn dimsOf(l: LV) -> vec3i { return vec3i(i32(l.nx), i32(l.ny), i32(l.nz)); }
fn inbOf(c: vec3i, d: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < d); }
fn idxOf(c: vec3i, d: vec3i) -> u32 { return u32(c.x + d.x * (c.y + d.y * c.z)); }
@group(0) @binding(0) var<uniform> L: LV;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> f: array<u32>;
//@STATS 4
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let d = dimsOf(L);
  let c = vec3i(gid);
  var r2 = 0.0; var b2 = 0.0;
  if (inbOf(c, d)) {
    let i = idxOf(c, d);
    if (f[i] != 1u) {
      var sum = 0.0; var cnt = 0.0;
      for (var k = 0; k < 6; k++) {
        let n = c + DIRS[k];
        if (n.y < 0) { continue; }
        if (!inbOf(n, d)) { cnt += 1.0; continue; }
        let j = idxOf(n, d);
        if (f[j] == 1u) { continue; }
        sum += p[j]; cnt += 1.0;
      }
      let r = b[i] - (sum - cnt * p[i]) / (L.h * L.h);
      r2 = r * r; b2 = b[i] * b[i];
    }
  }
  let a = wsum(li, r2);
  let e = wsum(li, b2);
  if (li == 0u) { sadd(10u, a); sadd(11u, e); }
}
