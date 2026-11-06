// ===== 公共定义：参数、索引、物性 =====
// 坐标约定：网格单位（1 = 一个单元边长 dx），单元 c 的中心在 c + 0.5；y 向上，地面 y = 0。
// 交错网格（MAC）：u 在 (i, j+½, k+½)，v 在 (i+½, j, k+½)，w 在 (i+½, j+½, k)；
// 三个速度分量都按 (nx+1)(ny+1)(nz+1) 的补齐尺寸存储，共用 vidx。
// 标量 S0 = (T [K], Y_F 燃料气质量分数, Y_O2, Y_s 碳烟)，S1 = (Y_m 白色气溶胶, q̇ [W/m³], S = ∇·u 源项 [1/s], 0)
// 固体 SOL = (材料号, 原始可燃质量 [kg/m³], 炭 [kg/m³], 表面温度 Ts [K])
// 热解 PYR = (挥发分质量通量 [kg/m²s], 炭氧化通量 [kg/m²s], 暴露面数, 0)

struct Material { a: vec4f, b: vec4f, c: vec4f }
// a = (密度, 气化热 L_g, 热解/沸腾温度 T_p, 类型 0 成炭 1 液池 2 惰性 3 不成炭)
// b = (成炭率, 发射率, 比热 c, 导热系数 k)
// c = (白烟产率, 燃烧速率倍率, 0, 0)

struct Params {
  nx: u32, ny: u32, nz: u32, frame: u32,
  dx: f32, dt: f32, time: f32, t0: f32,
  rho0: f32, g: f32, cp: f32, yo2: f32,
  vort: f32, cs: f32, buoy: f32, smoke_w: f32,
  wind_x: f32, wind_z: f32, wind_relax: f32, sponge: f32,
  t_ign: f32, s_o2: f32, dhc: f32, tau_min: f32,
  soot_form: f32, soot_ox: f32, soot_ox_t: f32, smoke_conv: f32,
  kap_gas: f32, h_conv: f32, ign_on: f32, ign_temp: f32,
  ign_x: f32, ign_y: f32, ign_z: f32, ign_r: f32,
  ign_flux: f32, ign_fuel: f32, char_a: f32, char_e: f32,
  dh_char: f32, burnout: f32, layer: f32, rad_len: f32,
  h_mass: f32, vort_t: f32, rad_frac: f32, seed: f32,
  mats: array<Material, 8>,
}
@group(0) @binding(0) var<uniform> P: Params;

const SIGMA: f32 = 5.670374e-8;
const RGAS: f32 = 8.314;
const RHO_SOOT: f32 = 1800.0;
var<private> DIRS: array<vec3i, 6> = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));

struct MM { lo: vec4f, hi: vec4f }

fn dimsI() -> vec3i { return vec3i(i32(P.nx), i32(P.ny), i32(P.nz)); }
fn vdimsI() -> vec3i { return dimsI() + vec3i(1); }
fn inb(c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < dimsI()); }
fn sidx(c: vec3i) -> u32 { let d = dimsI(); return u32(c.x + d.x * (c.y + d.y * c.z)); }
fn vidx(c: vec3i) -> u32 { let d = vdimsI(); return u32(c.x + d.x * (c.y + d.y * c.z)); }
fn clampC(c: vec3i) -> vec3i { return clamp(c, vec3i(0), dimsI() - vec3i(1)); }
fn pow4(x: f32) -> f32 { let y = x * x; return y * y; }
fn rhoOf(T: f32) -> f32 { return P.rho0 * P.t0 / max(T, 200.0); }
// Planck 平均吸收系数 [1/m]：碳烟 κ ≈ 1862·f_v·T，加 CO2/H2O（按燃烧产物比例）
fn kapP(T: f32, yo: f32, ys: f32) -> f32 {
  let fv = rhoOf(T) * max(ys, 0.0) / RHO_SOOT;
  return 1862.0 * fv * T + P.kap_gas * clamp(1.0 - yo / P.yo2, 0.0, 1.0);
}
// 燃烧产物定压比热随温度升高（CO2/H2O 振动自由度激发）：300 K 约 1.0，2000 K 约 1.4 kJ/kgK
fn cpOf(T: f32) -> f32 { return clamp(1000.0 + 0.24 * (T - 300.0), 1000.0, 1450.0); }
fn ambient0() -> vec4f { return vec4f(P.t0, 0.0, P.yo2, 0.0); }
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// 速度分量 k 的交错偏移与有效索引上界
fn compOff(k: u32) -> vec3f {
  if (k == 0u) { return vec3f(0.0, 0.5, 0.5); }
  if (k == 1u) { return vec3f(0.5, 0.0, 0.5); }
  return vec3f(0.5, 0.5, 0.0);
}
fn compHi(k: u32) -> vec3i {
  let d = dimsI() - vec3i(1);
  if (k == 0u) { return d + vec3i(1, 0, 0); }
  if (k == 1u) { return d + vec3i(0, 1, 0); }
  return d + vec3i(0, 0, 1);
}
fn compValid(k: u32, c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c <= compHi(k)); }
fn unitI(k: u32) -> vec3i {
  if (k == 0u) { return vec3i(1, 0, 0); }
  if (k == 1u) { return vec3i(0, 1, 0); }
  return vec3i(0, 0, 1);
}
// 开放边界外的距离（侧面与顶部）；>0 表示在计算域外
fn outside(gc: vec3f) -> f32 {
  let d = vec3f(dimsI());
  return max(max(-0.5 - gc.x, gc.x - (d.x - 0.5)), max(max(-0.5 - gc.z, gc.z - (d.z - 0.5)), gc.y - (d.y - 0.5)));
}
// 侧边与顶部吸收层权重
fn spongeSide(c: vec3i) -> f32 {
  let d = dimsI();
  let e = f32(min(min(c.x, d.x - 1 - c.x), min(c.z, d.z - 1 - c.z)));
  return 1.0 - smoothstep(0.0, P.sponge, e);
}
fn spongeTop(c: vec3i) -> f32 { let h = f32(dimsI().y); return smoothstep(h - 1.5 * P.sponge, h - 1.0, f32(c.y)); }
