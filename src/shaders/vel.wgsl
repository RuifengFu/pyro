// 速度场采样与 RK2 回溯（需要 vu/vv/vw 绑定与 triC_vu/vv/vw 模板）
fn velAt(p: vec3f) -> vec3f {
  return vec3f(
    triC_vu(p - compOff(0u), compHi(0u)),
    triC_vv(p - compOff(1u), compHi(1u)),
    triC_vw(p - compOff(2u), compHi(2u)));
}
fn traceBack(x: vec3f) -> vec3f {
  let k = P.dt / P.dx;
  let u1 = velAt(x);
  let u2 = velAt(x - 0.5 * k * u1);
  return x - k * u2;
}
fn traceFwd(x: vec3f) -> vec3f {
  let k = P.dt / P.dx;
  let u1 = velAt(x);
  let u2 = velAt(x + 0.5 * k * u1);
  return x + k * u2;
}
