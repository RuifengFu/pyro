// 单元中心 vec4 场 XBUF 的三线性采样（gc：以单元中心为整数点的网格坐标）
fn triS_XBUF(gc0: vec3f) -> vec4f {
  let d = dimsI();
  let g = clamp(gc0, vec3f(0.0), vec3f(d - vec3i(1)));
  let b = min(vec3i(floor(g)), d - vec3i(2));
  let f = g - vec3f(b);
  let sy = u32(d.x);
  let sz = u32(d.x * d.y);
  let i = sidx(b);
  let c00 = mix(XBUF[i], XBUF[i + 1u], f.x);
  let c10 = mix(XBUF[i + sy], XBUF[i + sy + 1u], f.x);
  let c01 = mix(XBUF[i + sz], XBUF[i + sz + 1u], f.x);
  let c11 = mix(XBUF[i + sy + sz], XBUF[i + sy + sz + 1u], f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
// 开放边界外取环境值
fn advS_XBUF(gc: vec3f, amb: vec4f) -> vec4f {
  return mix(triS_XBUF(gc), amb, clamp(outside(gc), 0.0, 1.0));
}
fn mmS_XBUF(gc0: vec3f) -> MM {
  let d = dimsI();
  let g = clamp(gc0, vec3f(0.0), vec3f(d - vec3i(1)));
  let b = min(vec3i(floor(g)), d - vec3i(2));
  var r: MM;
  r.lo = vec4f(1e30);
  r.hi = vec4f(-1e30);
  for (var j = 0; j < 8; j++) {
    let v = XBUF[sidx(b + vec3i(j & 1, (j >> 1) & 1, (j >> 2) & 1))];
    r.lo = min(r.lo, v);
    r.hi = max(r.hi, v);
  }
  return r;
}
