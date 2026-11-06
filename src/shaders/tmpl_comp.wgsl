// 交错速度分量 XBUF 的三线性采样（g：该分量节点为整数点的坐标；hi：索引上界）
fn triC_XBUF(g0: vec3f, hi: vec3i) -> f32 {
  let g = clamp(g0, vec3f(0.0), vec3f(hi));
  let b = min(vec3i(floor(g)), hi - vec3i(1));
  let f = g - vec3f(b);
  let d = vdimsI();
  let sy = u32(d.x);
  let sz = u32(d.x * d.y);
  let i = vidx(b);
  let c00 = mix(XBUF[i], XBUF[i + 1u], f.x);
  let c10 = mix(XBUF[i + sy], XBUF[i + sy + 1u], f.x);
  let c01 = mix(XBUF[i + sz], XBUF[i + sz + 1u], f.x);
  let c11 = mix(XBUF[i + sy + sz], XBUF[i + sy + sz + 1u], f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
fn mmC_XBUF(g0: vec3f, hi: vec3i) -> vec2f {
  let g = clamp(g0, vec3f(0.0), vec3f(hi));
  let b = min(vec3i(floor(g)), hi - vec3i(1));
  var r = vec2f(1e30, -1e30);
  for (var j = 0; j < 8; j++) {
    let v = XBUF[vidx(b + vec3i(j & 1, (j >> 1) & 1, (j >> 2) & 1))];
    r = vec2f(min(r.x, v), max(r.y, v));
  }
  return r;
}
