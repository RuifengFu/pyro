// 统计量：工作组内归约，再用 CAS 循环做全局浮点原子加
// 0 放热率 W | 1 辐射损失 W | 2 热解速率 kg/s | 3 碳烟质量 kg | 4 最高温度 K | 5 最大速度 m/s
// 6 燃料气质量 kg | 7 固体剩余质量 kg | 8 白烟质量 kg | 9 正在热解的表面积 m² | 16.. 每层放热率 W
@group(0) @binding(STATSB) var<storage, read_write> stats: array<atomic<u32>>;
var<workgroup> red: array<f32, 64>;
fn sadd(i: u32, v: f32) {
  var old = atomicLoad(&stats[i]);
  loop {
    let r = atomicCompareExchangeWeak(&stats[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
fn smax(i: u32, v: f32) { atomicMax(&stats[i], bitcast<u32>(max(v, 0.0))); }
fn wsum(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn wmax(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
