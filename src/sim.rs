//! GPU 流体-燃烧模拟：资源、每个子步的调度、统计回读

use crate::gpu::{Gpu, Kernel, assemble, b, groups3, split_kernels};
use crate::scene::{Params, Scene};
use anyhow::Result;

const ADVECT: &str = include_str!("shaders/advect.wgsl");
const FLUID: &str = include_str!("shaders/fluid.wgsl");
const COMBUST: &str = include_str!("shaders/combust.wgsl");
const MG: &str = include_str!("shaders/mg.wgsl");

pub const STATS_N: usize = 16 + 1024;
const MG_LEVELS: usize = 4;
const COARSE_ITERS: u32 = 80;

struct Level {
    dims: [u32; 3],
    u: wgpu::Buffer,
    phi: wgpu::Buffer,
    b: wgpu::Buffer,
    flags: wgpu::Buffer,
}

struct Kernels {
    solid: Kernel,
    curl: Kernel,
    adv_vel1: Kernel,
    adv_vel2: Kernel,
    adv_s1: Kernel,
    adv_s2: Kernel,
    react: Kernel,
    forces: Kernel,
    divergence: Kernel,
    project: Kernel,
    smooth: [Kernel; 2],
    coarse: [Kernel; 2],
    restrict: Kernel,
    prolong: Kernel,
    resid: Kernel,
}

pub struct Sim {
    pub p: Params,
    pub dims: [u32; 3],
    pub dx: f32,
    pub ubuf: wgpu::Buffer,
    pub vel: [[wgpu::Buffer; 3]; 2],
    hat: [wgpu::Buffer; 3],
    pub cur: usize,
    pub s0: wgpu::Buffer,
    pub s1: wgpu::Buffer,
    s0b: wgpu::Buffer,
    s1b: wgpu::Buffer,
    h0: wgpu::Buffer,
    h1: wgpu::Buffer,
    pub sol: [wgpu::Buffer; 2],
    pub sol_cur: usize,
    pyr: wgpu::Buffer,
    pub flags: wgpu::Buffer,
    curl: wgpu::Buffer,
    levels: Vec<Level>,
    stats: wgpu::Buffer,
    staging: wgpu::Buffer,
    k: Kernels,
    pub time: f32,
    pub cycles: u32,
    ign_until: f32,
}

/// 单元中心 vec4 场的 CPU 副本
pub type Field4 = Vec<[f32; 4]>;

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub hrr: f32,
    pub rad: f32,
    pub pyro: f32,
    pub soot: f32,
    pub max_t: f32,
    pub max_u: f32,
    /// 剩余可燃固体/液体质量 kg
    pub solid: f32,
    pub area: f32,
    pub resid: f32,
    /// 每层放热率 W
    pub layers: Vec<f32>,
}

impl Stats {
    /// 火焰高度：自下而上累计放热率达到总量 95% 的高度（米）
    pub fn flame_height(&self, dx: f32) -> f32 {
        let tot: f32 = self.layers.iter().sum();
        if tot <= 0.0 {
            return 0.0;
        }
        let mut acc = 0.0;
        for (j, v) in self.layers.iter().enumerate() {
            acc += v;
            if acc >= 0.95 * tot {
                return (j as f32 + 1.0) * dx;
            }
        }
        self.layers.len() as f32 * dx
    }
}

impl Sim {
    pub fn new(gpu: &Gpu, scene: &Scene) -> Result<Sim> {
        let p = scene.params()?;
        let dims = [p.nx, p.ny, p.nz];
        let dx = p.dx;
        let n = (dims[0] * dims[1] * dims[2]) as u64;
        let nv = ((dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)) as u64;
        let f4 = |l: &str| gpu.storage(l, n * 16);
        let fv = |l: &str| gpu.storage(l, nv * 4);
        let vel = [
            [fv("u0"), fv("v0"), fv("w0")],
            [fv("u1"), fv("v1"), fv("w1")],
        ];
        let hat = [fv("hu"), fv("hv"), fv("hw")];
        let s0 = f4("s0");
        let s1 = f4("s1");
        let s0b = f4("s0b");
        let s1b = f4("s1b");
        let h0 = f4("h0");
        let h1 = f4("h1");
        let sol = [f4("solA"), f4("solB")];
        let pyr = f4("pyr");
        let curl = f4("curl");
        let flags = gpu.storage("flags", n * 4);
        let ubuf = gpu.uniform("params", std::mem::size_of::<Params>() as u64);

        // 初始场：环境空气 + 体素化的固体
        let amb: Vec<[f32; 4]> = vec![[p.t0, 0.0, p.yo2, 0.0]; n as usize];
        gpu.queue.write_buffer(&s0, 0, bytemuck::cast_slice(&amb));
        let solv = scene.voxelize()?;
        gpu.queue
            .write_buffer(&sol[0], 0, bytemuck::cast_slice(&solv));
        gpu.queue
            .write_buffer(&sol[1], 0, bytemuck::cast_slice(&solv));

        // 多重网格层级
        let mut levels = Vec::new();
        let mut d = dims;
        let mut h = dx;
        for l in 0..MG_LEVELS {
            let cnt = (d[0] * d[1] * d[2]) as u64;
            let u = gpu.uniform("lv", 16);
            let lv: [u32; 4] = [d[0], d[1], d[2], h.to_bits()];
            gpu.queue.write_buffer(&u, 0, bytemuck::cast_slice(&lv));
            let flags_l = if l == 0 {
                flags.clone()
            } else {
                gpu.storage("mgf", cnt * 4)
            };
            levels.push(Level {
                dims: d,
                u,
                phi: gpu.storage("phi", cnt * 4),
                b: gpu.storage("b", cnt * 4),
                flags: flags_l,
            });
            if d.iter().min().copied().unwrap_or(0) < 16 {
                break;
            }
            d = d.map(|v| v.div_ceil(2));
            h *= 2.0;
        }

        let stats = gpu.storage("stats", (STATS_N * 4) as u64);
        let staging = gpu.readback("statsR", (STATS_N * 4) as u64);

        let mut src = split_kernels(ADVECT);
        src.extend(split_kernels(FLUID));
        src.extend(split_kernels(COMBUST));
        src.extend(split_kernels(MG));
        let mk = |name: &str, consts: &[(&str, f64)]| -> Kernel {
            let body = src.get(name).unwrap_or_else(|| panic!("缺少内核 {name}"));
            gpu.kernel(name, &assemble(body, !name.starts_with("mg_")), consts)
        };
        let k = Kernels {
            solid: mk("solid", &[]),
            curl: mk("curl", &[]),
            adv_vel1: mk("adv_vel1", &[]),
            adv_vel2: mk("adv_vel2", &[]),
            adv_s1: mk("adv_s1", &[]),
            adv_s2: mk("adv_s2", &[]),
            react: mk("react", &[]),
            forces: mk("forces", &[]),
            divergence: mk("divergence", &[]),
            project: mk("project", &[]),
            smooth: [
                mk("mg_smooth", &[("COLOR", 0.0), ("OMEGA", 1.0)]),
                mk("mg_smooth", &[("COLOR", 1.0), ("OMEGA", 1.0)]),
            ],
            coarse: [
                mk("mg_smooth", &[("COLOR", 0.0), ("OMEGA", 1.6)]),
                mk("mg_smooth", &[("COLOR", 1.0), ("OMEGA", 1.6)]),
            ],
            restrict: mk("mg_restrict", &[]),
            prolong: mk("mg_prolong", &[]),
            resid: mk("mg_resid", &[]),
        };
        let ign_until = scene.ignition.as_ref().map_or(0.0, |i| i.until);
        Ok(Sim {
            p,
            dims,
            dx,
            ubuf,
            vel,
            hat,
            cur: 0,
            s0,
            s1,
            s0b,
            s1b,
            h0,
            h1,
            sol,
            sol_cur: 0,
            pyr,
            flags,
            curl,
            levels,
            stats,
            staging,
            k,
            time: 0.0,
            cycles: scene.sim.pressure_cycles,
            ign_until,
        })
    }

    pub fn cells(&self) -> u64 {
        (self.dims[0] * self.dims[1] * self.dims[2]) as u64
    }

    fn g3(&self) -> [u32; 3] {
        groups3(self.dims, [4, 4, 4])
    }
    fn gv(&self) -> [u32; 3] {
        groups3(self.dims.map(|v| v + 1), [4, 4, 4])
    }

    /// 推进一个时间步 dt
    pub fn step(&mut self, gpu: &Gpu, dt: f32, want_stats: bool) {
        self.p.dt = dt;
        self.p.time = self.time;
        self.p.ign_on = if self.time < self.ign_until { 1.0 } else { 0.0 };
        gpu.queue
            .write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&self.p));
        let dev = &gpu.device;
        let u = self.ubuf.as_entire_binding();
        let mut enc = dev.create_command_encoder(&Default::default());
        if want_stats {
            enc.clear_buffer(&self.stats, 0, None);
        }
        let (g3, gv) = (self.g3(), self.gv());
        let va = &self.vel[self.cur];
        let vb = &self.vel[1 - self.cur];
        let (sa, sb) = (&self.sol[self.sol_cur], &self.sol[1 - self.sol_cur]);
        let k = &self.k;
        let bg_solid = k.solid.bind(
            dev,
            &[
                u.clone(),
                b(sa),
                b(sb),
                b(&self.pyr),
                b(&self.flags),
                b(&self.s0),
                b(&self.stats),
            ],
        );
        let bg_curl = k.curl.bind(
            dev,
            &[
                u.clone(),
                b(&va[0]),
                b(&va[1]),
                b(&va[2]),
                b(&self.curl),
                b(&self.stats),
            ],
        );
        let bg_av1 = k.adv_vel1.bind(
            dev,
            &[
                u.clone(),
                b(&va[0]),
                b(&va[1]),
                b(&va[2]),
                b(&self.hat[0]),
                b(&self.hat[1]),
                b(&self.hat[2]),
            ],
        );
        let bg_av2 = k.adv_vel2.bind(
            dev,
            &[
                u.clone(),
                b(&va[0]),
                b(&va[1]),
                b(&va[2]),
                b(&self.hat[0]),
                b(&self.hat[1]),
                b(&self.hat[2]),
                b(&vb[0]),
                b(&vb[1]),
                b(&vb[2]),
                b(&self.flags),
            ],
        );
        let bg_as1 = k.adv_s1.bind(
            dev,
            &[
                u.clone(),
                b(&va[0]),
                b(&va[1]),
                b(&va[2]),
                b(&self.s0),
                b(&self.s1),
                b(&self.h0),
                b(&self.h1),
            ],
        );
        let bg_as2 = k.adv_s2.bind(
            dev,
            &[
                u.clone(),
                b(&va[0]),
                b(&va[1]),
                b(&va[2]),
                b(&self.s0),
                b(&self.s1),
                b(&self.h0),
                b(&self.h1),
                b(&self.s0b),
                b(&self.s1b),
            ],
        );
        let bg_react = k.react.bind(
            dev,
            &[
                u.clone(),
                b(&self.s0b),
                b(&self.s1b),
                b(&self.s0),
                b(&self.s1),
                b(&self.flags),
                b(sb),
                b(&self.pyr),
                b(&self.curl),
                b(&self.stats),
            ],
        );
        let bg_forces = k.forces.bind(
            dev,
            &[
                u.clone(),
                b(&vb[0]),
                b(&vb[1]),
                b(&vb[2]),
                b(&self.s0),
                b(&self.s1),
                b(&self.curl),
                b(&self.flags),
            ],
        );
        let l0 = &self.levels[0];
        let bg_div = k.divergence.bind(
            dev,
            &[
                u.clone(),
                b(&vb[0]),
                b(&vb[1]),
                b(&vb[2]),
                b(&self.s1),
                b(&self.flags),
                b(&l0.b),
            ],
        );
        let bg_proj = k.project.bind(
            dev,
            &[
                u.clone(),
                b(&vb[0]),
                b(&vb[1]),
                b(&vb[2]),
                b(&l0.phi),
                b(&self.flags),
            ],
        );
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            k.solid.run(&mut pass, &bg_solid, g3);
            k.curl.run(&mut pass, &bg_curl, g3);
            k.adv_vel1.run(&mut pass, &bg_av1, gv);
            k.adv_vel2.run(&mut pass, &bg_av2, gv);
            k.adv_s1.run(&mut pass, &bg_as1, g3);
            k.adv_s2.run(&mut pass, &bg_as2, g3);
            k.react.run(&mut pass, &bg_react, g3);
            k.forces.run(&mut pass, &bg_forces, gv);
            k.divergence.run(&mut pass, &bg_div, g3);
            self.vcycles(dev, &mut pass, self.cycles);
            if want_stats {
                let bg = k.resid.bind(
                    dev,
                    &[b(&l0.u), b(&l0.phi), b(&l0.b), b(&l0.flags), b(&self.stats)],
                );
                k.resid.run(&mut pass, &bg, g3);
            }
            k.project.run(&mut pass, &bg_proj, gv);
        }
        if want_stats {
            enc.copy_buffer_to_buffer(&self.stats, 0, &self.staging, 0, (STATS_N * 4) as u64);
        }
        gpu.queue.submit([enc.finish()]);
        self.cur = 1 - self.cur;
        self.sol_cur = 1 - self.sol_cur;
        self.time += dt;
        self.p.frame += 1;
    }

    fn smooth(
        &self,
        dev: &wgpu::Device,
        pass: &mut wgpu::ComputePass,
        l: usize,
        iters: u32,
        coarse: bool,
    ) {
        let lv = &self.levels[l];
        let ks = if coarse {
            &self.k.coarse
        } else {
            &self.k.smooth
        };
        let bgs: Vec<wgpu::BindGroup> = ks
            .iter()
            .map(|k| k.bind(dev, &[b(&lv.u), b(&lv.b), b(&lv.flags), b(&lv.phi)]))
            .collect();
        let g = groups3([lv.dims[0].div_ceil(2), lv.dims[1], lv.dims[2]], [4, 4, 4]);
        for _ in 0..iters {
            for c in 0..2 {
                ks[c].run(pass, &bgs[c], g);
            }
        }
    }

    fn vcycles(&self, dev: &wgpu::Device, pass: &mut wgpu::ComputePass, cycles: u32) {
        let nl = self.levels.len();
        for _ in 0..cycles {
            for l in 0..nl - 1 {
                self.smooth(dev, pass, l, 2, false);
                let (f, c) = (&self.levels[l], &self.levels[l + 1]);
                let bg = self.k.restrict.bind(
                    dev,
                    &[
                        b(&f.u),
                        b(&f.phi),
                        b(&f.b),
                        b(&f.flags),
                        b(&c.b),
                        b(&c.phi),
                        b(&c.flags),
                        b(&c.u),
                    ],
                );
                self.k.restrict.run(pass, &bg, groups3(c.dims, [4, 4, 4]));
            }
            // 最粗层只有几千个单元：多迭代几次把它解透，粗网格修正的质量决定整个 V-cycle 的收敛率
            self.smooth(dev, pass, nl - 1, COARSE_ITERS, true);
            for l in (0..nl - 1).rev() {
                let (f, c) = (&self.levels[l], &self.levels[l + 1]);
                let bg = self.k.prolong.bind(
                    dev,
                    &[
                        b(&f.u),
                        b(&c.phi),
                        b(&f.flags),
                        b(&f.phi),
                        b(&c.u),
                        b(&c.flags),
                    ],
                );
                self.k.prolong.run(pass, &bg, groups3(f.dims, [4, 4, 4]));
                self.smooth(dev, pass, l, 2, false);
            }
        }
    }

    /// 读回 s0/s1（调试截面用）
    pub fn read_scalars(&self, gpu: &Gpu) -> Result<(Field4, Field4)> {
        let n = self.cells() * 16;
        let r0 = gpu.readback("r0", n);
        let r1 = gpu.readback("r1", n);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.s0, 0, &r0, 0, n);
        enc.copy_buffer_to_buffer(&self.s1, 0, &r1, 0, n);
        gpu.queue.submit([enc.finish()]);
        let a = gpu.read_buffer(&r0)?;
        let b = gpu.read_buffer(&r1)?;
        Ok((
            bytemuck::cast_slice(&a).to_vec(),
            bytemuck::cast_slice(&b).to_vec(),
        ))
    }

    pub fn read_stats(&self, gpu: &Gpu) -> Result<Stats> {
        let raw = gpu.read_buffer(&self.staging)?;
        let v: &[f32] = bytemuck::cast_slice(&raw);
        let ny = self.dims[1] as usize;
        Ok(Stats {
            hrr: v[0],
            rad: v[1],
            pyro: v[2],
            soot: v[3],
            max_t: v[4],
            max_u: v[5],
            solid: v[7],
            area: v[9],
            resid: if v[11] > 0.0 {
                (v[10] / v[11]).sqrt()
            } else {
                0.0
            },
            layers: v[16..16 + ny.min(1024)].to_vec(),
        })
    }
}
