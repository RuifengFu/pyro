//! 体积渲染器：打包模拟场 → 3D 纹理；火光重要性分布；光照缓存（烟雾、地面）；相机射线积分

use crate::color;
use crate::gpu::{Gpu, Kernel, assemble, b, groups3, split_kernels};
use crate::scene::{MATERIALS, RenderCfg, Scene};
use crate::sim::Sim;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::BindingResource as BR;

const SRC: &str = include_str!("shaders/render.wgsl");
const GROUND_RES: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RP {
    res: [u32; 4],
    dims: [u32; 4],
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    soot: [f32; 4],
    c: [f32; 4],
    alb: [[f32; 4]; 2],
    gc: [f32; 4],
}

pub struct Renderer {
    w: u32,
    h: u32,
    vol: wgpu::Texture,
    sol: wgpu::Texture,
    vel: wgpu::Texture,
    lv_a: wgpu::Texture,
    lv_b: wgpu::Texture,
    lv_dims: [u32; 3],
    ground: wgpu::Texture,
    gtmp: wgpu::Texture,
    t1: wgpu::Texture,
    t2: wgpu::Texture,
    blur_u: Vec<wgpu::Buffer>,
    k_blur: Kernel,
    sampler: wgpu::Sampler,
    bricks: wgpu::Buffer,
    lut: wgpu::Buffer,
    accum: wgpu::Buffer,
    readback: wgpu::Buffer,
    ubuf: wgpu::Buffer,
    k_pack: Kernel,
    k_bricks: Kernel,
    k_trace: Kernel,
    k_ecells: Kernel,
    k_lightvol: Kernel,
    k_lightground: Kernel,
    ecell: wgpu::Buffer,
    ecell_read: wgpu::Buffer,
    ecdf: wgpu::Buffer,
    chemi_u: wgpu::Buffer,
    ncells: usize,
    cache: bool,
    base: RP,
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-9);
    a.map(|v| v / l)
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn tex3(gpu: &Gpu, label: &str, d: [u32; 3]) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: d[0],
            height: d[1],
            depth_or_array_layers: d[2],
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

impl Renderer {
    pub fn new(gpu: &Gpu, scene: &Scene, sim: &Sim) -> Result<Renderer> {
        let cfg: &RenderCfg = &scene.render;
        let (w, h) = (cfg.width, cfg.height);
        let d = sim.dims;
        let lv_dims = d.map(|v| v.div_ceil(2));
        // 地面照度缓存用单层 3D 纹理，和体缓存共用模糊内核
        let gdims = [GROUND_RES, GROUND_RES, 1];
        let ground = tex3(gpu, "groundE", gdims);
        let gtmp = tex3(gpu, "groundT", gdims);
        let t1 = tex3(gpu, "lvT1", lv_dims);
        let t2 = tex3(gpu, "lvT2", lv_dims);
        // 模糊参数：(轴, 半径)
        let blur_u: Vec<wgpu::Buffer> = [(0u32, 2u32), (1, 2), (2, 2), (0, 3), (1, 3)]
            .iter()
            .map(|&(a, r)| {
                let u = gpu.uniform("blurU", 16);
                gpu.queue
                    .write_buffer(&u, 0, bytemuck::cast_slice(&[a, r, 0u32, 0u32]));
                u
            })
            .collect();
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nb = d.map(|v| v.div_ceil(8));
        let bricks = gpu.storage("bricks", (nb[0] * nb[1] * nb[2] * 4) as u64);
        let lutv = color::build_lut();
        let lut = gpu.storage("lut", (lutv.len() * 16) as u64);
        gpu.queue.write_buffer(&lut, 0, bytemuck::cast_slice(&lutv));
        let accum = gpu.storage("accum", (w * h * 16) as u64);
        let readback = gpu.readback("accumR", (w * h * 16) as u64);
        let ubuf = gpu.uniform("rp", std::mem::size_of::<RP>() as u64);
        let ks = split_kernels(SRC);
        let mk = |n: &str, common: bool| gpu.kernel(n, &assemble(&ks[n], common), &[]);
        let ncells = (d[0] * d[1] * d[2]) as usize;
        let ecell = gpu.storage("ecell", (ncells * 4) as u64);
        let ecell_read = gpu.readback("ecellR", (ncells * 4) as u64);
        let ecdf = gpu.storage("ecdf", ((ncells + 1) * 4) as u64);
        let chemi_u = gpu.uniform("chemi", 16);

        // 相机
        let pos = cfg.camera;
        let fwd = norm(sub(cfg.target, pos));
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let focus = if cfg.focus > 0.0 {
            cfg.focus
        } else {
            let t = sub(cfg.target, pos);
            (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()
        };
        let soot_alb = 0.25;
        let se = color::soot_extinction(soot_alb);
        let mut alb = [[0.0f32; 4]; 2];
        for (slot, &mi) in scene.used_materials()?.iter().enumerate() {
            let a = match MATERIALS[mi].name {
                "wood" => 0.32,
                "cardboard" => 0.45,
                "pu_foam" => 0.55,
                "pmma" => 0.1,
                "rubber" => 0.05,
                "steel" => 0.3,
                "concrete" => 0.35,
                _ => 0.05,
            };
            alb[slot / 4][slot % 4] = a;
        }
        // ecells 内核参数：化学发光强度、每 ppm 碳烟（绿光）与每 g/m³ 白烟的消光系数
        gpu.queue.write_buffer(
            &chemi_u,
            0,
            bytemuck::cast_slice(&[cfg.chemi, se[1], 5.0, 0.0]),
        );
        // 地面照度缓存覆盖计算域中心周围 3 倍宽的正方形
        let ext = [d[0] as f32 * sim.dx, d[2] as f32 * sim.dx];
        let side = 3.0 * ext[0].max(ext[1]);
        let cache = cfg.mode != "path";
        let base = RP {
            res: [w, h, 0, 0],
            dims: [d[0], d[1], d[2], 0],
            cam_pos: [pos[0], pos[1], pos[2], (cfg.fov.to_radians() * 0.5).tan()],
            cam_fwd: [fwd[0], fwd[1], fwd[2], w as f32 / h as f32],
            cam_right: [right[0], right[1], right[2], cfg.aperture],
            cam_up: [up[0], up[1], up[2], focus],
            a: [
                sim.dx,
                cfg.step,
                cfg.shutter / 360.0 / scene.sim.fps,
                cfg.sky,
            ],
            b: [
                if cfg.ground { 1.0 } else { 0.0 },
                cfg.ground_albedo,
                cfg.bounces as f32,
                cfg.chemi,
            ],
            soot: [se[0], se[1], se[2], soot_alb],
            c: [5.0, 0.92, 0.55, scene.sim.ambient_temp],
            alb,
            gc: [
                0.5 * ext[0] - 0.5 * side,
                0.5 * ext[1] - 0.5 * side,
                side,
                if cache { 1.0 } else { 0.0 },
            ],
        };
        Ok(Renderer {
            w,
            h,
            vol: tex3(gpu, "vol", d),
            sol: tex3(gpu, "sol", d),
            vel: tex3(gpu, "vel", d),
            lv_a: tex3(gpu, "lvA", lv_dims),
            lv_b: tex3(gpu, "lvB", lv_dims),
            lv_dims,
            ground,
            gtmp,
            t1,
            t2,
            blur_u,
            k_blur: mk("blur", false),
            sampler,
            bricks,
            lut,
            accum,
            readback,
            ubuf,
            k_pack: mk("pack", true),
            k_bricks: mk("bricks", true),
            k_trace: mk("trace", false),
            k_ecells: mk("ecells", true),
            k_lightvol: mk("lightvol", false),
            k_lightground: mk("lightground", false),
            ecell,
            ecell_read,
            ecdf,
            chemi_u,
            ncells,
            cache,
            base,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    /// 渲染一帧，返回线性 RGB（cd/m²）
    pub fn render(&self, gpu: &Gpu, sim: &Sim, frame: u32, spp: u32) -> Result<Vec<[f32; 3]>> {
        let dev = &gpu.device;
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (vv, sv, uv) = (view(&self.vol), view(&self.sol), view(&self.vel));
        let (lav, lbv, gv) = (view(&self.lv_a), view(&self.lv_b), view(&self.ground));
        let va = &sim.vel[sim.cur];
        let sa = &sim.sol[sim.sol_cur];
        let su = sim.ubuf.as_entire_binding();
        fn tv(v: &wgpu::TextureView) -> BR<'_> {
            BR::TextureView(v)
        }

        // 1) 打包、砖块占据、每单元发光功率
        let bg_pack = self.k_pack.bind(
            dev,
            &[
                su.clone(),
                b(&sim.s0),
                b(&sim.s1),
                b(sa),
                b(&sim.flags),
                b(&va[0]),
                b(&va[1]),
                b(&va[2]),
                tv(&vv),
                tv(&sv),
                tv(&uv),
            ],
        );
        let bg_bricks = self
            .k_bricks
            .bind(dev, &[su.clone(), tv(&vv), tv(&sv), b(&self.bricks)]);
        let bg_ec = self.k_ecells.bind(
            dev,
            &[
                su.clone(),
                tv(&vv),
                b(&self.lut),
                b(&self.ecell),
                self.chemi_u.as_entire_binding(),
            ],
        );
        {
            let mut enc = dev.create_command_encoder(&Default::default());
            enc.clear_buffer(&self.accum, 0, None);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                self.k_pack
                    .run(&mut pass, &bg_pack, groups3(sim.dims, [4, 4, 4]));
                self.k_bricks.run(
                    &mut pass,
                    &bg_bricks,
                    groups3(sim.dims.map(|v| v.div_ceil(8)), [4, 4, 4]),
                );
                self.k_ecells
                    .run(&mut pass, &bg_ec, groups3(sim.dims, [4, 4, 4]));
            }
            enc.copy_buffer_to_buffer(
                &self.ecell,
                0,
                &self.ecell_read,
                0,
                (self.ncells * 4) as u64,
            );
            gpu.queue.submit([enc.finish()]);
        }
        // 2) 发光分布 → 累积分布函数（CPU 前缀和，f64 累加）
        {
            let raw = gpu.read_buffer(&self.ecell_read)?;
            let e: &[f32] = bytemuck::cast_slice(&raw);
            let mut out = Vec::with_capacity(self.ncells + 1);
            let mut acc = 0f64;
            for &v in e {
                acc += v.max(0.0) as f64;
                out.push(acc);
            }
            let total = acc;
            let mut cdf: Vec<f32> = out
                .iter()
                .map(|&c| if total > 0.0 { (c / total) as f32 } else { 0.0 })
                .collect();
            cdf.push(total as f32);
            gpu.queue
                .write_buffer(&self.ecdf, 0, bytemuck::cast_slice(&cdf));
        }
        let mut rp = self.base;
        rp.dims[3] = frame;
        gpu.queue
            .write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&rp));
        let common = || -> Vec<(u32, BR)> {
            vec![
                (0, self.ubuf.as_entire_binding()),
                (1, BR::TextureView(&vv)),
                (2, BR::TextureView(&sv)),
                (3, BR::TextureView(&uv)),
                (4, BR::Sampler(&self.sampler)),
                (5, b(&self.bricks)),
                (6, b(&self.lut)),
                (8, b(&self.ecdf)),
            ]
        };
        // 3) 光照缓存
        if self.cache {
            let mut e1 = common();
            e1.push((12, BR::TextureView(&lav)));
            e1.push((13, BR::TextureView(&lbv)));
            let bg_lv = self.k_lightvol.bind_at(dev, &e1);
            let mut e2 = common();
            e2.push((14, BR::TextureView(&gv)));
            let bg_lg = self.k_lightground.bind_at(dev, &e2);
            let (t1v, t2v, gtv) = (view(&self.t1), view(&self.t2), view(&self.gtmp));
            // 模糊链：体缓存 x→y→z（src → t1 → t2 → src），地面 x→y（g → gt → g）
            let blur = |u: usize, src: &wgpu::TextureView, dst: &wgpu::TextureView| {
                self.k_blur.bind(
                    dev,
                    &[
                        self.blur_u[u].as_entire_binding(),
                        BR::TextureView(src),
                        BR::TextureView(dst),
                    ],
                )
            };
            let chain = [
                (blur(0, &lav, &t1v), self.lv_dims),
                (blur(1, &t1v, &t2v), self.lv_dims),
                (blur(2, &t2v, &lav), self.lv_dims),
                (blur(0, &lbv, &t1v), self.lv_dims),
                (blur(1, &t1v, &t2v), self.lv_dims),
                (blur(2, &t2v, &lbv), self.lv_dims),
                (blur(3, &gv, &gtv), [GROUND_RES, GROUND_RES, 1]),
                (blur(4, &gtv, &gv), [GROUND_RES, GROUND_RES, 1]),
            ];
            let mut enc = dev.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                self.k_lightvol
                    .run(&mut pass, &bg_lv, groups3(self.lv_dims, [4, 4, 4]));
                self.k_lightground.run(
                    &mut pass,
                    &bg_lg,
                    groups3([GROUND_RES, GROUND_RES, 1], [8, 8, 1]),
                );
                for (bg, d) in &chain {
                    self.k_blur.run(&mut pass, bg, groups3(*d, [4, 4, 4]));
                }
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait()?;
        }
        // 4) 相机射线：分块提交，避免单个命令缓冲运行过久
        let mut et = common();
        et.push((7, b(&self.accum)));
        et.push((9, BR::TextureView(&lav)));
        et.push((10, BR::TextureView(&lbv)));
        et.push((11, BR::TextureView(&gv)));
        let bg_trace = self.k_trace.bind_at(dev, &et);
        let chunk = if self.cache { 8u32 } else { 2u32 };
        let mut done = 0;
        while done < spp {
            let n = chunk.min(spp - done);
            rp.res[2] = done;
            rp.res[3] = n;
            gpu.queue
                .write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&rp));
            let mut enc = dev.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                self.k_trace.run(
                    &mut pass,
                    &bg_trace,
                    groups3([self.w, self.h, 1], [8, 8, 1]),
                );
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait()?;
            done += n;
        }
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            &self.accum,
            0,
            &self.readback,
            0,
            (self.w * self.h * 16) as u64,
        );
        gpu.queue.submit([enc.finish()]);
        let raw = gpu.read_buffer(&self.readback)?;
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        Ok(v.iter()
            .map(|p| {
                let n = p[3].max(1.0);
                [p[0] / n, p[1] / n, p[2] / n]
            })
            .collect())
    }
}
