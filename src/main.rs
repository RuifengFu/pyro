//! pyro：离线大规模火焰燃烧模拟与渲染

mod color;
mod gpu;
mod post;
mod render;
mod scene;
mod sim;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "pyro", about = "离线大规模火焰燃烧模拟与渲染")]
struct Cli {
    /// 场景文件（TOML）
    scene: String,
    /// 只模拟、打印统计量，不渲染
    #[arg(long)]
    sim_only: bool,
    /// 覆盖模拟时长（秒，不含预热）
    #[arg(long)]
    duration: Option<f32>,
    /// 覆盖网格尺寸（米）
    #[arg(long)]
    cell: Option<f32>,
    /// 覆盖每像素采样数
    #[arg(long)]
    spp: Option<u32>,
    /// 覆盖输出分辨率，如 1920x1080
    #[arg(long)]
    res: Option<String>,
    /// 覆盖预热时长（秒）
    #[arg(long)]
    preroll: Option<f32>,
    /// 只渲染第 t 秒的一张静帧
    #[arg(long)]
    still: Option<f32>,
    /// 输出目录
    #[arg(long)]
    out: Option<String>,
    /// 曝光补偿 EV
    #[arg(long)]
    ev: Option<f32>,
    /// 调试：在结束时输出中心竖直截面（T、燃料、氧、碳烟、放热率）
    #[arg(long)]
    slice: bool,
    /// 完成后用系统播放器打开视频（或静帧）
    #[arg(long)]
    open: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut scene = scene::load(&cli.scene)?;
    if let Some(d) = cli.duration {
        scene.sim.duration = d;
    }
    if let Some(c) = cli.cell {
        scene.domain.cell = c;
    }
    if let Some(s) = cli.spp {
        scene.render.spp = s;
    }
    if let Some(r) = &cli.res {
        let v: Vec<u32> = r.split('x').filter_map(|s| s.parse().ok()).collect();
        if v.len() == 2 {
            scene.render.width = v[0];
            scene.render.height = v[1];
        }
    }
    if let Some(p) = cli.preroll {
        scene.output.preroll = p;
    }
    if let Some(o) = &cli.out {
        scene.output.dir = o.clone();
    }
    if let Some(e) = cli.ev {
        scene.render.exposure = e;
    }
    let gpu = gpu::Gpu::new()?;
    let grid = scene.grid();
    println!(
        "GPU: {}  网格 {}×{}×{}（{:.1} M 单元，{:.1} cm）",
        gpu.name,
        grid.dims[0],
        grid.dims[1],
        grid.dims[2],
        (grid.dims[0] * grid.dims[1] * grid.dims[2]) as f32 / 1e6,
        grid.dx * 100.0
    );
    let mut sim = sim::Sim::new(&gpu, &scene)?;
    let rend = if cli.sim_only {
        None
    } else {
        Some(render::Renderer::new(&gpu, &scene, &sim)?)
    };
    // 场景文件里的相对输出路径以场景文件所在目录为基准（从任意工作目录启动都一致）
    let out_dir = {
        let p = PathBuf::from(&scene.output.dir);
        if p.is_relative() && cli.out.is_none() {
            std::path::Path::new(&cli.scene)
                .parent()
                .map(|d| d.join(&p))
                .unwrap_or(p)
        } else {
            p
        }
    };
    if rend.is_some() {
        std::fs::create_dir_all(&out_dir)?;
    }
    let mut cam = post::Camera::new(
        scene.render.auto_exposure,
        scene.render.exposure,
        scene.render.glare,
    );

    let frame_dt = 1.0 / scene.sim.fps;
    let (preroll, total) = match cli.still {
        Some(t) => (t, t + frame_dt * 0.5),
        None => (
            scene.output.preroll,
            scene.output.preroll + scene.sim.duration,
        ),
    };
    let frames = (total * scene.sim.fps).round() as u32;
    let mut max_u: f32 = 2.0;
    let t_start = Instant::now();
    let avg_from = preroll.max(0.4 * total);
    let (mut n_avg, mut hrr_avg, mut rad_avg, mut pyro_avg, mut area_avg) =
        (0u32, 0f64, 0f64, 0f64, 0f64);
    let mut layer_avg = vec![0f64; grid.dims[1] as usize];
    let mut out_idx = 0u32;
    let mut last_path: Option<PathBuf> = None;
    let mut render_time = 0.0f32;
    for f in 0..frames {
        let dt_cfl = scene.sim.cfl * sim.dx / max_u.max(1.0);
        let n = (frame_dt / dt_cfl).ceil().max(1.0) as u32;
        let dt = frame_dt / n as f32;
        for s in 0..n {
            sim.step(&gpu, dt, s == n - 1);
        }
        let st = sim.read_stats(&gpu)?;
        max_u = st.max_u.max(1.0);
        if sim.time >= avg_from {
            n_avg += 1;
            hrr_avg += st.hrr as f64;
            rad_avg += st.rad as f64;
            pyro_avg += st.pyro as f64;
            area_avg += st.area as f64;
            for (a, v) in layer_avg.iter_mut().zip(&st.layers) {
                *a += *v as f64;
            }
        }
        let rendering = match cli.still {
            Some(_) => f == frames - 1,
            None => sim.time >= preroll - 1e-4,
        };
        let mut line = format!(
            "t={:5.2}s 子步{:2} HRR {:7.1} kW 辐射{:4.0}% 热解 {:6.3} kg/s Tmax {:4.0} K umax {:4.1} 火焰高 {:4.2} m 碳烟 {:5.1} g 可燃物 {:.1} kg 残差 {:.0e}",
            sim.time,
            n,
            st.hrr / 1e3,
            if st.hrr > 0.0 {
                100.0 * st.rad / st.hrr
            } else {
                0.0
            },
            st.pyro,
            st.max_t,
            st.max_u,
            st.flame_height(sim.dx),
            st.soot * 1e3,
            st.solid,
            st.resid,
        );
        if let (Some(r), true) = (&rend, rendering) {
            let t0 = Instant::now();
            let img = r.render(&gpu, &sim, f, scene.render.spp)?;
            let (w, h) = r.size();
            let rgb = cam.develop(&img, w as usize, h as usize, frame_dt);
            let path = if cli.still.is_some() {
                out_dir.join(format!("still_{:.1}s.png", sim.time))
            } else {
                out_dir.join(format!("frame_{out_idx:05}.png"))
            };
            post::write_png(&path, &rgb, w, h)?;
            last_path = Some(path.clone());
            out_idx += 1;
            let rt = t0.elapsed().as_secs_f32();
            render_time += rt;
            line += &format!("  渲染 {rt:.1}s → {}", path.display());
        }
        if cli.sim_only || rendering || f % 12 == 0 {
            println!("{line}  [{:.0}s]", t_start.elapsed().as_secs_f32());
        }
    }
    if cli.slice {
        let (a, b) = sim.read_scalars(&gpu)?;
        let [nx, ny, nz] = sim.dims.map(|v| v as usize);
        let k = nz / 2;
        // 5 个面板并排：T(300–2000 K)、Y_F(0–0.3)、Y_O2(0–0.233)、碳烟 f_v(0–3 ppm)、q(0–5 MW/m³)
        let pw = nx * 5 + 8;
        let mut img = vec![0u8; pw * ny * 3];
        for j in 0..ny {
            for i in 0..nx {
                let c = a[i + nx * (j + ny * k)];
                let d = b[i + nx * (j + ny * k)];
                let rho = 1.204 * 293.0 / c[0].max(200.0);
                let vals = [
                    (c[0] - 300.0) / 1700.0,
                    c[1] / 0.3,
                    c[2] / 0.233,
                    rho * c[3] / 1800.0 * 1e6 / 3.0,
                    d[1] / 5e6,
                ];
                for (pi, v) in vals.iter().enumerate() {
                    let v = v.clamp(0.0, 1.0);
                    let col = [
                        (v * 3.0).min(1.0),
                        (v * 3.0 - 1.0).clamp(0.0, 1.0),
                        (v * 3.0 - 2.0).clamp(0.0, 1.0),
                    ];
                    let x = pi * (nx + 2) + i;
                    let o = ((ny - 1 - j) * pw + x) * 3;
                    for ch in 0..3 {
                        img[o + ch] = (col[ch] * 255.0) as u8;
                    }
                }
            }
        }
        std::fs::create_dir_all(&out_dir)?;
        let p = out_dir.join("slice.png");
        post::write_png(&p, &img, pw as u32, ny as u32)?;
        println!("截面：{}", p.display());
    }
    if cli.sim_only && n_avg > 0 {
        let n = n_avg as f64;
        let q = hrr_avg / n;
        let tot: f64 = layer_avg.iter().sum();
        let h_at = |frac: f64| {
            let mut acc = 0.0;
            for (j, v) in layer_avg.iter().enumerate() {
                acc += v;
                if acc >= frac * tot {
                    return (j as f64 + 1.0) * sim.dx as f64;
                }
            }
            0.0
        };
        println!("--- 平均（t ≥ {avg_from:.1} s，{n_avg} 帧）---");
        println!(
            "HRR {:.0} kW  辐射分数 {:.2}  热解 {:.4} kg/s  燃烧面积 {:.3} m²  → 单位面积 {:.4} kg/m²s",
            q / 1e3,
            rad_avg / hrr_avg.max(1.0),
            pyro_avg / n,
            area_avg / n,
            pyro_avg / area_avg.max(1e-9)
        );
        println!(
            "放热累计 50% 高度 {:.2} m  90% {:.2} m  99% {:.2} m",
            h_at(0.5),
            h_at(0.9),
            h_at(0.99)
        );
        for o in &scene.objects {
            if o.kind == "pool" {
                let d = 2.0 * o.radius as f64;
                let l = 0.235 * (q / 1e3).powf(0.4) - 1.02 * d;
                println!(
                    "Heskestad 平均火焰高度（按实际 HRR）{l:.2} m，脉动频率参考 {:.2} Hz",
                    1.5 / d.sqrt()
                );
            }
        }
    }
    if out_idx > 1 && scene.output.video && cli.still.is_none() {
        let mp4 = out_dir.join("video.mp4");
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-framerate"])
            .arg(format!("{}", scene.sim.fps))
            .arg("-i")
            .arg(out_dir.join("frame_%05d.png"))
            .args([
                "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
            ])
            .arg(&mp4)
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("视频：{}", mp4.display());
                last_path = Some(mp4.clone());
            }
            _ => println!("ffmpeg 编码失败（帧仍在 {}）", out_dir.display()),
        }
    }
    if out_idx > 0 {
        println!(
            "共 {out_idx} 帧，渲染平均 {:.1} s/帧，总耗时 {:.0} s",
            render_time / out_idx as f32,
            t_start.elapsed().as_secs_f32()
        );
    }
    if let (true, Some(p)) = (cli.open, &last_path) {
        open_file(p);
    }
    Ok(())
}

/// 用系统默认程序打开文件
fn open_file(p: &std::path::Path) {
    let r = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(p).status()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(p)
            .status()
    } else {
        std::process::Command::new("xdg-open").arg(p).status()
    };
    if r.is_err() {
        eprintln!("无法自动打开 {}", p.display());
    }
}
