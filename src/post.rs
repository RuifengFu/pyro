//! 成像后期：自动曝光、镜头眩光、胶片式色调映射、写 PNG

use anyhow::Result;
use std::io::BufWriter;

pub struct Camera {
    /// 平滑后的曝光（log2），None 表示还没开始
    ev: Option<f32>,
    pub auto: bool,
    pub ev_offset: f32,
    pub glare: f32,
}

impl Camera {
    pub fn new(auto: bool, ev_offset: f32, glare: f32) -> Camera {
        Camera {
            ev: None,
            auto,
            ev_offset,
            glare,
        }
    }

    /// 返回 sRGB 8 位图像
    pub fn develop(&mut self, img: &[[f32; 3]], w: usize, h: usize, frame_dt: f32) -> Vec<u8> {
        // 曝光：让亮部（99.5 百分位）落在色调曲线的肩部以上；像相机自动曝光一样随时间平滑
        let exposure = if self.auto {
            let mut l: Vec<f32> = img
                .iter()
                .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
                .collect();
            let k = ((l.len() as f32) * 0.995) as usize;
            let k = k.min(l.len() - 1);
            let (_, v, _) = l.select_nth_unstable_by(k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            // 像真实视频那样让火焰核心接近饱和，周围被火光照亮的环境才看得见
            let target = (4.0 / v.max(1e-3)).log2();
            let ev = match self.ev {
                None => target,
                Some(e) => e + (target - e) * (1.0 - (-frame_dt / 0.6).exp()),
            };
            self.ev = Some(ev);
            ev + self.ev_offset
        } else {
            self.ev_offset
        };
        let k = 2f32.powf(exposure);
        let mut lin: Vec<[f32; 3]> = img.iter().map(|p| [p[0] * k, p[1] * k, p[2] * k]).collect();
        if self.glare > 0.0 {
            let g = glare(&lin, w, h);
            for (p, q) in lin.iter_mut().zip(&g) {
                for c in 0..3 {
                    p[c] = p[c] * (1.0 - self.glare) + q[c] * self.glare;
                }
            }
        }
        let mut out = Vec::with_capacity(w * h * 3);
        for (i, p) in lin.iter().enumerate() {
            // 抖动，避免 8 位色带
            let n = (hash(i as u32) - 0.5) / 255.0;
            for &ch in p {
                let v = srgb(aces(ch)) + n;
                out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        out
    }
}

/// ACES 胶片曲线（Narkowicz 拟合）
fn aces(x: f32) -> f32 {
    let x = x * 0.6;
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}
fn srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}
fn hash(v: u32) -> f32 {
    let mut x = v.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    x = (x >> 22) ^ x;
    (x >> 8) as f32 / 16777216.0
}

/// 镜头眩光：三个尺度的高斯模糊加权（近似镜头的点扩散函数长尾）
fn glare(img: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; w * h];
    for (r, wt) in [(0.003f32, 0.5f32), (0.012, 0.3), (0.04, 0.2)] {
        let rad = ((r * w as f32) as usize).max(1);
        let mut buf = img.to_vec();
        for _ in 0..3 {
            box_h(&mut buf, w, h, rad);
            box_v(&mut buf, w, h, rad);
        }
        for (o, p) in out.iter_mut().zip(&buf) {
            for c in 0..3 {
                o[c] += p[c] * wt;
            }
        }
    }
    out
}

fn box_h(b: &mut [[f32; 3]], w: usize, h: usize, r: usize) {
    let mut row = vec![[0.0f32; 3]; w];
    let norm = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        let line = &b[y * w..(y + 1) * w];
        let mut acc = [0.0f32; 3];
        for px in &line[..=r.min(w - 1)] {
            for c in 0..3 {
                acc[c] += px[c];
            }
        }
        for c in 0..3 {
            acc[c] += line[0][c] * r as f32;
        }
        for (x, out) in row.iter_mut().enumerate() {
            *out = acc.map(|v| v * norm);
            let add = line[(x + r + 1).min(w - 1)];
            let rem = line[x.saturating_sub(r)];
            for c in 0..3 {
                acc[c] += add[c] - rem[c];
            }
        }
        b[y * w..(y + 1) * w].copy_from_slice(&row);
    }
}

fn box_v(b: &mut [[f32; 3]], w: usize, h: usize, r: usize) {
    let mut col = vec![[0.0f32; 3]; h];
    let norm = 1.0 / (2 * r + 1) as f32;
    for x in 0..w {
        let mut acc = [0.0f32; 3];
        for i in 0..=r.min(h - 1) {
            for c in 0..3 {
                acc[c] += b[i * w + x][c];
            }
        }
        for c in 0..3 {
            acc[c] += b[x][c] * r as f32;
        }
        for y in 0..h {
            col[y] = acc.map(|v| v * norm);
            let add = b[(y + r + 1).min(h - 1) * w + x];
            let rem = b[y.saturating_sub(r) * w + x];
            for c in 0..3 {
                acc[c] += add[c] - rem[c];
            }
        }
        for y in 0..h {
            b[y * w + x] = col[y];
        }
    }
}

pub fn write_png(path: &std::path::Path, rgb: &[u8], w: u32, h: u32) -> Result<()> {
    let f = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header()?;
    wr.write_image_data(rgb)?;
    Ok(())
}
