//! 光谱 → 颜色：黑体、碳烟发射的查找表（CIE 1931 2° 配色函数，Wyman–Sloan–Shirley 多瓣高斯拟合）

const H: f64 = 6.626_070_15e-34;
const C: f64 = 2.997_924_58e8;
const KB: f64 = 1.380_649e-23;
/// 碳烟 Rayleigh 吸收：κ_λ = 6π E(m) f_v / λ，E(m) ≈ 0.37
pub const SOOT_K: f64 = 6.97;

fn lobe(l: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let s = if l < mu { s1 } else { s2 };
    (-0.5 * ((l - mu) / s).powi(2)).exp()
}

/// λ 单位 nm
pub fn cmf(l: f64) -> [f64; 3] {
    let x = 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7)
        - 0.065 * lobe(l, 501.1, 20.4, 26.2);
    let y = 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
    let z = 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8);
    [x, y, z]
}

/// 普朗克光谱辐亮度 W/(m²·sr·m)，λ 单位 m
pub fn planck(lm: f64, t: f64) -> f64 {
    2.0 * H * C * C / lm.powi(5) / ((H * C / (lm * KB * t)).exp() - 1.0)
}

pub fn xyz_to_rgb(x: f64, y: f64, z: f64) -> [f64; 3] {
    [
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ]
}

/// 色域外（饱和的橙红色）用保亮度的方式压回：负分量置零后按 Y 重新缩放
fn gamut(rgb: [f64; 3], y: f64) -> [f32; 3] {
    let c = rgb.map(|v| v.max(0.0));
    let ly = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let k = if ly > 0.0 { y / ly } else { 0.0 };
    c.map(|v| (v * k) as f32)
}

/// 返回 2048 个 vec4：前 1024 为碳烟（单位体积分数、每米路径的发射亮度，cd/m² 量纲），
/// 后 1024 为黑体表面亮度；温度 300–3300 K 线性分布
pub fn build_lut() -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(2048);
    for pass in 0..2 {
        for i in 0..1024 {
            let t = 300.0 + 3000.0 * i as f64 / 1023.0;
            let mut xyz = [0.0f64; 3];
            let mut l = 360.0;
            while l <= 830.0 {
                let lm = l * 1e-9;
                let mut e = planck(lm, t) * 1e-9 * 5.0;
                if pass == 0 {
                    e *= SOOT_K / lm;
                }
                let c = cmf(l);
                for k in 0..3 {
                    xyz[k] += 683.0 * e * c[k];
                }
                l += 5.0;
            }
            let rgb = gamut(xyz_to_rgb(xyz[0], xyz[1], xyz[2]), xyz[1]);
            out.push([rgb[0], rgb[1], rgb[2], 0.0]);
        }
    }
    out
}

/// 每 ppm 碳烟在 R/G/B 代表波长处的消光系数 [1/m]（吸收 / (1 − 反照率)）
pub fn soot_extinction(albedo: f32) -> [f32; 3] {
    [610.0f64, 550.0, 465.0].map(|l| (SOOT_K / (l * 1e-9) * 1e-6 / (1.0 - albedo as f64)) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lum_at(lut: &[[f32; 4]], off: usize, t: f64) -> f32 {
        let i = (((t - 300.0) / 3000.0) * 1023.0).round() as usize;
        let c = lut[off + i];
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    #[test]
    fn blackbody_luminance_matches_known_values() {
        // 黑体亮度：2042 K（铂凝固点，坎德拉旧定义）约 6×10⁵ cd/m²；3000 K 约 1.5×10⁶–2×10⁶ 量级以上
        let lut = build_lut();
        let l2042 = lum_at(&lut, 1024, 2042.0);
        assert!((4.5e5..8.0e5).contains(&l2042), "2042 K 黑体亮度 {l2042}");
        // 单调递增
        assert!(lum_at(&lut, 1024, 1500.0) < lum_at(&lut, 1024, 2000.0));
    }

    #[test]
    fn blackbody_color_turns_from_red_to_yellow() {
        let lut = build_lut();
        let at = |t: f64| lut[1024 + (((t - 300.0) / 3000.0) * 1023.0).round() as usize];
        let c1 = at(1000.0);
        let c2 = at(2500.0);
        // 低温偏红（g/r 小），高温更偏黄白（g/r 大）
        assert!(c1[1] / c1[0] < c2[1] / c2[0]);
        assert!(c1[0] > c1[2]);
    }

    #[test]
    fn soot_extinction_scale() {
        // 1 ppm 碳烟在 550 nm 的消光约 15 /m（FDS 质量消光系数 8700 m²/kg × 1.8 g/m³ ≈ 15.7 /m）
        let e = soot_extinction(0.25);
        assert!((12.0..20.0).contains(&e[1]), "{e:?}");
        assert!(e[2] > e[1] && e[1] > e[0]);
    }
}
