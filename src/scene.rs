//! 场景描述（TOML）、材料库、体素化

use anyhow::{Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

// ---------------- 场景文件 ----------------
#[derive(Deserialize, Clone)]
pub struct Scene {
    pub domain: Domain,
    #[serde(default)]
    pub sim: SimCfg,
    #[serde(default)]
    pub wind: Wind,
    #[serde(default)]
    pub ignition: Option<Ignition>,
    #[serde(default)]
    pub fuel: FuelCfg,
    #[serde(default, rename = "object")]
    pub objects: Vec<Object>,
    #[serde(default)]
    pub render: RenderCfg,
    #[serde(default)]
    pub output: OutputCfg,
}

#[derive(Deserialize, Clone)]
pub struct Domain {
    /// 计算域尺寸 [x, y, z]，米；y 向上
    pub size: [f32; 3],
    /// 网格尺寸，米
    pub cell: f32,
}

#[derive(Deserialize, Clone)]
#[serde(default)]
pub struct SimCfg {
    pub fps: f32,
    pub duration: f32,
    pub cfl: f32,
    pub vorticity: f32,
    pub smagorinsky: f32,
    pub pressure_cycles: u32,
    pub seed: u32,
    pub ambient_temp: f32,
}
impl Default for SimCfg {
    fn default() -> Self {
        SimCfg {
            fps: 24.0,
            duration: 10.0,
            cfl: 1.5,
            vorticity: 3.0,
            smagorinsky: 0.17,
            pressure_cycles: 4,
            seed: 1,
            ambient_temp: 293.0,
        }
    }
}

#[derive(Deserialize, Clone, Default)]
#[serde(default)]
pub struct Wind {
    pub speed: f32,
    /// 风向（度，0 = +x 方向）
    pub dir: f32,
}

#[derive(Deserialize, Clone)]
pub struct Ignition {
    pub pos: [f32; 3],
    #[serde(default = "d_ign_r")]
    pub radius: f32,
    #[serde(default = "d_ign_flux")]
    pub flux: f32,
    #[serde(default = "d_ign_temp")]
    pub temp: f32,
    #[serde(default = "d_ign_until")]
    pub until: f32,
    #[serde(default)]
    pub fuel: f32,
}
fn d_ign_r() -> f32 {
    0.25
}
fn d_ign_flux() -> f32 {
    40e3
}
fn d_ign_temp() -> f32 {
    1300.0
}
fn d_ign_until() -> f32 {
    4.0
}

/// 气相燃料（热解气）性质；不写就取场景里第一个可燃物的材料
#[derive(Deserialize, Clone, Default)]
#[serde(default)]
pub struct FuelCfg {
    pub material: Option<String>,
    pub heat_of_combustion: Option<f32>,
    pub soot_yield: Option<f32>,
    /// 火焰内碳烟生成量 / 最终逸出的碳烟产率
    pub soot_formation_ratio: Option<f32>,
    pub soot_oxidation: Option<f32>,
    pub radiative_fraction: Option<f32>,
}

#[derive(Deserialize, Clone)]
pub struct Object {
    pub kind: String,
    pub material: String,
    #[serde(default)]
    pub center: [f32; 3],
    #[serde(default)]
    pub size: [f32; 3],
    #[serde(default)]
    pub radius: f32,
    #[serde(default)]
    pub depth: f32,
    #[serde(default)]
    pub a: [f32; 3],
    #[serde(default)]
    pub b: [f32; 3],
    /// 绕 y 轴旋转（度）
    #[serde(default)]
    pub yaw: f32,
    /// 托盘堆/木垛：层数
    #[serde(default)]
    pub count: u32,
    /// 木垛：每层根数、木条截面边长
    #[serde(default)]
    pub sticks: u32,
    #[serde(default)]
    pub stick: f32,
}

#[derive(Deserialize, Clone)]
#[serde(default)]
pub struct RenderCfg {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub camera: [f32; 3],
    pub target: [f32; 3],
    pub fov: f32,
    /// 曝光补偿（EV）；auto_exposure = false 时是绝对曝光
    pub exposure: f32,
    pub auto_exposure: bool,
    /// 快门角（度），180 = 半帧时间
    pub shutter: f32,
    pub ground: bool,
    pub ground_albedo: f32,
    /// 夜空亮度（cd/m²）
    pub sky: f32,
    pub bounces: u32,
    pub step: f32,
    pub glare: f32,
    pub aperture: f32,
    pub focus: f32,
    /// 化学发光（火焰根部的蓝色 CH*/C2*）强度
    pub chemi: f32,
    /// "cache"：火光照明用光照缓存（快、无噪点）；"path"：纯路径追踪（无偏，用于对照）
    pub mode: String,
}
impl Default for RenderCfg {
    fn default() -> Self {
        RenderCfg {
            width: 1280,
            height: 720,
            spp: 64,
            camera: [0.0, 1.5, -6.0],
            target: [0.0, 2.0, 0.0],
            fov: 45.0,
            exposure: 0.0,
            auto_exposure: true,
            shutter: 180.0,
            ground: true,
            ground_albedo: 0.12,
            sky: 0.003,
            bounces: 2,
            step: 0.6,
            glare: 0.04,
            aperture: 0.0,
            focus: 0.0,
            chemi: 0.05,
            mode: "cache".into(),
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(default)]
pub struct OutputCfg {
    pub dir: String,
    /// 开始渲染前先模拟的秒数（火势发展阶段）
    pub preroll: f32,
    pub video: bool,
}
impl Default for OutputCfg {
    fn default() -> Self {
        OutputCfg {
            dir: "out".into(),
            preroll: 0.0,
            video: true,
        }
    }
}

pub fn load(path: &str) -> Result<Scene> {
    let txt = std::fs::read_to_string(path).map_err(|e| anyhow!("读取场景 {path} 失败: {e}"))?;
    let s: Scene = toml::from_str(&txt).map_err(|e| anyhow!("场景文件格式错误: {e}"))?;
    Ok(s)
}

// ---------------- 材料库 ----------------
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub name: &'static str,
    pub label: &'static str,
    /// 0 成炭固体 1 液池 2 惰性 3 不成炭固体
    pub kind: u32,
    pub density: f32,
    /// 有效气化热 J/kg
    pub lg: f32,
    /// 热解温度（固体）/ 沸点（液体），K
    pub tp: f32,
    pub char_yield: f32,
    pub emissivity: f32,
    pub cp: f32,
    pub k: f32,
    pub smoke: f32,
    /// 挥发分有效燃烧热 J/kg
    pub dhc: f32,
    /// 碳烟产率（质量分数）
    pub soot: f32,
}

pub const MATERIALS: &[Material] = &[
    Material {
        name: "air",
        label: "空气",
        kind: 2,
        density: 0.0,
        lg: 1.0,
        tp: 1e4,
        char_yield: 0.0,
        emissivity: 0.9,
        cp: 1000.0,
        k: 0.03,
        smoke: 0.0,
        dhc: 0.0,
        soot: 0.0,
    },
    Material {
        name: "wood",
        label: "木材",
        kind: 0,
        density: 480.0,
        lg: 1.5e6,
        tp: 600.0,
        char_yield: 0.25,
        emissivity: 0.9,
        cp: 1500.0,
        k: 0.15,
        smoke: 0.02,
        dhc: 15.0e6,
        soot: 0.015,
    },
    Material {
        name: "cardboard",
        label: "纸板",
        kind: 0,
        density: 150.0,
        lg: 1.8e6,
        tp: 580.0,
        char_yield: 0.2,
        emissivity: 0.9,
        cp: 1400.0,
        k: 0.08,
        smoke: 0.02,
        dhc: 14.0e6,
        soot: 0.012,
    },
    Material {
        name: "pu_foam",
        label: "聚氨酯泡沫（沙发/床垫）",
        kind: 3,
        density: 35.0,
        lg: 1.4e6,
        tp: 570.0,
        char_yield: 0.0,
        emissivity: 0.9,
        cp: 1400.0,
        k: 0.04,
        smoke: 0.0,
        dhc: 24.0e6,
        soot: 0.10,
    },
    Material {
        name: "pmma",
        label: "有机玻璃 PMMA",
        kind: 3,
        density: 1190.0,
        lg: 1.6e6,
        tp: 630.0,
        char_yield: 0.0,
        emissivity: 0.9,
        cp: 1500.0,
        k: 0.19,
        smoke: 0.0,
        dhc: 24.9e6,
        soot: 0.022,
    },
    Material {
        name: "rubber",
        label: "橡胶（轮胎）",
        kind: 3,
        density: 1100.0,
        lg: 1.7e6,
        tp: 620.0,
        char_yield: 0.0,
        emissivity: 0.95,
        cp: 1700.0,
        k: 0.2,
        smoke: 0.0,
        dhc: 30.0e6,
        soot: 0.15,
    },
    Material {
        name: "heptane",
        label: "正庚烷油池",
        kind: 1,
        density: 684.0,
        lg: 0.48e6,
        tp: 371.0,
        char_yield: 0.0,
        emissivity: 0.95,
        cp: 2240.0,
        k: 0.14,
        smoke: 0.0,
        dhc: 44.6e6,
        soot: 0.037,
    },
    Material {
        name: "gasoline",
        label: "汽油油池",
        kind: 1,
        density: 740.0,
        lg: 0.42e6,
        tp: 345.0,
        char_yield: 0.0,
        emissivity: 0.95,
        cp: 2200.0,
        k: 0.13,
        smoke: 0.0,
        dhc: 43.7e6,
        soot: 0.059,
    },
    Material {
        name: "diesel",
        label: "柴油油池",
        kind: 1,
        density: 830.0,
        lg: 0.6e6,
        tp: 460.0,
        char_yield: 0.0,
        emissivity: 0.95,
        cp: 2100.0,
        k: 0.13,
        smoke: 0.0,
        dhc: 44.4e6,
        soot: 0.07,
    },
    Material {
        name: "steel",
        label: "钢（不燃）",
        kind: 2,
        density: 7850.0,
        lg: 1.0,
        tp: 1e4,
        char_yield: 0.0,
        emissivity: 0.7,
        cp: 500.0,
        k: 45.0,
        smoke: 0.0,
        dhc: 0.0,
        soot: 0.0,
    },
    Material {
        name: "concrete",
        label: "混凝土（不燃）",
        kind: 2,
        density: 2300.0,
        lg: 1.0,
        tp: 1e4,
        char_yield: 0.0,
        emissivity: 0.9,
        cp: 900.0,
        k: 1.4,
        smoke: 0.0,
        dhc: 0.0,
        soot: 0.0,
    },
];

pub fn material(name: &str) -> Result<(usize, &'static Material)> {
    MATERIALS
        .iter()
        .enumerate()
        .find(|(_, m)| m.name == name)
        .ok_or_else(|| {
            let all: Vec<String> = MATERIALS
                .iter()
                .skip(1)
                .map(|m| format!("{}（{}）", m.name, m.label))
                .collect();
            anyhow!("未知材料 {name}，可选：{}", all.join("、"))
        })
}

// ---------------- GPU 参数块（与 common.wgsl 的 Params 一一对应） ----------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub frame: u32,
    pub dx: f32,
    pub dt: f32,
    pub time: f32,
    pub t0: f32,
    pub rho0: f32,
    pub g: f32,
    pub cp: f32,
    pub yo2: f32,
    pub vort: f32,
    pub cs: f32,
    pub buoy: f32,
    pub smoke_w: f32,
    pub wind_x: f32,
    pub wind_z: f32,
    pub wind_relax: f32,
    pub sponge: f32,
    pub t_ign: f32,
    pub s_o2: f32,
    pub dhc: f32,
    pub tau_min: f32,
    pub soot_form: f32,
    pub soot_ox: f32,
    pub soot_ox_t: f32,
    pub smoke_conv: f32,
    pub kap_gas: f32,
    pub h_conv: f32,
    pub ign_on: f32,
    pub ign_temp: f32,
    pub ign_x: f32,
    pub ign_y: f32,
    pub ign_z: f32,
    pub ign_r: f32,
    pub ign_flux: f32,
    pub ign_fuel: f32,
    pub char_a: f32,
    pub char_e: f32,
    pub dh_char: f32,
    pub burnout: f32,
    pub layer: f32,
    pub rad_len: f32,
    pub h_mass: f32,
    pub vort_t: f32,
    /// 燃烧区辐射分数 χr（放热中以辐射形式散失的比例）
    pub rad_frac: f32,
    pub seed: f32,
    pub mats: [[f32; 12]; 8],
}

pub struct Grid {
    pub dims: [u32; 3],
    pub dx: f32,
}

impl Scene {
    pub fn grid(&self) -> Grid {
        let dx = self.domain.cell;
        // 各维取 16 的倍数，保证多重网格能减半 4 次
        let d = self
            .domain
            .size
            .map(|s| (((s / dx) / 16.0).round().max(1.0) as u32) * 16);
        Grid { dims: d, dx }
    }

    /// 场景中用到的材料（材料表下标），最多 8 种
    pub fn used_materials(&self) -> Result<Vec<usize>> {
        let mut v = vec![0usize];
        for o in &self.objects {
            let (i, _) = material(&o.material)?;
            if !v.contains(&i) {
                v.push(i);
            }
        }
        if v.len() > 8 {
            bail!("一个场景最多 7 种材料");
        }
        Ok(v)
    }

    pub fn fuel_material(&self) -> Result<&'static Material> {
        if let Some(n) = &self.fuel.material {
            return Ok(material(n)?.1);
        }
        for o in &self.objects {
            let (_, m) = material(&o.material)?;
            if m.kind != 2 {
                return Ok(m);
            }
        }
        bail!("场景里没有可燃物")
    }

    pub fn params(&self) -> Result<Params> {
        let g = self.grid();
        let fm = self.fuel_material()?;
        let dhc = self.fuel.heat_of_combustion.unwrap_or(fm.dhc);
        let soot = self.fuel.soot_yield.unwrap_or(fm.soot);
        let ratio = self.fuel.soot_formation_ratio.unwrap_or(2.5);
        let mut mats = [[0f32; 12]; 8];
        for (slot, &mi) in self.used_materials()?.iter().enumerate() {
            let m = &MATERIALS[mi];
            mats[slot] = [
                m.density,
                m.lg,
                m.tp,
                m.kind as f32, //
                m.char_yield,
                m.emissivity,
                m.cp,
                m.k, //
                m.smoke,
                1.0,
                0.0,
                0.0,
            ];
        }
        let wr = self.wind.dir.to_radians();
        let ign = self.ignition.clone();
        // 特征火焰厚度（辐射逃逸因子用）：取可燃物水平尺度的一半
        let mut span: f32 = 0.5;
        for o in &self.objects {
            let s = match o.kind.as_str() {
                "pool" | "sphere" => 2.0 * o.radius,
                _ => o.size[0].max(o.size[2]).max(0.3),
            };
            span = span.max(s);
        }
        Ok(Params {
            nx: g.dims[0],
            ny: g.dims[1],
            nz: g.dims[2],
            frame: 0,
            dx: g.dx,
            dt: 0.005,
            time: 0.0,
            t0: self.sim.ambient_temp,
            rho0: 1.204 * 293.0 / self.sim.ambient_temp,
            g: 9.81,
            cp: 1100.0,
            yo2: 0.233,
            vort: self.sim.vorticity,
            cs: self.sim.smagorinsky,
            buoy: 1.0,
            smoke_w: 0.0,
            wind_x: self.wind.speed * wr.cos(),
            wind_z: self.wind.speed * wr.sin(),
            wind_relax: if self.wind.speed > 0.0 { 0.3 } else { 0.0 },
            sponge: 6.0,
            t_ign: 750.0,
            s_o2: dhc / 13.1e6,
            dhc,
            tau_min: 2e-3,
            soot_form: soot * ratio,
            soot_ox: self.fuel.soot_oxidation.unwrap_or(4.0),
            soot_ox_t: 1150.0,
            smoke_conv: 0.5,
            kap_gas: 0.1,
            h_conv: 15.0,
            ign_on: if ign.is_some() { 1.0 } else { 0.0 },
            ign_temp: ign.as_ref().map_or(0.0, |i| i.temp),
            ign_x: ign.as_ref().map_or(0.0, |i| i.pos[0]),
            ign_y: ign.as_ref().map_or(0.0, |i| i.pos[1]),
            ign_z: ign.as_ref().map_or(0.0, |i| i.pos[2]),
            ign_r: ign.as_ref().map_or(0.0, |i| i.radius),
            ign_flux: ign.as_ref().map_or(0.0, |i| i.flux),
            ign_fuel: ign.as_ref().map_or(0.0, |i| i.fuel),
            char_a: 465.0,
            char_e: 68e3,
            dh_char: 32.8e6,
            burnout: 2.0,
            layer: 0.005,
            rad_len: 0.5 * span,
            h_mass: 0.02,
            vort_t: 500.0,
            rad_frac: self.fuel.radiative_fraction.unwrap_or(match fm.name {
                // 文献值（Tewarson、SFPE 手册）：碳烟越多辐射分数越高
                "heptane" => 0.33,
                "gasoline" => 0.35,
                "diesel" => 0.35,
                "pu_foam" => 0.42,
                "rubber" => 0.42,
                "pmma" => 0.31,
                _ => 0.30,
            }),
            seed: self.sim.seed as f32,
            mats,
        })
    }

    /// 体素化：返回每个单元的 SOL = (材料槽, 可燃质量, 炭, Ts)
    pub fn voxelize(&self) -> Result<Vec<[f32; 4]>> {
        let g = self.grid();
        let [nx, ny, nz] = g.dims.map(|v| v as usize);
        let dx = g.dx;
        let t0 = self.sim.ambient_temp;
        let used = self.used_materials()?;
        let mut sol = vec![[0.0, 0.0, 0.0, t0]; nx * ny * nz];
        let mut shapes: Vec<(Shape, usize)> = Vec::new();
        for o in &self.objects {
            let (mi, _) = material(&o.material)?;
            let slot = used.iter().position(|&u| u == mi).unwrap();
            for s in expand(o, dx)? {
                shapes.push((s, slot));
            }
        }
        let seed = self.sim.seed;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = [
                        (i as f32 + 0.5) * dx,
                        (j as f32 + 0.5) * dx,
                        (k as f32 + 0.5) * dx,
                    ];
                    for (s, slot) in &shapes {
                        if s.inside(p, dx) {
                            let m = &MATERIALS[used[*slot]];
                            let idx = i + nx * (j + ny * k);
                            let h = hash(idx as u32 ^ seed.wrapping_mul(0x9e3779b9));
                            let jit = 0.85 + 0.3 * h;
                            let mass = if m.kind == 2 { 1.0 } else { m.density * jit };
                            sol[idx] = [*slot as f32, mass, 0.0, t0];
                        }
                    }
                }
            }
        }
        Ok(sol)
    }
}

fn hash(v: u32) -> f32 {
    let mut x = v.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    x = (x >> 22) ^ x;
    (x as f32) / (u32::MAX as f32)
}

// ---------------- 几何 ----------------
#[derive(Clone, Copy)]
enum Shape {
    /// 中心、半尺寸、绕 y 旋转角（弧度）
    Box {
        c: [f32; 3],
        h: [f32; 3],
        yaw: f32,
    },
    Cyl {
        a: [f32; 3],
        b: [f32; 3],
        r: f32,
    },
    Sphere {
        c: [f32; 3],
        r: f32,
    },
    /// 液池：圆盘，y ∈ [0, depth]
    Pool {
        c: [f32; 3],
        r: f32,
        depth: f32,
    },
}

impl Shape {
    fn inside(&self, p: [f32; 3], dx: f32) -> bool {
        match *self {
            Shape::Box { c, h, yaw } => {
                let (s, co) = yaw.sin_cos();
                let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                let lx = co * d[0] + s * d[2];
                let lz = -s * d[0] + co * d[2];
                // 薄件至少占一格，避免体素化时丢掉
                let hh = h.map(|v| v.max(0.5 * dx));
                lx.abs() <= hh[0] && d[1].abs() <= hh[1] && lz.abs() <= hh[2]
            }
            Shape::Cyl { a, b, r } => {
                let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let l2 = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
                let t = (pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / l2;
                if !(0.0..=1.0).contains(&t) {
                    return false;
                }
                let q = [pa[0] - ba[0] * t, pa[1] - ba[1] * t, pa[2] - ba[2] * t];
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() <= r.max(0.5 * dx)
            }
            Shape::Sphere { c, r } => {
                let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= r
            }
            Shape::Pool { c, r, depth } => {
                let dxz = ((p[0] - c[0]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
                dxz <= r && p[1] <= depth.max(dx * 0.99)
            }
        }
    }
}

fn expand(o: &Object, dx: f32) -> Result<Vec<Shape>> {
    let yaw = o.yaw.to_radians();
    let c = o.center;
    Ok(match o.kind.as_str() {
        "box" => vec![Shape::Box {
            c,
            h: o.size.map(|v| v * 0.5),
            yaw,
        }],
        "cylinder" => vec![Shape::Cyl {
            a: o.a,
            b: o.b,
            r: o.radius,
        }],
        "sphere" => vec![Shape::Sphere { c, r: o.radius }],
        "pool" => vec![Shape::Pool {
            c,
            r: o.radius,
            depth: o.depth,
        }],
        // 木垛：count 层，每层 sticks 根截面 stick×stick 的木条，长度 size[0]，层间交替方向
        "crib" => {
            let n = o.sticks.max(2);
            let w = o.stick.max(dx);
            let len = o.size[0].max(0.2);
            let mut v = Vec::new();
            for l in 0..o.count.max(1) {
                let y = c[1] + w * (l as f32 + 0.5);
                for s in 0..n {
                    let off = -0.5 * len + 0.5 * w + (len - w) * s as f32 / (n - 1) as f32;
                    let (hx, hz, ox, oz) = if l % 2 == 0 {
                        (0.5 * len, 0.5 * w, 0.0, off)
                    } else {
                        (0.5 * w, 0.5 * len, off, 0.0)
                    };
                    v.push(Shape::Box {
                        c: rot([ox, 0.0, oz], yaw, [c[0], y, c[2]]),
                        h: [hx, 0.5 * w, hz],
                        yaw,
                    });
                }
            }
            v
        }
        // 托盘堆：count 个 1.2 m × 1.0 m 的木托盘叠放（底板 3 条、垫块梁 3 条、面板 7 条）
        "pallets" => {
            let mut v = Vec::new();
            let ph = 0.144;
            for l in 0..o.count.max(1) {
                let y0 = c[1] + ph * l as f32;
                for z in [-0.45f32, 0.0, 0.45] {
                    v.push(Shape::Box {
                        c: rot([0.0, 0.0, z], yaw, [c[0], y0 + 0.011, c[2]]),
                        h: [0.6, 0.011, 0.05],
                        yaw,
                    });
                    v.push(Shape::Box {
                        c: rot([0.0, 0.0, z], yaw, [c[0], y0 + 0.061, c[2]]),
                        h: [0.6, 0.039, 0.05],
                        yaw,
                    });
                }
                for s in 0..7 {
                    let x = -0.55 + 1.1 * s as f32 / 6.0;
                    v.push(Shape::Box {
                        c: rot([x, 0.0, 0.0], yaw, [c[0], y0 + 0.122, c[2]]),
                        h: [0.05, 0.011, 0.5],
                        yaw,
                    });
                }
            }
            v
        }
        // 大篝火：外圈 count 根原木斜靠成圆锥帐篷（底半径 radius、高 size[1]），内圈一半数量的短原木，
        // 中心垫 3 层交错木垛作燃料床；原木直径 stick，位置、角度、粗细带随机扰动
        "bonfire" => {
            let r = o.radius.max(0.5);
            let h = if o.size[1] > 0.0 { o.size[1] } else { 1.3 * r };
            let n = o.count.max(6);
            let d = if o.stick > 0.0 { o.stick } else { 0.08 * r };
            let mut v = Vec::new();
            let mut k = 0u32;
            let mut rnd = || {
                k += 1;
                hash(k.wrapping_mul(2654435761) ^ 0x5bd1e995)
            };
            let tau = std::f32::consts::TAU;
            for (ring, cnt, rr, hh) in [(0, n, r, h), (1, n / 2, 0.6 * r, 0.75 * h)] {
                for i in 0..cnt {
                    let th =
                        tau * (i as f32 + 0.5 * ring as f32) / cnt as f32 + 0.25 * (rnd() - 0.5);
                    let rb = rr * (0.88 + 0.17 * rnd());
                    let a = [c[0] + rb * th.cos(), c[1] + 0.5 * d, c[2] + rb * th.sin()];
                    // 顶端在中轴附近交叉，略过中轴，原木伸出顶点一截
                    let tt = th + std::f32::consts::PI + 0.6 * (rnd() - 0.5);
                    let off = 0.07 * r * (0.5 + rnd());
                    let top = [
                        c[0] + off * tt.cos(),
                        c[1] + hh * (0.9 + 0.15 * rnd()),
                        c[2] + off * tt.sin(),
                    ];
                    let ext = 1.08 + 0.1 * rnd();
                    let b = [
                        a[0] + (top[0] - a[0]) * ext,
                        a[1] + (top[1] - a[1]) * ext,
                        a[2] + (top[2] - a[2]) * ext,
                    ];
                    v.push(Shape::Cyl {
                        a,
                        b,
                        r: 0.5 * d * (0.75 + 0.5 * rnd()),
                    });
                }
            }
            for l in 0..3 {
                let y = c[1] + d * (l as f32 + 0.5);
                for s in 0..5 {
                    let off = -0.5 * r + r * s as f32 / 4.0 + 0.1 * r * (rnd() - 0.5);
                    let len = 0.7 * r;
                    let ang = if l % 2 == 0 {
                        0.0
                    } else {
                        std::f32::consts::FRAC_PI_2
                    } + 0.3 * (rnd() - 0.5);
                    let (sa, ca) = ang.sin_cos();
                    let p0 = [c[0] - sa * off, y, c[2] + ca * off];
                    let a = [p0[0] - ca * len, y, p0[2] - sa * len];
                    let b = [p0[0] + ca * len, y, p0[2] + sa * len];
                    v.push(Shape::Cyl {
                        a,
                        b,
                        r: 0.5 * d * (0.7 + 0.4 * rnd()),
                    });
                }
            }
            v
        }
        k => bail!("未知物体类型 {k}（可选 box/cylinder/sphere/pool/crib/pallets/bonfire）"),
    })
}

fn rot(p: [f32; 3], yaw: f32, c: [f32; 3]) -> [f32; 3] {
    let (s, co) = yaw.sin_cos();
    // Shape::Box 用的是 lx = co·dx + s·dz，这里取其逆变换
    [
        c[0] + co * p[0] - s * p[2],
        c[1] + p[1],
        c[2] + s * p[0] + co * p[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(objects: &str) -> Scene {
        toml::from_str(&format!(
            "[domain]\nsize = [1.6, 1.6, 1.6]\ncell = 0.05\n{objects}"
        ))
        .unwrap()
    }

    #[test]
    fn grid_dims_are_multiples_of_16() {
        let s = scene("");
        let g = s.grid();
        assert_eq!(g.dims, [32, 32, 32]);
    }

    #[test]
    fn box_voxel_volume() {
        let s = scene(
            "[[object]]\nkind = \"box\"\nmaterial = \"wood\"\ncenter = [0.8, 0.4, 0.8]\nsize = [0.4, 0.4, 0.4]\n",
        );
        let sol = s.voxelize().unwrap();
        let n = sol.iter().filter(|c| c[1] > 0.0).count();
        // 0.4 m / 0.05 m = 8 格 → 512 格（边界取等号会多出一圈的一部分）
        assert!((512..=729).contains(&n), "{n}");
    }

    #[test]
    fn thin_parts_and_shallow_pools_keep_one_cell() {
        let s = scene(
            "[[object]]\nkind = \"box\"\nmaterial = \"wood\"\ncenter = [0.8, 0.8, 0.8]\nsize = [0.5, 0.01, 0.5]\n\
             [[object]]\nkind = \"pool\"\nmaterial = \"heptane\"\ncenter = [0.8, 0.0, 0.8]\nradius = 0.3\ndepth = 0.001\n",
        );
        let sol = s.voxelize().unwrap();
        let [nx, ny, _] = s.grid().dims.map(|v| v as usize);
        let at = |i: usize, j: usize, k: usize| sol[i + nx * (j + ny * k)];
        assert!(
            at(16, 16, 16)[1] > 0.0 || at(16, 15, 16)[1] > 0.0,
            "薄板被丢掉了"
        );
        assert!(at(16, 0, 16)[1] > 0.0, "浅油池被丢掉了");
        assert_eq!(at(16, 1, 16)[1], 0.0);
    }

    #[test]
    fn unknown_material_lists_choices() {
        let e = material("lava").unwrap_err().to_string();
        assert!(e.contains("heptane"));
    }
}
