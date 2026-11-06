//! wgpu 设备、缓冲与计算内核的薄封装

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        // 默认：macOS 用 Metal，Windows 用 DX12/Vulkan，Linux 用 Vulkan；可用环境变量 WGPU_BACKEND=vulkan|dx12|metal 指定
        desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("找不到可用的 GPU")?;
        let info = adapter.get_info();
        let name = format!("{}（{:?}）", info.name, info.backend);
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pyro"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .context("创建 GPU 设备失败")?;
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("GPU 错误: {e}");
            std::process::exit(2);
        }));
        Ok(Gpu {
            device,
            queue,
            name,
        })
    }

    pub fn storage(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn readback(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// 同步读回一个 MAP_READ 缓冲（调用前应已提交拷贝命令）
    pub fn read_buffer(&self, buf: &wgpu::Buffer) -> Result<Vec<u8>> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| anyhow!("GPU 等待失败: {e:?}"))?;
        rx.recv()?.map_err(|e| anyhow!("映射失败: {e:?}"))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| anyhow!("映射失败: {e:?}"))?
            .to_vec();
        buf.unmap();
        Ok(data)
    }

    pub fn wait(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| anyhow!("GPU 等待失败: {e:?}"))?;
        Ok(())
    }

    pub fn kernel(&self, label: &str, src: &str, constants: &[(&str, f64)]) -> Kernel {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let pipe = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });
        Kernel {
            pipe,
            label: label.to_string(),
        }
    }
}

pub struct Kernel {
    pub pipe: wgpu::ComputePipeline,
    pub label: String,
}

impl Kernel {
    /// 按顺序把资源绑到 binding 0..n
    pub fn bind(&self, dev: &wgpu::Device, res: &[wgpu::BindingResource]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = res
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect();
        dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&self.label),
            layout: &self.pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// 按给定的 binding 号绑定（用于绑定号不连续的内核）
    pub fn bind_at(
        &self,
        dev: &wgpu::Device,
        res: &[(u32, wgpu::BindingResource)],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = res
            .iter()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: *i,
                resource: r.clone(),
            })
            .collect();
        dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&self.label),
            layout: &self.pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    pub fn run(&self, pass: &mut wgpu::ComputePass, bg: &wgpu::BindGroup, groups: [u32; 3]) {
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups[0].max(1), groups[1].max(1), groups[2].max(1));
    }
}

// ---------- WGSL 组装：一个文件里多个内核，用指令拼入公共代码与模板 ----------
const COMMON: &str = include_str!("shaders/common.wgsl");
const T_SCALAR: &str = include_str!("shaders/tmpl_scalar.wgsl");
const T_COMP: &str = include_str!("shaders/tmpl_comp.wgsl");
const VEL: &str = include_str!("shaders/vel.wgsl");
const STATS: &str = include_str!("shaders/stats.wgsl");
const TRACE: &str = include_str!("shaders/trace_lib.wgsl");

pub fn split_kernels(file: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut name: Option<String> = None;
    let mut body = String::new();
    for line in file.lines() {
        if let Some(n) = line.strip_prefix("//#KERNEL ") {
            if let Some(prev) = name.take() {
                out.insert(prev, std::mem::take(&mut body));
            }
            name = Some(n.trim().to_string());
        } else if name.is_some() {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(prev) = name {
        out.insert(prev, body);
    }
    out
}

/// 按段内指令拼装：//@COMP 名…、//@SCAL 名…、//@VEL、//@STATS 绑定号；with_common 时前置 common.wgsl
pub fn assemble(body: &str, with_common: bool) -> String {
    let mut s = String::new();
    if with_common {
        s.push_str(COMMON);
    }
    for line in body.lines() {
        let t = line.trim();
        if let Some(r) = t.strip_prefix("//@COMP ") {
            for n in r.split_whitespace() {
                s.push_str(&T_COMP.replace("XBUF", n));
            }
        } else if let Some(r) = t.strip_prefix("//@SCAL ") {
            for n in r.split_whitespace() {
                s.push_str(&T_SCALAR.replace("XBUF", n));
            }
        } else if t == "//@VEL" {
            s.push_str(VEL);
        } else if t == "//@TRACE" {
            s.push_str(TRACE);
        } else if let Some(r) = t.strip_prefix("//@STATS ") {
            s.push_str(&STATS.replace("STATSB", r.trim()));
        }
    }
    s.push_str(body);
    s
}

pub fn groups3(d: [u32; 3], wg: [u32; 3]) -> [u32; 3] {
    [
        d[0].div_ceil(wg[0]),
        d[1].div_ceil(wg[1]),
        d[2].div_ceil(wg[2]),
    ]
}

pub fn b(x: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    x.as_entire_binding()
}
