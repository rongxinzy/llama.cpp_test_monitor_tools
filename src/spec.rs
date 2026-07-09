use crate::utils::{hostname, now_str};
use anyhow::Result;
use csv::WriterBuilder;
use std::path::Path;

pub const HARDWARE_SPEC_HEADER: [&str; 5] = ["测试批次", "模型", "精度", "项目", "值"];

#[derive(Debug, Clone)]
pub struct HardwareSpec {
    pub item: String,
    pub value: String,
}

pub async fn collect_hardware_specs() -> Vec<HardwareSpec> {
    let mut specs = vec![
        HardwareSpec {
            item: "采集时间".to_string(),
            value: now_str(),
        },
        HardwareSpec {
            item: "主机名".to_string(),
            value: hostname(),
        },
    ];

    specs.extend(system_specs().await);
    specs.extend(cpu_specs().await);
    specs.extend(memory_specs().await);
    specs.extend(gpu_specs().await);
    specs.extend(gpu_topo_specs().await);
    specs.extend(cuda_specs().await);

    specs
}

async fn run_cmd(program: &str, args: &[&str]) -> String {
    match tokio::process::Command::new(program)
        .args(args)
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => String::new(),
    }
}

async fn system_specs() -> Vec<HardwareSpec> {
    let mut specs = Vec::new();
    let uname = run_cmd("uname", &["-srm"]).await;
    if !uname.is_empty() {
        specs.push(HardwareSpec {
            item: "操作系统".to_string(),
            value: uname,
        });
    }
    specs.push(HardwareSpec {
        item: "内核版本".to_string(),
        value: run_cmd("uname", &["-r"]).await,
    });
    specs
}

async fn cpu_specs() -> Vec<HardwareSpec> {
    let mut specs = Vec::new();
    let cpuinfo = run_cmd("bash", &["-lc", "cat /proc/cpuinfo 2>/dev/null || true"]).await;
    let model = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.splitn(2, ':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let cores = cpuinfo
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count();
    let sockets = cpuinfo
        .lines()
        .filter(|l| l.starts_with("physical id"))
        .collect::<std::collections::HashSet<_>>()
        .len();

    if !model.is_empty() {
        specs.push(HardwareSpec {
            item: "CPU 型号".to_string(),
            value: model,
        });
    }
    if cores > 0 {
        specs.push(HardwareSpec {
            item: "CPU 逻辑核心数".to_string(),
            value: cores.to_string(),
        });
    }
    if sockets > 0 {
        specs.push(HardwareSpec {
            item: "CPU 物理插槽数".to_string(),
            value: sockets.to_string(),
        });
    }
    specs
}

async fn memory_specs() -> Vec<HardwareSpec> {
    let mut specs = Vec::new();
    let meminfo = run_cmd("bash", &["-lc", "cat /proc/meminfo 2>/dev/null || true"]).await;
    if let Some(line) = meminfo.lines().find(|l| l.starts_with("MemTotal:")) {
        if let Some(v) = line.splitn(2, ':').nth(1) {
            specs.push(HardwareSpec {
                item: "内存总量".to_string(),
                value: v.trim().to_string(),
            });
        }
    }
    specs.push(HardwareSpec {
        item: "内存总量(可读)".to_string(),
        value: run_cmd("free", &["-h"])
            .await
            .lines()
            .nth(1)
            .map(|l| l.split_whitespace().nth(1).unwrap_or("").to_string())
            .unwrap_or_default(),
    });
    specs
}

async fn gpu_specs() -> Vec<HardwareSpec> {
    let mut specs = Vec::new();
    let list = run_cmd("nvidia-smi", &["-L"]).await;
    if !list.is_empty() {
        let names: Vec<String> = list
            .lines()
            .filter(|l| l.starts_with("GPU "))
            .filter_map(|l| l.split(':').nth(1))
            .map(|s| s.split("(UUID").next().unwrap_or(s).trim().to_string())
            .collect();
        if !names.is_empty() {
            specs.push(HardwareSpec {
                item: "GPU 数量".to_string(),
                value: names.len().to_string(),
            });
            specs.push(HardwareSpec {
                item: "GPU 型号".to_string(),
                value: names.join(" / "),
            });
        }
    }

    let query = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
    )
    .await;
    if !query.is_empty() {
        for (idx, line) in query.lines().enumerate() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let prefix = format!("GPU[{}] ", idx);
                specs.push(HardwareSpec {
                    item: format!("{}名称", prefix),
                    value: parts[0].to_string(),
                });
                specs.push(HardwareSpec {
                    item: format!("{}显存(MB)", prefix),
                    value: parts[1].to_string(),
                });
                specs.push(HardwareSpec {
                    item: format!("{}驱动版本", prefix),
                    value: parts[2].to_string(),
                });
                if parts.len() >= 5 {
                    specs.push(HardwareSpec {
                        item: format!("{}PCIe 最大链路", prefix),
                        value: format!("Gen{} x{}", parts[3], parts[4]),
                    });
                }
            }
        }
    }
    specs
}

async fn gpu_topo_specs() -> Vec<HardwareSpec> {
    let mut specs = Vec::new();
    let topo = strip_ansi_escape_sequences(&run_cmd("nvidia-smi", &["topo", "-m"]).await);
    if !topo.is_empty() {
        specs.push(HardwareSpec {
            item: "GPU 拓扑结构".to_string(),
            value: topo,
        });
    }
    specs
}

fn strip_ansi_escape_sequences(text: &str) -> String {
    // Remove ANSI escape sequences such as color codes.
    let re = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    re.replace_all(text, "").to_string()
}

async fn cuda_specs() -> Vec<HardwareSpec> {
    let mut specs = Vec::new();
    let nvcc = run_cmd(
        "bash",
        &["-lc", "nvcc --version 2>/dev/null | grep release || true"],
    )
    .await;
    if !nvcc.is_empty() {
        specs.push(HardwareSpec {
            item: "CUDA 版本".to_string(),
            value: nvcc.lines().next().unwrap_or(&nvcc).trim().to_string(),
        });
    }
    specs
}

pub fn write_hardware_spec_csv(
    path: &Path,
    run_id: &str,
    model_name: &str,
    dtype: &str,
    specs: &[HardwareSpec],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wtr = WriterBuilder::new().from_path(path)?;
    wtr.write_record(&HARDWARE_SPEC_HEADER)?;
    for spec in specs {
        wtr.write_record([run_id, model_name, dtype, &spec.item, &spec.value])?;
    }
    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_hardware_spec_csv() {
        let tmp = std::env::temp_dir().join("llama-test-matrix-hardware-spec-test.csv");
        let _ = std::fs::remove_file(&tmp);
        let specs = vec![
            HardwareSpec {
                item: "CPU 型号".to_string(),
                value: "AMD EPYC 7763".to_string(),
            },
            HardwareSpec {
                item: "GPU 数量".to_string(),
                value: "8".to_string(),
            },
        ];
        write_hardware_spec_csv(&tmp, "r1", "qwen", "q3", &specs).unwrap();
        let text = std::fs::read_to_string(&tmp).unwrap();
        assert!(text.contains("测试批次"));
        assert!(text.contains("AMD EPYC 7763"));
        assert!(text.contains("r1,qwen,q3"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_parse_gpu_list() {
        let line = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-xxx)";
        let name = line
            .split(':')
            .nth(1)
            .map(|s| s.split("(UUID").next().unwrap_or(s).trim())
            .unwrap_or("");
        assert_eq!(name, "NVIDIA GeForce RTX 4090");
    }

    #[test]
    fn test_strip_ansi_escape_sequences() {
        assert_eq!(strip_ansi_escape_sequences("\x1b[4mGPU0\x1b[0m"), "GPU0");
        assert_eq!(strip_ansi_escape_sequences("no ansi"), "no ansi");
    }
}
