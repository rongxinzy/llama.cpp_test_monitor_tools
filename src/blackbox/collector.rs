use super::TriggerEvent;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc::Sender;
use tokio::time::{Duration, sleep};

fn now_log() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%.3f%z")
        .to_string()
}

fn looks_like_error(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("err")
        || lower.contains("unknown error")
        || lower.contains("unable to determine")
        || lower.contains("failed to initialize nvml")
        || lower.contains("no devices were found")
        || lower.contains("has fallen off")
}

async fn append_log(path: &Path, text: String) {
    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = file.write_all(text.as_bytes()).await;
}

pub async fn gpu_metrics_loop(
    run_dir: &Path,
    interval: f64,
    stop_flag: Arc<AtomicBool>,
    tx: Sender<TriggerEvent>,
) {
    let fields = [
        "timestamp",
        "index",
        "name",
        "pci.bus_id",
        "temperature.gpu",
        "temperature.memory",
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "pcie.link.gen.current",
        "pcie.link.width.current",
        "clocks_throttle_reasons.active",
    ]
    .join(",");
    let outfile = run_dir.join("metrics").join("nvidia_smi_gpu.csv");
    let _ = append_log(&outfile, format!("collector_ts,{}\n", fields)).await;

    while !stop_flag.load(Ordering::SeqCst) {
        match Command::new("nvidia-smi")
            .args(["--query-gpu", &fields, "--format=csv,noheader,nounits"])
            .output()
            .await
        {
            Ok(output) => {
                let text = String::from_utf8_lossy(&output.stdout);
                let now = now_log();
                let mut buf = String::new();
                for line in text.lines() {
                    if !line.trim().is_empty() {
                        buf.push_str(&format!("{},{}\n", now, line));
                    }
                }
                let _ = append_log(&outfile, buf).await;
                if output.status.success() {
                    if looks_like_error(&text) {
                        let _ = tx
                            .send(TriggerEvent {
                                source: "nvidia-smi-query".to_string(),
                                line: text.to_string(),
                            })
                            .await;
                    }
                } else {
                    let err = String::from_utf8_lossy(&output.stderr);
                    let _ = tx
                        .send(TriggerEvent {
                            source: "nvidia-smi-query".to_string(),
                            line: format!("rc={} {}", output.status, err),
                        })
                        .await;
                }
            }
            Err(e) => {
                let _ = tx
                    .send(TriggerEvent {
                        source: "nvidia-smi-query".to_string(),
                        line: format!("spawn failed: {}", e),
                    })
                    .await;
            }
        }
        sleep(Duration::from_secs_f64(interval)).await;
    }
}

pub async fn nvidia_smi_table_loop(
    run_dir: &Path,
    interval: f64,
    stop_flag: Arc<AtomicBool>,
    tx: Sender<TriggerEvent>,
) {
    let outfile = run_dir.join("logs").join("nvidia_smi_table.log");
    while !stop_flag.load(Ordering::SeqCst) {
        match Command::new("nvidia-smi").output().await {
            Ok(output) => {
                let text = String::from_utf8_lossy(&output.stdout);
                let header = format!(
                    "\n===== {} nvidia-smi rc={} =====\n",
                    now_log(),
                    output.status
                );
                let _ = append_log(&outfile, format!("{}{}\n", header, text)).await;
                if !output.status.success() || looks_like_error(&text) {
                    let _ = tx
                        .send(TriggerEvent {
                            source: "nvidia-smi-table".to_string(),
                            line: format!("rc={} {}", output.status, text),
                        })
                        .await;
                }
            }
            Err(e) => {
                let _ = tx
                    .send(TriggerEvent {
                        source: "nvidia-smi-table".to_string(),
                        line: format!("spawn failed: {}", e),
                    })
                    .await;
            }
        }
        sleep(Duration::from_secs_f64(interval)).await;
    }
}

pub async fn compute_apps_loop(
    run_dir: &Path,
    interval: f64,
    stop_flag: Arc<AtomicBool>,
    tx: Sender<TriggerEvent>,
) {
    let outfile = run_dir.join("metrics").join("nvidia_compute_apps.csv");
    let _ = append_log(
        &outfile,
        "collector_ts,pid,process_name,used_gpu_memory,gpu_uuid\n".to_string(),
    )
    .await;
    while !stop_flag.load(Ordering::SeqCst) {
        match Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await
        {
            Ok(output) => {
                let text = String::from_utf8_lossy(&output.stdout);
                let now = now_log();
                let mut buf = String::new();
                for line in text.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty()
                        && !trimmed.to_lowercase().contains("no running processes")
                    {
                        buf.push_str(&format!("{},{}\n", now, trimmed));
                    }
                }
                let _ = append_log(&outfile, buf).await;
                if !output.status.success() || looks_like_error(&text) {
                    let _ = tx
                        .send(TriggerEvent {
                            source: "nvidia-compute-apps".to_string(),
                            line: format!("rc={} {}", output.status, text),
                        })
                        .await;
                }
            }
            Err(e) => {
                let _ = tx
                    .send(TriggerEvent {
                        source: "nvidia-compute-apps".to_string(),
                        line: format!("spawn failed: {}", e),
                    })
                    .await;
            }
        }
        sleep(Duration::from_secs_f64(interval)).await;
    }
}

pub async fn process_system_light_loop(run_dir: &Path, interval: f64, stop_flag: Arc<AtomicBool>) {
    let outfile = run_dir.join("metrics").join("process_system_light.log");
    while !stop_flag.load(Ordering::SeqCst) {
        let mut buf = format!("\n===== {} process/system light =====\n", now_log());
        for cmd in [
            vec!["uptime"],
            vec!["free", "-h"],
            vec![
                "ps",
                "-eo",
                "pid,ppid,stat,psr,%cpu,%mem,etime,comm,args",
                "--sort=-%cpu",
            ],
            vec![
                "ps",
                "-eo",
                "pid,ppid,stat,psr,%cpu,%mem,etime,comm,args",
                "--sort=-%mem",
            ],
        ] {
            if let Ok(output) = Command::new(&cmd[0]).args(&cmd[1..]).output().await {
                buf.push_str(&format!(
                    "--- {} ---\n{}",
                    cmd.join(" "),
                    String::from_utf8_lossy(&output.stdout)
                ));
            }
        }
        let _ = append_log(&outfile, buf).await;
        sleep(Duration::from_secs_f64(interval)).await;
    }
}

pub async fn system_slow_loop(run_dir: &Path, interval: f64, stop_flag: Arc<AtomicBool>) {
    let outfile = run_dir.join("metrics").join("system_slow.log");
    while !stop_flag.load(Ordering::SeqCst) {
        let mut buf = format!("\n===== {} system slow =====\n", now_log());
        for cmd in [
            vec!["vmstat", "1", "2"],
            vec!["iostat", "-xz", "1", "2"],
            vec!["sensors"],
            vec!["lsmod"],
        ] {
            if let Ok(output) = Command::new(&cmd[0]).args(&cmd[1..]).output().await {
                buf.push_str(&format!(
                    "--- {} ---\n{}",
                    cmd.join(" "),
                    String::from_utf8_lossy(&output.stdout)
                ));
            }
        }
        let _ = append_log(&outfile, buf).await;
        sleep(Duration::from_secs_f64(interval)).await;
    }
}
