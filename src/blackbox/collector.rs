use super::TriggerEvent;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::fs::{self, OpenOptions};
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
    let _ = file.flush().await;
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

pub async fn pcie_throughput_loop(run_dir: &Path, interval: f64, stop_flag: Arc<AtomicBool>) {
    let outfile = run_dir.join("metrics").join("nvidia_smi_pcie.csv");
    let _ = append_log(
        &outfile,
        "collector_ts,gpu_index,rxpci_mbps,txpci_mbps\n".to_string(),
    )
    .await;
    let delay = interval.max(1.0).round() as u64;

    while !stop_flag.load(Ordering::SeqCst) {
        let mut cmd = Command::new("nvidia-smi");
        cmd.args(["dmon", "-s", "t", "-d", &delay.to_string()])
            .stdout(Stdio::piped());
        match cmd.spawn() {
            Ok(mut child) => {
                if let Some(stdout) = child.stdout.take() {
                    use tokio::io::{AsyncBufReadExt, BufReader};
                    let mut reader = BufReader::new(stdout).lines();
                    while let Ok(Some(line)) = reader.next_line().await {
                        if stop_flag.load(Ordering::SeqCst) {
                            break;
                        }
                        let trimmed = line.trim();
                        if trimmed.is_empty() || trimmed.starts_with('#') {
                            continue;
                        }
                        let parts: Vec<&str> = trimmed.split_whitespace().collect();
                        if parts.len() >= 3 {
                            let _ = append_log(
                                &outfile,
                                format!("{},{},{},{}\n", now_log(), parts[0], parts[1], parts[2]),
                            )
                            .await;
                        }
                    }
                }
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
            Err(e) => {
                let _ =
                    append_log(&outfile, format!("# nvidia-smi dmon spawn failed: {}\n", e)).await;
                sleep(Duration::from_secs(delay)).await;
            }
        }
    }
}

async fn topo_string() -> String {
    match Command::new("nvidia-smi")
        .args(["topo", "-m"])
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).to_string()
        }
        _ => String::new(),
    }
}

fn normalize_topo(text: &str) -> String {
    // Strip ANSI color codes and collapse repeated whitespace so small
    // formatting differences do not cause false positives.
    let re = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    re.replace_all(text, "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub async fn gpu_topo_loop(
    run_dir: &Path,
    interval: f64,
    stop_flag: Arc<AtomicBool>,
    tx: Sender<TriggerEvent>,
) {
    let baseline_path = run_dir.join("snapshots").join("topo_baseline.txt");
    let logfile = run_dir.join("logs").join("nvidia_smi_topo.log");

    if !baseline_path.exists() {
        let baseline = topo_string().await;
        let _ = fs::write(&baseline_path, &baseline).await;
    }
    let baseline = fs::read_to_string(&baseline_path).await.unwrap_or_default();
    let normalized_baseline = normalize_topo(&baseline);

    let delay = interval.max(5.0).round() as u64;
    while !stop_flag.load(Ordering::SeqCst) {
        let current = topo_string().await;
        let ts = now_log();
        let _ = append_log(&logfile, format!("\n===== {} =====\n{}\n", ts, current)).await;

        if !normalized_baseline.is_empty() && normalize_topo(&current) != normalized_baseline {
            let _ = tx
                .send(TriggerEvent {
                    source: "gpu-topo-diff".to_string(),
                    line: format!("GPU topology changed at {}", ts),
                })
                .await;
        }
        sleep(Duration::from_secs(delay)).await;
    }
}

pub async fn pcie_aer_loop(run_dir: &Path, interval: f64, stop_flag: Arc<AtomicBool>) {
    let outfile = run_dir.join("metrics").join("pcie_aer.csv");
    let _ = append_log(
        &outfile,
        "collector_ts,pci_address,total_cor,total_nonfatal,total_fatal\n".to_string(),
    )
    .await;

    let script = r#"for dev in /sys/bus/pci/devices/*; do
  if [ -f "$dev/vendor" ] && [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x10de" ]; then
    addr=$(basename "$dev")
    cor=$(awk '/^TOTAL_ERR_COR/{print $2}' "$dev/aer_dev_correctable" 2>/dev/null)
    non=$(awk '/^TOTAL_ERR_NONFATAL/{print $2}' "$dev/aer_dev_nonfatal" 2>/dev/null)
    fat=$(awk '/^TOTAL_ERR_FATAL/{print $2}' "$dev/aer_dev_fatal" 2>/dev/null)
    echo "$addr ${cor:-} ${non:-} ${fat:-}"
  fi
done"#;

    let delay = interval.max(10.0).round() as u64;
    while !stop_flag.load(Ordering::SeqCst) {
        let now = now_log();
        let mut buf = String::new();
        if let Ok(output) = Command::new("bash").args(["-lc", script]).output().await {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 4 {
                    buf.push_str(&format!(
                        "{},{},{},{},{}\n",
                        now, parts[0], parts[1], parts[2], parts[3]
                    ));
                }
            }
        }
        let _ = append_log(&outfile, buf).await;
        sleep(Duration::from_secs(delay)).await;
    }
}
