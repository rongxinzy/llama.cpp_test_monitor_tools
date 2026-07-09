use crate::config::BlackboxConfig;
use crate::utils::{file_ts, hostname};
use anyhow::Result;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{Duration, sleep};

pub mod collector;
pub mod logs;
pub mod snapshot;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TriggerEvent {
    pub source: String,
    pub line: String,
}

#[allow(dead_code)]
pub struct BlackboxHandle {
    pub run_dir: PathBuf,
    pub trigger_count: Arc<AtomicUsize>,
    pub stop_flag: Arc<AtomicBool>,
    tasks: Vec<JoinHandle<()>>,
}

impl BlackboxHandle {
    pub async fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        for task in self.tasks.drain(..) {
            let _ = task.await;
        }
    }

    pub fn triggers(&self) -> usize {
        self.trigger_count.load(Ordering::SeqCst)
    }
}

pub async fn start_blackbox(config: BlackboxConfig) -> Result<BlackboxHandle> {
    let host = hostname();
    let host = host.replace(
        |c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.' && c != '-',
        "_",
    );
    let run_id = format!("{}-{}", file_ts(), host);
    let run_dir = PathBuf::from(&config.out).join(&run_id);
    fs::create_dir_all(run_dir.join("logs")).await?;
    fs::create_dir_all(run_dir.join("metrics")).await?;
    fs::create_dir_all(run_dir.join("snapshots")).await?;
    fs::create_dir_all(run_dir.join("incidents")).await?;
    fs::create_dir_all(run_dir.join("events")).await?;
    fs::create_dir_all(run_dir.join("meta")).await?;
    fs::create_dir_all(run_dir.join("final")).await?;

    // create latest symlink
    if let Ok(canonical) = fs::canonicalize(&run_dir).await {
        let latest = PathBuf::from(&config.out).join("latest");
        let _ = std::fs::remove_file(&latest);
        let _ = std::os::unix::fs::symlink(canonical, latest);
    }

    let log_path = run_dir.join("blackbox.log");
    let triggers_path = run_dir.join("events").join("triggers.log");
    let trigger_file = run_dir.join(".trigger");
    let trigger_re = Regex::new(&config.trigger_regex)?;

    log(
        &log_path,
        &format!("gpu-blackbox rust version; run_dir={}", run_dir.display()),
    )
    .await;

    let trigger_count = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let (tx, mut rx) = mpsc::channel::<TriggerEvent>(1024);

    let run_dir2 = run_dir.clone();
    let stop_flag2 = stop_flag.clone();
    let trigger_count2 = trigger_count.clone();
    let cooldown = config.cooldown;
    let trigger_file2 = trigger_file.clone();
    let triggers_path2 = triggers_path.clone();
    let log_path2 = log_path.clone();
    let snapshot_handle = tokio::spawn(async move {
        let mut last_capture: Option<std::time::Instant> = None;
        while !stop_flag2.load(Ordering::SeqCst) {
            let triggered = rx.try_recv().is_ok() || trigger_file2.exists();
            if triggered {
                let now = std::time::Instant::now();
                let allowed = match last_capture {
                    Some(t) => now.duration_since(t).as_secs() >= cooldown,
                    None => true,
                };
                if allowed {
                    trigger_count2.fetch_add(1, Ordering::SeqCst);
                    let _ = fs::remove_file(&trigger_file2).await;
                    let incident_id = file_ts();
                    log(
                        &log_path2,
                        &format!("capturing incident: {} (trigger)", incident_id),
                    )
                    .await;
                    let _ =
                        snapshot::capture_incident(&run_dir2, &incident_id, &triggers_path2).await;
                    last_capture = Some(now);
                } else {
                    let _ = fs::remove_file(&trigger_file2).await;
                    log(
                        &log_path2,
                        &format!("trigger suppressed by cooldown={}s", cooldown),
                    )
                    .await;
                }
            }
            sleep(Duration::from_secs(1)).await;
        }
    });

    let mut tasks = vec![snapshot_handle];

    // nvidia-smi gpu metrics
    let run_dir3 = run_dir.clone();
    let stop_flag3 = stop_flag.clone();
    let tx3 = tx.clone();
    let interval = config.interval;
    tasks.push(tokio::spawn(async move {
        collector::gpu_metrics_loop(&run_dir3, interval, stop_flag3, tx3).await;
    }));

    // nvidia-smi table
    let run_dir4 = run_dir.clone();
    let stop_flag4 = stop_flag.clone();
    let tx4 = tx.clone();
    let detail_interval = config.detail_interval;
    tasks.push(tokio::spawn(async move {
        collector::nvidia_smi_table_loop(&run_dir4, detail_interval, stop_flag4, tx4).await;
    }));

    // compute apps
    let run_dir5 = run_dir.clone();
    let stop_flag5 = stop_flag.clone();
    let tx5 = tx.clone();
    let ps_interval = config.ps_interval;
    tasks.push(tokio::spawn(async move {
        collector::compute_apps_loop(&run_dir5, ps_interval, stop_flag5, tx5).await;
    }));

    // log followers
    let run_dir6 = run_dir.clone();
    let stop_flag6 = stop_flag.clone();
    let tx6 = tx.clone();
    let trigger_re2 = trigger_re.clone();
    tasks.push(tokio::spawn(async move {
        logs::start_log_followers(&run_dir6, stop_flag6, tx6, trigger_re2).await;
    }));

    // process/system light
    let run_dir7 = run_dir.clone();
    let stop_flag7 = stop_flag.clone();
    tasks.push(tokio::spawn(async move {
        collector::process_system_light_loop(&run_dir7, ps_interval, stop_flag7).await;
    }));

    // system slow
    let run_dir8 = run_dir.clone();
    let stop_flag8 = stop_flag.clone();
    tasks.push(tokio::spawn(async move {
        collector::system_slow_loop(&run_dir8, detail_interval, stop_flag8).await;
    }));

    // PCIe throughput (nvidia-smi dmon -s t)
    let run_dir9 = run_dir.clone();
    let stop_flag9 = stop_flag.clone();
    let interval = config.interval;
    tasks.push(tokio::spawn(async move {
        collector::pcie_throughput_loop(&run_dir9, interval, stop_flag9).await;
    }));

    // GPU topology diff
    let run_dir10 = run_dir.clone();
    let stop_flag10 = stop_flag.clone();
    let tx10 = tx.clone();
    tasks.push(tokio::spawn(async move {
        collector::gpu_topo_loop(&run_dir10, detail_interval, stop_flag10, tx10).await;
    }));

    // PCIe AER counters
    let run_dir11 = run_dir.clone();
    let stop_flag11 = stop_flag.clone();
    tasks.push(tokio::spawn(async move {
        collector::pcie_aer_loop(&run_dir11, detail_interval, stop_flag11).await;
    }));

    // baseline snapshot
    let _ = snapshot::collect_baseline(&run_dir).await;

    Ok(BlackboxHandle {
        run_dir,
        trigger_count,
        stop_flag,
        tasks,
    })
}

pub async fn log(path: &Path, message: &str) {
    let line = format!(
        "[{}] {}\n",
        chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%z"),
        message
    );
    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = file.write_all(line.as_bytes()).await;
}

pub async fn finalize_blackbox(handle: BlackboxHandle) {
    handle.stop().await;
}
