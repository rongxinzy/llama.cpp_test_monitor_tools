use super::log;
use std::path::Path;
use tokio::fs;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

fn ts() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%.3f%z")
        .to_string()
}

async fn run_to_file(path: &Path, program: &str, args: &[&str]) {
    let _ = fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).await;
    let output = Command::new(program).args(args).output().await;
    let mut file = match OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await
    {
        Ok(f) => f,
        Err(_) => return,
    };
    match output {
        Ok(o) => {
            let _ = file.write_all(&o.stdout).await;
            if !o.stderr.is_empty() {
                let _ = file.write_all(b"\n--- stderr ---\n").await;
                let _ = file.write_all(&o.stderr).await;
            }
        }
        Err(e) => {
            let _ = file.write_all(format!("failed: {}", e).as_bytes()).await;
        }
    }
}

async fn append_cmd_output(file: &mut fs::File, label: &str, program: &str, args: &[&str]) {
    let _ = file
        .write_all(format!("\n===== {} {} =====\n", ts(), label).as_bytes())
        .await;
    match Command::new(program).args(args).output().await {
        Ok(o) => {
            let _ = file.write_all(&o.stdout).await;
            if !o.stderr.is_empty() {
                let _ = file.write_all(b"\n--- stderr ---\n").await;
                let _ = file.write_all(&o.stderr).await;
            }
        }
        Err(e) => {
            let _ = file.write_all(format!("failed: {}\n", e).as_bytes()).await;
        }
    }
}

pub async fn collect_baseline(run_dir: &Path) {
    let snap = run_dir.join("snapshots");
    run_to_file(&snap.join("system_baseline.txt"),
        "bash",
        &[
            "-lc",
            "date --iso-8601=ns 2>/dev/null || date; echo '--- identity ---'; hostnamectl 2>/dev/null || hostname; echo '--- kernel ---'; uname -a; echo '--- uptime/free ---'; uptime; free -h",
        ],
    )
    .await;

    run_to_file(
        &snap.join("nvidia_baseline.txt"),
        "bash",
        &[
            "-lc",
            "date --iso-8601=ns 2>/dev/null || date; echo '--- nvidia-smi ---'; nvidia-smi 2>/dev/null || true; echo '--- nvidia-smi -q ---'; nvidia-smi -q 2>/dev/null || true",
        ],
    )
    .await;

    run_to_file(
        &snap.join("topo_baseline.txt"),
        "nvidia-smi",
        &["topo", "-m"],
    )
    .await;

    run_to_file(
        &snap.join("pci_baseline.txt"),
        "bash",
        &[
            "-lc",
            "lspci -nn 2>/dev/null || true; echo; lspci -tv 2>/dev/null || true; echo; echo '--- lspci -vv NVIDIA ---'; lspci -vv -d 10de: 2>/dev/null || true",
        ],
    )
    .await;
}

pub async fn capture_incident(
    run_dir: &Path,
    incident_id: &str,
    _triggers_path: &Path,
) -> Result<(), std::io::Error> {
    let dir = run_dir.join("incidents").join(incident_id);
    fs::create_dir_all(&dir).await?;

    let mut reason = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(dir.join("reason.txt"))
        .await?;
    reason
        .write_all(
            format!(
                "time={}\nreason=trigger-file-or-log-pattern\nrun_dir={}\n",
                ts(),
                run_dir.display()
            )
            .as_bytes(),
        )
        .await?;

    // tail current logs/metrics into incident dir
    for pattern in [
        "logs/*.log",
        "metrics/*.log",
        "metrics/*.csv",
        "events/*.log",
    ] {
        let glob_path = run_dir.join(pattern);
        if let Ok(entries) = glob::glob(glob_path.to_string_lossy().as_ref()) {
            for entry in entries.flatten() {
                let dst = dir.join(format!(
                    "tail_{}",
                    entry.file_name().unwrap_or_default().to_string_lossy()
                ));
                let _ = tail_file(&entry, &dst, 5000).await;
            }
        }
    }

    let mut snap = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("nvidia_snapshot.txt"))
        .await?;
    for (label, cmd) in [
        ("nvidia-smi -L", vec!["-L"]),
        ("nvidia-smi", vec![]),
        ("nvidia-smi -q", vec!["-q"]),
        ("nvidia-smi topo -m", vec!["topo", "-m"]),
        ("nvidia-smi pmon", vec!["pmon", "-c", "1", "-s", "um"]),
    ] {
        let args: Vec<&str> = cmd.iter().map(|s| *s).collect();
        append_cmd_output(&mut snap, label, "nvidia-smi", &args).await;
    }

    append_cmd_output(
        &mut snap,
        "lspci -vv NVIDIA",
        "bash",
        &["-lc", "lspci -vv -d 10de: 2>/dev/null || true"],
    )
    .await;

    let mut kernel = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("kernel_snapshot.txt"))
        .await?;
    append_cmd_output(
        &mut kernel,
        "dmesg tail",
        "bash",
        &["-lc", "dmesg -T 2>/dev/null | tail -n 5000 || true"],
    )
    .await;

    let mut system = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("system_snapshot.txt"))
        .await?;
    for (label, script) in [
        (
            "ps top cpu",
            "ps -eo pid,ppid,stat,psr,%cpu,%mem,etime,comm,args --sort=-%cpu | head -n 120 || true",
        ),
        ("free", "free -h || true"),
        ("vmstat", "vmstat 1 3 || true"),
        ("sensors", "sensors || true"),
    ] {
        append_cmd_output(&mut system, label, "bash", &["-lc", script]).await;
    }

    // summary md
    let mut summary = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(dir.join("SUMMARY.md"))
        .await?;
    summary
        .write_all(
            format!(
                "# GPU Blackbox Incident Summary\n\nrun_dir: {}\nincident_dir: {}\ntime: {}\n",
                run_dir.display(),
                dir.display(),
                ts()
            )
            .as_bytes(),
        )
        .await?;

    log(
        &run_dir.join("blackbox.log"),
        &format!("incident captured: {}", dir.display()),
    )
    .await;

    // tar.gz
    let parent = run_dir.parent().unwrap_or(Path::new("."));
    let base = run_dir.file_name().unwrap_or_default().to_string_lossy();
    let archive = parent.join(format!("{}.tar.gz", base));
    let _ = Command::new("tar")
        .args([
            "-C",
            &parent.to_string_lossy(),
            "-czf",
            &archive.to_string_lossy(),
            &base,
        ])
        .output()
        .await;

    Ok(())
}

async fn tail_file(src: &Path, dst: &Path, n: usize) -> Result<(), std::io::Error> {
    let text = fs::read_to_string(src).await?;
    let lines: Vec<&str> = text.lines().collect();
    let tail: Vec<&str> = lines.iter().rev().take(n).rev().copied().collect();
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(dst)
        .await?;
    file.write_all(tail.join("\n").as_bytes()).await?;
    Ok(())
}
