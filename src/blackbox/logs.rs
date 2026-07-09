use super::TriggerEvent;
use regex::Regex;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::Sender;

fn looks_like_trigger(line: &str, re: &Regex) -> bool {
    re.is_match(line)
}

async fn follow_command(
    run_dir: &Path,
    name: &str,
    program: &str,
    args: &[&str],
    stop_flag: Arc<AtomicBool>,
    tx: Sender<TriggerEvent>,
    trigger_re: Regex,
) {
    let outfile = run_dir.join("logs").join(format!("{}.log", name));
    let mut child = match Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return,
    };
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => return,
    };
    let mut reader = BufReader::new(stdout).lines();
    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&outfile)
        .await
    {
        Ok(f) => f,
        Err(_) => return,
    };

    while !stop_flag.load(Ordering::SeqCst) {
        tokio::select! {
            biased;
            _ = async { while !stop_flag.load(Ordering::SeqCst) { tokio::time::sleep(tokio::time::Duration::from_secs(1)).await; } } => {
                break;
            }
            line = reader.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        let ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%z");
                        let out = format!("{} {}\n", ts, line);
                        let _ = tokio::io::AsyncWriteExt::write_all(&mut file, out.as_bytes()).await;
                        if looks_like_trigger(&line, &trigger_re) {
                            let _ = tx.send(TriggerEvent {
                                source: name.to_string(),
                                line,
                            }).await;
                        }
                    }
                    _ => break,
                }
            }
        }
    }
    let _ = child.start_kill();
}

pub async fn start_log_followers(
    run_dir: &Path,
    stop_flag: Arc<AtomicBool>,
    tx: Sender<TriggerEvent>,
    trigger_re: Regex,
) {
    let followers: Vec<(&str, &str, Vec<&str>)> = vec![
        ("dmesg_follow", "dmesg", vec!["-wT"]),
        (
            "journal_kernel",
            "journalctl",
            vec!["-kf", "-o", "short-iso", "--no-pager"],
        ),
        (
            "journal_nvidia_units",
            "journalctl",
            vec![
                "-f",
                "-o",
                "short-iso",
                "--no-pager",
                "-u",
                "nvidia-persistenced",
                "-u",
                "nvidia-fabricmanager",
                "-u",
                "nvidia-dcgm",
            ],
        ),
    ];

    for (name, program, args) in followers {
        if which::which(program).is_ok() {
            let stop = stop_flag.clone();
            let tx = tx.clone();
            let re = trigger_re.clone();
            let run_dir = run_dir.to_path_buf();
            let args: Vec<&str> = args;
            tokio::spawn(async move {
                follow_command(&run_dir, name, program, &args, stop, tx, re).await;
            });
        }
    }

    for log_path in ["/var/log/kern.log", "/var/log/messages", "/var/log/syslog"] {
        if std::path::Path::new(log_path).is_file() {
            let name = format!(
                "tail_{}",
                std::path::Path::new(log_path)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .replace('.', "_")
            );
            let stop = stop_flag.clone();
            let tx = tx.clone();
            let re = trigger_re.clone();
            let run_dir = run_dir.to_path_buf();
            let log_path = log_path.to_string();
            tokio::spawn(async move {
                follow_command(
                    &run_dir,
                    &name,
                    "tail",
                    &["-n", "0", "-F", &log_path],
                    stop,
                    tx,
                    re,
                )
                .await;
            });
        }
    }
}
