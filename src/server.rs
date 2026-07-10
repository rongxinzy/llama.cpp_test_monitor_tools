use crate::utils::tail_file;
use anyhow::{Result, bail};
use reqwest::Client;
use serde_json::Value;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

pub struct LlamaServer {
    pub process: Child,
    pub _log_path: PathBuf,
}

pub fn build_server_command(
    llama_server_bin: &str,
    model_path: &str,
    model_name: &str,
    port: u16,
    gpu_devices: &[String],
    parallel_size: usize,
    ctx_size: usize,
    batch_size: usize,
    gpu_layers: usize,
) -> Vec<String> {
    let mut cmd = vec![
        llama_server_bin.to_string(),
        "--model".to_string(),
        model_path.to_string(),
        "--alias".to_string(),
        model_name.to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--parallel".to_string(),
        parallel_size.to_string(),
        "--ctx-size".to_string(),
        ctx_size.to_string(),
        "--batch-size".to_string(),
        batch_size.to_string(),
        "--cont-batching".to_string(),
        "--metrics".to_string(),
        "-ngl".to_string(),
        gpu_layers.to_string(),
    ];
    if !gpu_devices.is_empty() && gpu_devices[0] != "none" {
        cmd.push("--device".to_string());
        cmd.push(gpu_devices.join(","));
    }
    cmd
}

pub async fn start_server(
    llama_server_bin: &str,
    model_path: &str,
    model_name: &str,
    port: u16,
    gpu_devices: &[String],
    parallel_size: usize,
    ctx_size: usize,
    batch_size: usize,
    gpu_layers: usize,
    log_path: &Path,
) -> Result<LlamaServer> {
    let cmd = build_server_command(
        llama_server_bin,
        model_path,
        model_name,
        port,
        gpu_devices,
        parallel_size,
        ctx_size,
        batch_size,
        gpu_layers,
    );
    if let Some(parent) = log_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let log_file = File::create(log_path)?;

    let mut command = Command::new(&cmd[0]);
    command
        .args(&cmd[1..])
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file))
        .kill_on_drop(true);
    unsafe {
        command.pre_exec(|| {
            // Put llama-server into its own process group so killpg
            // only targets the server subtree, not the parent tool.
            libc::setpgid(0, 0);
            // If the parent tool dies unexpectedly, make sure llama-server
            // is terminated instead of becoming an orphan process.
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    let process = command.spawn()?;

    Ok(LlamaServer {
        process,
        _log_path: log_path.to_path_buf(),
    })
}

pub async fn wait_for_ready(
    port: u16,
    process: &mut Child,
    log_path: &Path,
    model_name: &str,
    ready_timeout_sec: u64,
) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(ready_timeout_sec);

    while tokio::time::Instant::now() < deadline {
        if let Some(status) = process.try_wait()? {
            let tail = tail_file(log_path, 80);
            bail!("llama-server 提前退出 (rc={})，日志尾部:\n{}", status, tail);
        }
        match fetch_models(&client, port).await {
            Ok((200, Some(data))) => {
                if model_response_matches(&data, model_name) {
                    return Ok(());
                }
                bail!(
                    "端口 {} 返回了 200，但模型名不是当前启动的 {}",
                    port,
                    model_name
                );
            }
            Ok((status, body)) => {
                tracing::debug!("models endpoint status={} body={:?}", status, body);
            }
            Err(e) => {
                tracing::debug!("models endpoint error: {}", e);
            }
        }
        sleep(Duration::from_secs(5)).await;
    }
    bail!("等待服务启动超时: http://127.0.0.1:{}/v1/models", port)
}

async fn fetch_models(client: &Client, port: u16) -> Result<(u16, Option<Value>)> {
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let resp = client.get(&url).send().await;
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            let text = r.text().await.unwrap_or_default();
            let data = serde_json::from_str::<Value>(&text).ok();
            Ok((status, data))
        }
        Err(e) => Err(anyhow::anyhow!("{}", e)),
    }
}

fn model_response_matches(data: &Value, model_name: &str) -> bool {
    let candidates = data
        .get("data")
        .or_else(|| data.get("models"))
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    let items = match candidates {
        Value::Array(arr) => arr,
        Value::Object(_) => vec![Value::Object(candidates.as_object().unwrap().clone())],
        _ => return false,
    };
    for item in items {
        let names: Vec<String> = vec![
            item.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            item.get("name").and_then(|v| v.as_str()).unwrap_or(""),
            item.get("model").and_then(|v| v.as_str()).unwrap_or(""),
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
        let aliases: Vec<String> = item
            .get("aliases")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        for name in names.iter().chain(aliases.iter()) {
            if name == model_name {
                return true;
            }
        }
    }
    false
}

pub async fn is_port_open(port: u16) -> bool {
    match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(_) => false,
        Err(_) => true,
    }
}

pub async fn stop_server(server: Option<LlamaServer>) {
    if let Some(mut server) = server {
        let _ = terminate_process_group(&mut server.process).await;
    }
}

async fn terminate_process_group(process: &mut Child) -> Result<()> {
    let pid = process.id().unwrap_or(0) as i32;
    if pid <= 0 {
        let _ = process.start_kill();
        let _ = timeout(Duration::from_secs(5), process.wait()).await;
        return Ok(());
    }

    let own_pgid = unsafe { libc::getpgid(0) };
    unsafe {
        let pgid = libc::getpgid(pid);
        // Only killpg if the child is in a different process group from us.
        if pgid > 0 && pgid != own_pgid {
            let _ = libc::killpg(pgid, libc::SIGTERM);
        } else {
            let _ = libc::kill(pid, libc::SIGTERM);
        }
    }
    let wait = timeout(Duration::from_secs(30), process.wait()).await;
    if wait.is_err() {
        unsafe {
            let pgid = libc::getpgid(pid);
            if pgid > 0 && pgid != own_pgid {
                let _ = libc::killpg(pgid, libc::SIGKILL);
            } else {
                let _ = libc::kill(pid, libc::SIGKILL);
            }
        }
        let _ = timeout(Duration::from_secs(10), process.wait()).await;
    }
    Ok(())
}

pub async fn append_log(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
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
    if !text.ends_with('\n') {
        let _ = file.write_all(b"\n").await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_server_command_no_gpu() {
        let cmd = build_server_command(
            "/bin/llama-server",
            "/tmp/m.gguf",
            "m",
            18080,
            &[],
            4,
            4096,
            512,
            99,
        );
        assert_eq!(cmd[0], "/bin/llama-server");
        assert!(cmd.contains(&"--model".to_string()));
        assert!(cmd.contains(&"/tmp/m.gguf".to_string()));
        assert!(cmd.contains(&"--parallel".to_string()));
        assert!(cmd.contains(&"4".to_string()));
        assert!(!cmd.contains(&"--device".to_string()));
    }

    #[test]
    fn test_build_server_command_with_gpu() {
        let cmd = build_server_command(
            "/bin/llama-server",
            "/tmp/m.gguf",
            "m",
            18080,
            &["CUDA0".to_string(), "CUDA1".to_string()],
            4,
            4096,
            512,
            99,
        );
        let idx = cmd.iter().position(|s| s == "--device").unwrap();
        assert_eq!(cmd[idx + 1], "CUDA0,CUDA1");
    }

    #[test]
    fn test_build_server_command_none_gpu() {
        let cmd = build_server_command(
            "/bin/llama-server",
            "/tmp/m.gguf",
            "m",
            18080,
            &["none".to_string()],
            4,
            4096,
            512,
            99,
        );
        assert!(!cmd.contains(&"--device".to_string()));
    }

    #[test]
    fn test_model_response_matches_id() {
        let data = json!({"data": [{"id": "m"}]});
        assert!(model_response_matches(&data, "m"));
        assert!(!model_response_matches(&data, "other"));
    }

    #[test]
    fn test_model_response_matches_aliases() {
        let data = json!({"data": [{"id": "x", "aliases": ["m", "y"]}]});
        assert!(model_response_matches(&data, "m"));
    }

    #[test]
    fn test_model_response_matches_models_array() {
        let data = json!({"models": [{"name": "m"}]});
        assert!(model_response_matches(&data, "m"));
    }

    #[test]
    fn test_model_response_matches_empty() {
        let data = json!({"data": []});
        assert!(!model_response_matches(&data, "m"));
    }
}
