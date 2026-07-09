use crate::matrix::Case;
use anyhow::{Result, bail};
use futures::future::join_all;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::timeout;

#[derive(Debug, Clone, Default)]
pub struct BenchmarkMetrics {
    pub duration_s: f64,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub request_tps: f64,
    pub output_tok_s: f64,
    pub total_tok_s: f64,
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub prompt_token_source: String,
    pub output_token_source: String,
}

#[derive(Debug, Clone)]
pub struct BenchmarkOutput {
    pub text: String,
    pub metrics: BenchmarkMetrics,
}

pub fn detect_benchmark_mode(mode: &str) -> String {
    if mode != "auto" {
        return mode.to_string();
    }
    if std::path::Path::new("benchmark_serving.py").is_file() {
        return "benchmark_serving".to_string();
    }
    if which::which("vllm").is_ok() {
        return "vllm_cli".to_string();
    }
    "builtin".to_string()
}

pub async fn run_benchmark(
    mode: &str,
    host: &str,
    port: u16,
    model_path: &str,
    model_name: &str,
    case: &Case,
) -> Result<BenchmarkOutput> {
    match mode {
        "builtin" => run_builtin_benchmark(host, port, model_name, case).await,
        "benchmark_serving" => {
            run_external_benchmark(
                "benchmark_serving",
                host,
                port,
                model_path,
                model_name,
                case,
            )
            .await
        }
        "vllm_cli" => {
            run_external_benchmark("vllm_cli", host, port, model_path, model_name, case).await
        }
        _ => bail!("未知 benchmark mode: {}", mode),
    }
}

fn stable_case_seed(case: &Case, request_index: usize, salt: u64) -> u64 {
    let value = case.parallel_size as u64 * 1_000_003
        + case.num_prompts as u64 * 10_007
        + case.input_len as u64 * 101
        + case.output_len as u64 * 17
        + request_index as u64
        + salt;
    value & 0xFFFFFFFF
}

fn make_prompt(input_len: usize, seed: u64) -> String {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let prefix = format!(" request {}", rng.gen_range(0..1_000_000));
    let body_tokens = input_len.max(1);
    format!("{}{}", prefix, " a".repeat(body_tokens))
}

async fn read_sse_completion(
    client: Arc<Client>,
    host: &str,
    port: u16,
    model_name: &str,
    prompt: String,
    prompt_len: usize,
    output_len: usize,
) -> RequestResult {
    let url = format!("http://{}:{}/v1/completions", host, port);
    let payloads = vec![
        json!({
            "model": model_name,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0,
            "stream": true,
            "ignore_eos": true,
            "stream_options": {"include_usage": true},
        }),
        json!({
            "model": model_name,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0,
            "stream": true,
        }),
    ];

    let started = Instant::now();
    let mut last_http_error = String::new();

    for payload in payloads {
        let builder = client.post(&url).json(&payload);
        let mut es = match EventSource::new(builder) {
            Ok(es) => es,
            Err(e) => {
                last_http_error = format!("eventsource new failed: {}", e);
                continue;
            }
        };
        let mut first_token_at: Option<Instant> = None;
        let mut _last_token_at = started;
        let mut output_chunks = 0usize;
        let mut completion_tokens: Option<usize> = None;
        let mut prompt_tokens: Option<usize> = None;
        let mut generated_text = String::new();

        loop {
            match timeout(Duration::from_secs(3600), futures::StreamExt::next(&mut es)).await {
                Ok(Some(Ok(Event::Open))) => {}
                Ok(Some(Ok(Event::Message(message)))) => {
                    let chunk = message.data.trim();
                    if chunk == "[DONE]" {
                        break;
                    }
                    match serde_json::from_str::<Value>(chunk) {
                        Ok(parsed) => {
                            if let Some(usage) = parsed.get("usage") {
                                if let Some(v) =
                                    usage.get("completion_tokens").and_then(|v| v.as_u64())
                                {
                                    completion_tokens = Some(v as usize);
                                }
                                if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64())
                                {
                                    prompt_tokens = Some(v as usize);
                                }
                            }
                            if let Some(choices) = parsed.get("choices").and_then(|v| v.as_array())
                            {
                                if let Some(choice) = choices.first() {
                                    if let Some(text) = choice.get("text").and_then(|v| v.as_str())
                                    {
                                        let now = Instant::now();
                                        if first_token_at.is_none() {
                                            first_token_at = Some(now);
                                        }
                                        _last_token_at = now;
                                        output_chunks += text
                                            .chars()
                                            .filter(|c| !c.is_whitespace())
                                            .count()
                                            .max(1);
                                        generated_text.push_str(text);
                                    }
                                }
                            }
                        }
                        Err(_) => {}
                    }
                }
                Ok(Some(Err(e))) => {
                    last_http_error = format!("eventsource error: {}", e);
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    last_http_error = "SSE read timeout".to_string();
                    break;
                }
            }
        }
        es.close();

        if first_token_at.is_some() {
            let ended = Instant::now();
            let actual_output = completion_tokens.unwrap_or(output_chunks);
            let actual_prompt = prompt_tokens.unwrap_or(prompt_len);
            let ttft = first_token_at
                .unwrap()
                .duration_since(started)
                .as_secs_f64();
            let latency = ended.duration_since(started).as_secs_f64();
            let tpot = if actual_output > 1 {
                (latency - ttft) / (actual_output - 1) as f64
            } else {
                0.0
            };
            return RequestResult::success(
                actual_prompt,
                actual_output,
                ttft,
                tpot,
                prompt_tokens.is_some(),
                completion_tokens.is_some(),
            );
        }
    }

    RequestResult::error(format!("HTTP/SSE request failed: {}", last_http_error))
}

#[derive(Debug, Clone)]
struct RequestResult {
    success: bool,
    prompt_tokens: usize,
    output_tokens: usize,
    ttft: f64,
    tpot: f64,
    prompt_from_usage: bool,
    output_from_usage: bool,
    error: String,
}

impl RequestResult {
    fn error(msg: String) -> Self {
        Self {
            success: false,
            prompt_tokens: 0,
            output_tokens: 0,
            ttft: 0.0,
            tpot: 0.0,
            prompt_from_usage: false,
            output_from_usage: false,
            error: msg,
        }
    }

    fn success(
        prompt_tokens: usize,
        output_tokens: usize,
        ttft: f64,
        tpot: f64,
        prompt_from_usage: bool,
        output_from_usage: bool,
    ) -> Self {
        Self {
            success: true,
            prompt_tokens,
            output_tokens,
            ttft,
            tpot,
            prompt_from_usage,
            output_from_usage,
            error: String::new(),
        }
    }
}

async fn run_builtin_benchmark(
    host: &str,
    port: u16,
    model_name: &str,
    case: &Case,
) -> Result<BenchmarkOutput> {
    let client = Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(3600))
            .build()?,
    );
    let started = Instant::now();
    let results: Arc<Mutex<Vec<RequestResult>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for idx in 0..case.num_prompts {
        let client = client.clone();
        let results = results.clone();
        let prompt = make_prompt(case.input_len, stable_case_seed(case, idx, 0));
        let host = host.to_string();
        let model_name = model_name.to_string();
        let input_len = case.input_len;
        let output_len = case.output_len;
        handles.push(tokio::spawn(async move {
            let res = read_sse_completion(
                client,
                &host,
                port,
                &model_name,
                prompt,
                input_len,
                output_len,
            )
            .await;
            results.lock().await.push(res);
        }));
    }

    join_all(handles).await;
    let duration = started.elapsed().as_secs_f64();
    let results = results.lock().await.clone();
    format_builtin_result(results, duration)
}

fn format_builtin_result(results: Vec<RequestResult>, duration: f64) -> Result<BenchmarkOutput> {
    let successes: Vec<&RequestResult> = results.iter().filter(|r| r.success).collect();
    let failures: Vec<&RequestResult> = results.iter().filter(|r| !r.success).collect();

    let total_input: usize = successes.iter().map(|r| r.prompt_tokens).sum();
    let total_output: usize = successes.iter().map(|r| r.output_tokens).sum();
    let ttfts: Vec<f64> = successes.iter().map(|r| r.ttft * 1000.0).collect();
    let tpots: Vec<f64> = successes.iter().map(|r| r.tpot * 1000.0).collect();

    let prompt_sources: std::collections::HashSet<&str> = successes
        .iter()
        .map(|r| {
            if r.prompt_from_usage {
                "usage"
            } else {
                "estimated"
            }
        })
        .collect();
    let output_sources: std::collections::HashSet<&str> = successes
        .iter()
        .map(|r| {
            if r.output_from_usage {
                "usage"
            } else {
                "chunk_count"
            }
        })
        .collect();
    let prompt_source = if prompt_sources.len() > 1 {
        "mixed"
    } else {
        prompt_sources.into_iter().next().unwrap_or("estimated")
    };
    let output_source = if output_sources.len() > 1 {
        "mixed"
    } else {
        output_sources.into_iter().next().unwrap_or("chunk_count")
    };

    let req_tps = if duration > 0.0 {
        successes.len() as f64 / duration
    } else {
        0.0
    };
    let out_tps = if duration > 0.0 {
        total_output as f64 / duration
    } else {
        0.0
    };
    let total_tps = if duration > 0.0 {
        (total_input + total_output) as f64 / duration
    } else {
        0.0
    };
    let mean_ttft = if !ttfts.is_empty() {
        ttfts.iter().sum::<f64>() / ttfts.len() as f64
    } else {
        0.0
    };
    let mean_tpot = if !tpots.is_empty() {
        tpots.iter().sum::<f64>() / tpots.len() as f64
    } else {
        0.0
    };

    let mut lines = vec![
        "============ Serving Benchmark Result ============".to_string(),
        format!(
            "Successful requests:                     {}",
            successes.len()
        ),
        format!(
            "Failed requests:                         {}",
            failures.len()
        ),
        format!("Benchmark duration (s):                  {:.2}", duration),
        format!("Total input tokens:                      {}", total_input),
        format!("Total generated tokens:                  {}", total_output),
        format!("Request throughput (req/s):              {:.2}", req_tps),
        format!("Output token throughput (tok/s):         {:.2}", out_tps),
        format!("Total Token throughput (tok/s):          {:.2}", total_tps),
        format!("Prompt token count source:               {}", prompt_source),
        format!("Output token count source:               {}", output_source),
        "---------------Time to First Token----------------".to_string(),
        format!("Mean TTFT (ms):                          {:.2}", mean_ttft),
        format!(
            "Median TTFT (ms):                        {:.2}",
            median(&ttfts)
        ),
        format!(
            "P99 TTFT (ms):                           {:.2}",
            percentile(&ttfts, 99.0)
        ),
        "-----Time per Output Token (excl. 1st token)------".to_string(),
        format!("Mean TPOT (ms):                          {:.2}", mean_tpot),
        format!(
            "Median TPOT (ms):                        {:.2}",
            median(&tpots)
        ),
        format!(
            "P99 TPOT (ms):                           {:.2}",
            percentile(&tpots, 99.0)
        ),
        "==================================================".to_string(),
    ];
    if !failures.is_empty() {
        lines.push("Failed requests during benchmark run detected (capping to 10):".to_string());
        for (idx, failure) in failures.iter().take(10).enumerate() {
            lines.push(format!("Error {}: {}", idx, failure.error));
        }
    }

    Ok(BenchmarkOutput {
        text: lines.join("\n") + "\n",
        metrics: BenchmarkMetrics {
            duration_s: duration,
            total_input_tokens: total_input,
            total_output_tokens: total_output,
            request_tps: req_tps,
            output_tok_s: out_tps,
            total_tok_s: total_tps,
            ttft_ms: mean_ttft,
            tpot_ms: mean_tpot,
            successful_requests: successes.len(),
            failed_requests: failures.len(),
            prompt_token_source: prompt_source.to_string(),
            output_token_source: output_source.to_string(),
        },
    })
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (v.len() - 1) as f64 * pct / 100.0;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    if low == high {
        return v[low];
    }
    let weight = rank - low as f64;
    v[low] * (1.0 - weight) + v[high] * weight
}

#[derive(Serialize, Deserialize, Debug)]
struct ServingResult {
    #[serde(default)]
    text: String,
}

async fn run_external_benchmark(
    mode: &str,
    host: &str,
    port: u16,
    model_path: &str,
    model_name: &str,
    case: &Case,
) -> Result<BenchmarkOutput> {
    let cmd = if mode == "benchmark_serving" {
        vec![
            "python3".to_string(),
            "./benchmark_serving.py".to_string(),
            "--backend".to_string(),
            "llama.cpp".to_string(),
            "--host".to_string(),
            host.to_string(),
            "--port".to_string(),
            port.to_string(),
            "--model".to_string(),
            model_path.to_string(),
            "--tokenizer".to_string(),
            model_path.to_string(),
            "--served-model-name".to_string(),
            model_name.to_string(),
            "--dataset-name".to_string(),
            "random".to_string(),
            "--num-prompts".to_string(),
            case.num_prompts.to_string(),
            "--random-input-len".to_string(),
            case.input_len.to_string(),
            "--random-output-len".to_string(),
            case.output_len.to_string(),
            "--ignore-eos".to_string(),
        ]
    } else {
        vec![
            "vllm".to_string(),
            "bench".to_string(),
            "serve".to_string(),
            "--backend".to_string(),
            "llama.cpp".to_string(),
            "--host".to_string(),
            host.to_string(),
            "--port".to_string(),
            port.to_string(),
            "--model".to_string(),
            model_path.to_string(),
            "--tokenizer".to_string(),
            model_path.to_string(),
            "--served-model-name".to_string(),
            model_name.to_string(),
            "--dataset-name".to_string(),
            "random".to_string(),
            "--num-prompts".to_string(),
            case.num_prompts.to_string(),
            "--random-input-len".to_string(),
            case.input_len.to_string(),
            "--random-output-len".to_string(),
            case.output_len.to_string(),
            "--ignore-eos".to_string(),
        ]
    };

    let output = tokio::process::Command::new(&cmd[0])
        .args(&cmd[1..])
        .output()
        .await?;
    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        text.push('\n');
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() {
        text.push_str(&format!("\nbenchmark 命令退出码: {}\n", output.status));
    }
    let metrics = parse_metrics(&text);
    Ok(BenchmarkOutput { text, metrics })
}

pub fn parse_metrics(output: &str) -> BenchmarkMetrics {
    use regex::Regex;
    let number_re = Regex::new(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)").unwrap();
    let mut metrics = BenchmarkMetrics::default();
    let mapping: [(&str, fn(&mut BenchmarkMetrics, f64)); 8] = [
        ("Benchmark duration (s):", |m, v| m.duration_s = v),
        ("Total input tokens:", |m, v| {
            m.total_input_tokens = v as usize
        }),
        ("Total generated tokens:", |m, v| {
            m.total_output_tokens = v as usize
        }),
        ("Request throughput (req/s):", |m, v| m.request_tps = v),
        ("Output token throughput (tok/s):", |m, v| {
            m.output_tok_s = v
        }),
        ("Total Token throughput (tok/s):", |m, v| m.total_tok_s = v),
        ("Mean TTFT (ms):", |m, v| m.ttft_ms = v),
        ("Mean TPOT (ms):", |m, v| m.tpot_ms = v),
    ];
    for line in output.lines() {
        for (prefix, setter) in &mapping {
            if line.contains(prefix) {
                if let Some(m) = number_re.find(line.split(prefix).nth(1).unwrap_or("")) {
                    setter(&mut metrics, m.as_str().parse().unwrap_or(0.0));
                }
            }
        }
        if line.contains("Successful requests:") {
            if let Some(m) = number_re.find(line.split("Successful requests:").nth(1).unwrap_or(""))
            {
                metrics.successful_requests = m.as_str().parse().unwrap_or(0) as usize;
            }
        }
        if line.contains("Failed requests:") {
            if let Some(m) = number_re.find(line.split("Failed requests:").nth(1).unwrap_or("")) {
                metrics.failed_requests = m.as_str().parse().unwrap_or(0) as usize;
            }
        }
        if line.contains("Prompt token count source:") {
            metrics.prompt_token_source = line
                .split("Prompt token count source:")
                .nth(1)
                .unwrap_or("")
                .trim()
                .to_string();
        }
        if line.contains("Output token count source:") {
            metrics.output_token_source = line
                .split("Output token count source:")
                .nth(1)
                .unwrap_or("")
                .trim()
                .to_string();
        }
    }
    if metrics.prompt_token_source.is_empty() {
        metrics.prompt_token_source = "external".to_string();
    }
    if metrics.output_token_source.is_empty() {
        metrics.output_token_source = "external".to_string();
    }
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics_builtin_output() {
        let text = r#"============ Serving Benchmark Result ============
Successful requests:                     8
Failed requests:                         2
Benchmark duration (s):                  12.50
Total input tokens:                      1024
Total generated tokens:                  2048
Request throughput (req/s):              0.64
Output token throughput (tok/s):         163.84
Total Token throughput (tok/s):          245.76
Prompt token count source:               usage
Output token count source:               usage
---------------Time to First Token----------------
Mean TTFT (ms):                          123.45
Median TTFT (ms):                        120.00
P99 TTFT (ms):                           200.00
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.67
Median TPOT (ms):                        44.00
P99 TPOT (ms):                           80.00
=================================================="#;
        let m = parse_metrics(text);
        assert_eq!(m.successful_requests, 8);
        assert_eq!(m.failed_requests, 2);
        assert!((m.duration_s - 12.5).abs() < 0.001);
        assert_eq!(m.total_input_tokens, 1024);
        assert_eq!(m.total_output_tokens, 2048);
        assert!((m.request_tps - 0.64).abs() < 0.01);
        assert!((m.output_tok_s - 163.84).abs() < 0.01);
        assert!((m.ttft_ms - 123.45).abs() < 0.01);
        assert!((m.tpot_ms - 45.67).abs() < 0.01);
        assert_eq!(m.prompt_token_source, "usage");
        assert_eq!(m.output_token_source, "usage");
    }

    #[test]
    fn test_parse_metrics_defaults_to_external() {
        let m = parse_metrics("no metrics here");
        assert_eq!(m.prompt_token_source, "external");
        assert_eq!(m.output_token_source, "external");
    }

    #[test]
    fn test_median_and_percentile() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[1.0, 3.0, 5.0]), 3.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);

        assert_eq!(percentile(&[], 99.0), 0.0);
        assert_eq!(percentile(&[10.0], 50.0), 10.0);
        assert!((percentile(&[0.0, 10.0, 20.0, 30.0, 40.0], 50.0) - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_detect_benchmark_mode_explicit() {
        assert_eq!(detect_benchmark_mode("builtin"), "builtin");
        assert_eq!(detect_benchmark_mode("vllm_cli"), "vllm_cli");
    }

    #[test]
    fn test_make_prompt_length() {
        let p = make_prompt(10, 12345);
        assert!(p.starts_with(" request "));
        // " request N" + " a" * 10 tokens-ish
        assert!(p.len() > 10);
    }

    #[test]
    fn test_stable_case_seed_deterministic() {
        let case = Case {
            parallel_size: 1,
            num_prompts: 2,
            input_len: 64,
            output_len: 64,
            input_bucket: "short".to_string(),
            output_bucket: "short".to_string(),
            concurrency_bucket: "multi".to_string(),
        };
        assert_eq!(stable_case_seed(&case, 0, 0), stable_case_seed(&case, 0, 0));
        assert_ne!(stable_case_seed(&case, 0, 0), stable_case_seed(&case, 1, 0));
    }

    #[test]
    fn test_format_builtin_result_all_failed() {
        let results = vec![RequestResult::error("boom".to_string())];
        let out = format_builtin_result(results, 1.0).unwrap();
        assert!(
            out.text
                .contains("Failed requests:                         1")
        );
        assert_eq!(out.metrics.successful_requests, 0);
        assert_eq!(out.metrics.failed_requests, 1);
    }

    #[test]
    fn test_format_builtin_result_empty() {
        let out = format_builtin_result(vec![], 0.0).unwrap();
        assert_eq!(out.metrics.successful_requests, 0);
        assert_eq!(out.metrics.failed_requests, 0);
    }
}
