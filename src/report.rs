use crate::benchmark::BenchmarkMetrics;
use crate::matrix::Case;
use crate::utils::{normalize_error, now_str};
use anyhow::Result;
use csv::WriterBuilder;
use regex::Regex;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const SUMMARY_HEADER: [&str; 23] = [
    "timestamp",
    "model_name",
    "gpu_devices",
    "parallel_size",
    "ctx_size",
    "batch_size",
    "num_prompts",
    "input_len",
    "output_len",
    "input_bucket",
    "output_bucket",
    "concurrency_bucket",
    "case_id",
    "benchmark_mode",
    "status",
    "error",
    "prompt_token_source",
    "output_token_source",
    "output_tok_s",
    "ttft_ms",
    "tpot_ms",
    "successful_requests",
    "failed_requests",
];

pub const COMPANY_REPORT_HEADER: [&str; 21] = [
    "测试时间",
    "机型",
    "GPU",
    "模型",
    "精度",
    "物理卡数",
    "逻辑卡数",
    "模式",
    "请求并发数",
    "输入",
    "输出",
    "总输入",
    "总输出",
    "请求吞吐",
    "输出吞吐",
    "总吞吐",
    "首Token延时(ms)",
    "每Token延时(ms)",
    "总耗时(s)",
    "平均每用户输出吞吐",
    "备注",
];

pub fn ensure_summary_header(path: &Path) -> Result<()> {
    if path.exists() {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let first_line = text.lines().next().unwrap_or("");
        if first_line.split(',').collect::<Vec<_>>() == SUMMARY_HEADER.to_vec() {
            return Ok(());
        }
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = path.extension().unwrap_or_default().to_string_lossy();
        let backup = path.with_file_name(format!(
            "{}.pre-status-{}.{}",
            stem,
            chrono::Local::now().format("%Y%m%d-%H%M%S"),
            ext
        ));
        std::fs::rename(path, backup)?;
    }
    let mut wtr = WriterBuilder::new().from_path(path)?;
    wtr.write_record(&SUMMARY_HEADER)?;
    wtr.flush()?;
    Ok(())
}

pub fn append_summary(
    path: &Path,
    model_name: &str,
    gpu_devices_label: &str,
    case: &Case,
    ctx_size: usize,
    batch_size: usize,
    benchmark_mode: &str,
    metrics: &BenchmarkMetrics,
    status: &str,
    error: &str,
) -> Result<()> {
    let mut wtr =
        WriterBuilder::new().from_writer(std::fs::OpenOptions::new().append(true).open(path)?);
    wtr.write_record([
        now_str(),
        model_name.to_string(),
        gpu_devices_label.to_string(),
        case.parallel_size.to_string(),
        ctx_size.to_string(),
        batch_size.to_string(),
        case.num_prompts.to_string(),
        case.input_len.to_string(),
        case.output_len.to_string(),
        case.input_bucket.clone(),
        case.output_bucket.clone(),
        case.concurrency_bucket.clone(),
        case.case_id(),
        benchmark_mode.to_string(),
        status.to_string(),
        normalize_error(error),
        metrics.prompt_token_source.clone(),
        metrics.output_token_source.clone(),
        fmt_or_empty(metrics.output_tok_s),
        fmt_or_empty(metrics.ttft_ms),
        fmt_or_empty(metrics.tpot_ms),
        metrics.successful_requests.to_string(),
        metrics.failed_requests.to_string(),
    ])?;
    wtr.flush()?;
    Ok(())
}

fn fmt_or_empty(v: f64) -> String {
    if v == 0.0 {
        String::new()
    } else {
        format!("{:.2}", v)
    }
}

pub fn company_mode(case: &Case) -> String {
    let labels: HashMap<&str, &str> = [("short", "短"), ("mid", "中"), ("long", "长")]
        .into_iter()
        .collect();
    format!(
        "{}输入{}输出",
        labels.get(case.input_bucket.as_str()).unwrap_or(&"mid"),
        labels.get(case.output_bucket.as_str()).unwrap_or(&"mid")
    )
}

fn format_intish(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else if (value - value.round()).abs() < 0.000001 {
        format!("{}", value.round() as i64)
    } else {
        format!("{:.2}", value)
    }
}

fn format_report_float(value: f64, digits: usize) -> String {
    if !value.is_finite() {
        String::new()
    } else {
        format!("{:.*}", digits, value)
    }
}

fn float_metric(metrics: &BenchmarkMetrics, key: &str) -> f64 {
    match key {
        "successful_requests" => metrics.successful_requests as f64,
        "failed_requests" => metrics.failed_requests as f64,
        "duration_s" => metrics.duration_s,
        "total_input_tokens" => metrics.total_input_tokens as f64,
        "total_output_tokens" => metrics.total_output_tokens as f64,
        "request_tps" => metrics.request_tps,
        "output_tok_s" => metrics.output_tok_s,
        "total_tok_s" => metrics.total_tok_s,
        "ttft_ms" => metrics.ttft_ms,
        "tpot_ms" => metrics.tpot_ms,
        _ => 0.0,
    }
}

pub fn build_company_report_row(
    test_time: &str,
    machine_type: &str,
    gpu_name: &str,
    report_model_name: &str,
    report_precision: &str,
    physical_cards: &str,
    logical_cards: &str,
    case: &Case,
    ctx_size: usize,
    batch_size: usize,
    metrics: &BenchmarkMetrics,
    benchmark_mode: &str,
    gpu_devices_label: &str,
    error: &str,
) -> Option<Vec<String>> {
    let successful = float_metric(metrics, "successful_requests");
    let duration_s = float_metric(metrics, "duration_s");
    let total_input = float_metric(metrics, "total_input_tokens");
    let total_output = float_metric(metrics, "total_output_tokens");
    if successful <= 0.0 || duration_s <= 0.0 || total_output <= 0.0 {
        return None;
    }
    let input_per_request = total_input / successful;
    let output_per_request = total_output / successful;
    let request_tps = successful / duration_s;
    let output_tps = total_output / duration_s;
    let total_tps = (total_input + total_output) / duration_s;
    let avg_user_output_tps = output_tps / successful;

    let mut remarks = vec![
        format!("backend=llama.cpp"),
        format!("benchmark={}", benchmark_mode),
        format!("server_slots={}", case.parallel_size),
        format!("ctx={}", ctx_size),
        format!("batch={}", batch_size),
        format!("gpu_devices={}", gpu_devices_label),
    ];
    if !metrics.prompt_token_source.is_empty() || !metrics.output_token_source.is_empty() {
        remarks.push(format!(
            "token_source={} / {}",
            if metrics.prompt_token_source.is_empty() {
                "-"
            } else {
                &metrics.prompt_token_source
            },
            if metrics.output_token_source.is_empty() {
                "-"
            } else {
                &metrics.output_token_source
            }
        ));
    }
    if metrics.failed_requests > 0 {
        remarks.push(format!("failed_requests={}", metrics.failed_requests));
    }
    if !error.is_empty() {
        remarks.push(format!("error={}", normalize_error(error)));
    }

    Some(vec![
        test_time.to_string(),
        machine_type.to_string(),
        gpu_name.to_string(),
        report_model_name.to_string(),
        report_precision.to_string(),
        physical_cards.to_string(),
        logical_cards.to_string(),
        company_mode(case),
        format_intish(case.num_prompts as f64),
        format_intish(input_per_request),
        format_intish(output_per_request),
        format_intish(total_input),
        format_intish(total_output),
        format_report_float(request_tps, 4),
        format_report_float(output_tps, 2),
        format_report_float(total_tps, 2),
        format_report_float(metrics.ttft_ms, 2),
        format_report_float(metrics.tpot_ms, 2),
        format_report_float(duration_s, 2),
        format_report_float(avg_user_output_tps, 2),
        remarks.join("; "),
    ])
}

pub fn collect_company_report_rows(
    result_logs: &[std::path::PathBuf],
    run_id: &str,
    cases: &[Case],
    test_time: &str,
    machine_type: &str,
    gpu_name: &str,
    report_model_name: &str,
    report_precision: &str,
    physical_cards: &str,
    logical_cards: &str,
    benchmark_mode: &str,
    gpu_devices_label: &str,
) -> Vec<Vec<String>> {
    let case_by_id: HashMap<String, &Case> = cases.iter().map(|c| (c.case_id(), c)).collect();
    let case_order: HashMap<String, usize> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| (c.case_id(), i))
        .collect();
    let marker = format!("===== run_id={} ", run_id);
    let header_re = Regex::new(
        r"开始\s+\d+/\d+\s+(?P<case_id>\S+)\s+ctx=(?P<ctx_size>\d+)\s+batch=(?P<batch_size>\d+)\s+dtype=(?P<dtype>\S+)\s+测试"
    ).unwrap();
    let status_re = Regex::new(r"(?m)^status=(?P<status>\S+)\s+error=(?P<error>.*)$").unwrap();

    let mut collected: HashMap<String, (usize, usize, Vec<String>)> = HashMap::new();
    let mut parse_order = 0usize;

    for result_log in result_logs {
        if !result_log.exists() {
            continue;
        }
        let text = std::fs::read_to_string(result_log).unwrap_or_default();
        let marker_index = text.rfind(&marker);
        if marker_index.is_none() {
            continue;
        }
        let text = &text[marker_index.unwrap()..];
        let matches: Vec<regex::Match> = header_re.find_iter(text).collect();
        for (idx, m) in matches.iter().enumerate() {
            parse_order += 1;
            let caps = header_re.captures(m.as_str()).unwrap();
            let case_id = caps.name("case_id").unwrap().as_str().to_string();
            let case = match case_by_id.get(&case_id) {
                Some(c) => *c,
                None => continue,
            };
            let block_start = m.end();
            let block_end = matches
                .get(idx + 1)
                .map(|m| m.start())
                .unwrap_or(text.len());
            let block = &text[block_start..block_end];
            let status_match = status_re.captures(block);
            if status_match.is_none()
                || status_match
                    .as_ref()
                    .unwrap()
                    .name("status")
                    .unwrap()
                    .as_str()
                    != "completed"
            {
                continue;
            }
            let error = status_match
                .as_ref()
                .unwrap()
                .name("error")
                .unwrap()
                .as_str();
            let metrics = crate::benchmark::parse_metrics(block);
            let ctx_size = caps.name("ctx_size").unwrap().as_str().parse().unwrap_or(0);
            let batch_size = caps
                .name("batch_size")
                .unwrap()
                .as_str()
                .parse()
                .unwrap_or(0);
            if let Some(row) = build_company_report_row(
                test_time,
                machine_type,
                gpu_name,
                report_model_name,
                report_precision,
                physical_cards,
                logical_cards,
                case,
                ctx_size,
                batch_size,
                &metrics,
                benchmark_mode,
                gpu_devices_label,
                error,
            ) {
                let order = case_order.get(&case_id).copied().unwrap_or(parse_order);
                collected.insert(case_id, (order, parse_order, row));
            }
        }
    }

    let mut rows: Vec<_> = collected.into_values().collect();
    rows.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    rows.into_iter().map(|(_, _, row)| row).collect()
}

pub fn write_company_report_csv(path: &Path, rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all("\u{FEFF}".as_bytes())?;
    let mut wtr = WriterBuilder::new().from_writer(file);
    wtr.write_record(&COMPANY_REPORT_HEADER)?;
    for row in rows {
        wtr.write_record(row)?;
    }
    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkMetrics;

    fn sample_case() -> Case {
        Case {
            parallel_size: 1,
            num_prompts: 8,
            input_len: 512,
            output_len: 512,
            input_bucket: "short".to_string(),
            output_bucket: "short".to_string(),
            concurrency_bucket: "multi".to_string(),
        }
    }

    fn sample_metrics() -> BenchmarkMetrics {
        BenchmarkMetrics {
            duration_s: 10.0,
            total_input_tokens: 4096,
            total_output_tokens: 4096,
            request_tps: 0.8,
            output_tok_s: 409.6,
            total_tok_s: 819.2,
            ttft_ms: 100.0,
            tpot_ms: 50.0,
            successful_requests: 8,
            failed_requests: 0,
            prompt_token_source: "usage".to_string(),
            output_token_source: "usage".to_string(),
        }
    }

    #[test]
    fn test_company_mode() {
        let c = sample_case();
        assert_eq!(company_mode(&c), "短输入短输出");
    }

    #[test]
    fn test_build_company_report_row() {
        let row = build_company_report_row(
            "2024-01-01 10:00:00",
            "8F",
            "RTX 4090",
            "Qwen",
            "Q3_K_XL",
            "8",
            "8",
            &sample_case(),
            4096,
            512,
            &sample_metrics(),
            "builtin",
            "all",
            "",
        )
        .unwrap();
        assert_eq!(row[0], "2024-01-01 10:00:00");
        assert_eq!(row[1], "8F");
        assert_eq!(row[3], "Qwen");
        assert_eq!(row[7], "短输入短输出");
        assert_eq!(row[8], "8");
        assert_eq!(row[9], "512");
        assert_eq!(row[10], "512");
        assert_eq!(row[11], "4096");
        assert_eq!(row[12], "4096");
    }

    #[test]
    fn test_build_company_report_row_skips_failed() {
        let mut metrics = sample_metrics();
        metrics.successful_requests = 0;
        assert!(
            build_company_report_row(
                "2024-01-01 10:00:00",
                "",
                "",
                "",
                "",
                "",
                "",
                &sample_case(),
                4096,
                512,
                &metrics,
                "builtin",
                "all",
                ""
            )
            .is_none()
        );
    }

    #[test]
    fn test_write_company_report_csv() {
        let tmp = std::env::temp_dir().join("llama-test-matrix-company-report-test.csv");
        let row = build_company_report_row(
            "2024-01-01 10:00:00",
            "",
            "",
            "Qwen",
            "Q3",
            "8",
            "8",
            &sample_case(),
            4096,
            512,
            &sample_metrics(),
            "builtin",
            "all",
            "",
        )
        .unwrap();
        write_company_report_csv(&tmp, &[row]).unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(&bytes[0..3], &[0xEF, 0xBB, 0xBF]);
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("测试时间"));
        assert!(text.contains("Qwen"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_fmt_or_empty() {
        assert_eq!(fmt_or_empty(0.0), "");
        assert_eq!(fmt_or_empty(1.5), "1.50");
    }

    #[test]
    fn test_format_intish() {
        assert_eq!(format_intish(0.0), "0");
        assert_eq!(format_intish(8.0), "8");
        assert_eq!(format_intish(8.5), "8.50");
    }

    #[test]
    fn test_format_report_float() {
        assert_eq!(format_report_float(1.2345, 2), "1.23");
        assert_eq!(format_report_float(f64::NAN, 2), "");
        assert_eq!(format_report_float(f64::INFINITY, 2), "");
    }

    #[test]
    fn test_float_metric() {
        let m = sample_metrics();
        assert_eq!(float_metric(&m, "successful_requests"), 8.0);
        assert_eq!(float_metric(&m, "duration_s"), 10.0);
        assert_eq!(float_metric(&m, "output_tok_s"), 409.6);
        assert_eq!(float_metric(&m, "unknown"), 0.0);
    }

    #[test]
    fn test_ensure_summary_header_creates_file() {
        let tmp = std::env::temp_dir().join("llama-test-matrix-summary-header-test.csv");
        let _ = std::fs::remove_file(&tmp);
        ensure_summary_header(&tmp).unwrap();
        let text = std::fs::read_to_string(&tmp).unwrap();
        assert!(text.starts_with("timestamp"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_ensure_summary_header_backups_wrong_header() {
        let tmp = std::env::temp_dir().join("llama-test-matrix-summary-backup-test.csv");
        std::fs::write(&tmp, "wrong,header\n").unwrap();
        ensure_summary_header(&tmp).unwrap();
        let text = std::fs::read_to_string(&tmp).unwrap();
        assert!(text.starts_with("timestamp"));
        let backups: Vec<_> = std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("llama-test-matrix-summary-backup-test.pre-status-")
            })
            .collect();
        assert!(!backups.is_empty());
        for b in backups {
            let _ = std::fs::remove_file(b.path());
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_append_summary_and_collect_rows() {
        let tmp = std::env::temp_dir().join("llama-test-matrix-summary-collect-test.csv");
        let _ = std::fs::remove_file(&tmp);
        ensure_summary_header(&tmp).unwrap();

        let case = sample_case();
        append_summary(
            &tmp,
            "m",
            "all",
            &case,
            4096,
            512,
            "builtin",
            &sample_metrics(),
            "completed",
            "",
        )
        .unwrap();

        let result_log = std::env::temp_dir().join("llama-test-matrix-result-collect-test.log");
        let run_id = "r1";
        let header = format!(
            "===== run_id={} started_at=2024-01-01 10:00:00 model=m dtype=q3 =====\n",
            run_id
        );
        let block = format!(
            "\n******************************开始 1/1 {} ctx=4096 batch=512 dtype=q3 测试*******************************\n{}\nstatus=completed error=\n",
            case.case_id(),
            format_builtin_result_text()
        );
        std::fs::write(&result_log, format!("{}{}", header, block)).unwrap();

        let rows = collect_company_report_rows(
            &[result_log.clone()],
            run_id,
            &[case.clone()],
            "2024-01-01 10:00:00",
            "8F",
            "RTX",
            "Qwen",
            "Q3",
            "8",
            "8",
            "builtin",
            "all",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][3], "Qwen");

        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&result_log);
    }

    fn format_builtin_result_text() -> String {
        "============ Serving Benchmark Result ============\n\
         Successful requests:                     8\n\
         Failed requests:                         0\n\
         Benchmark duration (s):                  10.00\n\
         Total input tokens:                      4096\n\
         Total generated tokens:                  4096\n\
         Request throughput (req/s):              0.80\n\
         Output token throughput (tok/s):         409.60\n\
         Total Token throughput (tok/s):          819.20\n\
         Prompt token count source:               usage\n\
         Output token count source:               usage\n\
         ---------------Time to First Token----------------\n\
         Mean TTFT (ms):                          100.00\n\
         Median TTFT (ms):                        90.00\n\
         P99 TTFT (ms):                           150.00\n\
         -----Time per Output Token (excl. 1st token)------\n\
         Mean TPOT (ms):                          50.00\n\
         Median TPOT (ms):                        48.00\n\
         P99 TPOT (ms):                           80.00\n\
         =================================================="
            .to_string()
    }
}
