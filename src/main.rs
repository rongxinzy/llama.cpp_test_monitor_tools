mod benchmark;
mod blackbox;
mod cli;
mod config;
mod matrix;
mod progress;
mod report;
mod server;
mod spec;
mod utils;

use anyhow::{Result, bail};
use clap::Parser;
use cli::{BlackboxArgs, Cli, Commands, RunArgs};
use config::{MatrixConfig, build_blackbox_config, build_matrix_config};
use matrix::{
    Case, auto_batch_size, bucket, grouped_by_required_context, nice_points, parallel_points,
    parse_gpu_devices, parse_int_list, parse_range, prompt_points,
};
use progress::ProgressTracker;
use report::{
    append_summary, collect_company_report_rows, ensure_summary_header, write_company_report_csv,
};
use server::{append_log, is_port_open, start_server, stop_server, wait_for_ready};
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{Duration, sleep};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => {
            if let Err(e) = run_matrix(args).await {
                eprintln!("Error: {:#}", e);
                std::process::exit(1);
            }
        }
        Commands::Blackbox(args) => {
            if let Err(e) = run_blackbox(args).await {
                eprintln!("Error: {:#}", e);
                std::process::exit(1);
            }
        }
    }
}

async fn run_blackbox(args: BlackboxArgs) -> Result<()> {
    let config = build_blackbox_config(args);
    let handle = blackbox::start_blackbox(config.clone()).await?;
    let mut wrapped: Option<tokio::process::Child> = None;
    if !config.command.is_empty() {
        let mut cmd = Command::new(&config.command[0]);
        cmd.args(&config.command[1..]);
        wrapped = Some(cmd.spawn()?);
    }
    let mut stop_after = false;
    while !stop_after {
        sleep(Duration::from_secs(1)).await;
        if config.stop_after_trigger && handle.triggers() > 0 {
            stop_after = true;
        }
        if let Some(ref mut child) = wrapped {
            if child.try_wait()?.is_some() {
                break;
            }
        }
    }
    blackbox::finalize_blackbox(handle).await;
    if let Some(mut child) = wrapped {
        let _ = child.kill().await;
    }
    Ok(())
}

async fn run_matrix(args: RunArgs) -> Result<()> {
    let cfg = build_matrix_config(args.clone())?;

    // validate binary
    let bin_path = if std::path::Path::new(&cfg.llama_server_bin).is_file() {
        cfg.llama_server_bin.clone()
    } else if let Ok(p) = which::which(&cfg.llama_server_bin) {
        p.to_string_lossy().to_string()
    } else {
        bail!("找不到 llama-server: {}", cfg.llama_server_bin);
    };

    fs::create_dir_all(&cfg.result_dir).await?;

    let (gpu_devices_label, gpu_devices) = parse_gpu_devices(&cfg.gpu_devices)?;

    let input_lens = if let Some(list) = &cfg.input_lens {
        parse_int_list(list, "输入长度")?
    } else {
        let range = parse_range(
            cfg.input_len_range.as_deref().unwrap_or("64-4096"),
            "输入长度",
        )?;
        nice_points(range.0, range.1, cfg.io_points)
    };

    let output_lens = if let Some(list) = &cfg.output_lens {
        parse_int_list(list, "输出长度")?
    } else {
        let range = parse_range(
            cfg.output_len_range.as_deref().unwrap_or("64-4096"),
            "输出长度",
        )?;
        nice_points(range.0, range.1, cfg.io_points)
    };

    let prompts = if let Some(list) = &cfg.num_prompts {
        parse_int_list(list, "请求并发")?
    } else {
        let range = parse_range(
            cfg.num_prompts_range.as_deref().unwrap_or("1-32"),
            "请求并发",
        )?;
        prompt_points(range.0, range.1, cfg.prompt_points)
    };

    let parallels = if let Some(list) = &cfg.parallel_sizes {
        parse_int_list(list, "llama-server 并发 slot")?
    } else {
        let range = parse_range(
            cfg.parallel_range.as_deref().unwrap_or("1-8"),
            "llama-server 并发 slot",
        )?;
        parallel_points(range.0, range.1)
    };

    if cfg.pair_parallel_with_num_prompts && parallels.len() != prompts.len() {
        bail!("--pair-parallel-with-num-prompts 要求 --parallel-sizes 和 --num-prompts 数量一致");
    }
    if cfg.pair_input_output_lens && input_lens.len() != output_lens.len() {
        bail!("--pair-input-output-lens 要求输入长度点和输出长度点数量一致");
    }

    let parallel_prompt_pairs: Vec<(usize, usize)> = if cfg.pair_parallel_with_num_prompts {
        parallels
            .iter()
            .zip(prompts.iter())
            .map(|(p, n)| (*p, *n))
            .collect()
    } else {
        parallels
            .iter()
            .flat_map(|p| prompts.iter().map(move |n| (*p, *n)))
            .collect()
    };

    let input_output_pairs: Vec<(usize, usize)> = if cfg.pair_input_output_lens {
        input_lens
            .iter()
            .zip(output_lens.iter())
            .map(|(i, o)| (*i, *o))
            .collect()
    } else {
        input_lens
            .iter()
            .flat_map(|i| output_lens.iter().map(move |o| (*i, *o)))
            .collect()
    };

    let mut cases: Vec<Case> = Vec::new();
    for (parallel_size, num_prompts) in &parallel_prompt_pairs {
        for (input_len, output_len) in &input_output_pairs {
            cases.push(Case {
                parallel_size: *parallel_size,
                num_prompts: *num_prompts,
                input_len: *input_len,
                output_len: *output_len,
                input_bucket: bucket(*input_len, &input_lens),
                output_bucket: bucket(*output_len, &output_lens),
                concurrency_bucket: if *num_prompts == 1 {
                    "single".to_string()
                } else {
                    "multi".to_string()
                },
            });
        }
    }

    let benchmark_mode = benchmark::detect_benchmark_mode(&cfg.benchmark_mode);
    let max_input_len = input_lens.iter().copied().max().unwrap_or(64);
    let batch_size = auto_batch_size(max_input_len, cfg.max_batch_size);

    let plan_path = cfg.result_dir.join(format!(
        "{}-{}-matrix-plan.jsonl",
        cfg.model_name, cfg.dtype
    ));
    let mut plan_file = fs::File::create(&plan_path).await?;
    for case in &cases {
        let mut item = serde_json::json!(case);
        item["required_context_per_slot"] = serde_json::json!(case.required_context_per_slot());
        item["gpu_devices"] = serde_json::json!(&gpu_devices_label);
        let line = format!("{}\n", serde_json::to_string(&item)?);
        plan_file.write_all(line.as_bytes()).await?;
    }

    let summary_path = cfg.result_dir.join(format!(
        "{}-{}-matrix-summary.csv",
        cfg.model_name, cfg.dtype
    ));
    ensure_summary_header(&summary_path)?;

    let company_report_path = cfg.company_report_path.clone().unwrap_or_else(|| {
        cfg.result_dir.join(format!(
            "{}-{}-company-report.csv",
            cfg.model_name, cfg.dtype
        ))
    });

    let report_gpu_name = cfg
        .report_gpu_name
        .clone()
        .unwrap_or_else(|| infer_gpu_name(&gpu_devices_label, &gpu_devices).unwrap_or_default());
    let physical_cards = cfg
        .physical_cards
        .map(|v| v.to_string())
        .or_else(|| std::env::var("LLAMA_CPP_PHYSICAL_CARDS").ok())
        .unwrap_or_else(|| {
            infer_visible_gpu_count(&gpu_devices_label, &gpu_devices).unwrap_or_default()
        });
    let logical_cards = cfg
        .logical_cards
        .map(|v| v.to_string())
        .or_else(|| std::env::var("LLAMA_CPP_LOGICAL_CARDS").ok())
        .unwrap_or_else(|| physical_cards.clone());

    println!("llama.cpp 自动矩阵测试 (Rust)");
    println!("  GPU devices:  {}", gpu_devices_label);
    println!("  GPU model:    {}", report_gpu_name);
    println!("  physical GPUs:{}", physical_cards);
    println!("  logical GPUs: {}", logical_cards);
    println!("  slot parallel:{:?}", parallels);
    println!("  input_lens:   {:?}", input_lens);
    println!("  output_lens:  {:?}", output_lens);
    println!("  num_prompts:  {:?}", prompts);
    println!("  case 数量:    {}", cases.len());
    println!("  benchmark:    {}", benchmark_mode);
    println!("  ctx_strategy: {}", cfg.ctx_strategy);
    println!("  progress:     {}", cfg.progress);
    println!("  plan:         {}", plan_path.display());
    println!("  summary:      {}", summary_path.display());
    if !cfg.no_company_report {
        println!("  company csv:  {}", company_report_path.display());
    }

    let mut progress = ProgressTracker::new(cases.len(), cfg.progress.clone());
    let run_started_at = utils::now_str();
    let run_id = utils::run_id();

    // start blackbox
    let blackbox_handle = if cfg.blackbox.enabled {
        Some(blackbox::start_blackbox(cfg.blackbox.clone()).await?)
    } else {
        None
    };

    let mut global_skip: Vec<(usize, usize)> = Vec::new();
    let mut result_logs: Vec<PathBuf> = Vec::new();
    let exit_code = 0;

    for parallel_size in &parallels {
        let parallel_cases: Vec<Case> = cases
            .iter()
            .filter(|c| c.parallel_size == *parallel_size)
            .cloned()
            .collect();
        let result_log = cfg.result_dir.join(format!(
            "{}-{}-{}.log",
            cfg.model_name, parallel_size, cfg.dtype
        ));
        result_logs.push(result_log.clone());
        append_log(
            &result_log,
            &format!(
                "\n\n===== run_id={} started_at={} model={} dtype={} =====\n",
                run_id, run_started_at, cfg.model_name, cfg.dtype
            ),
        )
        .await;

        let grouped = grouped_by_required_context(&parallel_cases);

        if cfg.ctx_strategy == "progressive" {
            append_log(
                &result_log,
                &format!(
                    "\n\n===== parallel={} ctx_strategy=progressive =====\n",
                    parallel_size
                ),
            )
            .await;
            run_progressive_context_groups(
                &cfg,
                &bin_path,
                &gpu_devices_label,
                &gpu_devices,
                *parallel_size,
                batch_size,
                &grouped,
                &summary_path,
                &result_log,
                &benchmark_mode,
                parallel_cases.len(),
                &mut progress,
                &mut global_skip,
            )
            .await?;
            continue;
        }

        // max-first
        let max_context_per_slot = input_lens.iter().max().unwrap_or(&64)
            + output_lens.iter().max().unwrap_or(&64)
            + matrix::CTX_MARGIN;
        let ctx_size = max_context_per_slot * parallel_size;
        append_log(
            &result_log,
            &format!(
                "\n\n===== parallel={} ctx_strategy=max-first first_try_ctx={} =====\n",
                parallel_size, ctx_size
            ),
        )
        .await;
        let ok = run_case_group(
            &cfg,
            &bin_path,
            &gpu_devices_label,
            &gpu_devices,
            *parallel_size,
            ctx_size,
            batch_size,
            &parallel_cases,
            &summary_path,
            &result_log,
            &benchmark_mode,
            parallel_cases.len(),
            &mut progress,
            false,
        )
        .await?;
        if !ok {
            append_log(
                &result_log,
                "\n最大 ctx 启动失败，进入自适应降级：按每个 case 所需 ctx 分组，从短到长继续测试；第一次启动失败后跳过更长 ctx。\n",
            )
            .await;
            run_progressive_context_groups(
                &cfg,
                &bin_path,
                &gpu_devices_label,
                &gpu_devices,
                *parallel_size,
                batch_size,
                &grouped,
                &summary_path,
                &result_log,
                &benchmark_mode,
                parallel_cases.len(),
                &mut progress,
                &mut global_skip,
            )
            .await?;
        }
    }

    if let Some(handle) = blackbox_handle {
        blackbox::finalize_blackbox(handle).await;
    }

    progress.close();

    if exit_code != 0 {
        return Ok(());
    }

    if !cfg.no_company_report {
        let rows = collect_company_report_rows(
            &result_logs,
            &run_id,
            &cases,
            &run_started_at,
            &cfg.report_machine_type,
            &report_gpu_name,
            &cfg.report_model_name,
            &cfg.report_precision,
            &physical_cards,
            &logical_cards,
            &benchmark_mode,
            &gpu_devices_label,
        );
        write_company_report_csv(&company_report_path, &rows)?;
        println!(
            "\n公司格式 CSV: {} ({} rows)",
            company_report_path.display(),
            rows.len()
        );

        let spec_path = cfg.result_dir.join(format!(
            "{}-{}-hardware-spec.csv",
            cfg.model_name, cfg.dtype
        ));
        let specs = spec::collect_hardware_specs().await;
        spec::write_hardware_spec_csv(
            &spec_path,
            &run_id,
            &cfg.report_model_name,
            &cfg.report_precision,
            &specs,
        )?;
        println!(
            "硬件规格 CSV: {} ({} items)",
            spec_path.display(),
            specs.len()
        );
    }

    println!("\n全部 case 完成");
    println!(
        "结果目录: {}",
        cfg.result_dir
            .canonicalize()
            .unwrap_or(cfg.result_dir.clone())
            .display()
    );
    Ok(())
}

async fn run_progressive_context_groups(
    cfg: &MatrixConfig,
    bin_path: &str,
    gpu_devices_label: &str,
    gpu_devices: &[String],
    parallel_size: usize,
    batch_size: usize,
    grouped_cases: &[(usize, Vec<Case>)],
    summary_path: &std::path::Path,
    result_log: &std::path::Path,
    benchmark_mode: &str,
    global_total: usize,
    progress: &mut ProgressTracker,
    global_skip: &mut Vec<(usize, usize)>,
) -> Result<()> {
    for (group_index, (required_context_per_slot, cases)) in grouped_cases.iter().enumerate() {
        let ctx_size = required_context_per_slot * parallel_size;

        if let Some((failed_parallel, failed_ctx)) =
            matching_failure(global_skip, *required_context_per_slot, parallel_size)
        {
            let error = format!(
                "required_context_per_slot={} 已达到或超过 parallel={} 的启动失败阈值 {}；当前 parallel 不小于失败档位，自动跳过",
                required_context_per_slot, failed_parallel, failed_ctx
            );
            record_skipped_cases(
                summary_path,
                result_log,
                &cfg.model_name,
                gpu_devices_label,
                &grouped_cases[group_index..],
                parallel_size,
                batch_size,
                "skipped_after_global_startup_limit",
                &error,
                progress,
                benchmark_mode,
            )
            .await;
            break;
        }

        let ok = run_case_group(
            cfg,
            bin_path,
            gpu_devices_label,
            gpu_devices,
            parallel_size,
            ctx_size,
            batch_size,
            cases,
            summary_path,
            result_log,
            benchmark_mode,
            global_total,
            progress,
            true,
        )
        .await?;

        if ok {
            continue;
        }

        mark_failed(global_skip, *required_context_per_slot, parallel_size);

        if group_index + 1 < grouped_cases.len() {
            let error = format!(
                "ctx={} 启动失败；按单调递增假设，更长上下文自动跳过",
                ctx_size
            );
            record_skipped_cases(
                summary_path,
                result_log,
                &cfg.model_name,
                gpu_devices_label,
                &grouped_cases[group_index + 1..],
                parallel_size,
                batch_size,
                "skipped_after_startup_limit",
                &error,
                progress,
                benchmark_mode,
            )
            .await;
        }
        break;
    }
    Ok(())
}

fn matching_failure(
    failures: &[(usize, usize)],
    required_context_per_slot: usize,
    parallel_size: usize,
) -> Option<(usize, usize)> {
    for (failed_parallel, failed_ctx) in failures {
        if parallel_size >= *failed_parallel && required_context_per_slot >= *failed_ctx {
            return Some((*failed_parallel, *failed_ctx));
        }
    }
    None
}

fn mark_failed(
    failures: &mut Vec<(usize, usize)>,
    required_context_per_slot: usize,
    parallel_size: usize,
) {
    for (failed_parallel, failed_ctx) in failures.iter() {
        if *failed_parallel <= parallel_size && *failed_ctx <= required_context_per_slot {
            return;
        }
    }
    failures.retain(|(fp, fc)| !(parallel_size <= *fp && required_context_per_slot <= *fc));
    failures.push((parallel_size, required_context_per_slot));
}

async fn run_case_group(
    cfg: &MatrixConfig,
    bin_path: &str,
    gpu_devices_label: &str,
    gpu_devices: &[String],
    parallel_size: usize,
    ctx_size: usize,
    batch_size: usize,
    cases: &[Case],
    summary_path: &std::path::Path,
    result_log: &std::path::Path,
    benchmark_mode: &str,
    global_total: usize,
    progress: &mut ProgressTracker,
    record_startup_failures: bool,
) -> Result<bool> {
    let mut attempt = 1usize;
    let service_log = cfg.result_dir.join(format!(
        "{}-{}-{}-ctx{}-attempt{}-service.log",
        cfg.model_name, parallel_size, cfg.dtype, ctx_size, attempt
    ));

    if is_port_open(cfg.port).await {
        let error = format!("端口 {} 已被占用，拒绝复用旧服务", cfg.port);
        progress.write(error.clone());
        append_log(result_log, &error).await;
        if record_startup_failures {
            for case in cases {
                record_case_status(
                    summary_path,
                    result_log,
                    &cfg.model_name,
                    gpu_devices_label,
                    case,
                    ctx_size,
                    batch_size,
                    "startup_failed",
                    &error,
                    &benchmark::BenchmarkMetrics::default(),
                    "",
                    progress,
                    benchmark_mode,
                )
                .await;
            }
        }
        return Ok(false);
    }

    let mut server = Some(
        match start_server(
            bin_path,
            &cfg.model_path,
            &cfg.model_name,
            cfg.port,
            gpu_devices,
            parallel_size,
            ctx_size,
            batch_size,
            cfg.gpu_layers,
            &service_log,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                let error = format!("启动 llama-server 失败: {}", e);
                progress.write(error.clone());
                append_log(result_log, &error).await;
                if record_startup_failures {
                    for case in cases {
                        record_case_status(
                            summary_path,
                            result_log,
                            &cfg.model_name,
                            gpu_devices_label,
                            case,
                            ctx_size,
                            batch_size,
                            "startup_failed",
                            &error,
                            &benchmark::BenchmarkMetrics::default(),
                            "",
                            progress,
                            benchmark_mode,
                        )
                        .await;
                    }
                }
                return Ok(false);
            }
        },
    );

    let ready_timeout = std::env::var("LLAMA_CPP_READY_TIMEOUT_SEC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3600u64);
    if let Err(e) = wait_for_ready(
        cfg.port,
        &mut server.as_mut().unwrap().process,
        &service_log,
        &cfg.model_name,
        ready_timeout,
    )
    .await
    {
        let error = format!("等待服务启动失败: {}", e);
        progress.write(error.clone());
        append_log(result_log, &error).await;
        stop_server(server.take()).await;
        if record_startup_failures {
            for case in cases {
                record_case_status(
                    summary_path,
                    result_log,
                    &cfg.model_name,
                    gpu_devices_label,
                    case,
                    ctx_size,
                    batch_size,
                    "startup_failed",
                    &error,
                    &benchmark::BenchmarkMetrics::default(),
                    "",
                    progress,
                    benchmark_mode,
                )
                .await;
            }
        }
        return Ok(false);
    }

    progress.write(format!(
        "模型已启动，parallel={}, ctx={}，开始预热",
        parallel_size, ctx_size
    ));

    if cfg.warmup_count > 0 && !cases.is_empty() {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        for _ in 0..cfg.warmup_count {
            let base = cases.choose(&mut rng).unwrap();
            let warmup = Case {
                parallel_size: base.parallel_size,
                num_prompts: base.num_prompts,
                input_len: base.input_len,
                output_len: base.output_len,
                input_bucket: "warmup".to_string(),
                output_bucket: "warmup".to_string(),
                concurrency_bucket: "warmup".to_string(),
            };
            let _ = benchmark::run_benchmark(
                benchmark_mode,
                &cfg.host,
                cfg.port,
                &cfg.model_path,
                &cfg.model_name,
                &warmup,
            )
            .await;
            sleep(Duration::from_secs(cfg.sleep_between_cases)).await;
        }
    }

    append_log(
        result_log,
        "\n\n******************************预热完成，马上开始性能测试*******************************\n",
    )
    .await;

    for (idx, case) in cases.iter().enumerate() {
        let current_index = idx + 1;
        let header = format!(
            "\n\n******************************开始 {}/{} {} ctx={} batch={} dtype={} 测试*******************************\n",
            current_index,
            global_total,
            case.case_id(),
            ctx_size,
            batch_size,
            cfg.dtype
        );
        progress.write(header.trim().to_string());
        append_log(result_log, &header).await;

        // check server alive
        if server.as_mut().unwrap().process.try_wait()?.is_some() {
            let error = "server crashed before case; restarting".to_string();
            append_log(result_log, &error).await;
            append_log(result_log, &utils::tail_file(&service_log, 120)).await;
            attempt += 1;
            let service_log2 = cfg.result_dir.join(format!(
                "{}-{}-{}-ctx{}-attempt{}-service.log",
                cfg.model_name, parallel_size, cfg.dtype, ctx_size, attempt
            ));
            stop_server(server.take()).await;
            match start_server(
                bin_path,
                &cfg.model_path,
                &cfg.model_name,
                cfg.port,
                gpu_devices,
                parallel_size,
                ctx_size,
                batch_size,
                cfg.gpu_layers,
                &service_log2,
            )
            .await
            {
                Ok(s) => server = Some(s),
                Err(e) => {
                    record_case_status(
                        summary_path,
                        result_log,
                        &cfg.model_name,
                        gpu_devices_label,
                        case,
                        ctx_size,
                        batch_size,
                        "restart_failed",
                        &format!("{}", e),
                        &benchmark::BenchmarkMetrics::default(),
                        "",
                        progress,
                        benchmark_mode,
                    )
                    .await;
                    continue;
                }
            }
            if wait_for_ready(
                cfg.port,
                &mut server.as_mut().unwrap().process,
                &service_log2,
                &cfg.model_name,
                ready_timeout,
            )
            .await
            .is_err()
            {
                record_case_status(
                    summary_path,
                    result_log,
                    &cfg.model_name,
                    gpu_devices_label,
                    case,
                    ctx_size,
                    batch_size,
                    "restart_failed",
                    "服务重启后未就绪",
                    &benchmark::BenchmarkMetrics::default(),
                    "",
                    progress,
                    benchmark_mode,
                )
                .await;
                continue;
            }
        }

        let mut status = "completed";
        let mut error = String::new();
        let mut output_text = String::new();
        let metrics = match benchmark::run_benchmark(
            benchmark_mode,
            &cfg.host,
            cfg.port,
            &cfg.model_path,
            &cfg.model_name,
            case,
        )
        .await
        {
            Ok(out) => {
                output_text = out.text;
                out.metrics
            }
            Err(e) => {
                status = "case_exception";
                error = format!("{}", e);
                benchmark::BenchmarkMetrics::default()
            }
        };

        if server.as_mut().unwrap().process.try_wait()?.is_some() {
            let original_status = status;
            let original_error = error.clone();
            status = "server_crashed";
            let server_error = utils::tail_file(&service_log, 120);
            error = if original_status != "completed" || !original_error.is_empty() {
                format!(
                    "case_status={}; case_error={}; server_error={}",
                    original_status, original_error, server_error
                )
            } else {
                server_error
            };
        } else if status == "completed" {
            if metrics.failed_requests > 0 && metrics.successful_requests == 0 {
                status = "case_failed";
                error = "all requests failed".to_string();
            }
        }

        record_case_status(
            summary_path,
            result_log,
            &cfg.model_name,
            gpu_devices_label,
            case,
            ctx_size,
            batch_size,
            status,
            &error,
            &metrics,
            &output_text,
            progress,
            benchmark_mode,
        )
        .await;

        if status == "server_crashed" {
            progress.write(format!(
                "case 导致服务崩溃，已记录并跳过: {}",
                case.case_id()
            ));
            stop_server(server.take()).await;
            attempt += 1;
            let service_log2 = cfg.result_dir.join(format!(
                "{}-{}-{}-ctx{}-attempt{}-service.log",
                cfg.model_name, parallel_size, cfg.dtype, ctx_size, attempt
            ));
            match start_server(
                bin_path,
                &cfg.model_path,
                &cfg.model_name,
                cfg.port,
                gpu_devices,
                parallel_size,
                ctx_size,
                batch_size,
                cfg.gpu_layers,
                &service_log2,
            )
            .await
            {
                Ok(s) => server = Some(s),
                Err(e) => {
                    append_log(result_log, &format!("restart_failed_after_crash: {}\n", e)).await;
                    continue;
                }
            }
            let _ = wait_for_ready(
                cfg.port,
                &mut server.as_mut().unwrap().process,
                &service_log2,
                &cfg.model_name,
                ready_timeout,
            )
            .await;
        }

        sleep(Duration::from_secs(cfg.sleep_between_cases)).await;
    }

    progress.write(format!(
        "停止 llama-server，parallel={}, ctx={}",
        parallel_size, ctx_size
    ));
    stop_server(server).await;
    sleep(Duration::from_secs(
        std::env::var("LLAMA_CPP_RESTART_SLEEP_SEC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30u64),
    ))
    .await;

    Ok(true)
}

async fn record_case_status(
    summary_path: &std::path::Path,
    result_log: &std::path::Path,
    model_name: &str,
    gpu_devices_label: &str,
    case: &Case,
    ctx_size: usize,
    batch_size: usize,
    status: &str,
    error: &str,
    metrics: &benchmark::BenchmarkMetrics,
    output: &str,
    progress: &mut ProgressTracker,
    benchmark_mode: &str,
) {
    let _ = append_summary(
        summary_path,
        model_name,
        gpu_devices_label,
        case,
        ctx_size,
        batch_size,
        benchmark_mode,
        metrics,
        status,
        error,
    );
    let _ = append_log(
        result_log,
        &format!("status={} error={}", status, utils::normalize_error(error)),
    )
    .await;
    if !output.is_empty() {
        let _ = append_log(result_log, output).await;
    }
    progress.update(case, status);
}

async fn record_skipped_cases(
    summary_path: &std::path::Path,
    result_log: &std::path::Path,
    model_name: &str,
    gpu_devices_label: &str,
    grouped_cases: &[(usize, Vec<Case>)],
    parallel_size: usize,
    batch_size: usize,
    status: &str,
    error: &str,
    progress: &mut ProgressTracker,
    benchmark_mode: &str,
) {
    let _ = append_log(result_log, &format!("{}: {}\n", status, error)).await;
    for (required_context_per_slot, cases) in grouped_cases {
        let ctx_size = required_context_per_slot * parallel_size;
        for case in cases {
            record_case_status(
                summary_path,
                result_log,
                model_name,
                gpu_devices_label,
                case,
                ctx_size,
                batch_size,
                status,
                error,
                &benchmark::BenchmarkMetrics::default(),
                "",
                progress,
                benchmark_mode,
            )
            .await;
        }
    }
}

fn infer_visible_gpu_count(gpu_devices_label: &str, gpu_devices: &[String]) -> Option<String> {
    if gpu_devices_label == "none" {
        return Some("0".to_string());
    }
    if !gpu_devices.is_empty() {
        return Some(gpu_devices.len().to_string());
    }
    if let Ok(v) = std::env::var("CUDA_VISIBLE_DEVICES") {
        if let Some(count) = parse_cuda_visible_device_count(&v) {
            return Some(count.to_string());
        }
    }
    if let Ok(output) = std::process::Command::new("nvidia-smi").arg("-L").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let count = text
                .lines()
                .filter(|l| l.trim().starts_with("GPU "))
                .count();
            return Some(count.to_string());
        }
    }
    None
}

fn infer_gpu_name(gpu_devices_label: &str, gpu_devices: &[String]) -> Option<String> {
    if gpu_devices_label == "none" {
        return Some(String::new());
    }
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let all_names: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if all_names.is_empty() {
        return None;
    }
    let selected: Vec<String> = if gpu_devices.is_empty() {
        all_names.clone()
    } else {
        gpu_devices
            .iter()
            .filter_map(|d| {
                d.chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| all_names.get(i).cloned())
            })
            .collect()
    };
    let names = if selected.is_empty() {
        all_names
    } else {
        selected
    };
    let mut unique = Vec::new();
    for n in names {
        if !unique.contains(&n) {
            unique.push(n);
        }
    }
    Some(unique.join(" / "))
}

fn parse_cuda_visible_device_count(value: &str) -> Option<usize> {
    let count = value
        .split(',')
        .filter(|s| !s.trim().is_empty() && s.trim() != "-1")
        .count();
    if count > 0 { Some(count) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching_failure() {
        let failures = vec![(2, 100)];
        assert_eq!(matching_failure(&failures, 100, 2), Some((2, 100)));
        assert_eq!(matching_failure(&failures, 100, 1), None);
        assert_eq!(matching_failure(&failures, 99, 2), None);
    }

    #[test]
    fn test_mark_failed() {
        let mut failures = Vec::new();
        mark_failed(&mut failures, 100, 2);
        assert_eq!(failures, vec![(2, 100)]);
        // duplicate should not add
        mark_failed(&mut failures, 100, 2);
        assert_eq!(failures, vec![(2, 100)]);
        // (1, 90) dominates (2, 100) -> replace
        mark_failed(&mut failures, 90, 1);
        assert_eq!(failures, vec![(1, 90)]);
        // (4, 200) is less severe -> skip
        mark_failed(&mut failures, 200, 4);
        assert_eq!(failures, vec![(1, 90)]);
        // (1, 50) is more severe -> replace
        mark_failed(&mut failures, 50, 1);
        assert_eq!(failures, vec![(1, 50)]);
    }

    #[test]
    fn test_infer_visible_gpu_count_from_devices() {
        assert_eq!(
            infer_visible_gpu_count("all", &["CUDA0".to_string(), "CUDA1".to_string()]),
            Some("2".to_string())
        );
        assert_eq!(infer_visible_gpu_count("none", &[]), Some("0".to_string()));
    }

    #[test]
    fn test_parse_cuda_visible_device_count() {
        assert_eq!(parse_cuda_visible_device_count("0,1,2"), Some(3));
        assert_eq!(parse_cuda_visible_device_count("0, -1, 1"), Some(2));
        assert_eq!(parse_cuda_visible_device_count(""), None);
        assert_eq!(parse_cuda_visible_device_count("-1"), None);
    }
}
