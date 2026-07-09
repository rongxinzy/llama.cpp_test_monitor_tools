use crate::cli::{BlackboxArgs, RunArgs};
use anyhow::{Result, bail};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct MatrixConfig {
    pub llama_server_bin: String,
    pub model_path: String,
    pub model_name: String,
    pub port: u16,
    pub dtype: String,
    pub gpu_devices: String,
    pub physical_cards: Option<usize>,
    pub logical_cards: Option<usize>,
    pub parallel_range: Option<String>,
    pub parallel_sizes: Option<String>,
    pub input_len_range: Option<String>,
    pub input_lens: Option<String>,
    pub output_len_range: Option<String>,
    pub output_lens: Option<String>,
    pub num_prompts_range: Option<String>,
    pub num_prompts: Option<String>,
    pub pair_parallel_with_num_prompts: bool,
    pub pair_input_output_lens: bool,
    pub report_model_name: String,
    pub report_precision: String,
    pub report_machine_type: String,
    pub report_gpu_name: Option<String>,
    pub company_report_path: Option<PathBuf>,
    pub no_company_report: bool,
    pub benchmark_mode: String,
    pub ctx_strategy: String,
    pub progress: String,
    pub host: String,
    pub result_dir: PathBuf,
    pub io_points: usize,
    pub prompt_points: usize,
    pub sleep_between_cases: u64,
    pub warmup_count: usize,
    pub max_batch_size: usize,
    pub gpu_layers: usize,
    pub blackbox: BlackboxConfig,
}

#[derive(Debug, Clone)]
pub struct BlackboxConfig {
    pub enabled: bool,
    pub out: String,
    pub interval: f64,
    pub ps_interval: f64,
    pub detail_interval: f64,
    pub cooldown: u64,
    pub stop_after_trigger: bool,
    pub _run_bug_report: bool,
    pub _run_dcgm_diag: bool,
    pub _auto_install_missing: bool,
    pub trigger_regex: String,
    pub command: Vec<String>,
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_opt(name: &str) -> Option<String> {
    env::var(name).ok().filter(|s| !s.is_empty())
}

pub fn build_matrix_config(args: RunArgs) -> Result<MatrixConfig> {
    let llama_server_bin = args
        .llama_server_bin
        .or_else(|| env_opt("LLAMA_SERVER_BIN"))
        .unwrap_or_else(|| "llama-server".to_string());

    let model_path = match args.model_path {
        Some(p) => p,
        None => bail!("--model-path is required (or use interactive shell wrapper)"),
    };
    let model_name = match args.model_name {
        Some(n) => n,
        None => bail!("--model-name is required"),
    };
    let port = args.port.unwrap_or(18080);
    let dtype = args.dtype.unwrap_or_else(|| "q3_k_xl".to_string());
    let gpu_devices = args.gpu_devices.unwrap_or_else(|| {
        env::var("LLAMA_CPP_DEVICES")
            .or_else(|_| env::var("LLAMA_CPP_GPU_DEVICES"))
            .unwrap_or_else(|_| "all".to_string())
    });

    let report_model_name = args.report_model_name.unwrap_or_else(|| model_name.clone());
    let report_precision = args
        .report_precision
        .unwrap_or_else(|| dtype.to_uppercase());
    let report_machine_type = args
        .report_machine_type
        .or_else(|| env_opt("LLAMA_CPP_REPORT_MACHINE_TYPE"))
        .unwrap_or_default();
    let report_gpu_name = args
        .report_gpu_name
        .or_else(|| env_opt("LLAMA_CPP_REPORT_GPU_NAME"));

    let benchmark_mode = args
        .benchmark_mode
        .or_else(|| env_opt("LLAMA_CPP_BENCH_MODE"))
        .unwrap_or_else(|| "auto".to_string());

    let ctx_strategy = args.ctx_strategy.to_lowercase();
    if !matches!(ctx_strategy.as_str(), "progressive" | "max-first") {
        bail!("--ctx-strategy must be progressive or max-first");
    }

    let progress = args.progress.to_lowercase();
    if !matches!(progress.as_str(), "plain" | "none") {
        bail!("--progress must be plain or none");
    }

    let company_report_path = args.company_report_path.map(PathBuf::from);

    let blackbox = BlackboxConfig {
        enabled: !args.no_blackbox,
        out: args.blackbox_out,
        interval: args.blackbox_interval,
        ps_interval: 5.0,
        detail_interval: 30.0,
        cooldown: args.blackbox_cooldown,
        stop_after_trigger: args.blackbox_stop_after_trigger,
        _run_bug_report: true,
        _run_dcgm_diag: false,
        _auto_install_missing: true,
        trigger_regex: args.blackbox_trigger_regex.unwrap_or_else(|| {
            r"NVRM|Xid|GPU has fallen off|fallen off the bus|PCIe Bus Error|AER:|pcieport|NVLink.*(error|Error)|ECC|uncorrectable|GSP|RmInit|Failed to initialize NVML|Unknown Error|ERR!?".to_string()
        }),
        command: vec![],
    };

    Ok(MatrixConfig {
        llama_server_bin,
        model_path,
        model_name,
        port,
        dtype,
        gpu_devices,
        physical_cards: args.physical_cards,
        logical_cards: args.logical_cards,
        parallel_range: args.parallel_range,
        parallel_sizes: args.parallel_sizes,
        input_len_range: args.input_len_range,
        input_lens: args.input_lens,
        output_len_range: args.output_len_range,
        output_lens: args.output_lens,
        num_prompts_range: args.num_prompts_range,
        num_prompts: args.num_prompts,
        pair_parallel_with_num_prompts: args.pair_parallel_with_num_prompts,
        pair_input_output_lens: args.pair_input_output_lens,
        report_model_name,
        report_precision,
        report_machine_type,
        report_gpu_name,
        company_report_path,
        no_company_report: args.no_company_report,
        benchmark_mode,
        ctx_strategy,
        progress,
        host: args.host,
        result_dir: PathBuf::from(args.result_dir),
        io_points: args.io_points.unwrap_or_else(|| {
            env_or("LLAMA_CPP_MATRIX_IO_POINTS", "4")
                .parse()
                .unwrap_or(4)
        }),
        prompt_points: args.prompt_points.unwrap_or_else(|| {
            env_or("LLAMA_CPP_MATRIX_PROMPT_POINTS", "3")
                .parse()
                .unwrap_or(3)
        }),
        sleep_between_cases: args.sleep_between_cases.unwrap_or_else(|| {
            env_or("LLAMA_CPP_SLEEP_BETWEEN_CASES", "10")
                .parse()
                .unwrap_or(10)
        }),
        warmup_count: args
            .warmup_count
            .unwrap_or_else(|| env_or("LLAMA_CPP_WARMUP_COUNT", "5").parse().unwrap_or(5)),
        max_batch_size: args.max_batch_size.unwrap_or_else(|| {
            env_or("LLAMA_CPP_MAX_BATCH_SIZE", "2048")
                .parse()
                .unwrap_or(2048)
        }),
        gpu_layers: args
            .gpu_layers
            .unwrap_or_else(|| env_or("LLAMA_CPP_GPU_LAYERS", "99").parse().unwrap_or(99)),
        blackbox,
    })
}

pub fn build_blackbox_config(args: BlackboxArgs) -> BlackboxConfig {
    BlackboxConfig {
        enabled: true,
        out: args.out,
        interval: args.interval,
        ps_interval: args.ps_interval,
        detail_interval: args.detail_interval,
        cooldown: args.cooldown,
        stop_after_trigger: args.stop_after_trigger,
        _run_bug_report: !args.no_bug_report,
        _run_dcgm_diag: args.dcgm_diag,
        _auto_install_missing: !args.no_install_missing,
        trigger_regex: args.trigger_regex.unwrap_or_else(|| {
            r"NVRM|Xid|GPU has fallen off|fallen off the bus|PCIe Bus Error|AER:|pcieport|NVLink.*(error|Error)|ECC|uncorrectable|GSP|RmInit|Failed to initialize NVML|Unknown Error|ERR!?".to_string()
        }),
        command: args.command,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_or_default() {
        assert_eq!(env_or("_LLAMA_TEST_NONEXISTENT_VAR", "def"), "def");
    }

    #[test]
    fn test_env_opt_missing() {
        assert_eq!(env_opt("_LLAMA_TEST_ANOTHER_NONEXISTENT"), None);
    }

    #[test]
    fn test_build_matrix_config_requires_model() {
        let args = RunArgs {
            llama_server_bin: None,
            model_path: None,
            model_name: Some("m".to_string()),
            ..RunArgs::default()
        };
        assert!(build_matrix_config(args).is_err());
    }

    #[test]
    fn test_build_matrix_config_requires_model_name() {
        let args = RunArgs {
            llama_server_bin: None,
            model_path: Some("/tmp/m.gguf".to_string()),
            model_name: None,
            ..RunArgs::default()
        };
        assert!(build_matrix_config(args).is_err());
    }

    #[test]
    fn test_build_matrix_config_defaults() {
        let args = RunArgs {
            llama_server_bin: None,
            model_path: Some("/tmp/m.gguf".to_string()),
            model_name: Some("m".to_string()),
            ..RunArgs::default()
        };
        let cfg = build_matrix_config(args).unwrap();
        assert_eq!(cfg.port, 18080);
        assert_eq!(cfg.dtype, "q3_k_xl");
        assert_eq!(cfg.gpu_devices, "all");
        assert_eq!(cfg.ctx_strategy, "progressive");
        assert_eq!(cfg.progress, "plain");
        assert_eq!(cfg.io_points, 4);
        assert_eq!(cfg.prompt_points, 3);
        assert_eq!(cfg.sleep_between_cases, 10);
        assert_eq!(cfg.warmup_count, 5);
        assert_eq!(cfg.max_batch_size, 2048);
        assert_eq!(cfg.gpu_layers, 99);
        assert!(cfg.blackbox.enabled);
    }

    #[test]
    fn test_build_matrix_config_invalid_ctx_strategy() {
        let args = RunArgs {
            llama_server_bin: None,
            model_path: Some("/tmp/m.gguf".to_string()),
            model_name: Some("m".to_string()),
            ctx_strategy: "invalid".to_string(),
            ..RunArgs::default()
        };
        assert!(build_matrix_config(args).is_err());
    }

    #[test]
    fn test_build_matrix_config_invalid_progress() {
        let args = RunArgs {
            llama_server_bin: None,
            model_path: Some("/tmp/m.gguf".to_string()),
            model_name: Some("m".to_string()),
            progress: "fancy".to_string(),
            ..RunArgs::default()
        };
        assert!(build_matrix_config(args).is_err());
    }

    #[test]
    fn test_build_blackbox_config_defaults() {
        let args = BlackboxArgs {
            out: "gpu-blackbox-runs".to_string(),
            interval: 1.0,
            ps_interval: 5.0,
            detail_interval: 30.0,
            cooldown: 60,
            stop_after_trigger: false,
            no_bug_report: false,
            dcgm_diag: true,
            no_install_missing: true,
            trigger_regex: None,
            command: vec![],
        };
        let cfg = build_blackbox_config(args);
        assert!(cfg._run_dcgm_diag);
        assert!(!cfg._auto_install_missing);
        assert!(cfg.trigger_regex.contains("NVRM"));
    }
}
