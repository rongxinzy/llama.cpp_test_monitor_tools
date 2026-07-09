use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "llama-test-matrix")]
#[command(about = "llama.cpp benchmark matrix + GPU blackbox monitor", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run the llama.cpp benchmark matrix.
    Run(RunArgs),
    /// Run the GPU blackbox monitor standalone.
    Blackbox(BlackboxArgs),
}

#[derive(Debug, Parser, Clone)]
pub struct RunArgs {
    /// Path to llama-server binary.
    #[arg(long)]
    pub llama_server_bin: Option<String>,

    /// Path to GGUF model.
    #[arg(long)]
    pub model_path: Option<String>,

    /// Served model alias.
    #[arg(long)]
    pub model_name: Option<String>,

    /// Server port.
    #[arg(long)]
    pub port: Option<u16>,

    /// Result label / dtype.
    #[arg(long)]
    pub dtype: Option<String>,

    /// GPU devices to pass to --device. "all" means do not pass --device.
    #[arg(long)]
    pub gpu_devices: Option<String>,

    /// Physical GPU card count (reporting).
    #[arg(long)]
    pub physical_cards: Option<usize>,

    /// Logical GPU card count (reporting).
    #[arg(long)]
    pub logical_cards: Option<usize>,

    /// llama-server --parallel range, e.g. 1-8.
    #[arg(long, alias = "slot-parallel-range")]
    pub parallel_range: Option<String>,

    /// Exact comma-separated --parallel values, e.g. 1,4,8.
    #[arg(long)]
    pub parallel_sizes: Option<String>,

    /// Input token length range, e.g. 64-4096.
    #[arg(long)]
    pub input_len_range: Option<String>,

    /// Exact comma-separated input lengths.
    #[arg(long)]
    pub input_lens: Option<String>,

    /// Output token length range, e.g. 64-4096.
    #[arg(long)]
    pub output_len_range: Option<String>,

    /// Exact comma-separated output lengths.
    #[arg(long)]
    pub output_lens: Option<String>,

    /// Request concurrency range, e.g. 1-32.
    #[arg(long)]
    pub num_prompts_range: Option<String>,

    /// Exact comma-separated request concurrency values.
    #[arg(long)]
    pub num_prompts: Option<String>,

    /// Pair --parallel-sizes and --num-prompts by position.
    #[arg(long)]
    pub pair_parallel_with_num_prompts: bool,

    /// Pair input and output length points by position.
    #[arg(long)]
    pub pair_input_output_lens: bool,

    /// Model name in company report.
    #[arg(long)]
    pub report_model_name: Option<String>,

    /// Precision in company report.
    #[arg(long)]
    pub report_precision: Option<String>,

    /// Machine type in company report.
    #[arg(long)]
    pub report_machine_type: Option<String>,

    /// GPU name in company report.
    #[arg(long)]
    pub report_gpu_name: Option<String>,

    /// Output path for company-format CSV.
    #[arg(long)]
    pub company_report_path: Option<String>,

    /// Do not generate the company-format CSV.
    #[arg(long)]
    pub no_company_report: bool,

    /// Benchmark mode: builtin, vllm_cli, benchmark_serving, auto.
    #[arg(long)]
    pub benchmark_mode: Option<String>,

    /// Context strategy: progressive or max-first.
    #[arg(long, default_value = "progressive")]
    pub ctx_strategy: String,

    /// Progress display: plain, none.
    #[arg(long, default_value = "plain")]
    pub progress: String,

    /// Host for benchmark requests.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Result directory.
    #[arg(long, default_value = "benchmark_results")]
    pub result_dir: String,

    /// Number of IO points to sample.
    #[arg(long)]
    pub io_points: Option<usize>,

    /// Number of prompt concurrency points to sample.
    #[arg(long)]
    pub prompt_points: Option<usize>,

    /// Sleep between cases (seconds).
    #[arg(long)]
    pub sleep_between_cases: Option<u64>,

    /// Warmup count.
    #[arg(long)]
    pub warmup_count: Option<usize>,

    /// Max batch-size.
    #[arg(long)]
    pub max_batch_size: Option<usize>,

    /// GPU layers (-ngl).
    #[arg(long)]
    pub gpu_layers: Option<usize>,

    /// Disable automatic GPU blackbox during run.
    #[arg(long)]
    pub no_blackbox: bool,

    /// Blackbox output root.
    #[arg(long, default_value = "gpu-blackbox-runs")]
    pub blackbox_out: String,

    /// Blackbox GPU metric interval (seconds).
    #[arg(long, default_value = "1")]
    pub blackbox_interval: f64,

    /// Blackbox cooldown between incident captures (seconds).
    #[arg(long, default_value = "60")]
    pub blackbox_cooldown: u64,

    /// Blackbox trigger regex.
    #[arg(long)]
    pub blackbox_trigger_regex: Option<String>,

    /// Stop benchmark after first blackbox trigger.
    #[arg(long)]
    pub blackbox_stop_after_trigger: bool,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            llama_server_bin: None,
            model_path: None,
            model_name: None,
            port: None,
            dtype: None,
            gpu_devices: None,
            physical_cards: None,
            logical_cards: None,
            parallel_range: None,
            parallel_sizes: None,
            input_len_range: None,
            input_lens: None,
            output_len_range: None,
            output_lens: None,
            num_prompts_range: None,
            num_prompts: None,
            pair_parallel_with_num_prompts: false,
            pair_input_output_lens: false,
            report_model_name: None,
            report_precision: None,
            report_machine_type: None,
            report_gpu_name: None,
            company_report_path: None,
            no_company_report: false,
            benchmark_mode: None,
            ctx_strategy: "progressive".to_string(),
            progress: "plain".to_string(),
            host: "127.0.0.1".to_string(),
            result_dir: "benchmark_results".to_string(),
            io_points: None,
            prompt_points: None,
            sleep_between_cases: None,
            warmup_count: None,
            max_batch_size: None,
            gpu_layers: None,
            no_blackbox: false,
            blackbox_out: "gpu-blackbox-runs".to_string(),
            blackbox_interval: 1.0,
            blackbox_cooldown: 60,
            blackbox_trigger_regex: None,
            blackbox_stop_after_trigger: false,
        }
    }
}

#[derive(Debug, Parser, Clone)]
pub struct BlackboxArgs {
    /// Output root directory.
    #[arg(long, default_value = "gpu-blackbox-runs")]
    pub out: String,

    /// GPU metric interval (seconds).
    #[arg(long, default_value = "1")]
    pub interval: f64,

    /// ps/proc sampler interval (seconds).
    #[arg(long, default_value = "5")]
    pub ps_interval: f64,

    /// Slower sampler interval (seconds).
    #[arg(long, default_value = "30")]
    pub detail_interval: f64,

    /// Cooldown between incident captures (seconds).
    #[arg(long, default_value = "60")]
    pub cooldown: u64,

    /// Stop after first incident, package logs, then exit.
    #[arg(long)]
    pub stop_after_trigger: bool,

    /// Do not run nvidia-bug-report.sh after incident.
    #[arg(long)]
    pub no_bug_report: bool,

    /// Run dcgmi diag -r 1 during incident capture if dcgmi exists.
    #[arg(long)]
    pub dcgm_diag: bool,

    /// Do not apt-get install missing diagnostic tools.
    #[arg(long)]
    pub no_install_missing: bool,

    /// Override trigger regex.
    #[arg(long)]
    pub trigger_regex: Option<String>,

    /// Optional command to wrap.
    #[arg(trailing_var_arg = true)]
    pub command: Vec<String>,
}

impl Default for BlackboxArgs {
    fn default() -> Self {
        Self {
            out: "gpu-blackbox-runs".to_string(),
            interval: 1.0,
            ps_interval: 5.0,
            detail_interval: 30.0,
            cooldown: 60,
            stop_after_trigger: false,
            no_bug_report: false,
            dcgm_diag: false,
            no_install_missing: false,
            trigger_regex: None,
            command: vec![],
        }
    }
}
