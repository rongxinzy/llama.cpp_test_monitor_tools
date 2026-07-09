# AGENTS.md — llama-test-matrix

> 本文件面向 AI 编程助手。读者应当被假设对该项目一无所知。以下信息全部来自当前代码库的实际内容，不做推断。

## 项目概述

`llama-test-matrix` 是一个用 Rust 编写的命令行工具，用于对 `llama.cpp` 做自动矩阵压测，并在压测过程中对 GPU 进行黑盒监控。它将原先分散的自动矩阵压测脚本和 GPU 黑盒监控脚本整合到一个二进制中。

主要功能：

- `run` 子命令：自动生成测试 case 矩阵、启动 `llama-server`、执行压测、汇总结果，并生成公司格式报告与硬件规格报告。
- `blackbox` 子命令：独立运行 GPU 黑盒监控，采集 `nvidia-smi`、系统日志，并在检测到 Xid / NVRM / PCIe 等异常时自动捕获快照。

项目信息：

- 语言：Rust（`edition = "2024"`）
- 包名：`llama-test-matrix`
- 版本：`0.1.0`
- 入口：`src/main.rs`
- 构建工具：`cargo`

## 技术栈与依赖

主要运行时依赖：

| 依赖 | 用途 |
|------|------|
| `anyhow` | 错误处理 |
| `chrono` | 时间戳 |
| `clap` | 命令行参数解析（derive + env） |
| `csv` | CSV 报告输出 |
| `futures` | 并发请求聚合 |
| `reqwest` + `reqwest-eventsource` | 调用 `/v1/completions` 和 `/v1/models` |
| `regex` | 日志解析与触发规则 |
| `serde` + `serde_json` | 配置与计划文件序列化 |
| `tokio` | 异步运行时 |
| `tracing` + `tracing-subscriber` | 日志/追踪 |
| `libc` | 进程组管理 |
| `which` | 查找外部命令 |
| `rand` | warmup prompt 加盐 |
| `glob` | 事件快照时匹配日志文件 |

开发依赖：

- `tokio-test`

## 代码组织

```
src/
├── main.rs          # 入口、子命令分发、run 主流程编排
├── cli.rs           # clap 命令行参数定义（RunArgs / BlackboxArgs）
├── config.rs        # 从 CLI / 环境变量构建 MatrixConfig / BlackboxConfig
├── matrix.rs        # case 结构、矩阵生成、长度采样、GPU 设备解析
├── server.rs        # 启动 / 停止 llama-server， readiness 检测
├── benchmark.rs     # builtin / vllm_cli / benchmark_serving 三种压测后端
├── report.rs        # 汇总 CSV、公司格式 CSV、结果解析
├── spec.rs          # 硬件规格采集与 CSV 输出
├── progress.rs      # 终端进度条/状态统计
├── utils.rs         # 时间、主机名、tail、格式化等通用工具
└── blackbox/        # GPU 黑盒监控模块
    ├── mod.rs       # 黑盒生命周期、触发事件通道、任务编排
    ├── collector.rs # nvidia-smi / ps / vmstat / iostat 等采集循环
    ├── logs.rs      # dmesg / journalctl / 系统日志跟踪与触发
    └── snapshot.rs  # baseline 与 incident 快照捕获、打包
```

## 构建命令

常规编译：

```bash
cargo build --release
```

产物路径：`./target/release/llama-test-matrix`

静态编译（用于 glibc 较老的远程机器）：

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

产物路径：`./target/x86_64-unknown-linux-musl/release/llama-test-matrix`

## 测试命令

运行全部单元测试：

```bash
cargo test
```

格式化代码：

```bash
cargo fmt
```

项目当前包含 55 个单元测试，分布于 `config.rs`、`matrix.rs`、`server.rs`、`benchmark.rs`、`report.rs`、`spec.rs`、`utils.rs`。测试覆盖范围包括矩阵采样、GPU 设备解析、CSV 输出、metrics 解析、硬件规格 CSV 生成等。

## 代码风格指南

- 使用 `cargo fmt` 自动格式化，不要手动调整缩进风格。
- 错误信息以中文输出为主（面向运维/测试人员），代码注释以中文为主。
- 文件名、结构体、函数使用 `snake_case`；结构体使用 `PascalCase`。
- 异步函数统一使用 `async fn`，由 `tokio::main` 驱动。
- 外部命令调用优先使用 `tokio::process::Command`，日志写入使用 `tokio::io::AsyncWriteExt`。
- 配置默认值优先从环境变量读取，CLI 参数可覆盖环境变量；环境变量名以 `LLAMA_CPP_` 为前缀。
- 新增环境变量时，在 `config.rs` 中集中管理，并在 `cli.rs` 添加对应参数。

## 子命令与核心流程

### `run` 子命令

1. 解析参数与环境变量，构建 `MatrixConfig`。
2. 验证 `llama-server` 可执行文件路径。
3. 生成输入长度、输出长度、slot 数、请求并发数等采样点。
4. 根据 `--pair-*` 参数决定是笛卡尔积还是按位置配对，生成 case 矩阵。
5. 输出 `matrix-plan.jsonl` 计划文件。
6. 按 `parallel_size` 循环，按 `--ctx-strategy` 策略启动 `llama-server`：
   - `progressive`（默认）：按 `required_context_per_slot` 从小到大启动，失败后跳过更长 ctx。
   - `max-first`：先尝试最大 ctx，失败后再降级为 progressive。
7. 每个 ctx 分组先执行 `--warmup-count` 次随机预热。
8. 逐个 case 运行 benchmark，记录 metrics、status、error 到 `matrix-summary.csv`。
9. case 之间休息 `--sleep-between-cases` 秒。
10. 全部完成后生成 `company-report.csv`（UTF-8 BOM）和 `hardware-spec.csv`。
11. 默认同时启动 blackbox 监控；可通过 `--no-blackbox` 关闭。

### `blackbox` 子命令

可独立运行，监控 GPU 与系统日志：

```bash
./target/release/llama-test-matrix blackbox \
  --out ./gpu-blackbox-runs \
  --stop-after-trigger \
  -- /path/to/cmd
```

内部任务：

- `gpu_metrics_loop`：每秒 `nvidia-smi --query-gpu` 采集 GPU 指标到 CSV。
- `nvidia_smi_table_loop`：定期采集完整 `nvidia-smi` 表格。
- `compute_apps_loop`：定期采集 GPU 计算进程。
- `process_system_light_loop` / `system_slow_loop`：采集 `uptime`、`free`、`ps`、`vmstat`、`iostat`、`sensors` 等。
- `start_log_followers`：跟踪 `dmesg -wT`、`journalctl -kf`、NVIDIA 服务单元日志、`/var/log/kern.log` 等。
- 触发事件通过 channel 发送给 `snapshot_handle`；满足冷却条件后调用 `snapshot::capture_incident` 保存快照并打包 `tar.gz`。

## 关键状态说明

`matrix-summary.csv` 中常见的 `status`：

| status | 含义 |
|--------|------|
| `completed` | 完成 |
| `case_failed` | 请求全部失败但服务未崩溃 |
| `case_exception` | 请求执行异常 |
| `server_crashed` | 运行中服务崩溃 |
| `startup_failed` | 该 ctx 分组启动失败 |
| `restart_failed` | 崩溃后重启失败 |
| `skipped_after_startup_limit` | 同一 parallel 下更长 ctx 被跳过 |
| `skipped_after_global_startup_limit` | 更小 parallel 已失败，更大 parallel 直接跳过 |

## 输出文件结构

运行结束后，`--result-dir`（默认 `benchmark_results/`）下生成：

```
benchmark_results/
  <model>-<dtype>-matrix-plan.jsonl       # 全部 case 计划
  <model>-<dtype>-matrix-summary.csv      # 每个 case 的原始结果与状态
  <model>-<dtype>-company-report.csv      # 公司格式汇总表
  <model>-<dtype>-hardware-spec.csv       # 硬件规格报告
  <model>-<parallel>-<dtype>.log          # 每个 parallel 的结果日志
  <model>-<parallel>-<dtype>-ctx<ctx>-attempt<N>-service.log  # 服务日志
```

## 环境变量速查

| 环境变量 | 对应 CLI 参数 | 默认值 |
|----------|--------------|--------|
| `LLAMA_SERVER_BIN` | `--llama-server-bin` | `llama-server` |
| `LLAMA_CPP_DEVICES` / `LLAMA_CPP_GPU_DEVICES` | `--gpu-devices` | `all` |
| `LLAMA_CPP_CTX_STRATEGY` | `--ctx-strategy` | `progressive` |
| `LLAMA_CPP_PROGRESS` | `--progress` | `plain` |
| `LLAMA_CPP_BENCH_MODE` | `--benchmark-mode` | `auto` |
| `LLAMA_CPP_MATRIX_IO_POINTS` | `--io-points` | `4` |
| `LLAMA_CPP_MATRIX_PROMPT_POINTS` | `--prompt-points` | `3` |
| `LLAMA_CPP_SLEEP_BETWEEN_CASES` | `--sleep-between-cases` | `10` |
| `LLAMA_CPP_WARMUP_COUNT` | `--warmup-count` | `5` |
| `LLAMA_CPP_MAX_BATCH_SIZE` | `--max-batch-size` | `2048` |
| `LLAMA_CPP_GPU_LAYERS` | `--gpu-layers` | `99` |
| `LLAMA_CPP_PHYSICAL_CARDS` | `--physical-cards` | 自动推断 |
| `LLAMA_CPP_LOGICAL_CARDS` | `--logical-cards` | 自动推断 / 同 physical |
| `LLAMA_CPP_REPORT_MACHINE_TYPE` | `--report-machine-type` | 空 |
| `LLAMA_CPP_REPORT_GPU_NAME` | `--report-gpu-name` | 自动推断 |
| `LLAMA_CPP_READY_TIMEOUT_SEC` | 无 | `3600` |
| `LLAMA_CPP_RESTART_SLEEP_SEC` | 无 | `30` |

## 部署与发布

- 推荐对远程目标使用 musl 静态链接构建，避免 glibc 版本不兼容。
- 二进制除 `llama-server` 外，还依赖 `nvidia-smi`、可选的 `nvidia-bug-report.sh`、`dcgmi`、`dmesg`、`journalctl`、`lspci` 等系统工具。
- 静态编译产物可单独拷贝到目标机器运行；结果目录默认生成在运行目录下。

## 安全与运维注意事项

- 工具会调用外部命令（`llama-server`、`nvidia-smi`、`dmesg`、`journalctl`、`tar`、`bash -lc ...` 等），请确保运行环境可信。
- `server.rs` 使用 `unsafe` 块调用 `libc::setpgid` 将子进程放入独立进程组，停止时通过 `killpg`/`kill` 发送 `SIGTERM`/`SIGKILL`。
- 黑盒监控的默认触发正则包含 `NVRM|Xid|PCIe Bus Error|AER` 等 GPU 故障关键字，可通过 `--trigger-regex` 自定义。
- 端口占用检查会拒绝复用已有服务，避免测试数据污染。
- `/v1/models` ready 检查会校验返回的模型 alias 是否等于 `--model-name`。
- 修改代码后请运行 `cargo test` 与 `cargo build --release` 验证。
