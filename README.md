# llama-test-matrix

llama.cpp 自动矩阵压测 + GPU 黑盒监控工具。

- 测试人员使用说明：见 [`MANUAL.md`](./MANUAL.md)。
- 开发和编译说明：见下文。

## 快速使用

本工具为单个二进制，直接运行即可：

```bash
llama-test-matrix run \
  --llama-server-bin /root/llama.cpp/build/bin/llama-server \
  --model-path /path/to/model.gguf \
  --model-name my-model \
  --port 18080 \
  --dtype q3_k_xl \
  --gpu-devices all \
  --parallel-sizes 1,4,8,16,32 \
  --num-prompts 1,4,8,16,32 \
  --pair-parallel-with-num-prompts \
  --input-len-range 512-50000 \
  --output-len-range 512-50000 \
  --pair-input-output-lens
```

完整参数、推荐命令与故障排查请见 [`MANUAL.md`](./MANUAL.md)。

---

## 开发

### 代码结构

```text
src/
├── main.rs          # 入口、子命令分发、run 主流程编排
├── cli.rs           # clap 命令行参数定义
├── config.rs        # 从 CLI / 环境变量构建配置
├── matrix.rs        # case 结构、矩阵生成、长度采样、GPU 设备解析
├── server.rs        # 启动 / 停止 llama-server，readiness 检测
├── benchmark.rs     # builtin / vllm_cli / benchmark_serving 压测后端
├── report.rs        # 汇总 CSV、公司格式 CSV
├── spec.rs          # 硬件规格采集与 CSV 输出
├── progress.rs      # 终端进度显示
├── utils.rs         # 时间、主机名、tail、格式化等工具
└── blackbox/        # GPU 黑盒监控模块
    ├── mod.rs
    ├── collector.rs
    ├── logs.rs
    └── snapshot.rs
```

### 依赖

- Rust 工具链（建议 1.85+）
- `cargo`
- 系统依赖：`llama-server`、`nvidia-smi`，以及可选的 `nvidia-bug-report.sh`、`dcgmi`、`dmesg`、`journalctl` 等

### 编译

常规编译：

```bash
cargo build --release
```

产物：`./target/release/llama-test-matrix`。

如需部署到 glibc 较老的远程机器，使用 musl 静态编译：

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

产物：`./target/x86_64-unknown-linux-musl/release/llama-test-matrix`。

### 测试与格式化

运行全部单元测试：

```bash
cargo test
```

格式化代码：

```bash
cargo fmt
```

当前项目包含 55 个单元测试，覆盖矩阵采样、GPU 设备解析、CSV 输出、metrics 解析、硬件规格 CSV 生成等。

### 关键设计

- `run` 子命令默认按 `progressive` 策略执行：按 `required_context_per_slot` 从小到大启动 `llama-server`，某个 ctx 分组启动失败后，同一 parallel 下更长 ctx 自动跳过。
- 默认同时后台启动 GPU blackbox 监控，可通过 `--no-blackbox` 关闭。
- 配置默认值优先从环境变量读取，CLI 参数可覆盖环境变量；环境变量名以 `LLAMA_CPP_` 为前缀。
- 错误信息面向运维/测试人员，以中文输出为主。
