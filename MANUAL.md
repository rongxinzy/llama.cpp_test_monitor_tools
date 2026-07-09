# llama-test-matrix 使用说明书

`llama-test-matrix` 是 llama.cpp 测试/监控工具，把自动矩阵压测脚本和 GPU 黑盒监控整合到一个二进制里。核心子命令：

- `run`：自动构造 case 矩阵、启动 llama-server、执行压测、生成汇总报告和硬件规格报告。
- `blackbox`：独立运行 GPU 黑盒监控，采集 `nvidia-smi`、系统日志，并在触发 Xid / NVRM / PCIe 等异常时自动快照。

目录：

- [1. 快速开始](#1-快速开始)
- [2. 子命令与常用参数](#2-子命令与常用参数)
- [3. case 矩阵生成规则](#3-case-矩阵生成规则)
- [4. 输出文件说明](#4-输出文件说明)
- [5. 公司格式报告与硬件规格报告](#5-公司格式报告与硬件规格报告)
- [6. GPU 黑盒监控](#6-gpu-黑盒监控)
- [7. 推荐 20 case 命令](#7-推荐-20-case-命令)
- [8. 故障排查](#8-故障排查)

---

## 1. 快速开始

下面是一条完整、可直接运行的最小命令示例：

```bash
llama-test-matrix run \
  --llama-server-bin /root/llama.cpp/build/bin/llama-server \
  --model-path /models/MiniMax-M2.7-UD-Q3_K_XL/UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf \
  --model-name minimax \
  --port 18080 \
  --dtype q3_k_xl \
  --gpu-devices all \
  --physical-cards 8 \
  --logical-cards 8 \
  --parallel-sizes 1,4,8,16,32 \
  --num-prompts 1,4,8,16,32 \
  --pair-parallel-with-num-prompts \
  --input-len-range 512-50000 \
  --output-len-range 512-50000 \
  --pair-input-output-lens
```

说明：

- `--gpu-devices all` 表示不额外传 `--device`，让 llama.cpp 使用 `CUDA_VISIBLE_DEVICES` 下所有可见 GPU。
- `--pair-parallel-with-num-prompts` 让 slot 和请求并发一一配对（p1/c1、p4/c4 ...）。
- `--pair-input-output-lens` 让输入/输出长度点一一配对，把 5×5×4×4 的交叉矩阵压缩到 5×4 = 20 个 case。

---

## 2. 子命令与常用参数

### 2.1 `run` 子命令

#### 必填参数

| 参数 | 说明 | 环境变量 |
|------|------|----------|
| `--llama-server-bin` | llama-server 可执行文件路径 | `LLAMA_SERVER_BIN` |
| `--model-path` | GGUF 模型文件路径 | - |
| `--model-name` | 服务对外暴露的模型 alias | - |

#### 核心测试参数

| 参数 | 说明 | 默认值 / 环境变量 |
|------|------|-------------------|
| `--port` | 服务端口 | `18080` |
| `--dtype` | 结果标签 / 精度 | `q3_k_xl` |
| `--gpu-devices` | GPU 设备范围，`all` 表示不传 `--device` | `all`（`LLAMA_CPP_DEVICES` / `LLAMA_CPP_GPU_DEVICES`） |
| `--physical-cards` | 物理卡数（仅报告） | 自动推断 |
| `--logical-cards` | 逻辑卡数（仅报告） | 自动推断 |
| `--parallel-range` | slot 范围，例如 `1-8` | `1-8` |
| `--parallel-sizes` | 精确 slot 列表，例如 `1,4,8` | - |
| `--input-len-range` | 输入长度范围，例如 `64-4096` | `64-4096` |
| `--input-lens` | 精确输入长度列表 | - |
| `--output-len-range` | 输出长度范围 | `64-4096` |
| `--output-lens` | 精确输出长度列表 | - |
| `--num-prompts-range` | 请求并发范围 | `1-32` |
| `--num-prompts` | 精确请求并发列表 | - |
| `--pair-parallel-with-num-prompts` | slot 与请求并发按位置配对 | - |
| `--pair-input-output-lens` | 输入与输出长度按位置配对 | - |

#### 执行策略参数

| 参数 | 说明 | 默认值 / 环境变量 |
|------|------|-------------------|
| `--ctx-strategy` | `progressive`（短 ctx 先跑）或 `max-first` | `progressive`（`LLAMA_CPP_CTX_STRATEGY`） |
| `--progress` | `plain` 或 `none` | `plain`（`LLAMA_CPP_PROGRESS`） |
| `--benchmark-mode` | `builtin` / `vllm_cli` / `benchmark_serving` / `auto` | `auto`（`LLAMA_CPP_BENCH_MODE`） |
| `--host` | benchmark 请求 host | `127.0.0.1` |
| `--result-dir` | 结果目录 | `benchmark_results` |
| `--io-points` | 输入/输出长度采样点数 | `4`（`LLAMA_CPP_MATRIX_IO_POINTS`） |
| `--prompt-points` | 请求并发采样点数 | `3`（`LLAMA_CPP_MATRIX_PROMPT_POINTS`） |
| `--sleep-between-cases` | case 之间休息秒数 | `10`（`LLAMA_CPP_SLEEP_BETWEEN_CASES`） |
| `--warmup-count` | 每个 ctx 分组预热次数 | `5`（`LLAMA_CPP_WARMUP_COUNT`） |
| `--max-batch-size` | 最大 batch-size | `2048`（`LLAMA_CPP_MAX_BATCH_SIZE`） |
| `--gpu-layers` | `-ngl` 层数 | `99`（`LLAMA_CPP_GPU_LAYERS`） |

#### 报告参数

| 参数 | 说明 |
|------|------|
| `--report-model-name` | 公司报告里的模型名（默认 `--model-name`） |
| `--report-precision` | 公司报告里的精度名（默认 `--dtype` 大写） |
| `--report-machine-type` | 机型（默认空） |
| `--report-gpu-name` | GPU 名称（默认从 `nvidia-smi` 推断） |
| `--company-report-path` | 公司报告 CSV 路径 |
| `--no-company-report` | 不生成公司报告和硬件规格 CSV |

#### 黑盒参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no-blackbox` | 不启动 GPU 黑盒监控 | 默认启用 |
| `--blackbox-out` | 黑盒输出目录 | `gpu-blackbox-runs` |
| `--blackbox-interval` | GPU 采样间隔（秒） | `1` |
| `--blackbox-cooldown` | 触发后冷却时间（秒） | `60` |
| `--blackbox-trigger-regex` | 触发正则 | 内置默认 |
| `--blackbox-stop-after-trigger` | 首次触发后停止压测 | - |

### 2.2 `blackbox` 子命令

```bash
llama-test-matrix blackbox [OPTIONS] [-- <COMMAND>]
```

常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out` | 输出根目录 | `gpu-blackbox-runs` |
| `--interval` | GPU 采样间隔 | `1` |
| `--ps-interval` | 进程采样间隔 | `5` |
| `--detail-interval` | 慢速采样间隔 | `30` |
| `--cooldown` | 触发冷却 | `60` |
| `--stop-after-trigger` | 触发后停止并打包 | - |
| `--no-bug-report` | 不运行 `nvidia-bug-report.sh` | - |
| `--dcgm-diag` | 触发时运行 `dcgmi diag -r 1` | - |
| `--no-install-missing` | 不自动安装缺失诊断工具 | - |
| `--trigger-regex` | 自定义触发正则 | 内置默认 |

独立运行示例：

```bash
llama-test-matrix blackbox \
  --out ./gpu-blackbox-runs \
  --stop-after-trigger \
  -- /path/to/cmd
```

---

## 3. case 矩阵生成规则

默认情况下，矩阵是四个维度的笛卡尔积：

```text
parallel_sizes × num_prompts × input_lens × output_lens
```

例如：

```text
slot  {1,2,4,8}      -> 4 个点
并发  {1,8,32}       -> 3 个点
输入  {64,512,2048,4096} -> 4 个点
输出  {64,512,2048,4096} -> 4 个点
case 总数 = 4 × 3 × 4 × 4 = 192
```

### 3.1 配对模式

- `--pair-parallel-with-num-prompts`：把 slot 列表和并发列表按位置配对，例如 `1/1, 4/4, 8/8, 16/16, 32/32`。
- `--pair-input-output-lens`：把输入列表和输出列表按位置配对，例如 `512/512, 2048/2048, 16384/16384, 50000/50000`。

两者同时开启时，20 case 命令可得到 5×4 = 20 个 case。

### 3.2 ctx-size 计算

每个 case 的 `required_context_per_slot` 为：

```text
input_len + output_len + 16
```

llama-server 实际启动参数：

```text
--ctx-size = required_context_per_slot × parallel_size
--batch-size 根据最大输入长度自动选 512 / 1024 / 2048
```

### 3.3 执行策略

- `progressive`（默认）：按 `required_context_per_slot` 从小到大启动服务。某个 ctx 分组启动失败后，同一 parallel 下更长 ctx 直接标记为 `skipped_after_startup_limit`。
- `max-first`：先尝试最大 ctx 一次性启动所有 case；失败后再降级到 progressive。

---

## 4. 输出文件说明

运行结束后，`--result-dir` 下会生成：

```text
benchmark_results/
  <model>-<dtype>-matrix-plan.jsonl      # 全部 case 的计划清单
  <model>-<dtype>-matrix-summary.csv     # 每个 case 的原始结果与状态
  <model>-<dtype>-company-report.csv     # 公司格式汇总表
  <model>-<dtype>-hardware-spec.csv      # 硬件规格报告（含 GPU 拓扑）
  <model>-<parallel>-<dtype>.log         # 每个 parallel 档的结果日志
  <model>-<parallel>-<dtype>-ctx<ctx>-attempt<N>-service.log  # 服务日志
```

### 4.1 matrix-summary.csv

包含每个 case 的完整信息，关键列：

```text
gpu_devices,benchmark_mode,status,error,prompt_token_source,output_token_source
```

常见 `status`：

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

查看非 completed 行：

```bash
awk -F, 'NR==1 || $15 != "completed" {print}' benchmark_results/*-matrix-summary.csv
```

### 4.2 服务日志

每次启动 llama-server 都会保留一个 `attempt<N>-service.log`，崩溃后可从日志尾部定位原因。

---

## 5. 公司格式报告与硬件规格报告

### 5.1 company-report.csv

测试结束后自动生成，UTF-8 BOM 编码，Excel 直接打开不乱码。表头：

```text
测试时间,机型,GPU,模型,精度,物理卡数,逻辑卡数,模式,请求并发数,输入,输出,总输入,总输出,请求吞吐,输出吞吐,总吞吐,首Token延时(ms),每Token延时(ms),总耗时(s),平均每用户输出吞吐,备注
```

可通过报告参数覆盖默认字段，例如：

```bash
--report-machine-type "8F" \
--report-gpu-name "NVIDIA GeForce RTX 4090" \
--report-model-name "MiniMax-M2.7" \
--report-precision "Q3_K_XL" \
--company-report-path benchmark_results/minimax-company-report.csv
```

### 5.2 hardware-spec.csv

硬件规格 CSV 与公司格式报告一起生成，格式为：

```text
测试批次,模型,精度,项目,值
```

采集内容包括：

- 采集时间、主机名
- 操作系统、内核版本
- CPU 型号、逻辑核心数、物理插槽数
- 内存总量
- GPU 数量、GPU 型号、每张卡的显存、驱动版本、PCIe 最大链路
- **GPU 拓扑结构**（`nvidia-smi topo -m`）
- CUDA 版本

如果 `--no-company-report` 被设置，则硬件规格 CSV 也不会生成。

---

## 6. GPU 黑盒监控

### 6.1 随矩阵测试自动启动

`run` 默认会在后台启动 blackbox，输出到 `./gpu-blackbox-runs`。检测到 NVRM/Xid/AER/PCIe 等异常时自动捕获 incident 快照。

禁用 blackbox：

```bash
--no-blackbox
```

自定义参数：

```bash
--blackbox-out ./gpu-blackbox-runs \
--blackbox-interval 1 \
--blackbox-cooldown 60 \
--blackbox-trigger-regex 'NVRM|Xid|AER' \
--blackbox-stop-after-trigger
```

### 6.2 独立运行

```bash
llama-test-matrix blackbox \
  --out ./gpu-blackbox-runs \
  --stop-after-trigger \
  -- /path/to/cmd
```

---

## 7. 推荐 20 case 命令

下面命令在 8 卡机器上跑 5 个并发档 × 4 个长度点 = 20 case。

### 7.1 MiniMax

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LLAMA_CPP_MATRIX_IO_POINTS=4 \
LLAMA_CPP_CTX_STRATEGY=progressive \
LLAMA_CPP_PROGRESS=plain \
llama-test-matrix run \
  --llama-server-bin /root/llama.cpp/build/bin/llama-server \
  --model-path /models/MiniMax-M2.7-UD-Q3_K_XL/UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf \
  --model-name minimax-m2-7-q3 \
  --port 18080 \
  --dtype q3_k_xl \
  --gpu-devices all \
  --physical-cards 8 \
  --logical-cards 8 \
  --parallel-sizes 1,4,8,16,32 \
  --num-prompts 1,4,8,16,32 \
  --pair-parallel-with-num-prompts \
  --input-len-range 512-50000 \
  --output-len-range 512-50000 \
  --pair-input-output-lens
```

### 7.2 Qwen

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LLAMA_CPP_MATRIX_IO_POINTS=4 \
LLAMA_CPP_CTX_STRATEGY=progressive \
LLAMA_CPP_PROGRESS=plain \
llama-test-matrix run \
  --llama-server-bin /root/llama.cpp/build/bin/llama-server \
  --model-path /models/Qwen3-235B-2507-UD-Q3_K_XL/UD-Q3_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q3_K_XL-00001-of-00003.gguf \
  --model-name qwen3-235b-2507-q3 \
  --port 18081 \
  --dtype q3_k_xl \
  --gpu-devices all \
  --physical-cards 8 \
  --logical-cards 8 \
  --parallel-sizes 1,4,8,16,32 \
  --num-prompts 1,4,8,16,32 \
  --pair-parallel-with-num-prompts \
  --input-len-range 512-50000 \
  --output-len-range 512-50000 \
  --pair-input-output-lens
```

### 7.3 更短的整机吞吐测试

如果目标是测整机 8 卡吞吐，不一定要用很多 slot 档：

```bash
--gpu-devices all \
--parallel-sizes 1,4 \
--num-prompts 32
```

客户端 32 并发会形成整机压力，而 `--parallel` 只控制 KV slot 数。

---

## 8. 故障排查

### 8.1 运行时错误

| 现象 | 原因 | 处理 |
|------|------|------|
| `端口 xxxx 已被占用，拒绝复用旧服务` | 目标端口已有服务 | 杀掉占用端口的进程再跑 |
| `启动 llama-server 失败` | 模型路径错误、CUDA 不可用、显存不足 | 检查 `--model-path`、CUDA、最小 ctx 是否能启动 |
| `startup_failed` | 当前 ctx 分组启动失败 | 正常现象，progressive 策略会跳过更长 ctx |
| `server_crashed` | case 运行中服务崩溃 | 查看对应 `attemptN-service.log` |
| `skipped_after_global_startup_limit` | 小 parallel 已失败，大 parallel 直接跳过 | 降低输入/输出长度或减少 slot 数 |

### 8.2 数据准确性

- 启动服务前会检查端口占用，避免复用旧服务。
- `/v1/models` ready 检查会校验返回的模型名/alias 是否等于 `--model-name`。
- builtin benchmark 优先使用响应里的 `usage.prompt_tokens` / `usage.completion_tokens` 统计；拿不到 usage 才回退估算，并在 summary 中标明来源。
- warmup prompt 会加 salt，避免 KV cache 命中污染 TTFT/prefill。

