#!/usr/bin/env bash
# quick_start.sh — 在 172.18.1.246 上启动 MiniMax 20 case 最大压力压测
set -e

cd /root/llm_test_tools
mkdir -p benchmark_results

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LLAMA_CPP_MATRIX_IO_POINTS=4 \
LLAMA_CPP_CTX_STRATEGY=max-first \
LLAMA_CPP_PROGRESS=plain \
nohup ./llama-test-matrix run \
  --llama-server-bin /root/llama.cpp/build/bin/llama-server \
  --model-path /root/models/chat/MiniMax-M2.7-UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf \
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
  --pair-input-output-lens \
  > benchmark_results/quick_start_20case.log 2>&1 &

echo "20 case max-pressure benchmark started in background, PID=$!"
echo "Log: /root/llm_test_tools/benchmark_results/quick_start_20case.log"
