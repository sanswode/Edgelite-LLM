#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_MODEL:?Please set VLLM_MODEL, for example Qwen/Qwen2.5-1.5B-Instruct}"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8003}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

vllm serve "${VLLM_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}"
