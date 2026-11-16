#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for rollouts (separate venv, separate GPU).
#   scripts/launch_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8011 1        # model, port, GPU
#   scripts/launch_vllm.sh HuggingFaceTB/SmolVLM2-2.2B-Instruct 8011 1 --max-model-len 8192   # extra args -> vllm serve
# Env: VLLM_BIN (default ~/venvs/vllm/bin/vllm), MAX_LORA_RANK (64), GPU_UTIL (0.85), MAX_LEN (32768)
set -euo pipefail
MODEL=${1:?model}; PORT=${2:-8000}; GPU=${3:-1}
VLLM_BIN=${VLLM_BIN:-$HOME/venvs/vllm/bin/vllm}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # enables /v1/load_lora_adapter
export VLLM_USE_FLASHINFER_SAMPLER=0           # avoid FlashInfer JIT (needs nvcc) for sampling
CUDA_VISIBLE_DEVICES=$GPU exec "$VLLM_BIN" serve "$MODEL" --port "$PORT" \
  --enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" --max-loras 2 \
  --enable-prefix-caching --max-model-len "${MAX_LEN:-32768}" \
  --limit-mm-per-prompt '{"image": 32}' --gpu-memory-utilization "${GPU_UTIL:-0.85}" "${@:4}"
