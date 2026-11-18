# Rollout backends

Generation is the bottleneck of multi-turn VLM RL. Every round re-sends the whole multi-image
context, and PEAR's episodes average 94k tokens. vlmrl has two backends behind the same `generate` interface.

| | `rollout=hf` | `rollout=vllm` |
|---|---|---|
| where | in the trainer process (`HFPolicy.generate`) | a separate vLLM OpenAI-compatible server |
| prefill reuse | none: the full context is re-prefilled every round | **automatic prefix caching** |
| batching | static batches; halves the batch on CUDA OOM and remembers it | continuous batching, 128 concurrent requests |
| weight sync | none needed | **LoRA hot-swap**, or an in-place **full-weight reload**, after each update |
| fine-tuning | LoRA or full | LoRA or full |
| GPUs | 1 | 2 (server + trainer) recommended |
| speed (Qwen3-VL-4B, A6000) | 32-episode eval ≈ 9 min | ≈ 1 min |

## vLLM server

```bash
scripts/launch_vllm.sh <model> <port> <gpu> [extra vllm serve args...]
scripts/launch_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8011 1
```

The launcher sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, `VLLM_SERVER_DEV_MODE=1` (full-weight sync) and
`VLLM_USE_FLASHINFER_SAMPLER=0`, listens on `${VLLM_HOST:-127.0.0.1}` (dev mode exposes admin endpoints), and
serves with `--enable-lora --max-lora-rank ${MAX_LORA_RANK:-64} --max-loras 2` (dropped with `ENABLE_LORA=0`) `--enable-prefix-caching
--max-model-len ${MAX_LEN:-32768} --limit-mm-per-prompt '{"image": 32}' --gpu-memory-utilization
${GPU_UTIL:-0.85}`. Extra arguments are passed through (later flags win).

Point the trainer at it:

```bash
python scripts/train.py rollout=vllm vllm_url=http://localhost:8011
```

### How weight sync works

**LoRA** (`lora_rank > 0`):

1. At startup (or after resuming) and after each optimizer step, the trainer saves the LoRA adapter to
   `out_dir/adapters/policy_<tag>_step{N}` and calls `POST /v1/load_lora_adapter`. `<tag>` is unique per
   trainer process: a sweep reuses one server across runs, and resumed runs restart at an old step, so vLLM
   would otherwise refuse a name that's still loaded.
2. Later requests use `model=policy_<tag>_step{N}`.
3. The previous adapter is unloaded and its directory deleted.

**Full fine-tuning** (`lora_rank=0`; start the server with `ENABLE_LORA=0`, which sweeps do automatically):

1. After each optimizer step, the trainer writes an HF-format checkpoint (`HFPolicy.save`, bf16, the same
   tensor names as the original checkpoint) to `vllm_sync_dir` (default `out_dir/adapters`) as
   `full_<tag>_step{N}`. Step 0 is skipped: the server already serves the base weights.
2. `POST /collective_rpc {"method": "reload_weights", "kwargs": {"weights_path": ...}}` reloads it in place on
   every worker, and `POST /reset_prefix_cache` drops KV blocks computed with the old weights.
3. Requests keep using the base model name. The previous checkpoint directory is deleted.

Both are simple and robust and need no NCCL. The full-weight path costs one disk round trip per step
(seconds for a few-B model on local NVMe). Verified on puget: perturbed weights change the server's
log-probs, and restoring them reproduces the original log-probs exactly.

### Messages and tokens

HF chat messages become OpenAI-format messages, with PNG data-URL images. Sampling sends `temperature`,
`top_p`, `top_k=-1` and `max_tokens`. The server's `usage` (prompt + completion tokens) feeds
`max_context_tokens`. Replies are re-tokenized by the trainer for the loss (see
[Algorithm](../concepts/algorithm.md#multi-turn-loss-masks-model-agnostic)).

!!! warning "One server, one experiment"
    The server is started with `--max-loras 2`. Two trainers sharing a server would evict each other's
    adapters. Give each concurrent run its own server and port.
