# Rollout backends

Generation is the bottleneck of multi-turn VLM RL. Every round re-sends the whole multi-image
context, and PEAR's episodes average 94k tokens. vlmrl has two backends behind the same `generate` interface.

| | `rollout=hf` | `rollout=vllm` |
|---|---|---|
| where | in the trainer process (`HFPolicy.generate`) | a separate vLLM OpenAI-compatible server |
| prefill reuse | none: the full context is re-prefilled every round | **automatic prefix caching** |
| batching | static batches; halves the batch on CUDA OOM and remembers it | continuous batching, 128 concurrent requests |
| weight sync | none needed | **LoRA hot-swap** after each update |
| fine-tuning | LoRA or full | LoRA only |
| GPUs | 1 | 2 (server + trainer) recommended |
| speed (Qwen3-VL-4B, A6000) | 32-episode eval ≈ 9 min | ≈ 1 min |

## vLLM server

```bash
scripts/launch_vllm.sh <model> <port> <gpu> [extra vllm serve args...]
scripts/launch_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8011 1
```

The launcher sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and `VLLM_USE_FLASHINFER_SAMPLER=0`, and
serves with `--enable-lora --max-lora-rank ${MAX_LORA_RANK:-64} --max-loras 2 --enable-prefix-caching
--max-model-len ${MAX_LEN:-32768} --limit-mm-per-prompt '{"image": 32}' --gpu-memory-utilization
${GPU_UTIL:-0.85}`. Extra arguments are passed through (later flags win).

Point the trainer at it:

```bash
python scripts/train.py rollout=vllm vllm_url=http://localhost:8011
```

### How weight sync works

1. At startup and after each optimizer step, the trainer saves the LoRA adapter to
   `out_dir/adapters/policy_step{N}` and calls `POST /v1/load_lora_adapter`.
2. Later requests use `model=policy_step{N}`.
3. The previous adapter is unloaded and its directory deleted.

This is simple and robust, and it needs no NCCL. Full-weight sync is on the [roadmap](../development/roadmap.md).

### Messages and tokens

HF chat messages become OpenAI-format messages, with PNG data-URL images. Sampling sends `temperature`,
`top_p`, `top_k=-1` and `max_tokens`. The server's `usage` (prompt + completion tokens) feeds
`max_context_tokens`. Replies are re-tokenized by the trainer for the loss (see
[Algorithm](../concepts/algorithm.md#multi-turn-loss-masks-model-agnostic)).

!!! warning "One server, one experiment"
    The server is started with `--max-loras 2`. Two trainers sharing a server would evict each other's
    adapters. Give each concurrent run its own server and port.
