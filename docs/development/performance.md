# Performance

The trainer is kept deliberately small. These are the optimizations it does have, and where they came from.

| optimization | where | why |
|---|---|---|
| global token-count normalization of summed micro-batch losses | `algo.policy_loss_sum`, `Trainer.update` | exact token-mean however sequences are split into micro-batches; a per-micro-batch mean would overweight short ones (FeynRL-style) |
| `logits_to_keep` from the first response token | `HFPolicy.token_logprobs` | skips the vocab projection over the system prompt and first image |
| chunked fp32 log-softmax + gather (also top-k / gather for score centering) | `chunked_token_stats` | never materializes [B,T,V] in fp32 (V ≈ 152k for Qwen) |
| length-sorted micro-batches | `Trainer.update` | less padding |
| lazily built, prefetched micro-batches (gradient rescaled by the global token count afterwards) | `Trainer._batches`, `Trainer.update` | a step of ~1000 episodes has ~5–20k per-turn samples; building all batches up front held every sample's pixel values (~45 GB for SmolVLM) |
| reply span from text templates (verified against the token ids) | `HFPolicy._last_reply_span` | per-turn samples skip re-running the image processor on the prefix twice per sample (649 s → 183 s update for ~5k SmolVLM samples) |
| drop images of DAPO-filtered groups | `strip_images` in `Trainer.collect` | filtered groups only feed metrics; peak trainer RAM 64 → 25 GB at 64 groups × 16 × 3 rounds |
| one MuJoCo simulator per maze, per-slot state swapped in | `PointMazeSimVecEnv` | procedural mazes drew a new env per slot (~0.1 s each): `reset` 103 s → 7 s for 1024 episodes, then ~0 |
| PNG `compress_level=1` for vLLM image payloads | `vllm_client._image_url` | lossless, several times faster to encode |
| vLLM `--api-server-count 4 --max-num-seqs 1024` (small models) | sweep `vllm_args` | each rollout turn sends new images; one API server is CPU-bound on preprocessing (39 → 130 replies/s for 1024 fresh SmolVLM requests) |
| skip zero-advantage trajectories | `Trainer.update` | they contribute zero gradient |
| adaptive generation batch | `HFPolicy.generate` | multi-image KV caches grow every round; halve on CUDA OOM and remember |
| LoRA (PEFT) + gradient checkpointing | `HFPolicy` | 2–8B models on one 48 GB GPU |
| vLLM server with automatic prefix caching | `vllm_client.py` | multi-round contexts share long prefixes (PEAR uses LMCache for the same reason, with 95%+ hits) |
| LoRA hot-swap weight sync | `VLLMServerGenerator.sync` | robust sync without NCCL |
| full-weight sync by in-place reload | `VLLMServerGenerator._sync_full` | full fine-tuning with vLLM rollouts, no NCCL |
| NaN/inf guards on loss and grad norm | `Trainer.update` | skip a bad micro-batch or step instead of corrupting weights |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `scripts/train.py` | fragmentation between generate and train phases |

## Measured

Qwen3-VL-4B LoRA on an A6000, PointMaze UMaze, ≤10 rounds:

| | HF backend | vLLM backend |
|---|---|---|
| 32-episode greedy eval | ≈ 9 min | ≈ 1 min |
| training-step rollouts | ≈ 210 s per 64 episodes | ≈ 120–150 s for up to 3 resampling rounds (~190 episodes), falling to ≈ 30 s as episodes shorten |
| update (LoRA r=64, ~20–30k tokens) | — | ≈ 30–150 s |

Once rollouts use vLLM, the update is the next bottleneck.
