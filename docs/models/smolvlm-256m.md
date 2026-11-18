# SmolVLM-256M-Instruct

**Presets:** `policy=smolvlm256m` (full fine-tuning; used by `experiment=smoke` and the `--run-slow` tests),
`policy=smolvlm256m_lora` (LoRA on `text_model`, servable by vLLM) ·
**HF:** [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

```yaml
model: HuggingFaceTB/SmolVLM-256M-Instruct
processor_kwargs: {do_image_splitting: false}
lora_rank: 0
lr: 1.0e-5
```

Small enough to run on a laptop CPU (~500 MB download). Originally for **testing the pipeline**; with the
`compact` prompt it is also being trained on procedural mazes (below).

| | |
|---|---|
| tokens per 336-px frame | 67 without tiling, 1,127 with tiling (the processor default) |
| chat template | prefix-stable; the multi-turn mask tests run on it |
| behavior | does not follow the action format: replies like `[0, 0]` or unterminated tags |

What it's used for:

- `make smoke`: one full RL step on CPU (`on_invalid=noop`, dense reward, so there's something to update).
- `make test-slow`: mask spans decode to exactly the assistant replies, `logits_to_keep` log-probs equal
  a naive full forward, padding invariance, CLI end-to-end, a real weight update.

## In the `pointmaze-small-v4` sweep

Both presets run on PointMaze medium to test whether a very small model can learn the task:

- **LoRA** (`smolvlm256m_lora`) with vLLM rollouts, in `configs/sweeps/pointmaze_small.yaml`. Serve with
  `--max-model-len 8192 --mm-processor-kwargs '{"do_image_splitting": false}'` (set in `vllm_args`).
- **Full fine-tuning** (`smolvlm256m`) in `configs/sweeps/pointmaze_small_fft.yaml`, with `rollout=hf`
  (in-process generation on the training GPU). A 256M model with AdamW fits easily on one 48 GB GPU.
  `rollout=vllm` also works for full fine-tuning now ([full-weight sync](../training/backends.md#how-weight-sync-works));
  it is the model used to verify that sync on puget.

Under the PEAR prompt, the one-step smoke run (2026-11-16) was 100% invalid: it echoes the state JSON
without a fenced block (`{"ball": {...}, "end": {...}}`). With `on_invalid=terminate` that ends every
episode after round 1 with identical scores, so groups have zero variance and nothing trains.

## Procedural mazes (`smolvlm-procgen-v1`, 2026-11-17)

Zero-shot on `experiment=pointmaze_procgen` (64 eval episodes per set, 20 rounds):

| prompt | valid turns | success |
|---|---|---|
| `pear` | 0% (flat `{"x": .., "y": ..}` echoing the state, no `"ball"` key) | 0 |
| `simple` (with / without "Think briefly") | 0% (prose: "The best path is to move the ball into the green circle.") | 0 |
| `compact`, greedy | 18% | 0 |
| `compact`, T=1 | 15% | 0 |

With `compact` and `on_invalid=noop` (an invalid reply is a no-op, the episode continues), ~2% of sampled
episodes reach the goal, enough for the **sparse** success reward with a large batch: in a 2-step pilot
(64 groups × 16, 3 resampling rounds, lr 1e-5) rollout success went 1.7% → 3.4% and the groups with signal
17 → 31 of 64.

Settings (`configs/sweeps/smolvlm_procgen_gpu{0,1}.yaml`, one per GPU, lr 1e-5 / 3e-5):

- full fine-tuning (`policy=smolvlm256m`), vLLM rollouts with full-weight sync; the trainer and its server
  share one GPU (`train_gpu == vllm_gpu`, `vllm_gpu_util: 0.35`), so each GPU runs its own sweep;
- vLLM `--api-server-count 4 --max-num-seqs 1024 --mm-processor-cache-gb 0`: every rollout turn sends new images, and a single API
  server process is CPU-bound on image preprocessing (39 → 130 replies/s for 1024 fresh-image requests); the
  per-process multimodal cache never hits (images never repeat) and cost ~59 GB RAM across 8 API processes;
- `max_new_tokens=64` (a move is ~15 tokens), `micro_batch_size=64`, 60 steps, eval every 5 steps.

A step (up to 3072 episodes, ~5–9k per-turn samples) takes ~9–11 min, with ~25 GB peak trainer RAM.
