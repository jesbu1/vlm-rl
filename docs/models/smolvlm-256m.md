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

Small enough to run on a laptop CPU (~500 MB download). It's for **testing the pipeline**, not for
learning the task.

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
