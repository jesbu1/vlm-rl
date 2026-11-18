# MiniCPM-V 4.6

**Presets:** `policy=minicpmv46_lora`, `policy=minicpmv46_full` · **HF:** [`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6)

```yaml
model: openbmb/MiniCPM-V-4.6
lora_rank: 64
lora_alpha: 128
lora_targets: '.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
lr: 2.0e-5
```

**1.3B** parameters in total (vision tower included). The language model is a hybrid: full-attention
layers (`self_attn.{q,k,v,o}_proj`) interleaved with linear-attention layers (`linear_attn.in_proj_*`,
`linear_attn.out_proj`). The preset adapts the full-attention projections and every MLP. The
linear-attention projections are not targeted.

## Gotchas

| | |
|---|---|
| chat template | **not prefix-stable across turns**: earlier assistant turns are re-rendered with the empty `<think></think>` block changed. Use `max_frames=<int>` or `max_images=<int>` (per-turn samples; checked exact for a single trailing reply). `experiment=pointmaze_q3vl4b_vllm` already sets `max_frames=1` |
| generation prompt | ends with an empty `<think>\n\n</think>` (non-thinking mode) |
| tokens per 336-px frame | 70 |
| shipped generation config | `temperature=0.7`, `top_k=0`, `top_p=1.0` (overridden by vlmrl) |
| vLLM | `MiniCPMV4_6ForConditionalGeneration` with LoRA, default server flags |

## Zero-shot behavior under the PEAR prompt

In the one-step smoke run (2026-11-16, 6 episodes) every reply was invalid: it wrote **unfenced** JSON
with invented keys, e.g. `{"ball": {"x": -1.43, "y": 2.73}, "robot": {"direction": "east",
"total_change": 0.5}}`. With `on_invalid=terminate` every episode ends after round 1 with the same score,
so groups have zero variance and nothing trains. Watch `rollout/end_invalid`.

## Status

In the `pointmaze-small-v4` sweep (see [Experiment log](../training/experiments.md)).

## Full fine-tuning

`policy=minicpmv46_full`: every parameter trained, with fp32 master weights and a bf16 autocast forward pass. See
the memory table in [Models](index.md#memory); it fits one 48 GB GPU. vLLM rollouts use the
[full-weight sync](../training/backends.md#how-weight-sync-works). The saved checkpoint has exactly the original
tensor names (checked on CPU). Shipped sweep: `configs/sweeps/batch_small_fft.yaml` (simple prompt, dense
reward, UMaze + medium), run on L40S machines through [AWS Batch](../training/batch.md).

