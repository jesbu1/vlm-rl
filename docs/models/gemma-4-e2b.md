# Gemma 4 E2B-it

**Presets:** `policy=gemma4_e2b_lora`, `policy=gemma4_e2b_full` (full fine-tuning) · **HF:** [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)

```yaml
model: google/gemma-4-E2B-it
lora_rank: 64
lora_alpha: 128
lora_targets: '.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
lr: 2.0e-5
```

"E2B" means about 2B *effective* parameters. The checkpoint holds **5.1B** in total, mostly per-layer
embeddings, so it is the largest model in the `pointmaze-small-v4` sweep by memory.

| | |
|---|---|
| tokens per 336-px frame | 258 |
| chat template | prefix-stable (whole-episode samples work) |
| LoRA targets | the Qwen regex works: language-model linears are `model.language_model.layers.N.*_proj`. Vision- and audio-tower linears end in `.linear` and are not matched |
| other LM linears | `per_layer_input_gate`, `per_layer_projection` (not targeted) |
| shipped generation config | `temperature=1.0`, `top_k=64`, `top_p=0.95` (overridden by vlmrl) |
| vLLM | `Gemma4ForConditionalGeneration` with LoRA, default server flags |

Checked on puget (2026-11-16): one-step smoke run through `scripts/sweep.py` (vLLM LoRA load, rollouts,
update, adapter hot-swap) under the PEAR prompt, 0% invalid actions and short replies (~44 chars).

## Status

In the `pointmaze-small-v4` sweep (see [Experiment log](../training/experiments.md)).

## Full fine-tuning

`policy=gemma4_e2b_full` (`configs/policy/gemma4_e2b_full.yaml`):

```yaml
model: google/gemma-4-E2B-it
lora_rank: 0
freeze: 'model\.(language_model\.embed_tokens(_per_layer)?|audio_tower|embed_audio)\..*'
fp32_master: true
lr: 1.0e-6
```

| parameters | count | trained? |
|---|---|---|
| per-layer embeddings `embed_tokens_per_layer` | 2.35B | frozen (bf16) |
| token embeddings `embed_tokens` (tied with the LM head) | 0.40B | frozen (bf16) |
| audio tower + `embed_audio` | ~0.35B | frozen (unused by PointMaze) |
| language-model transformer layers + per-layer projections | ~1.9B | **trained** (fp32 masters) |
| vision tower + `embed_vision` | ~0.17B | **trained** (fp32 masters) |

fp32 master weights cost ~16 bytes per trainable parameter with AdamW (weight + grad + 2 moments), so the
~2.1B trainable parameters plus the frozen bf16 tables take ≈ 40 GB before activations: one 80 GB GPU. Training
all 5.1B in fp32 would not fit. Freezing the embedding tables is also the usual choice for RL fine-tuning.

vLLM rollouts use the [full-weight sync](../training/backends.md#how-weight-sync-works). The saved checkpoint
has the original tensor names, except `k_proj`/`v_proj`/`k_norm` of the KV-shared layers (15 and up). Hugging Face
never builds those, and vLLM's Gemma 4 loader drops them too. Checked on CPU: a forward/backward pass on a
PointMaze turn under bf16 autocast, and the checkpoint's names.

**On a 48 GB GPU** (L40S has 44.4 GB usable) the first optimizer step ran out of memory: 43.7 GB allocated when AdamW
created its moments. `policy=gemma4_e2b_full_48gb` also freezes the vision tower (~1.9B trainable, ~36 GB), and the
trainer uses fused AdamW on CUDA (no parameter-sized temporaries in the step).

Shipped sweep: `configs/sweeps/sagemaker_gemma4_fft.yaml` (maze × lr, 4 lanes on one 8-GPU node, see
[SageMaker](../training/sagemaker.md)).

