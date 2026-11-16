# Gemma 4 E2B-it

**Preset:** `policy=gemma4_e2b_lora` · **HF:** [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)

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
