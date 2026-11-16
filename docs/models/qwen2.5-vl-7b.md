# Qwen2.5-VL-7B-Instruct (full fine-tuning)

**Preset:** `policy=qwen25vl7b_full` · used by `experiment=pear_recipe` · **HF:** [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

```yaml
model: Qwen/Qwen2.5-VL-7B-Instruct
lora_rank: 0        # full fine-tuning
lr: 1.0e-6          # PEAR's learning rate
```

This preset is the closest stand-in for PEAR's full-FT setup at a scale one node can hold. The processor
and generation config are the same as for [Qwen2.5-VL-3B](qwen2.5-vl-3b.md): 146 tokens per frame, and a
near-greedy shipped config that vlmrl overrides.

## Requirements

| | |
|---|---|
| memory | bf16 weights (~15 GB) + gradients (~15 GB) + AdamW states (~30 GB) ≈ **60 GB before activations**, so an 80 GB GPU |
| rollout backend | `rollout=hf` only. The vLLM backend syncs weights via LoRA adapters, and full-FT sync would need NCCL ([Roadmap](../development/roadmap.md)) |
| trainer | single process, no FSDP |

## Status

Untested on hardware. `experiment=pear_recipe` composes, and its settings are checked by tests.

```bash
python scripts/train.py experiment=pear_recipe groups_per_step=16     # scale the batch to your hardware
```
