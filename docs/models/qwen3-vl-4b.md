# Qwen3-VL-4B-Instruct

**Preset:** `policy=qwen3vl4b_lora` (the default) · **HF:** [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

```yaml
# configs/policy/qwen3vl4b_lora.yaml
model: Qwen/Qwen3-VL-4B-Instruct
lora_rank: 64
lora_alpha: 128
lora_targets: '.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
lr: 2.0e-5
```

## Facts

| | |
|---|---|
| tokens per 336-px frame | 102 |
| shipped generation config | `temperature=0.7, top_p=0.8, top_k=20` (overridden, see [Models](index.md#generation-config)) |
| chat template | prefix-stable: whole-episode sequences work (`max_images=null`) |
| processor requirement | `torchvision` (a core dependency) |
| vLLM | served with `--enable-lora --max-lora-rank 64`; LoRA hot-swap verified (vLLM 0.33) |
| memory | trainer fits on one 48 GB A6000 with LoRA r=64, `micro_batch_size=2`, gradient checkpointing; vLLM server on a second GPU at `--gpu-memory-utilization 0.85` |

## Results

The reference model for vlmrl. On PointMaze UMaze (8 groups × 8, `reward=success`, vLLM rollouts):

| step | 0 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| eval success (greedy, 32 fixed starts) | 0.625 | 0.656 | 0.906 | **1.000** | 1.000 |

Zero-shot, it already follows the action format (0% invalid). Its failure mode is pushing into a wall
repeatedly although its position doesn't change, and RL removes it. This run used the earlier
`prompt_style=simple`; see the [experiment log](../training/experiments.md).

## Throughput (A6000)

| | HF `generate` | vLLM server |
|---|---|---|
| 32-episode greedy eval, ≤10 rounds | ≈ 9 min | ≈ 1 min |
| one training step's rollouts | ≈ 210 s for 64 episodes | ≈ 30–150 s for up to 3 resampling rounds |

## Run it

```bash
scripts/launch_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8011 1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py experiment=pointmaze_q3vl4b_vllm
```
