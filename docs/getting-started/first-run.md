# Your first real run

This reproduces the verified run: **Qwen3-VL-4B + LoRA on PointMaze UMaze, vLLM rollouts**, on a
machine with two ≥48 GB GPUs (tested on 2×A6000). Expect eval success to climb from ~0.6 to ~1.0
within about 20 steps, at 2–4 min per step.

## 1. Start the rollout server (GPU 1)

```bash
scripts/launch_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8011 1        # model, port, GPU
```

The script enables runtime LoRA loading and prefix caching. Wait until `curl localhost:8011/v1/models`
answers.

## 2. Train (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py experiment=pointmaze_q3vl4b_vllm steps=40 \
    wandb_project=vlm-rl-pointmaze wandb_group=first-run wandb_run_name=q3vl4b-umaze
```

`experiment=pointmaze_q3vl4b_vllm` selects `policy=qwen3vl4b_lora`, `task=pointmaze`, `rollout=vllm`
and 8 groups × 8 rollouts per step. See [Configuration](../training/configuration.md).

## 3. Watch it

- `runs/pointmaze_q3vl4b_vllm/metrics.jsonl`: one JSON line per step (`eval/success` every 5 steps).
- `runs/pointmaze_q3vl4b_vllm/episodes/*.mp4|txt`: sample episode videos with full transcripts.
- wandb: success curves, episode videos, transcripts. See [Weights & Biases](../training/wandb.md).

## No second GPU?

Use the in-process HF backend on one GPU (≈4–9× slower rollouts):

```bash
python scripts/train.py policy=qwen3vl4b_lora rollout=hf steps=40 gen_batch_size=16
```

!!! note "Prompt style"
    The verified run predates the PEAR-protocol prompt that is now the default
    (`env_kwargs.prompt_style=pear`). It used `prompt_style=simple` with mean-centered advantages and
    no-op on invalid actions. To reproduce it exactly, add `env_kwargs.prompt_style=simple
    adv_mode=center on_invalid=noop env_kwargs.settle=false`. See the [experiment log](../training/experiments.md).
