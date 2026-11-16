# Sweeps

`scripts/sweep.py` runs a grid of training runs into one wandb group, for example reward functions ×
models.

```bash
python scripts/sweep.py configs/sweeps/pointmaze_small.yaml --dry_run    # list runs + their overrides
nohup python scripts/sweep.py configs/sweeps/pointmaze_small.yaml > sweep.log 2>&1 &
python scripts/make_wandb_report.py --group pointmaze-small-v4 --takeaways takeaways.md
```

## Sweep file

```yaml
group: pointmaze-small-v4          # wandb group + runs/<group>/<run>/
project: vlm-rl-pointmaze
train_gpu: 0
vllm_gpu: 1
vllm_port: 8011
overrides:                          # Hydra overrides applied to every run
  - experiment=pointmaze_q3vl4b_vllm
  - steps=30
  - max_new_tokens=512
  - env_kwargs.maze=medium
grid:                               # cartesian product; any config key or group
  policy: [gemma4_e2b_lora, qwen3vl4b_lora, minicpmv46_lora, smolvlm256m_lora]   # largest first
  reward: [success, "success+0.5*neg_dist", pref_oracle_dist]
vllm_args:                          # extra `vllm serve` flags per model
  HuggingFaceTB/SmolVLM-256M-Instruct: [--max-model-len, "8192", --mm-processor-kwargs, '{"do_image_splitting": false}']
```

**Full fine-tuning in a sweep.** vLLM weight sync is LoRA-only, so a full-FT sweep sets `rollout=hf`
in its overrides. No server is started, and generation runs in-process on the training GPU.
`configs/sweeps/pointmaze_small_fft.yaml` does this for `policy=smolvlm256m`, using the same wandb group as
the LoRA sweep so one report covers both.

## What it does

1. **Composes every run up front** with Hydra, so typos and type errors fail before anything launches.
   Run names are `key=value__key=value`. Each run gets `out_dir=runs/<group>/<name>`, `wandb_group`
   and `wandb_run_name`.
2. **Groups runs by model**, in the order models first appear in the grid. Each model's vLLM server
   (`rollout_backend=vllm`) is started once and reused for all of that model's runs.
3. **Skips finished runs** (`metrics.jsonl` already reached `steps`) and **resumes unfinished ones** from
   their last per-step checkpoint ([Resuming](index.md#resuming)), appending to `train.log`. Re-running
   the sweep after a crash or reboot picks up where it stopped.
4. **Stops servers robustly**: the server runs in its own process group, which is SIGTERMed and then
   SIGKILLed. vLLM can ignore SIGTERM during startup, and killing only the parent would orphan the
   GPU-holding engine process.

Per-model **training** settings (LoRA target names, processor kwargs) live in the
[policy presets](../models/index.md), so a grid over `policy` picks them up automatically.

Logs: `runs/<group>/<run>/train.log`, `runs/<group>/vllm_<model>.log`.
