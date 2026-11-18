# Sweeps

`scripts/sweep.py` runs a grid of training runs into one wandb group, for example reward functions ×
models.

```bash
python scripts/sweep.py configs/sweeps/pointmaze_small.yaml --dry_run    # list runs + their overrides
nohup python scripts/sweep.py configs/sweeps/pointmaze_small.yaml > sweep.log 2>&1 &
python scripts/make_wandb_report.py --group pointmaze-small-v5 --takeaways takeaways.md
```

## Sweep file

```yaml
group: pointmaze-small-v5          # wandb group + runs/<group>/<run>/
project: vlm-rl-pointmaze
train_gpu: 0
vllm_gpu: 1
vllm_port: 8011
overrides:                          # Hydra overrides applied to every run
  - experiment=pointmaze_q3vl4b_vllm
  - steps=30
  - reward=success
  - groups_per_step=16                # bigger RL batch: sparse rewards need mixed-outcome groups
  - group_size=16
  - max_new_tokens=512
grid:                               # cartesian product; any config key or group
  policy: [gemma4_e2b_lora, qwen3vl4b_lora, minicpmv46_lora, smolvlm256m_lora]   # largest first
  env_kwargs.maze: [umaze, medium]
  reward: [success, "success+0.5*neg_dist", pref_oracle_dist]
vllm_args:                          # extra `vllm serve` flags per model
  HuggingFaceTB/SmolVLM-256M-Instruct: [--max-model-len, "8192", --mm-processor-kwargs, '{"do_image_splitting": false}']
```

**Full fine-tuning in a sweep.** Works with either backend. With `rollout=vllm`, servers for `lora_rank=0`
runs are started with `ENABLE_LORA=0` and weights are synced by an in-place reload
([how](backends.md#how-weight-sync-works)). `configs/sweeps/pointmaze_small_fft.yaml` uses `rollout=hf` for
`policy=smolvlm256m` (generation in-process on the training GPU), with the same wandb group as the LoRA sweep
so one report covers both.

**Parallel lanes.** On a node with many GPUs, `--lanes N` runs N copies of the loop side by side. Lane *i*
trains on GPU 2*i* with its own vLLM server on GPU 2*i*+1 (port `vllm_port`+*i*); runs are dealt round-robin.
`--out_root DIR` moves the run dirs, `--sync_root DIR` puts the weights handed to vLLM elsewhere, `--set KEY=VALUE`
overrides a value for every run, and `--group NAME` renames the group. [SageMaker](sagemaker.md) jobs use all of these.

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

Logs: `runs/<group>/<run>/train.log`, `runs/<group>/vllm_<model>_<port>.log`.

**One sweep per GPU (small models).** With `train_gpu == vllm_gpu` the trainer and its server share a GPU;
cap the server with `vllm_gpu_util` (e.g. 0.35). Two such sweep files with different `train_gpu` / `vllm_port`
can run side by side in the same group, e.g. `configs/sweeps/smolvlm_procgen_gpu{0,1}.yaml` (SmolVLM-256M full
fine-tuning, one lr each). With `--lanes N --gpus_per_lane 1`, one sweep runs N such shared lanes itself.

## On a shared machine

Several sweeps, agents or people often share one box. Run anything memory-hungry (a full fine-tune, a large
model loaded on CPU, `make test-slow`) under a hard RAM cap, so a runaway process is killed instead of the
machine:

```bash
scripts/memcap.sh 64G python scripts/sweep.py configs/sweeps/pointmaze_small.yaml
```

`scripts/memcap.sh` uses a transient systemd user scope (cgroup v2 `MemoryMax`, no swap). It refuses to start
when less than the cap plus 10% is available (exit 75), and exits 137 if the command exceeded the cap. Start
long jobs detached (`setsid nohup … &` or tmux), so a dropped SSH session or VPN doesn't kill them.

