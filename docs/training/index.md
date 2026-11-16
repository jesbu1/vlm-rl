# Training

| page | |
|---|---|
| [Configuration (Hydra)](configuration.md) | config groups, overrides, presets, multirun; the full field list is in the [reference](../reference/config.md) |
| [Rollout backends](backends.md) | in-process HF `generate` vs a vLLM server with prefix caching and LoRA hot-swap |
| [Sweeps](sweeps.md) | reward × model grids with one vLLM server per model, resumable |
| [Weights & Biases](wandb.md) | what is logged, grouping, building comparison reports |
| [Experiment log](experiments.md) | every run so far and what it showed |

```bash
python scripts/train.py experiment=pointmaze_q3vl4b_vllm steps=40 wandb_project=vlm-rl-pointmaze
```

Outputs go to `out_dir/`: `.hydra/` (the resolved config and overrides), `config.json`,
`metrics.jsonl` (one JSON line per step), `episodes/*.mp4|txt` (sample episode videos with full transcripts; `video_format=gif` for GIFs),
`ckpt_step*/` (`save_pretrained` adapter or full weights + processor), `adapters/` (vLLM sync) and `resume/state.pt`.

## Resuming

After every step (`resume_every`, default 1), the trainer atomically writes `out_dir/resume/state.pt`:
trainable weights, optimizer state, the sampler RNG, and that step's metrics. **Re-running the same command
continues after the last checkpointed step**:

- `metrics.jsonl` is truncated to that step, so re-run steps aren't duplicated, and a step that was
  checkpointed but not yet logged is logged.
- The same wandb run is resumed (its id is kept in `out_dir/wandb_run_id`), and rows the server lost in
  a hard kill are re-sent.
- With the vLLM backend, the server gets the restored adapter under a fresh, unique name.

`resume=false` starts over. [Sweeps](sweeps.md) rely on this, so re-running a sweep after a crash or
reboot resumes it.
