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
`metrics.jsonl` (one JSON line per step), `episodes/*.gif|txt` (sample episodes with full transcripts),
`ckpt_step*/` (`save_pretrained` adapter or full weights + processor) and `adapters/` (vLLM sync).
