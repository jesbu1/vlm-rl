# Weights & Biases

```bash
python scripts/train.py ... wandb_project=vlm-rl-pointmaze wandb_entity=tri \
    wandb_group=reward-sweep wandb_run_name=q3vl4b-success wandb_tags='pointmaze,umaze,lora'
```

Logging is off unless `wandb_project` is set. The full resolved config becomes the run config, so runs
can be grouped and filtered by `config.reward`, `config.model`, `config.env_kwargs.maze`, and so on.

## Metrics

| prefix | keys |
|---|---|
| `rollout/` | `success`, `score`, `turns`, `invalid_frac`, `resp_chars`, `groups_kept`, `rounds`, `end_{success,invalid,max_turns,context,terminated,truncated}` |
| `train/` | `loss`, `grad_norm`, `tokens`, `sequences`, `mean_logp`, `clipfrac` (ppo_clip only), `skipped` |
| `eval/` | `success`, `turns`, `invalid_frac`, `end_*` (greedy by default, `eval_temperature`) |
| `time/` | `rollout`, `train` |
| media | `train/episode`, `eval/episode` (GIF) and `train/transcript`, `eval/transcript` (HTML), every `log_samples_every` / `eval_every` steps |

Reading them:

- **`rollout/groups_kept` falling** while `rollout/success` → 1 means GRPO is saturating, since all-success
  groups get filtered. Move to a harder task.
- **`rollout/end_invalid` high** means format problems. Try a format reward, or check the model's page.
- **`rollout/end_context` > 0** means episodes are hitting `max_context_tokens`.

## Reports

`scripts/make_wandb_report.py` builds a Report for a sweep group (needs `wandb-workspaces`). It contains
a setup section, a final-eval summary table computed from run histories (crashed and partial runs
included), success curves grouped by reward and by model, episode dynamics, wall-clock, and your
takeaways from a markdown file.

```bash
uv pip install -p .venv wandb-workspaces
python scripts/make_wandb_report.py --group pointmaze-small-v1 --takeaways takeaways.md [--dry_run]
```

Offline logging works too (`WANDB_MODE=offline`), and is covered by a test.
