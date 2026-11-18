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
| `eval/` | `success`, `turns`, `invalid_frac`, `end_*` (greedy by default, `eval_temperature`). With [`eval_sets`](../environments/pointmaze.md#eval-sets-in-and-out-of-distribution): `eval/<name>/…` per set, and `eval/…` is the mean over sets |
| `time/` | `rollout`, `train` |
| media | `train/episode`, `eval/episode` (**MP4 video**, H.264, 2 fps = one frame per round, captioned with success / score / rounds / end reason; upscaled with nearest-neighbour to ≥ `video_min_size` (672) px and encoded at `video_crf` (18) so thin lines and small dots stay sharp; `video_format=gif` and `video_fps` to change) and `train/transcript`, `eval/transcript` (HTML) for one episode, plus incremental tables **`train/samples`, `eval/samples`** with `samples_per_log` (4) episodes per dump: step, success, score, advantage, turns, invalid turns, end reason, mean reply chars, **full reasoning transcript**, video. Every `log_samples_every` / `eval_every` steps |

Reading them:

- **`rollout/groups_kept` falling** while `rollout/success` → 1 means GRPO is saturating, since all-success
  groups get filtered. Move to a harder task.
- **`rollout/end_invalid` high** means format problems. Try a format reward, or check the model's page.
- **`rollout/end_context` > 0** means episodes are hitting `max_context_tokens`.
- **Reading reasoning chains**: open the `eval/samples` table and filter by `episode`. Eval rows come from
  the same fixed start states every time, so one maze's reasoning can be followed across training. Train rows
  mix successes and failures. With `eval_sets`, each set has its own `eval/<name>/samples` table and
  `eval/<name>/episode` video panel.

## Reports

`scripts/make_wandb_report.py` builds a Report for a sweep group (needs `wandb-workspaces`). It contains
a setup section, a final-eval summary table computed from run histories (crashed and partial runs
included), success curves grouped by reward and by model, episode dynamics, wall-clock, and your
takeaways from a markdown file.

```bash
uv pip install -p .venv wandb-workspaces
python scripts/make_wandb_report.py --group pointmaze-small-v5 --takeaways takeaways.md [--dry_run]
```

`scripts/make_smolvlm_procgen_report.py` is a narrative report for one experiment (the SmolVLM procgen runs,
group `smolvlm-procgen-v1`): headline eval curve, runs grouped by `env_kwargs.prompt_style`, per-eval-set
generalization curves plus a latest-value bar chart, learning dynamics, behaviour, eval/train episode videos, cost,
setup and a run comparer. All panels are live, so curves keep updating while runs train; the takeaways and setup
text are static. `--url <report>` rewrites an existing report in place (e.g. to refresh the takeaways).

```bash
python scripts/make_smolvlm_procgen_report.py --takeaways runs/smolvlm-procgen-v1/takeaways.md [--url REPORT_URL]
```

Offline logging works too (`WANDB_MODE=offline`), and is covered by a test.
