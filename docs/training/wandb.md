# Weights & Biases

```bash
wandb login                     # or export WANDB_API_KEY=...; for a self-hosted server also WANDB_BASE_URL=...
python scripts/train.py ... wandb_project=vlm-rl-pointmaze \
    wandb_group=reward-sweep wandb_run_name=q3vl4b-success wandb_tags='pointmaze,umaze,lora'
```

Logging is off unless `wandb_project` (or `$WANDB_PROJECT`) is set. Runs go to your default entity unless you set
`wandb_entity=<team>` or `$WANDB_ENTITY`; **the environment variables win over the config**, so sweep files that name an
entity (some older ones say `tri`) still log to yours. `WANDB_MODE=offline` keeps everything local. The full resolved config becomes the run config, so runs
can be grouped and filtered by `config.reward`, `config.model`, `config.env_kwargs.maze`, and so on.

## Metrics

| prefix | keys |
|---|---|
| `rollout/` | `success`, `score`, `turns`, `invalid_frac`, `resp_chars`, `groups_kept`, `rounds`, `end_{success,invalid,max_turns,context,terminated,truncated}`, `vllm_logp` (vLLM rollouts: the server's mean log-prob of the sampled reply tokens, on the trained trajectories), `retok_mismatch` (fraction of trained per-turn replies whose re-tokenized text differs from the sampled ids) |
| `train/` | `loss`, `grad_norm`, `tokens`, `sequences`, `mean_logp`, `logp_gap` (`mean_logp` − `rollout/vllm_logp`), `tok_ratio` (trained tokens / sampled tokens), `clipfrac` (ppo_clip only), `skipped` |
| `eval/` | `success`, `turns`, `invalid_frac`, `end_*` (greedy by default, `eval_temperature`). With [`eval_sets`](../environments/pointmaze.md#eval-sets-in-and-out-of-distribution): `eval/<name>/…` per set, and `eval/…` is the mean over sets |
| `time/` | `rollout`, `train` |
| media | `train/episode`, `eval/episode` (**MP4 video**, H.264, 2 fps = one frame per round, captioned with success / score / rounds / end reason; upscaled with nearest-neighbour to ≥ `video_min_size` (672) px and encoded at `video_crf` (18) so thin lines and small dots stay sharp; `video_format=gif` and `video_fps` to change) and `train/transcript`, `eval/transcript` (HTML) for one episode, plus incremental tables **`train/samples`, `eval/samples`** with `samples_per_log` (4) episodes per dump: step, success, score, advantage, turns, invalid turns, end reason, mean reply chars, **full reasoning transcript**, video. Every `log_samples_every` / `eval_every` steps |

Reading them:

- **`rollout/groups_kept` falling** while `rollout/success` → 1 means GRPO is saturating, since all-success
  groups get filtered. Move to a harder task.
- **`rollout/end_invalid` high** means format problems. Try a format reward, or check the model's page.
- **`rollout/end_context` > 0** means episodes are hitting `max_context_tokens`.
- **`train/logp_gap` away from 0** means the trainer is not scoring what the rollout engine sampled. The trainer's
  forward pass runs before the optimizer step, so on-policy the two log-probs agree up to numerics (bf16 vs the
  server's kernels: a few hundredths). A gap that grows over training points at the weight sync. `train/tok_ratio`
  far from 1 points at re-tokenization: the replies are re-tokenized from text, plus one end-of-turn token per
  reply, so short replies sit slightly above 1. Both stay flat while `mean_logp` falls = a real drift of the policy
  (e.g. too high an lr), not a mismatch.
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

`scripts/make_smolvlm_procgen_report.py` is a narrative report for one experiment (the SmolVLM procgen runs, group
`smolvlm-procgen-v1`), written as bullets:
* takeaways (`scripts/report_notes/smolvlm_procgen_takeaways.md`);
* prompts compared: every prompt style as the model reads it (rendered from the code), what differs, what the models
  actually wrote, plus `scripts/report_notes/prompts_results.md`;
* related runs from other agents (`scripts/report_notes/smolvlm_procgen_related.md` + live panels);
* live panels: headline, by prompt, per-eval-set generalization, learning dynamics, behaviour, episodes (video +
  transcript + samples table per split), cost; then setup and a run comparer.

Panels are live; the markdown is static. By default the script rewrites the existing report in place (`--url ''`
creates a new one), so edit the notes files rather than the report UI, or UI edits are lost on the next run.

```bash
python scripts/make_smolvlm_procgen_report.py [--takeaways FILE] [--url REPORT_URL]
```

Offline logging works too (`WANDB_MODE=offline`), and is covered by a test.
