# PEAR ↔ vlmrl

How each element of the paper maps onto vlmrl's config keys and code. `experiment=pear_recipe`
selects the Table 1 settings in one go.

| PEAR | vlmrl | default / notes |
|---|---|---|
| GRPO advantages (Eq. 1) | `algo.group_advantages(mode="zscore")` | `adv_mode=zscore`; also `center` (Dr.GRPO), `rloo` |
| REINFORCE, bypass mode, no clipping | `algo.policy_loss_sum(old_logp=None)` | `policy_loss=reinforce`; `ppo_clip` (DAPO clip-higher) optional |
| score centering (Eq. 2) | `algo.score_centering_term` | `score_centering=true`; active whenever a batch is reused off-policy (`ppo_epochs>1`), exactly 0 and skipped on-policy. See [Algorithm](../concepts/algorithm.md#score-centering) |
| DAPO group filter, groups replaced | `algo.keep_group`, `Trainer.collect` | `filter_groups=true`, `max_resample_rounds` |
| 128 groups × 8 after filter | `groups_per_step`, `group_size` | 8 × 8 by default; 128 × 8 in `pear_recipe` |
| token-mean over the batch | summed micro-batch losses ÷ global token count | exact under any micro-batching |
| no KL / entropy, 1 update per step, AdamW | — | `ppo_epochs=1`; `lr` 1e-6 full FT, ≈2e-5 LoRA |
| temperature 1.0, full distribution | `temperature=1 top_p=1`, `top_k=0` forced | checkpoint generation configs are overridden (see [Models](../models/index.md#generation-config)) |
| whole episode = one sequence | `max_images=null` | `max_images=k` → per-turn samples |
| protocol (App. H) | `vlmrl/envs/protocol.py` | `env_kwargs.prompt_style=pear` |
| EXECUTION: until reached & at rest / blocked | `PointMazeSimVecEnv(settle=True)` | `env_kwargs.settle=true` |
| episode ends on invalid action | `rollout(on_invalid="terminate")` | `on_invalid=terminate` |
| episode ends at the context budget | `rollout(max_context_tokens=…)` | `max_context_tokens` |
| binary success reward | `reward=success` | plus shaped / preference rewards |
| curriculum | change `env_kwargs` between runs, resume from `ckpt_step*` | no automatic scheduler yet |
| veRL async, FSDP × SP, LMCache, TP rollouts | single-process trainer + vLLM server (prefix caching, LoRA hot-swap) | see [Backends](../training/backends.md), [Roadmap](../development/roadmap.md) |
