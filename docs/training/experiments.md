# Experiment log

All runs are on `puget` (2×RTX A6000 48 GB). wandb project: `tri/vlm-rl-pointmaze`.

## 2026-11-16 · Qwen3-VL-4B, PointMaze UMaze, success reward (wandb group `dev`)

`policy=qwen3vl4b_lora rollout=vllm`, 8 groups × 8, ≤10 rounds, lr 2e-5, LoRA r=64 on the language
model. **This predates the PEAR-faithful defaults:** `prompt_style=simple`, `adv_mode=center`,
`on_invalid=noop`, fixed 40-substep actions (`settle=false`).

| step | eval success (greedy, 32 fixed starts) | eval mean rounds | rollout success (T=1) | rollout mean rounds |
|---|---|---|---|---|
| 0 (zero-shot) | 0.625 | 6.03 | – | – |
| 5 | 0.656 | 5.66 | 0.54 | 6.35 |
| 10 | 0.906 | 4.94 | 0.93 | 4.26 |
| 15 | 0.969 | 5.16 | 0.99 | 4.40 |
| 20 | **1.000** | 5.16 | 0.99 | 4.74 |
| 40 | 1.000 | 4.91 | 1.00 | 4.51 |

- 2–4 min per step. Wall-clock **falls** as the policy improves: rollout time went 149 s → 30 s as
  episodes got shorter.
- Fewer groups survive the DAPO filter over time (all-success groups get dropped), the expected sign
  of GRPO saturating on an easy maze.
- Zero-shot failure mode: pushing into a wall repeatedly although the position doesn't change. RL removed it.

## 2026-11-16 · Toy WAM + preference rewards (HF backend)

`experiment=toy_wam_prefs` with Qwen3-VL-4B: Minari `D4RL/pointmaze/umaze-v2` start states, kinematic
toy WAM, win-rate preference rewards. One full update step completed (8 groups kept, 22.7k train tokens;
210 s rollout, 77 s train). This verified the WAM + preference path with a real model.

## 2026-11-16 · Verification of the PEAR-faithful code path (wandb group `verify`)

Qwen3-VL-2B LoRA, HF backend, PEAR protocol, z-score advantages, `on_invalid=terminate`,
`max_context_tokens=24000`, 3 steps × 2 groups × 4.

- With `reward=success`, every group failed and was filtered (no update). The model echoes the state JSON
  as its action (see [Qwen3-VL-2B](../models/qwen3-vl-2b.md)).
- With `reward='success+0.5*neg_dist'`: finite losses (grad norm 0.6–1.0), 0% invalid actions, rollout
  success 0 → 0.19 at step 3. That's 16 episodes, so it only shows the code path works.

## 2026-11-16 · `pointmaze-small-v1`–`v3`: stopped early (old code path)

All three ran **before** the PEAR-faithful defaults landed (`prompt_style=simple`, `adv_mode=center`,
`on_invalid=noop`, fixed 40-substep actions) and were stopped after a few steps of their first run,
Qwen3-VL-2B with `reward=success` on PointMaze **medium** (≤15 rounds, 8 groups × 8). None reached the
other grid points. They are **not comparable** with v4, which runs the current code. Each was stopped to
change one setting:

| group | change | why it was stopped |
|---|---|---|
| v1 | full multi-turn history, `max_new_tokens=256` | switched to single-frame context (`max_frames=1`) and added resumable training |
| v2 | `max_frames=1` | 2B replies truncated at 256 tokens before the action tag; raised the limit |
| v3 | `max_new_tokens=512` | repo moved to Hydra + PEAR-faithful defaults; restarted as v4 |

Partial results (rollout = temperature 1.0, 8 groups × 8 with DAPO resampling; eval = greedy, 32 fixed starts):

| group | step | eval success | rollout success | rollout invalid frac | mean reply chars | s/step |
|---|---|---|---|---|---|---|
| v1 | 0 / 2 | 0.062 | 0.062 → 0.081 | 0.76 → 0.66 | 791 → 766 | 633 |
| v2 | 0 / 4 | 0.062 | 0.025 → 0.364 | 0.82 → 0.32 | 870 → 707 | 280–380 |
| v3 | 0 / 4 | **0.25** | 0.116 → **0.483** | 0.65 → **0.09** | 1448 → 788 | 460–585 |

- **RL teaches brevity without a format reward.** Under `on_invalid=noop` a truncated reply wastes a round,
  so episodes with more short, valid replies succeed more often and get positive advantage. In v3,
  invalid actions fell from 65% to 9% and replies shrank by 45% in 4 steps.
- **The token limit mattered more than context.** Raising `max_new_tokens` 256 → 512 lifted zero-shot eval
  success from 0.06 to 0.25. Single-frame context (v1 → v2) halved step time but didn't change it.
- Zero-shot Qwen3-VL-2B under greedy decoding falls into repetition loops ("The ball is in a dark cell,
  so it cannot move…" ×12, although the ball is on free floor), which sampling at T=1 escapes more often.
  So early on, eval lags rollouts.
- wandb runs: v2 `5d6f2cac`, v3 `dcaba8b7` (state: crashed/killed). v2 also served as the live resume test:
  the stack was hard-killed mid-step twice and continued from the last checkpointed step.

## In progress · `pointmaze-small-v4` sweep (current code)

`configs/sweeps/pointmaze_small.yaml`: {Gemma 4 E2B, Qwen3-VL-4B, MiniCPM-V 4.6, SmolVLM-256M} (LoRA r=64,
largest model first) × {`success`, `success+0.5*neg_dist`, `pref_oracle_dist`}, plus
`configs/sweeps/pointmaze_small_fft.yaml`: SmolVLM-256M **full fine-tuning** (HF rollouts) × the same
rewards. PointMaze medium, ≤15 rounds, 30 steps, `max_frames=1`, `max_new_tokens=512`, current defaults
(PEAR prompt, z-score advantages, settle-until-at-rest, `on_invalid=terminate`). wandb group
`pointmaze-small-v4`.

One-step smoke runs before launch (6 episodes each, PEAR prompt): Gemma 4 E2B produced valid, short
actions. **MiniCPM-V 4.6 and SmolVLM-256M were 100% invalid** (unfenced or state-echo JSON). With
`on_invalid=terminate` such groups have zero variance and are filtered, so these runs may not learn at
all. That is reported as a finding; prompts and rewards are left unchanged.
