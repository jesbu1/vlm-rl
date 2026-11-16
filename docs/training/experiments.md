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

## In progress · `pointmaze-small-v1` sweep

{Qwen3-VL-2B, Qwen3-VL-4B, SmolVLM2-2.2B} × {`success`, `success+0.5*neg_dist`, `pref_oracle_dist`} on
PointMaze **medium**, 30 steps each (`configs/sweeps/pointmaze_small.yaml`). The comparison report
will be in wandb group `pointmaze-small-v1`.
