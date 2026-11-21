_Metaworld (Farama, Sawyer arm) through the same text protocol and trainer, AWS Batch; wandb project
`tri/vlm-rl-metaworld`. Numbers as of 2026-11-20._

**Setup**
* **Observation each turn:** one 224×224 frame, a center crop of a 640×480 render from the `corner` camera,
  plus `Gripper: open|closed. Last move: dx=…, dy=…, dz=… (cm). Move:`.
  * No absolute coordinates are given, and the goal is visible only in the image.
  * Only the current turn is in context (`max_frames=1`): earlier frames and replies are dropped.
* **Reply:** `{"dx", "dy", "dz", "grip"}` (`compact_rel`). The env moves the hand there (at most 10 cm per turn) and
  holds the grip. 15 turns per episode.
* **Reward `pref_mw`:** within each GRPO group, rollouts are compared on Metaworld's own ground truth.
  * Success beats failure; otherwise the higher dense reward at the final state wins (log scale, 2% tie margin).
  * An episode that ended on an invalid reply ranks last.
  * Win rates become the scores, so a group has a signal even when nothing succeeds.
* **Training:** 16 groups × 8 per step, 40-60 steps; per-turn samples trained on the exact sampled token ids.

**ML1 reach** (50 goal positions to train on; eval = 32 greedy episodes on held-out positions and on train positions)

| model | eval success, held-out / train positions | notes |
|---|---|---|
| **Qwen3.5-4B, LoRA on MLPs, lr 1e-5, `traj_mean`** (live) | 0 (steps 0-15) → **0.38 / 0.28** (step 20) → 0.31 / 0.41 → **0.41 / 0.34** (step 30) | holding: mean log-prob flat at −0.29 … −0.34, gradient norms ~0.2, 0% invalid |
| Qwen3.5-4B, LoRA on MLPs, lr 2e-5 | 0 → **0.34 / 0.50** (step 25) → 0.03 / 0.12 (step 40) | one update at step 24 sharpened the policy (mean log-prob −0.39 → −0.07), then it degraded |
| Qwen3.5-4B, lr 2e-5 + KL 0.05 (live) | 0 through step 30 | samples successes (rollouts 0.06-0.21) but greedy eval stays 0 |
| Qwen3-VL-2B, full FT, lr 5e-6, T=1.2 | 0 → **0.31 / 0.25** (step 10), 0.28 / 0.31 to step 40 | froze once the policy became deterministic (every group tied) |
| same, sparse success reward | 0 throughout | success starts at 0, so no group ever has a signal |

* Zero-shot, every model scores ~0 on ML1 push, ML10 and ML45 (Qwen3-VL-2B/4B, Gemma 4 E2B; `pear` and
  `compact` prompts). Before `compact_rel`, models copied the hand's absolute position as the "move".
* **Push has not learned yet** with any model or reward: the hand never gets near the puck, where the dense reward
  starts to vary.

**ML10** (train on its 10 tasks; eval on them and on 5 held-out tasks): flat so far at 0.02-0.03 on the
training tasks and 0 held out (steps 0-30, both stabilizers). With 16 groups over 10 tasks each task gets 1-2 groups
per step; a 32-group run at lr 1e-5 is live. All runs here: Qwen3.5-4B, per-trajectory loss weighting
(`loss_agg=traj_mean`).

**Bugs found on the way** (all fixed):
* Mirrored frames: a 180°-rolled camera needs a rotation, not a row flip.
* One GL context per simulator (out of GPU memory).
* `pref_mw` rewarding invalid endings.
* Re-tokenized replies: 90% of Qwen3.5's sampled replies use a non-canonical token split.
* Slow linear-attention kernels for Qwen3.5 in the trainer.
