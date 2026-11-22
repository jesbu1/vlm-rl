Metaworld (Farama, Sawyer arm), same trainer and text protocol; wandb project `tri/vlm-rl-metaworld`.

**What the model sees and does** (one turn = one image, no history)
* **Input:** the task instruction, a 224×224 corner-camera frame, and `Gripper: open. Last move: dx=+2.0, dy=…, dz=…`.
  No coordinates are given; the goal is visible only in the image.
* **Reply:** `{"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}`, a hand move in cm (at most 10 per turn).
* **Episode:** 15 turns, ending early on Metaworld's `success`.

**Reward: preferences from Metaworld's ground truth (`reward=pref_mw`)**

Each GRPO group is 8 rollouts from the same start (same goal position and seed).
1. **Score every rollout** by its final state: `s = 10·success + log(0.001 + r_final)`. Here `r_final` is Metaworld's
   dense reward (0–10) at the last step. An episode ended by an invalid reply gets `s = −20` (ranked last).
2. **Compare every pair:** `P(i beats j) = sigmoid((s_i − s_j) / 0.1)`, a tie (0.5) if `|s_i − s_j| < 0.02`.
   On the log scale that means "within 2%", so 0.026 vs 0.033 (hand 26 vs 24 cm from the puck) still counts.
3. **Win rate:** each rollout's reward is its mean `P(i beats j)` over the other 7, in [0, 1].
4. **GRPO:** advantages are z-scored within the group; a group with no spread (all tied) is dropped.

*Example (4 rollouts):*

| rollout | r_final | s | reward | advantage |
|---|---|---|---|---|
| invalid reply | — | −20 | 0 | −1.34 |
| no success | 0.03 | −3.5 | 1/3 | −0.45 |
| no success | 0.07 | −2.6 | 2/3 | +0.45 |
| success | 9.8 | 12.3 | 1 | +1.34 |

* **Why this works:** with temperature 0.1 on log rewards this is essentially a *ranking* by final progress (success
  first), soft only between near-equal outcomes. Every group has a signal even when nothing succeeds. A sparse success
  reward gave none (Metaworld success starts at 0%).
* **Loss:** each trajectory weighs the same (`loss_agg=traj_mean`), its turns trained on the exact sampled tokens.
* **Object-first variant (`reward=pref_mw_obj`), for tasks with an object.** Rollout *a* beats *b* at the first rule
  that separates them: (1) valid reply vs invalid; (2) success; (3) the object ended **≥ 1 cm closer to its goal**
  (Metaworld's `obj_to_target`); (4) `pref_mw`'s dense-reward rule. Rule 3 is needed on push: there the dense reward
  at the final state is highest for *gripping the puck and holding still* (≈ 5.5), while pushing it partway and
  losing contact scores < 2. Under `pref_mw`, RL taught exactly that (below).

**Reach** (eval = 32 greedy episodes on held-out / training goal positions; zero-shot is 0 for every model)

| model | eval, held-out / train | how |
|---|---|---|
| **Qwen3.5-4B** (LoRA on MLPs, lr 1e-5) | 0 (step 15) → **0.94 / 0.91** (95) → 0.78 / 0.88 (200) | RL only. Held-out plateaus at 0.5–0.78 after step 95 while training positions reach 0.97 |
| **Qwen3-VL-4B** (LoRA) | 0.28 / 0.28 after BC → 0.66 / 0.81 (30) → **0.75–0.78 / 0.91–1.00** (70–80) | BC then RL. RL alone: 0 (always dx = dy = 0) |
| Qwen3-VL-2B (full FT, lr 2e-6) | 0 → 0.22 / 0.22 (40) → 0.38 / 0.53 (100) | RL only, T = 1.6 (stuck at T = 1.2) |
| Gemma 4 E2B (full FT, lr 1e-6) | 0.09 / 0.09 after BC → 0.16–0.22 / 0.19–0.25 (15–25) | BC then RL. RL only at T = 1.6: 0.1–0.2 |

* **Exploration decides whether RL alone can start.** The preference reward needs rollouts in a group to differ. At
  T = 1.2 the mean log-prob of sampled tokens at step 1 was −0.6 for Qwen3.5-4B but −0.04 to −0.07 for the other
  three: near-deterministic, so groups tie. Higher temperature helped Qwen3-VL-2B and Gemma; Qwen3-VL-4B never moved
  sideways.
* **Behaviour cloning fixes the start.** Metaworld's scripted experts are run through the same text protocol, their
  moves written as the model's replies; the model is fine-tuned on the successful episodes (`bc_steps=30`, 64 episodes
  per step), then RL runs from the clone. RL then improves on it: Qwen3-VL-4B 0.28 → 0.75 in 70 steps.

**Push and ML10**
* **Push, RL only: 0** in every run, also from the reach adapter (the policy repeats one fixed move).
* **Push, BC:** the clone succeeds 0.22 / 0.03 (training / held-out), the first push successes; the expert succeeds
  0.61. The clone keeps sending the "drop onto the puck" move after its hand is already on the table, a state the
  expert never visits.
* **RL with `pref_mw` undid the clone** (rollouts 0.12 → 0.01 in 15 steps): it learned to grip the puck and hold it,
  the dense reward's local optimum. **With `pref_mw_obj` the clone holds** (rollouts ~0.1 through step 26) but has
  not improved yet.
* **DAgger made the clone worse** (0.22 → 0.00; ML10 0.12 → 0.04). The scripted experts are state machines with
  sharp thresholds on state the model cannot see (more than 2 cm from the puck means "go back up 20 cm"), so in the
  clone's own states nearly identical images get opposite labels.
* **DART helps push:** the expert drives with noise on each move and its clean replies are cloned, so the states stay
  near the expert's path but include small slips. 20 DART steps from the clone: 1 cm noise 0.25 / **0.16** (from
  0.22 / 0.03), 2 cm 0.09 / 0.16. On ML10 it hurt (0.12 → 0.02): its tasks need precision. RL from the DART clones
  is live.
* **RL is starting to improve push** (from the plain clone, `pref_mw_obj`): rollouts 0.06 → 0.16 over steps 28–32.
* **ML10, Qwen3.5-4B:** RL only reaches 0.12 on its training tasks; from the reach adapter 0.17, and 0.02 on the
  held-out tasks. BC then RL evals 0.04–0.12 with either reward; rollouts rising to 0.21 by step 32 (`pref_mw_obj`).
