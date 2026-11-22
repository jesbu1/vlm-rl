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

**Results, ML1 reach** (eval = 32 greedy episodes on held-out / training goal positions)

| run | eval, held-out / train | notes |
|---|---|---|
| **Qwen3.5-4B LoRA (MLPs), lr 1e-5** | 0 (steps 0–15) → 0.47 / 0.78 (60) → **0.94 / 0.91 (95)** → 0.75 / 0.91 (130) → 0.53–0.72 / 0.81–0.97 (135–185) | plateau on held-out positions while training positions keep rising; rollouts ~0.78. Greedy eval is jumpy (32 episodes) |
| Qwen3-VL-2B full FT, lr 2e-6, T=1.6 | 0 → 0.22 / 0.22 (40) → 0.19 / 0.34 (60) | slowly learning; rollouts 0.35 |
| Gemma 4 E2B full FT, lr 1e-6, T=1.6 | 0 → 0.19 / 0.22 (30) → 0.09–0.16 (35–50) | learned some, then slipped |
| Qwen3-VL-4B LoRA, T=2.0, lr 5e-6 | 0 | always answers dx = dy = 0 (never moves sideways), so every rollout ties |
| same, lr 2e-5 | 0 → 0.34 / 0.50 (25) → 0.03 / 0.12 (40) | one update sharpened the policy, then it degraded |
| same, lr 2e-5 + KL 0.05 | 0 throughout | KL too strong: held near the base model |
| Qwen3-VL-2B full FT, lr 5e-6 | 0 → 0.31 / 0.25 (10), flat after | became deterministic: every group tied |

* **Exploration decides whether a model can learn.** The preference reward needs rollouts in a group to differ. At
  T=1.2 the mean log-prob of sampled tokens at step 1 was −0.6 for Qwen3.5-4B but only −0.04 to −0.07 for
  Qwen3-VL-4B, Qwen3-VL-2B and Gemma 4 E2B: near-deterministic, so groups tie. At T=1.6, Gemma's rollouts went from 0
  to 0.25 and Qwen3-VL-2B's from 0.11 to 0.23 (eval 0.16). Qwen3-VL-4B still sharpened to −0.009 with no success, and
  is retried at T=2.0 with lr 5e-6.
* **Warm starts from the reach adapter:**
  * ML10: eval on its training tasks rises to 0.17 (step 70; from scratch at most 0.12), and the 5 held-out tasks
    reach 0.02 (steps 65–80), the first nonzero held-out-task score.
  * Push: 0 through step 50 even at T=1.6. The policy repeats one fixed move every turn
    (`{"dx": 0, "dy": 1.5, "dz": -1.5, "grip": "close"}`) whatever the image shows.
* **What blocked push and the weaker models was discovery, not the reward.** They almost never sampled the needed
  motion (move sideways, go to the puck first).
* **Behaviour cloning, then RL** (`bc_steps=30`, 64 expert episodes per step, successful ones only):
  * **How it works:** Metaworld's scripted experts are driven through the same text protocol and their moves written
    as the model's replies. The model is fine-tuned on them, then RL runs from the clone.
  * **After BC alone** (step-0 eval, held-out / train):

    | run | after BC alone | before BC |
    |---|---|---|
    | Qwen3-VL-4B reach | **0.28 / 0.28** | 0 |
    | Qwen3.5-4B push | **0.09 / 0.19** | 0 everywhere, the first push successes |
    | Gemma 4 E2B reach | 0.09 / 0.09 | 0 |
    | Qwen3.5-4B ML10 | 0.12 on training tasks, 0.02 held-out | — |

  * RL is running from these clones (live).
* **Zero-shot is ~0 for every model**, and push has not learned from scratch either.
