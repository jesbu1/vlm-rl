**Summary** (2026-11-22; panels below are live)

* **Mazes learn from sparse success reward**, and the skill transfers to held-out layouts.
  * SmolVLM-256M: the main study below.
  * Other models on procedural mazes, eval on training / held-out 11×11 / held-out 15×15 layouts at step 40:
    Qwen3-VL-2B 0.48 / 0.31 / 0.33, MiniCPM-V 0.42 / 0.25 / 0.30, Gemma 4 E2B 0.34 / 0.28 / 0.34.
  * Single fixed mazes: SmolVLM2-500M reaches 0.95 on medium, SmolVLM2-2.2B 0.98 on UMaze.
* **Metaworld reach learns, with four models.** Held-out goal positions:
  * Qwen3.5-4B, RL only: 0 → 0.94 at step 95, 0.78 at step 200.
  * Qwen3-VL-4B, behaviour cloning on scripted experts first: 0.28 → 0.75 in 70 RL steps (1.00 on training
    positions). RL alone gave 0.
  * Qwen3-VL-2B (RL only) 0.38 at step 100; Gemma 4 E2B (BC first) about 0.2.
* **Push and ML10 are not solved yet; the main blocker was our controller.** Its gain overshot, so even Metaworld's
  scripted expert succeeded only 0.63 on push through our protocol (0.97 at the new gain 0.5). Push and ML10 are being
  rerun with it: behaviour cloning, then RL. Earlier, at the old gain, RL with the dense-reward preference undid the
  cloned push policy (it learned to hold the puck still); ranking by the puck's progress first fixed that.
* **Most "collapses" were bugs**, all fixed:
  * Trained a template token the model never generates.
  * Re-tokenized replies instead of the sampled ids.
  * Mirrored Metaworld frames.
  * A preference reward that rewarded giving up.
  * Mismatched Qwen3.5 kernels.
* **What remains is optimization and exploration.**
  * Token-mean loss lets long failed episodes outweigh short successes. Per-trajectory weighting (`traj_mean`) fixes
    most of it.
  * Too high an lr sharpens the policy until it degrades: lr 1e-5 rather than 2e-5 for Qwen3.5 LoRA.
  * Near-deterministic models never sample the moves RL needs. Higher temperature or behaviour cloning fixes that.
