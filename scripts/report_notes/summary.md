**Summary** (2026-11-21; panels below are live)

* **Mazes learn from sparse success reward**, and the skill transfers to held-out layouts.
  * SmolVLM-256M: the main study below.
  * Other models on procedural mazes, eval on training / held-out 11×11 / held-out 15×15 layouts at step 40:
    Qwen3-VL-2B 0.48 / 0.31 / 0.33, MiniCPM-V 0.42 / 0.25 / 0.30, Gemma 4 E2B 0.34 / 0.28 / 0.34.
  * Single fixed mazes: SmolVLM2-500M reaches 0.95 on medium, SmolVLM2-2.2B 0.98 on UMaze.
* **Metaworld reach learns** with Qwen3.5-4B: held-out goal positions go from 0 to 0.94 (step 95).
  * Push, ML10 (10 tasks) and sparse rewards have not worked yet.
  * Live: reach to 200 steps; push and ML10 started from the trained reach adapter; Qwen3-VL-4B / 2B and Gemma 4 on
    reach.
* **Most "collapses" were bugs**, all fixed:
  * Trained a template token the model never generates.
  * Re-tokenized replies instead of the sampled ids.
  * Mirrored Metaworld frames.
  * A preference reward that rewarded giving up.
  * Mismatched Qwen3.5 kernels.
* **What remains is optimization.**
  * Token-mean loss lets long failed episodes outweigh short successes. Per-trajectory weighting (`traj_mean`) fixes
    most of it.
  * Too high an lr sharpens the policy until it degrades: lr 1e-5 rather than 2e-5 for Qwen3.5 LoRA.
