Same trainer and setup as the SmolVLM study above, other models, AWS Batch. Runs from before the loss-mask fix are
in wandb groups `procgen-fft-larger-v1*` and `pointmaze-small-fft-batch-v*`; this section shows only runs after it.

**Procedural mazes** (`compact_rel`, sparse success, 32 groups × 16, 40 steps; eval on training / held-out 11×11 /
held-out 15×15 layouts, 64 episodes each)

| model | lr | eval at step 40 | notes |
|---|---|---|---|
| Qwen3-VL-2B (full FT) | 5e-6 | **0.48 / 0.31 / 0.33** | best overall |
| MiniCPM-V 4.6 | 1e-6 | 0.42 / 0.25 / 0.30 | still rising |
| Gemma 4 E2B (full FT, vision frozen) | 1e-6 | 0.34 / 0.28 / 0.34 | stable throughout |
| SmolVLM2-2.2B, `traj_mean` | 5e-6 | ~0.37 at step 25 | without `traj_mean` it collapses at step 5 (below) |

**Single fixed mazes, 100 steps** (same settings): SmolVLM2-500M reaches 0.95 on medium and 1.00 on UMaze;
SmolVLM2-2.2B with `traj_mean` reaches 0.98 on UMaze.

**Why SmolVLM2-2.2B collapsed, and the fix**
* **What happens:** success rises while replies shrink to minimal JSON. Then one update flips the policy to bare words
  (`right`), which aren't valid actions (~95% invalid).
* **Not the cause:** maze sampling (it happens on a fixed maze too), and a lower lr, warmup or a format penalty only
  delay it.
* **Cause:** the token-mean loss. Failures use all 20 turns, so they carry more tokens than short successes, and each
  update leans negative, pushing probability toward unsampled outputs.
* **Fix:** weighting every trajectory equally (`loss_agg=traj_mean`) fixes UMaze and procedural mazes. Adding a KL
  penalty (0.05) also stabilizes medium.
