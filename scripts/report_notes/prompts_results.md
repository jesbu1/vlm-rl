**What happened with each prompt** (eval = greedy success; "invalid" = replies with no parseable action)
* **`pear`**: LoRA sweep v5 (fixed mazes, sparse reward) and the zero-shot protocol ablation.
  * 4B-class models learn it (LoRA): Gemma 4 E2B UMaze 0.03 -> 1.00, medium 0 -> 0.56 (peak); Qwen3-VL-4B UMaze
    0.22 -> 0.84 (peak).
  * Zero-shot it costs ~3x on UMaze and 5-17x on medium vs `simple` (Qwen3-VL-4B: 0.22 vs 0.66, 0.03 vs 0.53).
  * Smaller models fail the format: Qwen3-VL-2B mostly echoes the state JSON instead of writing a change;
    SmolVLM-256M and MiniCPM-V are ~100% invalid.
  * Procedural mazes, Gemma 4 E2B full FT: valid replies but no progress (eval 0.02-0.03 by step 13, few groups with
    signal), while the same model with `compact_rel` is at 0.11 by step 5.
* **`simple`**: full-FT batch sweep v1 (fixed mazes, dense reward).
  * ~2B models learn it after first learning the format (invalid 62-83% -> 0% in 5-10 steps): Qwen3-VL-2B 0.84 / 0.78
    (UMaze / medium), SmolVLM2-2.2B 0.78 / 0.72, MiniCPM-V UMaze 0.62.
  * Small models never get a signal: SmolVLM-256M (both mazes) and MiniCPM-V medium stay ~100% invalid, so every GRPO
    group scores the same.
* **`compact`**: SmolVLM-256M, procedural mazes, sparse reward.
  * Collapses to echoing the goal's absolute coordinates as the move (1-3 distinct replies per episode); eval peaks at
    6.8%, then falls.
* **`compact_rel`**: SmolVLM-256M procedural (this report); SmolVLM-256M + MiniCPM-V fixed mazes (batch v2).
  * Works for the smallest models: the same echo is now the straight-line move.
  * Procedural: 1.6% -> ~21-25% eval, held-out layouts within a few points of the training ones.
  * Fixed mazes: SmolVLM-256M 0 -> 0.66 (UMaze) / 0.56 (medium); MiniCPM-V medium 0.12 -> 0.53.
  * Procedural, larger models (running): Qwen3-VL-2B 0.22 by step 5, Gemma 4 E2B 0.11, MiniCPM-V (lr 1e-6) 0.10.
  * Invalid replies 1-21%. Plateaus where a wall blocks the straight line.

**Rule of thumb so far:** the less a model can do zero-shot, the more of the work the prompt has to do. `pear` needs a
~4B model, `simple` a ~2B one, and sub-1B models need the goal handed to them as the move (`compact_rel`).
