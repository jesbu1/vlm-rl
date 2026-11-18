_Companion runs by the AWS Batch agent: same trainer (full fine-tuning, fp32 master weights, GRPO + DAPO), other
models, fixed D4RL mazes. wandb groups `pointmaze-small-fft-batch-v1` (simple prompt) and
`pointmaze-small-fft-batch-v2` (follow-ups)._

**How these runs differ from the procedural runs**
* **Dense** reward (`success+0.5*neg_dist`) instead of sparse success.
* **One fixed layout** per run (UMaze or medium); eval = new start/goal pairs on the same layout, so these are not
  generalization tests.
* 16 groups x 16 per step, 40 steps; eval on 32 greedy episodes (SE ~8 points at 50%); one L40S 48 GB per run.

**A. The prompt finding holds on fixed mazes, and for MiniCPM-V 4.6**

| model | maze | v1: `simple` prompt | v2: `compact_rel` |
|---|---|---|---|
| SmolVLM-256M | UMaze | 0.00 throughout (99-100% invalid) | 0.00 -> **0.66** (step 20), 0.62 at step 35 |
| SmolVLM-256M | medium | 0.00 -> 0.03, crashed at step 16 (literal-`<image>` bug) | 0.00 -> **0.56** (step 10), 0.56 at step 25 |
| MiniCPM-V 4.6 (1.3B) | medium | 0.00 throughout (100% invalid) | 0.12 -> 0.19 at step 5 (running) |

* With `simple`, small models almost never produce a parseable action; with terminate-on-invalid every group scores
  the same, so GRPO has no signal.
* `compact_rel` fixes the format immediately (invalid 1-21%).
* On a single fixed layout SmolVLM-256M gets well past the ~25% procedural plateau: straight-line moves plus wall
  sliding solve most UMaze/medium start-goal pairs, while procedural layouts need real routing.

**B. ~2B models do not need the compact prompt** (under `simple` they learn the format first, invalid 62-83% -> 0% in
5-10 steps, then the task)

| model | UMaze, eval 0 -> 40 (best) | medium, eval 0 -> 40 (best) |
|---|---|---|
| Qwen3-VL-2B | 0.00 -> 0.84 at step 5, ends 0.62 | 0.03 -> 0.78 at step 15, ends 0.59 |
| SmolVLM2-2.2B | 0.00 -> 0.78 (rising at the end) | 0.00 -> 0.72 (rising at the end) |
| MiniCPM-V 4.6 | 0.00 -> 0.62 at step 5, ends 0.56 | 0.00 (never valid enough to train; see A) |

**C. Peak-then-drift, like lr=3e-5 above**
* Qwen3-VL-2B at lr=5e-6 peaks within 5-15 steps and drifts down ~20 points (without the gibberish collapse).
* The v2 rerun at lr=2e-6 is early (step 4, invalid 81% -> 51-60%); it will tell whether the drift is the lr or noise.
* SmolVLM2-2.2B at the same lr is still improving at step 40.

**D. Infrastructure**
* One L40S per run: SmolVLM-256M and MiniCPM-V share it with a vLLM server (30% of memory; full weights reloaded into
  vLLM after every update); ~2B models need most of the 48 GB for fp32 masters + AdamW and generate in-process
  (6-10 min/step).
* Bugs found and fixed: vLLM's encoder cache after an in-place weight reload, MiniCPM-V ignoring `logits_to_keep`,
  SmolVLM2's processor returning lists, an S3 sync race between array jobs.
* Gemma 4 E2B full fine-tuning (maze x lr, 2 GPUs per run on 4x L40S machines) is queued on a multi-GPU Batch queue,
  waiting for g6e.12xlarge capacity.
