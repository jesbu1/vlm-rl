_Companion runs by the AWS Batch agent: the same trainer (full fine-tuning with fp32 master weights, GRPO + DAPO),
other models and the fixed D4RL mazes instead of procedural ones. wandb groups `pointmaze-small-fft-batch-v1` (simple
prompt) and `pointmaze-small-fft-batch-v2` (follow-ups). Differences from the runs above: a **dense** reward
(`success+0.5*neg_dist`) rather than sparse success; **one fixed layout** per run (UMaze or medium; eval = new
start/goal pairs on the same layout, so these are not generalization tests); 16 groups x 16 per step; 40 steps; eval
on 32 greedy episodes (SE ~8 points at 50%); one L40S 48 GB per run on AWS Batch._

**A. The prompt finding (takeaway 2) holds for SmolVLM-256M on the fixed mazes too, and for MiniCPM-V 4.6.**

| model | maze | v1: simple prompt | v2: `compact_rel` |
|---|---|---|---|
| SmolVLM-256M | UMaze | 0.00 throughout (99-100% invalid; nothing to train on) | 0.00 -> **0.66** (step 20), 0.62 at step 35 |
| SmolVLM-256M | medium | 0.00 -> 0.03, crashed at step 16 (the literal-`<image>` bug) | 0.00 -> **0.56** (step 10), 0.56 at step 25 |
| MiniCPM-V 4.6 (1.3B) | medium | 0.00 throughout (100% invalid) | 0.12 -> 0.19 at step 5 (running) |

With the simple prompt the small models almost never produce a parseable action, and with terminate-on-invalid every
group scores the same, so GRPO has no signal at all. `compact_rel` fixes the format problem immediately (invalid
replies 1-21%), and on a single fixed layout SmolVLM-256M gets well past the ~25% plateau seen on procedural mazes:
straight-line moves plus wall sliding solve most UMaze/medium start-goal pairs, while the procedural layouts need
real routing (takeaway 3).

**B. ~2B models do not need the compact prompt.** Under the simple prompt they first learn the action format
(invalid replies 62-83% -> 0% within 5-10 steps), then the task:

| model | UMaze, eval steps 0 -> 40 (best) | medium, eval steps 0 -> 40 (best) |
|---|---|---|
| Qwen3-VL-2B | 0.00 -> 0.84 at step 5, ends 0.62 | 0.03 -> 0.78 at step 15, ends 0.59 |
| SmolVLM2-2.2B | 0.00 -> 0.78 (rising at the end) | 0.00 -> 0.72 (rising at the end) |
| MiniCPM-V 4.6 | 0.00 -> 0.62 at step 5, ends 0.56 | 0.00 (never valid enough to train; see A) |

**C. Peak-then-drift, like lr=3e-5 in takeaway 4.** Qwen3-VL-2B at lr=5e-6 peaks within 5-15 steps and drifts down
by ~20 points (without the gibberish collapse). The v2 rerun at lr=2e-6 is still early (step 4, invalid 81% -> 51-60%)
and will tell whether the drift is the learning rate or eval noise. SmolVLM2-2.2B at the same lr is still improving at
step 40.

**D. Infrastructure.** Each run had one L40S: SmolVLM-256M and MiniCPM-V share it with a vLLM server (30% of memory;
full weights reloaded into vLLM after every update), the ~2B models need most of the 48 GB for fp32 masters + AdamW and
generate in-process (6-10 min/step). Bugs found and fixed on the way: vLLM's encoder cache after an in-place weight
reload, MiniCPM-V ignoring `logits_to_keep`, SmolVLM2's processor returning lists, and an S3 sync race between array
jobs. Gemma 4 E2B full fine-tuning (maze x lr, 2 GPUs per run on 4x L40S machines) is queued on a dedicated
multi-GPU Batch queue, waiting for g6e.12xlarge capacity.
