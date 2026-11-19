_Companion runs by the AWS Batch agent: same trainer (full fine-tuning, fp32 master weights, GRPO + DAPO), other
models. Fixed D4RL mazes: wandb groups `pointmaze-small-fft-batch-v1` (simple prompt), `pointmaze-small-fft-batch-v2`
(follow-ups), `pointmaze-gemma4-fft-batch-v1` (Gemma 4 E2B). Procedural mazes, same setup as this report:
`procgen-fft-larger-v1` and `procgen-fft-larger-v1-lr1e-6` (before the mask fix), `procgen-fft-v2` and
`procgen-fft-larger-v2-maskfix` (after it). Sections B-F are from before the fix (numbers as of 2026-11-18 14:40)._

**A0. After the loss-mask fix: one run per model, SmolVLM2-500M to Gemma 4 E2B** (`compact_rel`, sparse success,
`on_invalid=noop`, 32 groups x 16, 40 steps, no KL; launched 2026-11-18 ~22:30, running; see the "after the mask fix"
panels)

| model | lr | why this lr |
|---|---|---|
| SmolVLM2-500M | 5e-6 | same as the other <=2.2B models |
| MiniCPM-V 4.6 | 1e-6 | at 5e-6, even with the fix, 2 updates sharpen it into a near-deterministic policy (`mean_logp` -0.24 -> -0.04, success 6% -> 1%); then no group has signal and training stops. Invalid replies stayed at 0-2% (before the fix: 99% by step 4) |
| SmolVLM2-2.2B | 5e-6 | the validation run of the fix (`procgen-fft-larger-v2-maskfix`); before the fix this lr collapsed to 97% invalid by step 6 |
| Qwen3-VL-2B | 5e-6 | stable before the fix as well |
| Gemma 4 E2B | 1e-6 | as before (vision tower frozen to fit 48 GB) |


**How the fixed-maze runs differ from the procedural runs**
* **Dense** reward (`success+0.5*neg_dist`; Gemma: sparse success) instead of sparse success.
* **One fixed layout** per run (UMaze or medium); eval = new start/goal pairs on the same layout, so these are not
  generalization tests.
* 16 groups x 16 per step, 40 steps; eval on 32 greedy episodes (SE ~8 points at 50%); L40S 48 GB GPUs.

**A. Larger models on the procedural generalization setup, before the fix** ( settings as above but 32 groups x 16 per step,
fp32-master full fine-tuning, 64 eval episodes per set)

| model, prompt, lr | step | eval success (mean of 3 sets) | latest train / held-out / larger | notes |
|---|---|---|---|---|
| Qwen3-VL-2B, `compact_rel`, 5e-6 | 8 | 0.01 -> **0.22** (step 5) | 0.30 / 0.23 / 0.14 | rollouts 32-40%; on par with SmolVLM-256M here at half the batch |
| Gemma 4 E2B, `compact_rel`, 1e-6 | 8 | 0.08 -> 0.11 (step 5) | 0.14 / 0.09 / 0.11 | rollouts 10% -> 25-30%, 22-25 of 32 groups with signal |
| Gemma 4 E2B, `pear`, 1e-6 | 13 | 0.03 -> 0.02 -> 0.03 | 0.03 / 0.02 / 0.03 | valid replies but only 4-8 groups with signal: the paper prompt does not get going on procedural mazes |
| MiniCPM-V 4.6, `compact_rel`, 1e-6 | 5 | 0.03 -> **0.10** | 0.12 / 0.11 / 0.06 | stable rerun (lr 5e-6 collapsed, see B) |
| SmolVLM2-2.2B, `compact_rel`, 1e-6 | - | queued | | rerun (lr 5e-6 collapsed, see B) |

* Held-out 11x11 layouts score close to the training layouts for every model that learns, as for SmolVLM-256M above.
* Gemma 4 E2B (vision tower frozen to fit 48 GB) learns slower per step than Qwen3-VL-2B at a 5x lower lr.

**B. lr 5e-6 collapsed the 1-2B models on procedural mazes, like SmolVLM-256M at lr 3e-5 above: the mask bug**
* MiniCPM-V 4.6: invalid replies 0% -> 99% between steps 1 and 4 (grad norm 33 at step 1), replies shrink to ~9
  characters, no group has signal afterwards, so it never recovers.
* SmolVLM2-2.2B: invalid 31% -> 96% -> 97% at steps 4-6.
* Qwen3-VL-2B is stable at the same lr. Reruns at lr 1e-6 (MiniCPM-V so far) are stable.
* On fixed mazes the same lr did not collapse anything (dense reward, smaller task), but see D for the drift.
* Cause: the loss-mask bug (top of the report). The collapsing runs show the signature: grad norms of 20-185 and `train/mean_logp` far below vLLM's sampled log-probs. Dense rewards keep advantages closer to balanced, which is why the fixed mazes mostly escaped.

**C. The prompt finding holds on fixed mazes, and for MiniCPM-V 4.6** (final, step 40)

| model | maze | v1: `simple` prompt | v2: `compact_rel` |
|---|---|---|---|
| SmolVLM-256M | UMaze | 0.00 throughout (99-100% invalid) | 0.00 -> **0.66** (step 20), ends 0.62 |
| SmolVLM-256M | medium | 0.00 -> 0.03, crashed at step 16 (literal-`<image>` bug) | 0.00 -> 0.56 (step 10), ends **0.56** |
| MiniCPM-V 4.6 (1.3B) | medium | 0.00 throughout (100% invalid) | 0.12 -> **0.53** (step 25), ends 0.53 |

* With `simple`, small models almost never produce a parseable action; with terminate-on-invalid every group scores
  the same, so GRPO has no signal. `compact_rel` fixes the format immediately (invalid 0-20%).
* On a single fixed layout SmolVLM-256M gets well past the ~25% procedural plateau: straight-line moves plus wall
  sliding solve most UMaze/medium start-goal pairs, while procedural layouts need real routing.

**D. ~2B models do not need the compact prompt, and a lower lr removes the drift**

| model | lr | UMaze, eval 0 -> 40 | medium, eval 0 -> 40 |
|---|---|---|---|
| Qwen3-VL-2B | 5e-6 | 0.00 -> 0.84 (step 5), ends 0.62 | 0.03 -> 0.78 (step 15), ends 0.59 |
| Qwen3-VL-2B | **2e-6** | 0.00 -> 0.53 -> 0.78 ... ends **0.94** (still rising) | 0.03 -> 0.59 (step 5), 0.56 at step 25 (running) |
| SmolVLM2-2.2B | 5e-6 | 0.00 -> 0.78 (rising at the end) | 0.00 -> 0.72 (rising at the end) |
| MiniCPM-V 4.6 | 5e-6 | 0.00 -> 0.62 (step 5), ends 0.56 | 0.00 (never valid enough to train; see C) |

* Under `simple` they learn the format first (invalid 62-83% -> 0% in 5-10 steps), then the task.
* Qwen3-VL-2B at lr 5e-6 peaks within 5-15 steps and drifts down ~20 points; at lr 2e-6 UMaze climbs steadily to
  0.94, the best run of all.

**E. Gemma 4 E2B full fine-tuning on fixed mazes** (48 GB GPUs: vision tower frozen too, ~1.9B trainable; sparse
reward; running)
* UMaze, lr 5e-6: 0.00 -> 0.25 -> 0.06 -> 0.31 -> **0.38** (step 20) -> 0.34 (step 25).
* Medium, lr 5e-6: no success at all for ~3 steps (no signal), then 0.06 -> **0.22** at step 10.
* lr 1e-6: UMaze 0.16 at step 5; medium still 0 at step 16 (rollouts 3-5%).

**F. Infrastructure** (details in the docs: AWS Batch page)
* Small models share one L40S with their vLLM server; ~2B models and Gemma use 2 GPUs per run (trainer + server) on
  4-8 GPU machines. Full weights are reloaded into vLLM after every update.
* Bugs found and fixed on the way: vLLM's encoder cache after a weight reload; MiniCPM-V ignoring `logits_to_keep`;
  SmolVLM2's processor returning lists; an S3 sync race between array jobs; the default AdamW step running Gemma out
  of memory (now fused AdamW); reload headroom on the server; two jobs on one machine reaching each other's servers.
