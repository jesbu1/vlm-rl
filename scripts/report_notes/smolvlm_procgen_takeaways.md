**Bottom line**
* SmolVLM-256M (full fine-tuning, GRPO, **sparse success reward only**) learns the procedural mazes once the prompt
  gives the goal **relative to the ball** (`compact_rel`): greedy eval goes from 1.6% to **~23%** in 15 steps.
* The skill **transfers to unseen mazes**: held-out 11x11 layouts and larger held-out 15x15 mazes score within a few
  points of the training layouts.
* It has learned **"head straight for the goal"**, not maze navigation: episodes fail where a wall blocks the straight
  line. Getting past ~25-35% needs routing around walls, which hasn't appeared yet.
* **lr=1e-5 works; lr=3e-5 learns faster, then degenerates into gibberish** (eval 24.5% at step 10, 0% at step 20).

**Eval success by run** (greedy, mean of 3 sets x 64 episodes; SE ~5 points at 25%)

| run | step 0 | 5 | 10 | 15 | 20 | status |
|---|---|---|---|---|---|---|
| `compact_rel`, lr=1e-5 | 1.6 | 10.4 | 20.8 | **22.9** | | running (GPU1) |
| `compact_rel`, lr=3e-5, **kl_coef=0.05** | 1.6 | | | | | running (GPU0): tests the KL fix on the setting that collapsed |
| `compact_rel`, lr=1e-5, top_p=0.95 | 1.6 | 5.7 | | | | stopped @5 (behind top_p=1 at step 5) |
| `compact_rel`, lr=3e-5 | 1.6 | 22.9 | **24.5** | 18.2 | 0.0 | degenerated, stopped @20 |
| `compact`, lr=3e-5 | 0.0 | 2.6 | 6.8 | 5.7 | 4.2 | stopped @25 (2.6) |
| `compact`, lr=1e-5 | 0.5 | 1.6 | 4.7 | 5.2 | | stopped @18 |

**Why the prompt mattered most**
* `compact` (absolute goal): both lrs collapsed to **echoing the goal's coordinates as the move**, e.g.
  `x=2.24, y=-4.10. Move right.` every turn while the ball sits against a wall (1-3 distinct greedy replies per
  20-turn episode). T=1 rollouts still hit 11-20% from sampling noise; greedy eval never passed 6.8%.
* `compact_rel` (goal relative to ball): **the same echo is now the straight-line move**
  (`Goal: x=-1.78, y=-2.98 from the ball.` -> `x=-1.78, y=-2.98.`), so it pays off immediately: ~4x the best `compact`
  eval within 5-10 steps.

**Stability (lr and sampling)**
* lr=3e-5 had a one-step wobble at step 5 (88% invalid, grad norm 20.8; recovered), then **degenerated at steps 17-20**:
  * mean log-prob of sampled tokens -2.3 -> -4.9 -> -8.3; replies became `X=+ Haguepite facequeenue...`;
  * 99.9% invalid, 0 of 64 groups with a GRPO signal, so no gradient and no way back.
  * Likely cause: no KL to the reference policy and T=1 sampling over the full vocabulary.
* lr=1e-5: no collapse so far, but mean log-prob slid -3.8 -> -5.3 (steps 13-17), then held near -5.0 through step
  19; invalid replies 18% -> 23-31%; rollout success still 30-36%. **Watch it.**
* `top_p=0.95` (lr=1e-5) was slower (5.7% vs 10.4% eval at step 5), so it was stopped.
* **New: `kl_coef`** (k3 KL to the pretrained model, off by default as in PEAR). GPU0 now runs lr=3e-5 with
  `kl_coef=0.05`: if it keeps lr=3e-5's fast start without the step-17 collapse, it is the setting to use.

**Scale / cost**
* <= 3072 episodes per step (64 groups x 16, up to 3 DAPO rounds), 13-16 min/step, one A6000 per run (trainer +
  vLLM), ~32-37 GB host RAM per trainer; 50-58 of 64 groups carry signal per step.

**Caveats**
* One seed per config; 64 episodes per eval set.
* Stopped runs show as "crashed" in wandb because they were killed on purpose to free GPUs. The `compact` lr=1e-5 run
  also hit a real bug at step 17 (a reply containing a literal `<image>`), since fixed.
