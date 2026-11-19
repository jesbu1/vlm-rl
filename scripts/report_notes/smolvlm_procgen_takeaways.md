**All numbers below are from `smolvlm-procgen-v1`, which predates the loss-mask fix (2e2b7d2)**
* Every v1 run trained the newline after `<end_of_utterance>` that the model never samples; its log-prob ran away
  and dragged the policy. So the v1 **collapses, the late drift and the case for `kl_coef` are confounded** by the bug.
* The prompt finding (`compact` vs `compact_rel`) and the ~25% straight-line ceiling are the most likely to survive.
* **Reruns from scratch, group `smolvlm-procgen-v2`** (started 21:15, no KL):
  * GPU0: SmolVLM-256M, `compact_rel`, lr=3e-5: the v1 setting that collapsed fastest. Does it still collapse?
  * GPU1: SmolVLM2-500M, `compact_rel`, lr=1e-5, `max_new_tokens=192` + `parse_last` (final move of a reasoning reply).
* Stopped v1 runs at the fix: SmolVLM2-500M 64 tokens (eval 3.6 / 22.9 / 5.7 / 20.8% at steps 0-15) and 192 tokens +
  `parse_last` (10.4% zero-shot vs 3.6%; stopped at step 2).

**Bottom line**
* SmolVLM-256M (full fine-tuning, GRPO, **sparse success reward only**) learns the procedural mazes once the prompt
  gives the goal **relative to the ball** (`compact_rel`): greedy eval goes from 1.6% to **~23%** in 15 steps.
* The skill **transfers to unseen mazes**: held-out 11x11 layouts and larger held-out 15x15 mazes score within a few
  points of the training layouts.
* It has learned **"head straight for the goal"**, not maze navigation: episodes fail where a wall blocks the straight
  line. Getting past ~25-35% needs routing around walls, which hasn't appeared yet.
* **Without a KL penalty both lrs degenerate into gibberish**: lr=3e-5 peaks at 24.5% (step 10) and is at 0% by step 20;
  lr=1e-5 peaks at 22.9% (step 15) and is at 0.5% by step 25. The fix being tested is `kl_coef=0.05` (k3 KL to the
  pretrained model), on both lrs.

**Eval success by run** (greedy, mean of 3 sets x 64 episodes; SE ~5 points at 25%)

| run | step 0 | 5 | 10 | 15 | 20 | status |
|---|---|---|---|---|---|---|
| `compact_rel`, lr=3e-5, **kl_coef=0.05** | 1.6 | 13.5 | 24.0 | **25.0** | **25.0** | 24.5 @25, 17.2 @30, **3.6 @35**; stopped @37 |
| `compact_rel`, lr=1e-5, **kl_coef=0.05** | 1.6 | 4.7 | 20.3 | 22.9 | | stable; stopped @19 for SmolVLM2-500M |
| **SmolVLM2-500M**, `compact_rel`, lr=1e-5, kl_coef=0.05 | 3.6 | **22.9** | 5.7 | | | running (GPU1): reasoning cut off at 64 tokens |
| **SmolVLM2-500M**, same + 192 new tokens + `parse_last` | | | | | | running (GPU0), started ~20:00 |
| `compact_rel`, lr=1e-5 | 1.6 | 10.4 | 20.8 | **22.9** | 20.8 | degenerated at step 25 (0.5%), stopped |
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
* lr=1e-5 degenerated the same way, 5 steps later: mean log-prob -3.8 (step 13) -> -5.0 (steps 18-20) -> -6.9 (step 25);
  rollout invalid 15% -> 73%; greedy eval invalid 98% at step 25 (eval 0.5%). Rollout success peaked at 39% (step 20).
  * The mean log-prob slide preceded the eval drop by ~8 steps in both runs: it is the early warning.
* `top_p=0.95` (lr=1e-5) was slower (5.7% vs 10.4% eval at step 5), so it was stopped.
* **New: `kl_coef`** (k3 KL to the pretrained model, off by default as in PEAR); both GPUs now run `kl_coef=0.05`.
  * **KL stops the collapse so far**: lr=3e-5 + KL is at 25.0% eval at step 20 (no-KL: 0%), 0% invalid eval replies;
    mean log-prob -1.8 to -2.2 through step 20 (no-KL: -8.3); KL 0.1-0.4 per token; grad norm <= 2 after step 3.
  * Cost: slower start (step 5: 13.5% vs 22.9% at lr=3e-5; 4.7% vs 10.4% at lr=1e-5); both caught up by step 10.
  * lr=1e-5 + KL matched its no-KL peak (22.9% at step 15) with mean log-prob -2.1 -> -1.75 (no drift).
  * Eval is flat at 25.0% for steps 15-20 (identical per-set scores): greedy policy solves the same straight-line
    episodes; rollout invalid replies 30% -> 44-57% (steps 11-20) without the log-prob drifting.
  * One spike at step 2 (KL 222, grad norm 5157, clipped to 1.0) from a few tokens with a huge k3 ratio; recovered by step 4.

**Next: model size** (GPU1)
* KL fixed stability, not the ~25% ceiling (lr=3e-5 + KL: 24.0 / 25.0 / 25.0 / 24.5% at steps 10-25).
* lr=3e-5 + KL slipped at step 30 (eval 17.2%, eval invalid 0% -> 49%) without a log-prob slide (-2.2): replies
  drifted back to pretrained-style prose (`The goal is to move the ball to the right 3.81 units.`, repeated), which
  doesn't parse. Rollout invalid replies had crept 30% -> 55% since step 11.
* Step 35: eval **3.6%** with 0% invalid: every greedy reply is the system prompt's example move
  `{"x": 1.0, "y": -0.5}`, repeated. T=1 rollouts are still fine (24.5% success, log-prob -1.8).
  * So with KL the sampled policy is stable, but its **greedy mode keeps flipping** (offset echo -> prose ->
    prompt-example echo): greedy eval is fragile at lr=3e-5 even when training looks healthy.
  * A prompt fix to try: drop the concrete example values from the system prompt (the model copies them).
* GPU1 now runs **SmolVLM2-500M** (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, 2x the parameters) with the same
  prompt, batch, eval and lr=1e-5 + `kl_coef=0.05`: does a larger model learn to route around walls?
  * Zero-shot 3.6% (256M: 1.6%), 64% invalid (256M: 95%).
  * **22.9% eval at step 5** (training 29.7, held-out 11x11 18.8, 15x15 20.3); 256M needed 10-15 steps for that.
  * ~20 min/step (256M: 13-16); GPU1 at 33 of 48 GB.
  * Step 10: eval **5.7%** while T=1 rollouts kept improving (27%). The greedy reply became step-by-step reasoning
    (`The ball is at (3.14, 1.60) and the goal is at (-0.9, -5.71). The ball moves by 3 units to the right ...`)
    that hits `max_new_tokens=64` before a move; the parser reads the *first* x/y, so 48% of eval turns are invalid.
* GPU0 now runs SmolVLM2-500M with `max_new_tokens=192` and `env_kwargs.parse_last=true` (new option: read the last
  x/y of the reply, i.e. the final move): lets the model finish reasoning and be scored on its answer.
* Common pattern in all three runs: T=1 rollouts stay healthy while the **greedy** reply flips between modes
  (offset echo, prose, prompt-example echo, truncated reasoning); greedy eval is the fragile part.

**Scale / cost**
* <= 3072 episodes per step (64 groups x 16, up to 3 DAPO rounds), 13-16 min/step, one A6000 per run (trainer +
  vLLM), ~32-37 GB host RAM per trainer; 50-58 of 64 groups carry signal per step.

**Caveats**
* One seed per config; 64 episodes per eval set.
* Stopped runs show as "crashed" in wandb because they were killed on purpose to free GPUs. The `compact` lr=1e-5 run
  also hit a real bug at step 17 (a reply containing a literal `<image>`), since fixed.
