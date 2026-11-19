**All numbers below are from `smolvlm-procgen-v1`, which predates the loss-mask fix (2e2b7d2)**
* Every v1 run trained the newline after `<end_of_utterance>` that the model never samples; its log-prob ran away
  and dragged the policy. So the v1 **collapses, the late drift and the case for `kl_coef` are confounded** by the bug.
* The prompt finding (`compact` vs `compact_rel`) and the ~25% straight-line ceiling are the most likely to survive.
* **Reruns from scratch, group `smolvlm-procgen-v2`** (started 21:15, no KL):
  * GPU0: SmolVLM-256M, `compact_rel`, lr=3e-5: the v1 setting that collapsed fastest. Does it still collapse?
  * GPU1: SmolVLM2-500M, `compact_rel`, lr=1e-5, `max_new_tokens=192` + `parse_last` (final move of a reasoning reply).
  * Step-0 evals match v1 (256M 1.6%, 500M 11.5%). **The panels below now show `smolvlm-procgen-v2`**; v1 numbers
    are in the table further down.
  * **256M at lr=3e-5 collapsed again with the fix**, just later and from higher: step-5 eval 27.6% (0% invalid),
    mean log-prob rising to -0.46 at step 3 (over-confident), then steps 7-10 rollout invalid 2% -> 90%, grad norm
    1 -> 5.8, groups with signal 60 -> 21, gibberish replies (`2.80, 6.90-off way to right! Ya rezured ...`), eval
    **1.0% at step 10**. Stopped at step 10. So the mask bug was not the whole story: lr=3e-5 is too high for 256M
    full fine-tuning. Replaced by **lr=1e-5, no KL** (GPU0, from 00:15).
    * lr=1e-5 step 5: eval **22.4%** (v1 lr=1e-5: 10.4%), rollout 34%, invalid 80% -> 4%, mean log-prob -0.5.
    * Step 10: eval 19.3% (training 21.9, held-out 11x11 7.8, 15x15 28.1), eval invalid 25%; mean log-prob -0.5 -> -1.8,
      grad norm 0.4 -> 2.3, rollout invalid 45%: the same early signs as lr=3e-5, but slower.
    * Step 15: eval **10.9%** (invalid 11%); rollouts 15-20%; log-prob bottomed at -2.5 (step 12) and is back to -1.7,
      grad norm ~2.7. Not collapsed (lr=3e-5 reached 90% invalid by step 8), but it has lost half its step-5 eval.
    * Step 20: **collapsed** too, just later than lr=3e-5: eval 7.3% (training 0, held-out 11x11 1.6, 15x15 20.3), eval
      invalid 74%; rollout invalid 82% -> 91%, grad norm 6.8. Stopped at step 20. So the mask fix alone does not stop
      256M full fine-tuning from collapsing; it only delays it at a lower lr. The max_frames=2 swap was conditional on
      this run holding, so it did not happen as planned.
    * GPU0 now (queue, 40 steps each, a run is aborted if rollout invalid >85% two steps running after step 5):
      256M `compact_rel` **lr=5e-6** (max_frames=1) -> same with **max_frames=2** (the A/B) -> `compact` -> `simple`,
      all at lr=5e-6.
    * lr=5e-6 step 5: eval **16.2%** (training 23.4, held-out 11x11 12.5, 15x15 12.5), eval invalid 12%; rollouts 28%,
      invalid 79% -> 34%, grad norm ~0.5. Slower than lr=1e-5 (22.4% at step 5), as expected; the question is step 20+.
    * lr=5e-6 step 10: eval **24.0%** (training 29.7, held-out 11x11 20.3, 15x15 21.9), 0.1% invalid; rollouts 30-34%,
      invalid 7%, log-prob -0.7, grad norm ~0.65 (lr=1e-5 at step 10: 19.3%, invalid 25%, grad norm 2.3).
    * Step 15: eval **12.5%** (12.5 on every set), eval invalid 13%, while rollouts stay healthy (30%, invalid 9%,
      grad norm 0.9). Greedy eval replies repeat one move for the whole episode: the goal offset echoed back
      (`x=+3.10, y=+3.89 from the ball.`), and in one episode written with a Unicode minus (`x=−3.51`, U+2212), which
      the parser does not accept, so all 19 turns were no-ops. Same greedy-fragility theme, not a collapse (so far).
  * 500M (192 tokens + `parse_last`): step-5 eval **28.6%** (training 31.2, held-out 11x11 21.9, held-out 15x15
    **32.8**), 0% invalid eval replies; v1 500M with 64 tokens was 22.9% at step 5 and then lost it to truncated
    reasoning. Rollout success 30%, invalid 71% -> 33%, log-prob -1.8 -> -1.0.
    Step 10: eval 21.3% (training 21.9, held-out 11x11 12.5, 15x15 29.7), 0% invalid; rollouts 39%, invalid 4%;
    mean log-prob -0.42 (as confident as 256M lr=3e-5 just before it collapsed): watching.
    Step 15: eval **6.8%** with 0% invalid, while rollouts are at their best (36-43%, log-prob -0.5 to -0.7).
    * Cause: `parse_last` now misreads the greedy reply. The reply *starts* with the right move, then adds the
      ball's position and commentary: `{"x": -0.90, "y": -5.71, "goal": {"x": 3.14, "y": 1.60}}` + `The ball is at
      x=-0.90 and y=-5.71, and the goal is at x=3.14 and y=1.60 ...` -> the last x/y is the ball's position.
    * Same theme as every run: the sampled policy is fine, the greedy reply drifts into a form the parser misreads.
    * Step 20: eval back to **22.4%** (training 32.8, held-out 11x11 14.1, 15x15 20.3), 0% invalid; rollouts 37-47%,
      log-prob ~-0.9, grad norm ~0.1 (stable). The step-15 dip was the parser, not the policy.
    * Step 25: eval **29.7%**, the best of any run so far (training 29.7, held-out 11x11 **26.6**, 15x15 32.8), 0% invalid;
      rollouts 37%, log-prob -0.8, grad norm 0.2. The held-out 11x11 gap (9-14% before) has nearly closed.
    * Steps 26-29: early drift, the 256M pattern: rollout invalid 0.4% -> 24%, success 39% -> 25%, grad norm 0.14 ->
      0.37. Step-30 training replies drop the JSON (`2.40, -2.07.`), add arithmetic prose, or turn into gibberish
      (`akoIfillinthisdiskarexpandedheliopolis`). Queued 500M `simple`/`pear` runs moved to lr=5e-6.
    * Step 30: eval still **27.6%** (training 31.2, held-out 11x11 23.4, 15x15 28.1), eval invalid 4%; rollouts 32%,
      invalid 20%, grad norm 0.68 (rising). Greedy eval holds while sampled rollouts degrade; watching.
    * Steps 31-32: grad norm 0.98 -> 2.5. Stopped at step 32 for the terminate-on-invalid A/B below.
* **Why the runs collapse (hypothesis, not yet tested in isolation):** with `on_invalid=noop` an unparseable reply is
  a free no-op, and the trajectory-level advantage is broadcast to every turn, so junk replies inside a successful
  episode get the same positive push as the good moves (z-scored: ~+3.9 when 1 of 16 succeeds). Nothing anchors the
  policy (full FT, no KL, REINFORCE without clipping), and the token-mean loss weights long rambles more. Lower lr only
  delays it (3e-5: steps 7-10; 1e-5: ~20 for 256M, ~26 for 500M).
* **Test: PEAR's rule, an invalid reply ends the episode** (`on_invalid=terminate` in training; eval stays `noop` via
  the new `eval_on_invalid` so numbers compare with the noop runs). Same settings as the noop runs otherwise:
  * GPU1 (now): SmolVLM2-500M, compact_rel, lr=1e-5, 192 tokens + parse_last (noop twin: peak 29.7% @25, drift from 26).
  * GPU0 (after the lr=5e-6 noop baseline): SmolVLM-256M, compact_rel, lr=5e-6. Risk from scratch: ~80% invalid
    replies at step 0, so few episodes survive to the goal; aborted after step 3 if <6 groups carry a signal.
  * 500M terminate, steps 1-5: groups with signal 7 -> 53 of 64, rollout invalid 85% -> 33%, success 0.3% -> 12%,
    turns/episode 1.4 -> 2.7+. Step-5 eval **26.0%** (training 28.1, held-out 11x11 21.9, 15x15 28.1), 0% invalid
    (noop twin at step 5: 28.6%). Starting from scratch under terminate cost almost nothing for 500M.
  * Queued on GPU0: once the 256M **lr=1e-5** run passes step 20 (v1 collapsed by then) with eval >= 10% and mean log-prob
    >= -4, it is replaced by the same run with **`max_frames=2`** (sees the previous round too; default 1 = only
    the current image and text, so it cannot tell a move into a wall left it stuck).
* Stopped v1 runs at the fix: SmolVLM2-500M 64 tokens (eval 3.6 / 22.9 / 5.7 / 20.8% at steps 0-15) and 192 tokens +
  `parse_last` (10.4% zero-shot vs 3.6%; stopped at step 2).
  * **Follow-up queue** (40 steps each, no KL, starts when a GPU frees up; a run is dropped after step 3 if < 6
    groups carried signal):
    * GPU1, SmolVLM2-500M back to the richer prompts: `simple` (think, then `<action>`; 256 tokens), then `pear`
      (full PEAR protocol, fenced json; 384 tokens).
    * GPU0, SmolVLM-256M after `max_frames=2`: `compact` (absolute goal; does `compact_rel`'s edge survive the fix?),
      then `simple` (128 tokens); both at lr=1e-5.

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
