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
    * max_frames=2 (noop, lr=5e-6) started 13:18 after the terminate run was stopped. Step-0 eval
      2.1% (eval invalid 77%; max_frames=1 at step 0: 1.6%, 94% invalid): seeing the previous turn already helps format.
      Step 5: eval **14.6%** (training 17.2, held-out 11x11 14.1, 15x15 12.5), eval invalid **2.6%** (max_frames=1 at
      step 5: 16.2%, 11.6% invalid); rollouts 21%, invalid 73% -> 23%, log-prob -0.25 (very confident early).
      Steps 6-8: rollouts 18% -> 26%, invalid 22% -> 10%, 54-58 of 64 groups with signal (max_frames=1 at the same
      steps: ~28-30%). About 28 min/step (2x the image tokens), so 40 steps take ~19 h.
      Step 10: eval **24.0%** (training 29.7, held-out 11x11 20.3, 15x15 21.9), 0% invalid; rollouts 36%, invalid 6.5%,
      grad norm ~0.4 -- identical to max_frames=1 at step 10 (24.0%). The real test is steps 15-20, where max_frames=1
      got stuck at walls (step 15) and then collapsed.
      Steps 11-12: rollouts 29-40%, invalid 5%, grad norm 0.4-0.5, log-prob -0.34 -> -0.72 (less confident).
      Steps 13-14: rollouts 36% -> 30%, grad norm 0.8 -> 1.0 (rising, the usual pre-collapse sign).
      Step 15: eval **13.0%** (training 14.1, held-out 11x11 **3.1**, 15x15 21.9), eval invalid **25%** (max_frames=1
      at step 15: 12.5%, 13% invalid). The second frame did not fix the greedy loops; it gave them a new form. The
      model sees its own previous reply and copies it, with a tail: `x=+1.60, y=+0.54. No change. Move: x=+1.60,
      y=+0.54. No change. ...` repeated to the token limit, or a run of alternating pairs (`x=4.40, y=-4.40,
      x=-3.51, y=+2.16, x=3.87, ...`). Valid-looking junk again, now fed by the extra context.
      Step 16: **collapsing**: rollouts 30% -> 16%, invalid 6% -> 38%, grad norm 1.44. Stopped at step 16.
      Verdict: max_frames=2 matched max_frames=1 through step 10 (24.0% both) at 2x the cost per step, and collapsed at
      the same point (~step 16-18). Seeing the previous turn does not stop the drift; it gives the model its own
      text to copy.
    * lr=5e-6 step 5: eval **16.2%** (training 23.4, held-out 11x11 12.5, 15x15 12.5), eval invalid 12%; rollouts 28%,
      invalid 79% -> 34%, grad norm ~0.5. Slower than lr=1e-5 (22.4% at step 5), as expected; the question is step 20+.
    * lr=5e-6 step 10: eval **24.0%** (training 29.7, held-out 11x11 20.3, 15x15 21.9), 0.1% invalid; rollouts 30-34%,
      invalid 7%, log-prob -0.7, grad norm ~0.65 (lr=1e-5 at step 10: 19.3%, invalid 25%, grad norm 2.3).
    * Step 15: eval **12.5%** (12.5 on every set), eval invalid 13%, while rollouts stay healthy (30%, invalid 9%,
      grad norm 0.9). Greedy eval replies repeat one move for the whole episode: the goal offset echoed back
      (`x=+3.10, y=+3.89 from the ball.`), and in one episode written with a Unicode minus (`x=−3.51`, U+2212), which
      the parser does not accept, so all 19 turns were no-ops. Same greedy-fragility theme, not a collapse (so far).
    * Steps 16-19: now collapsing too, later than lr=1e-5: rollout invalid 9% -> 77%, grad norm 0.9 -> 2.2, success
      30% -> 9%. So under `on_invalid=noop` every lr tried collapses (3e-5: ~step 8, 1e-5: ~20, 5e-6: ~18). Stopped
      after its step-20 eval to start the 256M terminate run.
      Step-20 eval: **9.4%** (training 6.2, held-out 11x11 6.2, 15x15 15.6), eval invalid 47%; rollout invalid 61%.
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
  * 500M terminate step 10: eval **26.6%** (training 29.7, held-out 11x11 21.9, 15x15 28.1), 0% invalid; rollouts
    38%, invalid 2%, all 53-64 groups with signal, grad norm 0.2-0.3 (noop twin at step 10: 21.3%).
    Step 15: eval **28.6%** (training 34.4, held-out 11x11 20.3, 15x15 31.2), 0% invalid; rollouts 36%, invalid 0.6%,
    grad norm 0.12 (noop twin at step 15: 6.8%, the parse_last misread; terminate's greedy reply stays clean).
    Step 20: eval **26.0%** (training 31.2, held-out 11x11 20.3, 15x15 26.6), 0% invalid; rollouts 35%, invalid 0.3%,
    grad norm 0.2, log-prob steady at -0.55 to -0.6 (noop twin at step 20: 22.4%, rollout invalid already rising soon after).
    Step 25: eval **14.1%** (training 17.2, held-out 11x11 7.8, 15x15 17.2), eval invalid 3%; rollouts still 30-39%,
    invalid 0.2%, grad norm 0.28, log-prob rising -0.55 -> -0.40 (more confident). Replies became one JSON object with
    *two* x/y pairs, `{"x": 4.40, "y": -2.24, "x": -3.51, "y": 0.00};`; `parse_last` executes the second pair. Greedy
    repeats one such reply 30-45 times per eval episode (stuck at a wall, identical observation). Valid-format junk again:
    the first pair is never executed and never penalized.
    Steps 26-28: degrading: rollout success 39% -> 18-25%, grad norm 0.25 -> 0.79, groups with signal 63 -> 39.
    Steps 29-30: **collapsing** in training: rollout success 5%, invalid 46%, turns/episode 16 -> 4.6, grad norm
    1.3-2.3, log-prob -0.4 -> -1.06; the greedy eval still read 26.0% at step 30 (eval lags the sampled policy).
    Stopped at step 30. Verdict for 500M: terminate delayed the drift by ~3 steps (noop: ~26, terminate: ~26-29), it
    did not prevent it. Both models find junk the reward never checks (tails after the move, a second x/y pair).
  * 256M terminate (lr=5e-6), steps 1-5: survived the step-3 signal check: groups with signal 10 -> 34 of 64, rollout
    invalid ~85% -> 60%, turns/episode 1.4 -> 2.6. Step-5 eval **13.0%** (training 15.6, held-out 11x11 12.5, 15x15
    10.9), eval invalid 14% (noop twin at step 5: 16.2%). Pre-clip grad norm is high (2-4; clipped to 1.0).
    Step 10: eval **22.4%** (training 29.7, held-out 11x11 15.6, 15x15 21.9), eval invalid 2%; rollouts 27-30%,
    invalid 6%, grad norm back to 0.5 (noop twin at step 10: 24.0%).
    Step 15: eval **16.7%** (training 18.8, held-out 11x11 12.5, 15x15 18.8), eval invalid 3%, while rollouts improve
    (33%, invalid 4%, grad norm 0.5-0.7). Not a collapse: the greedy reply gets stuck. When a move hits a wall the
    ball does not move, the next observation is identical (max_frames=1), so greedy repeats the same move for the
    rest of the episode (19 identical replies, e.g. `x=+3.65, y=+0.54`). Sampled rollouts escape by chance. This is
    the case for a sampled eval and for max_frames=2 (queued next on GPU0).
    Steps 18-20: degrading after all: rollout success 33% -> 14%, invalid 4% -> 14%, grad norm 0.5 -> 1.7, log-prob
    -0.62 -> -0.94; step-20 eval **13.0%** (eval invalid 12%). The junk now sits *after* a valid move, inside the same
    reply: `x=+3.81, y=+0.94. neighboring cells with number equal to x and y ... the final answer is 3.81.` The
    parser reads the leading move, so the reply is valid and its tail shares the episode's advantage; terminate only
    removes junk that replaces the move. (Also seen again: a Unicode minus, `X=−1.32`, which now ends the episode.)
    Step 25: **collapsed**: eval 7.8% (eval invalid 36%), rollouts 8%, invalid 24%, grad norm 2.4, log-prob -1.33.
    Stopped at step 25. Verdict for 256M: terminate-on-invalid delays nothing (noop at the same lr collapsed ~step 18,
    this one ~step 18-20); it only changes where the junk goes. Proposed next: a strict format check (any text beyond
    the move is invalid) so junk tails also end the episode.
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
      then `simple` (128 tokens); both at lr=5e-6.
    * 500M `simple` (lr=5e-6, 256 tokens, noop): step-0 eval **1.0%**, eval invalid 50% (500M compact_rel at step 0:
      11.5%). Steps 1-4: 9-16 of 64 groups with signal (passed the step-3 check), rollout success ~1%, invalid 75% ->
      68%. Step-5 eval **0.5%** with eval invalid 0.3%: the `<action>` format is learned, but the moves don't reach
      the goal yet.
      Steps 6-9: groups with signal 27 -> 41, rollout invalid 51% -> 22%, success 2% -> 6%. Step-10 eval **2.6%**
      (training 3.1, held-out 11x11 1.6, 15x15 3.1), 0% invalid. It has dropped the "think briefly" part: replies are
      just the tag, mostly valid (`<action>{"dx": 1.5, "dy": 1.5}</action>`) with some junk keys
      (`<action>{"velocity": -0.14, "direction": ""]</action>`). Absolute coordinates, no reasoning: the same task
      as `compact` with a harder format, so it should trail compact_rel (500M compact_rel at step 10: 21.3%).
      Steps 11-14: rollouts 6-8%, invalid 12-14%, 42-53 groups with signal, grad norm 0.12-0.15 (very stable).
      Step 15: eval **2.1%** (training 1.6, held-out 11x11 1.6, 15x15 3.1), 0% invalid; rollout success 10%. Learning
      slowly in sampling, not yet in greedy eval.
      Steps 16-18: rollouts 12-15% (up from 7%), invalid 10%, 55-58 groups with signal, grad norm 0.14-0.2.
      Step 20: eval **5.2%** (training 7.8, held-out 11x11 3.1, 15x15 4.7), 0.3% invalid; rollouts 15%, grad norm 0.15.
      Rising steadily (1.0 -> 0.5 -> 2.6 -> 2.1 -> 5.2%) and the most stable run so far (no drift by step 20, unlike
      every compact/compact_rel run), but far below 500M compact_rel at step 20 (22.4% noop, 26.0% terminate).
      Step 25: eval **6.2%** (training 7.8, held-out 11x11 6.2, 15x15 4.7), eval invalid 2%, while rollouts are at
      14-20% (grad norm 0.14-0.19, still no drift). The greedy reply echoes the goal instead of a move:
      `<action>{"goal_position": {x: 0.88, y: -2.23}}</action>`, 20 times per episode. It is not valid JSON, so the
      lenient fallback takes the first two numbers and executes the *absolute goal* as a relative move. The same
      parser-loophole theme as compact (the echoed goal offset) and compact_rel (tails, second pairs).
      Steps 26-30: rollouts 14-23%, invalid 9% -> 13%, grad norm 0.12-0.17; step-30 eval **6.2%** (same goal-echo
      replies). Plateaued at ~6% greedy / ~18% sampled; stable through step 30 (the compact_rel 500M twin drifted at ~26).
      Step 35: eval **6.2%** again (rollouts 13-19%, invalid 10-13%, grad norm 0.11-0.14): flat since step 25.
      Steps 36-39: **drift starts**: rollout invalid 10% -> 44%, log-prob -0.75 -> -1.29, grad norm 0.14 -> 1.9,
      success 22% -> 12%. So the simple prompt delays the drift (~step 37 vs ~26 for 500M compact_rel) but does not
      prevent it. Finished at step 40: final eval **7.3%** (training 9.4, held-out 11x11 4.7, 15x15 7.8), eval invalid
      6.5%; rollouts 6.6%. Verdict: stable for ~35 steps but plateaued at ~6-7% greedy, a third of compact_rel's peak.
    * 500M `pear` (full PEAR protocol, fenced json, 384 tokens, lr=5e-6, noop) started 06:18. Step-0 eval **0%**,
      **100% invalid**: the model never writes the fenced json action. It restates the task (`The task is to move the
      ball into the GREEN circle. The circle is centred at x=-0.82, y=3.82.`), every turn, so the ball never moves.
      Steps 1-3: rollout invalid 99%, 0 of 64 groups with signal, updates skipped; ~20 min/step of pure rollout.
      **Dropped after step 3** by the queue. The full PEAR protocol is out of reach for 500M from scratch without SFT
      or a laxer parser (e.g. accepting a bare `{"ball": ...}` without the fence).
    * 500M compact_rel **terminate at lr=5e-6** (192 tokens + parse_last; started 07:30 on the free GPU1): the lr=1e-5
      terminate run peaked at 28.6% and collapsed at steps 29-30; does halving the lr delay that, as it did for 256M?
      Step 0: eval 7.8% (54% invalid). Steps 1-4: groups with signal 6 -> 14 of 64 (lr=1e-5 twin: 7 -> 53), rollout
      invalid 86% -> 82%, turns/episode 1.3-1.5: most episodes end on the first, invalid, reply. Step-5 eval **14.6%**
      (63% invalid; twin 26.0%, 0% invalid). Half the lr learns the format far more slowly under terminate.
      Steps 6-9: groups with signal 27-28, rollout invalid 76% -> 67% (clear of the queue's 85% abort), turns 1.7 -> 2.0.
      Step-10 eval **19.3%** (58% invalid; twin 26.6%). Improving steadily, about 5 steps behind the lr=1e-5 twin.
      Steps 11-14: 37-48 groups with signal, rollout invalid 61% -> 55%, turns 2.3 -> 2.7, grad norm 0.14-0.21 (low).
      Step-15 eval **25.0%** (training 31.2, held-out 11x11 20.3, 15x15 23.4), eval invalid 45% (twin at 15: 28.6%).
      Steps 16-19: 45-52 groups with signal, rollout invalid 50% -> 45%, success 6% -> 12%, turns 3.1 -> 3.4, grad
      norm 0.12-0.14 (the lowest of any v2 run). Step-20 eval **32.3%**, the **best of any run so far** (training 37.5,
      held-out 11x11 29.7, 15x15 29.7), eval invalid 18% (twin at 20: 26.0%; previous best 29.7%, 500M noop step 25).
      Steps 21-24: 53-58 groups, rollout invalid 41% -> 33%, success 9% -> 15%, turns 3.9 -> 5.0, grad norm 0.09.
      Step-25 eval **31.8%** (training 37.5, held-out 11x11 26.6, 15x15 31.2), eval invalid 11% (twin at 25: 14.1%,
      the double-x/y-pair junk). Holding at the top; no sign of the twin's drift yet.
      Steps 26-30: **past the twin's collapse point** (lr=1e-5 collapsed at 29-30): rollouts 14% -> 22%, invalid 30% ->
      26%, 57-62 groups with signal, turns 5.7 -> 6.5, grad norm flat at 0.08. Step-30 eval **29.2%** (training 32.8,
      held-out 11x11 26.6, 15x15 28.1), eval invalid 4%. Halving the lr under terminate is the first change that has
      held 500M stable through step 30 while still improving.
      Steps 31-35: rollouts 20-28% (rollout success at step 35: 28%), invalid 24% -> 22%, turns 6.9 -> 7.7, grad norm
      0.07-0.09. Step-35 eval **28.6%** (training 35.9, held-out 11x11 26.6, 15x15 23.4), eval invalid 2%.
      Steps 36-39: rollouts 24% -> 29% (highest training success of any v2 run), invalid 20% -> 17%, grad norm
      0.07, mean logp -0.95 (steady). Final step-40 eval **31.2%** (training 39.1, held-out 11x11 29.7, 15x15 25.0),
      eval invalid 0.6%; rollouts 25%. **Finished without collapse**: best 32.3% (step 20), 28.6-32.3% from step 20 on.
      This is the best v2 recipe so far.
    * 500M compact_rel **noop at lr=5e-6** (192 tokens + parse_last; started 11:52 on GPU1): the noop twin of the run
      above. Isolates on_invalid: is the stability from terminate or from the lower lr? (Distinct from the AWS run,
      which uses 64 tokens and no parse_last.)
      Step-0 eval 7.3% (57% invalid; terminate twin 7.8%). Steps 1-3: rollouts 12-13%, invalid 71% -> 66%, 50-57 groups
      with signal (terminate twin: 6-14 groups, 86% invalid): noop starts learning much faster than terminate.
      Step 4: rollouts 18%, invalid 65%. ~21 min/step (episodes run ~19 turns vs 1.4 under terminate early on), so 40
      steps take ~14 h.
      Step 5: rollouts 24%, invalid 55%, 60 groups with signal. Step-5 eval **13.0%** (training 12.5, held-out 11x11
      12.5, 15x15 14.1), 47% invalid (terminate twin at 5: 14.6%, 63% invalid; noop lr=1e-5 at 5: 28.6%, 0% invalid).
      Steps 6-9: rollouts 24% -> **36%** (highest rollout success of any v2 run), invalid 52% -> 20%, grad norm 0.08-0.19,
      mean logp -1.52 -> -0.81 (sharpening quickly).
      Step 10: rollouts **37.5%**, invalid 15%. Step-10 eval **24.5%** (training 29.7, held-out 11x11 18.8, 15x15 25.0),
      eval invalid **0%** (terminate twin at 10: 19.3%; noop lr=1e-5 at 10: 21.3%).
      Steps 11-13: rollouts 38-42%, invalid 11% -> 7%, grad norm 0.08, mean logp -0.74 -> -0.65.
      Step-15 eval **25.5%** (eval invalid 0%), rollouts 37.5%: eval flat vs step 10 (24.5%) while rollouts stay at ~40%.
      Steps 16-17: rollouts 37-39%, invalid 4% -> 3%, grad norm 0.08-0.10.
      Steps 18-20: rollouts 40-45%, invalid 3%, grad norm 0.09-0.14. Step-20 eval **26.0%** (training 32.8, held-out
      11x11 20.3, 15x15 25.0), 0% invalid. Greedy eval has plateaued at 24.5-26.0% since step 10 while sampled rollouts
      climb to ~45% (terminate seed 0 at 20: 32.3%).
      Steps 21-23: rollouts 38-47%, but rollout invalid ticks up 3% -> 4% -> 5% -> 7% (grad norm 0.08-0.12). Small, but
      the lr=1e-5 noop run started drifting at step 26; watching.
      Steps 24-25: **drift begins**: rollout invalid 11% -> 31%, rollouts 39% -> 33%, grad norm 0.27/0.22, mean logp
      -0.51 -> -0.80. Step-25 eval still **26.6%** (2% invalid). Eval replies show the format eroding: bare numbers
      without labels (`-7.79.`, `-0.01, 7.79.`), JSON creeping back (`{ "x": 4.77, "y": 8.33}.`), and one move repeated
      every turn (`x=+0.00, y=-1.83.` x22). Same onset step as the lr=1e-5 noop run (26), so halving the lr did not delay
      the noop drift, while terminate at 5e-6 (3 seeds) shows none.
      Step 26: rollout invalid 30%, grad norm 0.32, mean logp -0.90 (still falling).
      Steps 27-29: **recovers**: rollout invalid 23% -> 17% -> 11%, rollouts back to 36-44%, grad norm 0.17-0.27, mean
      logp -0.81 -> -0.64. A wobble rather than a collapse so far (the lr=1e-5 noop run did not recover).
      Step 30: rollouts 43%, invalid 6%. Step-30 eval **26.6%** (training 32.8, held-out 11x11 21.9, 15x15 25.0), 0%
      invalid: greedy eval flat at 24.5-26.6% since step 10 (terminate 3 seeds at 30: 29.2 / 26.6 / 31.8%).
      Steps 31-35: rollouts 45% -> 36%, invalid 3-4%, grad norm 0.16-0.29, mean logp -0.55 -> -0.30 (sharpening fast).
      Step-35 eval **27.1%** (training 35.9, held-out 11x11 20.3, 15x15 25.0), 0% invalid.
      Steps 36-40: rollouts 41% -> **47.5%**, invalid 4% -> 2%, grad norm 0.20-0.24, mean logp back down to -0.93.
      Final step-40 eval **28.6%** (training 31.2, held-out 11x11 25.0, 15x15 29.7), 0% invalid. Finished without
      collapse (one wobble at steps 24-26 that recovered). Verdict, noop vs terminate at lr=5e-6 (192 tokens + parse_last):
      noop learns the format faster (eval 24.5% at step 10 vs 19.3 / 10.9 / 18.2%) and reaches far higher sampled rollout
      success (~45% vs ~25%), but its greedy eval plateaus at 25-29% from step 10, slightly below terminate's ~30% at step
      40 (31.2 / 30.2 / 30.2%), with grad norm ~3-4x higher and a mid-run wobble. Most of the stability comes from the
      lower lr (noop at 1e-5 drifted from step 26 and did not recover); terminate adds a smoother run and a few points.
      (vLLM EngineDeadError in the log is from shutdown after the run ended.)
    * 500M compact_rel terminate lr=5e-6 with **64 tokens, no parse_last** (GPU1, started 23:24): does the 192-token
      reasoning budget matter? Step-0 eval **4.2%** (192 tokens + parse_last, seed 0: 7.8%). Eval invalid 62%.
      Steps 1-5: groups 7 -> 15, rollout invalid 87% -> 78%, grad norm 1.1-1.9 (higher than 192 tokens' 0.1-0.4 at this
      stage). Step-5 eval **15.6%** (training 17.2, held-out 11x11 20.3, 15x15 9.4), 60% invalid (192-token seeds at 5:
      14.6 / 5.7 / 6.8%). ~75 s/step of rollout vs ~100-125 s for 192 tokens.
      Steps 6-10: format clicks in fast: rollout invalid 74% -> 29%, groups 25 -> 56, rollouts 1% -> 15%, grad norm
      1.2 -> 0.33. Step-10 eval **26.6%** (training 31.2, held-out 11x11 21.9, 15x15 26.6), eval invalid **0%** (192-token
      seeds at 10: 19.3 / 10.9 / 18.2%). Fastest terminate start so far.
      Steps 11-14: rollouts 21% -> 37%, invalid 20% -> 5%, grad norm 0.29 -> 0.16.
      Step-15 eval **25.0%** (training 29.7, held-out 11x11 20.3, 15x15 25.0), 0% invalid; rollouts 34%, invalid 3%.
      Note the episodes are now long (step time 800 s vs 75 s at step 5), like noop: few invalid replies end episodes.
      Steps 16-20: rollouts 34-44%, invalid 2% -> 1%, grad norm 0.13-0.16, mean logp -0.43 -> -0.33 (sharpening).
      Step-20 eval **25.0%** again (training 29.7, held-out 11x11 20.3, 15x15 25.0): greedy eval flat at 25% since step 10.
      Steps 21-25: rollouts 31-42%, invalid <1%, grad norm 0.11-0.18, logp -0.37 -> -0.50. Step-25 eval **25.0%**, same
      per-split numbers as steps 15 and 20 (29.7 / 20.3 / 25.0), though transcripts differ, so weights are updating.
      Transcripts show why it plateaus: with 64 tokens the model writes **no reasoning**, just a JSON move
      (`{"x": -0.00, "y": -1.83}`); often it repeats the pair until the cap (64 of 142 eval replies at step 25, 0 at
      step 20, 72 at step 10) and the parser takes the first one. It learns quickly but has no way to plan.
      Steps 26-30: rollouts 35-39%, invalid <1%, grad norm 0.13-0.20, logp -0.53 -> -0.54. Step-30 eval **24.5%**
      (training 28.1, held-out 11x11 20.3, 15x15 25.0), 0% invalid (192-token seeds at 30: 29.2 / 26.6 / 31.8%).
      Steps 31-35: rollouts 35-46%, invalid <1%, grad norm 0.09-0.23 (one spike at 31, then back). Step-35 eval
      **26.0%** (training 32.8, held-out 11x11 20.3, 15x15 25.0), 0% invalid.
      Finished at step 40: final eval **26.6%** (training 32.8, held-out 11x11 20.3, 15x15 26.6), 0% invalid; rollouts
      37-46%, invalid <1%, grad norm ~0.1, no drift. **Verdict:** 64 tokens learns the format fastest (26.6% by step 10)
      and is stable, but plateaus at 24.5-26.6% vs ~30% for 192 tokens + parse_last (3 seeds): the model drops reasoning
      and answers reflexively, which costs ~4 points. The 192-token reasoning budget is worth keeping.
    * **Queued next** (each starts when its GPU's current run ends): GPU1: 500M **`simple` + terminate** (lr=5e-6,
      256 tokens): does terminate rescue the reasoning-first prompt (noop simple plateaued at 6-7%)? GPU0: 500M
      **`compact` (absolute goal) + terminate** (best recipe otherwise): is the relative goal what makes compact_rel work?
    * 500M `simple` + terminate (GPU1, started 06:52): step-0 eval **1.0%**, 51% invalid (same as the noop simple run).
      Step 1: rollout invalid 95%, success 0%, no update. With terminate, the first bad reply ends the episode, so
      the queue's step-3 signal check (>= 6 groups) decides whether it continues.
      Steps 1-5: rollout invalid 95-96%, success 0%, **0 groups with signal, no updates**; step-5 eval 1.0% (51%
      invalid). **Aborted by the queue at 06:59.** With terminate, the reasoning-first prompt never produces a valid
      `<action>` often enough to learn from: noop's extra turns per episode were what let simple get off the ground.
      Terminate needs a starting format hit rate like compact_rel's (~46% valid at step 0), or an SFT warm start.
    * Replacement on GPU1: best recipe with a **2x batch** (128 groups x 16 = 2048 episodes/step): does a less noisy
      gradient lift the ~30% plateau?
      Started 07:01. Step-0 eval **10.9%** (seed 0, same eval set as the 64-group seed-0 run, which read 7.8% at step
      0: greedy vLLM evals of the same weights vary by a few points with batching, another reason to read single
      evals with a ~3-point error bar).
      Steps 1-5: rollout invalid 85% -> 74%, grad norm 1.07 -> 0.34, ~214 s/step (64 groups: ~100-125 s). Step-5 eval
      **11.5%** (training 17.2, held-out 11x11 6.2, 15x15 10.9), 52% invalid (64-group seeds at 5: 14.6 / 5.7 / 6.8%).
      Steps 6-10: rollout invalid 72% -> 59%, groups with signal ~81 of 128, grad norm 0.27 -> 0.12, ~420 s/step.
      Step-10 eval **21.4%** (training 25.0, held-out 11x11 21.9, 15x15 17.2), 47% invalid (64-group seeds at 10:
      19.3 / 10.9 / 18.2%).
      Steps 11-15: rollout invalid 57% -> 49%, rollouts 4% -> 7%, 90 of 128 groups with signal, grad norm ~0.10, ~505
      s/step. Step-15 eval **26.6%** (training 26.6, held-out 11x11 23.4, 15x15 29.7), 26% invalid (64 x 16 seeds at
      15: 25.0 / 14.1 / 26.0%). Ahead of or level with the best 64-group seed per step so far, at ~2x the wall-clock.
      Steps 16-20: rollout invalid 47% -> 34%, rollouts 8% -> 15%, 115 of 128 groups with signal, grad norm 0.07-0.11.
      Step-20 eval **31.8%** (training 39.1, held-out 11x11 26.6, 15x15 29.7), 6% invalid (64 x 16 seeds at 20: 32.3 /
      25.0 / 27.1%). Still climbing, with a third of rollouts still invalid: room to pass the ~30% plateau.
      Steps 21-25: rollout invalid 31% -> 22%, rollouts 16% -> 25%, 120 of 128 groups with signal, grad norm ~0.06.
      Step-25 eval **32.8%** (training 37.5, held-out 11x11 23.4, 15x15 37.5), **0.8% invalid**: the best v2 eval so far
      (previous peak 32.3%, 64 x 16 seed 0 at step 20). Steps now ~1165 s as episodes lengthen. RAM steady
      (trainer RSS ~30 GB, ~108 GB available).
      Steps 26-30: rollout invalid 21% -> 16%, rollouts 25-26%, grad norm ~0.057, ~1380 s/step. Step-30 eval **30.2%**
      (training 32.8, held-out 11x11 28.1, 15x15 29.7), 0.3% invalid (64 x 16 seeds at 30: 31.8 / 26.6 / 31.8%). Back in
      the ~30% band; one eval is +-3 points, so steps 35-40 decide whether the batch lifts the plateau.
      Steps 31-35: rollout invalid 14% -> 9%, rollouts 27-33%, grad norm ~0.05, ~1700 s/step. Step-35 eval **31.8%**
      (training 34.4, held-out 11x11 25.0, 15x15 35.9), 0.3% invalid (64 x 16 seeds at 35: 29.7 / 30.2 / 29.7%).
      Finished at step 40: final eval **32.3%** (training 37.5, held-out 11x11 25.0, 15x15 34.4), 0.1% invalid; rollouts
      33%, invalid 5%, grad norm 0.03. Mean eval over steps 30-40: **31.4%** (group_size 32: 32.5%; 64 x 16 seeds:
      29.7 / 29.0 / 30.6%). Both 2048-episode layouts end ~2 points above the 1024-episode seeds, and the held-out
      15x15 split gains most (31-36% vs 19-25%), at ~2x the wall-clock (~10 h for 40 steps vs ~5 h).
    * Next on GPU1: **seed 1 of group_size 32**, so each 2048-episode layout has two seeds.
      Started 17:34. Step-0 eval 7.3% (59% invalid). Steps 1-5: rollout invalid 81% -> 73%, grad norm ~0.3-0.5,
      28 groups with signal at step 5. Step-5 eval **11.5%**, 58% invalid (64 x 16 seed 1 at 5: 5.7%; 128 x 16 seed 1: 9.9%).
      Steps 6-10: rollout invalid 70% -> 60%, 47 groups with signal, grad norm 0.20 -> 0.13. Step-10 eval **16.7%**
      (training 20.3, held-out 11x11 14.1, 15x15 15.6), 42% invalid (64 x 16 seed 1: 10.9%; 128 x 16 seed 1: 17.2%).
      Steps 11-15: rollout invalid 58% -> 53%, rollouts 4-7%, grad norm ~0.10. Step-15 eval **25.5%** (training 32.8,
      held-out 11x11 20.3, 15x15 23.4), 33% invalid (64 x 16 seed 1: 14.1%; 128 x 16 seed 1: 22.9%).
      Steps 16-20: rollout invalid 51% -> 42%, rollouts 5-8%, grad norm 0.08-0.10. Step-20 eval **30.2%** (training 32.8,
      held-out 11x11 26.6, 15x15 31.2), 23% invalid (64 x 16 seed 1: 25.0%; 128 x 16 seed 1: 27.6%).
      Steps 21-25: rollout invalid 40% -> 30%, rollouts 11-13%, grad norm 0.05-0.07. Step-25 eval **29.7%** (training
      31.2, held-out 11x11 26.6, 15x15 31.2), 18% invalid (64 x 16 seed 1: 25.0%; 128 x 16 seed 1: 34.4%). Flat vs step 20
      (within the ~3-point eval noise); still the most invalid of the three seed-1 runs.
      Steps 26-30: rollout invalid 29% -> 22%, rollouts 16-24%, grad norm ~0.05. Step-30 eval **32.3%** (training
      31.2, held-out 11x11 31.2, 15x15 34.4), 4% invalid (64 x 16 seed 1 at 30: 26.6%).
      Steps 31-35: rollouts 18-32%, invalid 21% -> 17%, grad norm 0.04-0.05. Step-35 eval **33.9%** (training 28.1,
      held-out 11x11 31.2, 15x15 **42.2%**), 1% invalid (64 x 16 seed 1 at 35: 30.2%).
      Finished at step 40: final eval **27.1%** (training 26.6, held-out 11x11 23.4, 15x15 31.2), 0.1% invalid; rollouts
      30%, invalid 11%, grad norm 0.04. The eval fell 7 points from step 35 while rollouts held at ~30%, the same
      greedy-eval / rollout split seen in 128 x 16 seed 1. Mean eval over steps 30-40: **31.1%** vs 29.0% for 64 x 16
      seed 1 (seed 0: 32.5% vs 29.7%). The group_size-32 edge holds on both seeds at ~2-3 points, but it is about the
      size of the eval noise and the final evals are noisy.
    * Next on GPU1: **seed 2 of group_size 32**, paired with the 64 x 16 seed-2 run (30.6% over steps 30-40), as a
      third seed for the group_size-32 edge.
      Started 02:26. Step-0 eval **11.5%** (64 x 16 seed 2: 11.5%; same eval set and weights).
      Steps 1-5: rollout invalid 85% -> 75%, 25 groups with signal at step 5. Step-5 eval **9.9%**, 66% invalid
      (64 x 16 seed 2 at 5: 6.8%, 78% invalid).
      Steps 6-10: rollout invalid 72% -> 61%, rollouts 2-4%, grad norm 0.23 -> 0.11, 48 groups with signal. Step-10 eval
      **12.0%**, but 79% invalid (64 x 16 seed 2 at 10: 18.2%, 54% invalid). Behind on eval format so far, though
      training invalid is falling at the usual rate.
      Steps 11-15: rollout invalid 60% -> 52%, rollouts 4-6%, grad norm ~0.10. Step-15 eval **21.4%** (training 28.1,
      held-out 11x11 26.6, 15x15 9.4), 37% invalid (64 x 16 seed 2 at 15: 26.0%, 38% invalid).
      Steps 16-20: rollout invalid 50% -> 41%, rollouts 7-11%, grad norm ~0.08. Step-20 eval **24.5%** (training 28.1,
      held-out 11x11 29.7, 15x15 15.6), 5% invalid (64 x 16 seed 2 at 20: 27.1%, 17% invalid).
      Steps 21-25: rollout invalid 39% -> 32%, rollouts 10-15%, grad norm ~0.06. Step-25 eval **27.1%** (training 31.2,
      held-out 11x11 31.2, 15x15 18.8), 2% invalid (64 x 16 seed 2 at 25: 29.2%).
      Steps 26-30: rollout invalid 29% -> 25%, rollouts 17-19%, grad norm ~0.05. Step-30 eval **30.7%** (training 31.2,
      held-out 11x11 **43.8%**, 15x15 17.2), 1% invalid (64 x 16 seed 2 at 30: 31.8%).
      Steps 31-35: rollout invalid 23% -> 18%, rollouts 21-25%, grad norm ~0.04. Step-35 eval **28.1%** (training 31.2,
      held-out 11x11 32.8, 15x15 20.3), 1% invalid (64 x 16 seed 2 at 35: 29.7%).
      Finished at step 40: final eval **26.6%** (training 28.1, held-out 11x11 29.7, 15x15 21.9), 0.5% invalid; rollouts
      24-29%, invalid 14%, grad norm 0.04. Mean eval over steps 30-40: **28.5%** vs 30.6% for 64 x 16 seed 2.
      Greedy eval fell 30.7 -> 26.6% over steps 30-40 while rollouts rose 17% -> 24-29%: the third 2048-episode run
      with this pattern.
    * **Updated verdict, group_size 32 vs 64 x 16 (mean greedy eval over steps 30-40):** +2.8 (seed 0), +2.1 (seed 1),
      **-2.1 (seed 2)**; average +0.9 points (30.7% vs 29.8%). **No reliable gain** from doubling episodes per step,
      at ~2x wall-clock. 64 x 16 stays the default.
    * Next on GPU1: **group_size 32 seed 2 + `kl_coef=0.05`** (k3 KL to the pretrained model), paired with the run
      above: does a KL anchor stop the late greedy-eval drop?
      Started 11:03. Step-0 eval **12.5%**, 56% invalid (no-KL seed-2 run: 11.5%, 54%).
      Steps 1-5: rollout invalid 85% -> 73% (no KL: 85% -> 75%), 30 groups with signal, KL 0.001 -> 0.019 per token.
      Step-5 eval **6.8%**, 71% invalid (no KL: 9.9%, 66%). Same training pace so far; the eval gap is within noise.
      Steps 6-10: rollout invalid 72% -> 61% (no KL: same), rollouts 1-4%, KL 0.023 -> 0.028. Step-10 eval **15.1%**
      (training 18.8, held-out 11x11 15.6, 15x15 10.9), 48% invalid (no KL: 12.0%, 79%).
      Steps 11-15: rollout invalid 61% -> 52% (no KL: 60% -> 52%), rollouts 3-7%, KL 0.029 -> 0.047, grad norm ~0.2
      (no KL: ~0.1; includes the KL term). Step-15 eval **20.3%** (training 25.0, held-out 11x11 20.3, 15x15 15.6),
      12% invalid (no KL: 21.4%, 37%).
      Steps 16-20: rollout invalid 45% -> 36% (no KL: 50% -> 41%), rollouts 11-17% (no KL: 7-11%), KL ~0.07. Step-20
      eval **18.8%** (training 23.4, held-out 11x11 21.9, 15x15 10.9), 7% invalid (no KL: 24.5%, 5%). Training slightly
      ahead, greedy eval slightly behind; both within noise.
      Steps 21-25: rollout invalid 35-37%, rollouts 12-18% (no KL: 10-15%), KL 0.076 -> 0.060. Step-25 eval **24.5%**
      (training 26.6, held-out 11x11 26.6, 15x15 20.3), 2% invalid (no KL: 27.1%, 2%).
      Steps 26-30: rollout invalid 35% -> 31% (no KL: 29% -> 25%), rollouts 14-17% (no KL: 17-19%), KL 0.055 -> 0.050.
      Step-30 eval **26.6%** (training 29.7, held-out 11x11 31.3, 15x15 18.8), 0% invalid (no KL: 30.7%, peak of that run).
      Still ~3 points behind, within noise. The test is steps 35-40, where the no-KL run fell to 28.1 / 26.6.
      Steps 31-35: rollouts 15-21% (no KL: ~17-19%), invalid ~30%, KL ~0.045. Step-35 eval **23.4%** (training 25.0,
      held-out 11x11 29.7, 15x15 15.6), 0% invalid (no KL: 28.1%). KL has not improved on no-KL at any step so far.
      Step-40 eval **22.9%** (training 26.6, held-out 11x11 28.1, 15x15 14.1), 0% invalid (no KL: 26.6%); rollouts 14-20%,
      invalid stuck at ~30% (no KL: ~25%). **Verdict: KL does not help.** Mean over steps 30-40 **24.3%** vs **28.5%**
      without KL (training 27.1 vs 30.2, held-out 11x11 29.7 vs 35.4, 15x15 16.1 vs 19.8). The KL run is flatter late
      (26.6 -> 23.4 -> 22.9) but sits lower throughout, so it trades learning speed for stability rather than fixing the
      drop. One seed pair; the gap is about the size of eval noise.
    * Next on GPU1 (from 18:15): **`env_kwargs.train_mazes=1024`** (default recipe, seed 0; 64 layouts -> 1024). Tests
      whether more layout diversity closes the train/held-out gap. Baseline seed 0, steps 30-40: 29.7% overall, held-out
      11x11 27.6%, 15x15 25.5%.
      Started 18:08. Step-0 eval **11.5%** (training 12.5, held-out 11x11 9.4, 15x15 12.5), 53% invalid (same pretrained
      model; within noise of the other runs' step-0 evals).
      Steps 1-5: rollout invalid 86% -> 76%, 9-21 groups with signal, ~2 min/step. Step-5 eval **14.1%** (training 17.2,
      held-out 11x11 9.4, 15x15 15.6), 52% invalid. Normal start.
      Steps 6-10: rollout invalid 75% -> 64%, 23-38 groups with signal. Step-10 eval **21.4%** (training 23.4, held-out
      11x11 14.1, 15x15 26.6), 50% invalid. Same pace as the 64-layout seed-0 run (19.3%, held-out 11x11 15.6).
      Steps 11-15: rollout invalid 63% -> 58%, rollouts 3-5%, 36-44 groups with signal. Step-15 eval **21.4%** (training
      25.0, held-out 11x11 15.6, 15x15 23.4), 45% invalid (64 layouts: 25.0%, held-out 11x11 20.3).
      Steps 16-20: rollout invalid 55% -> 49%, rollouts 5-7%. Step-20 eval **26.0%** (training 25.0, held-out 11x11 20.3,
      15x15 32.8), 27% invalid (64 layouts: 32.3%, held-out 11x11 29.7). Rollouts lag the 64-layout run.
      Steps 21-25: rollout invalid 46% -> 37%, rollouts 7-13%. Step-25 eval **30.2%** (training 29.7, held-out 11x11 25.0,
      15x15 35.9), 12% invalid (64 layouts: 31.8%, held-out 11x11 26.6). Caught up on eval despite lower rollouts.
      Steps 26-30: rollout invalid 39% -> 32%, rollouts 10-13%. Step-30 eval **31.8%** (training 37.5, held-out 11x11 28.1,
      15x15 29.7), 4% invalid (64 layouts: 29.2%, held-out 11x11 26.6).
      Steps 31-35: rollout invalid 29% -> 26%, rollouts 16-22%. Step-35 eval **28.6%** (training 31.2, held-out 11x11 23.4,
      15x15 31.2), 3% invalid (64 layouts: 28.6%, held-out 11x11 26.6).
      Step-40 eval **27.6%** (training 32.8, held-out 11x11 20.3, 15x15 29.7), 1% invalid (64 layouts: 31.2%); rollouts
      20-21%, invalid 20% (64 layouts: 25-29%, 15%). **Verdict: more layouts change nothing measurable.** Steps 30-40 mean
      **29.3%** vs **29.7%** (training 33.9 vs 35.9, held-out 11x11 24.0 vs 27.6, 15x15 30.2 vs 25.5): the 11x11 and 15x15
      differences go opposite ways and are within noise. The 64-layout policy is not memorizing layouts, so layout
      diversity is not what caps it at ~30%. Rollouts were slower to rise (the training set no longer repeats).
    * Next on GPU1 (from 22:10): **`env_kwargs.include_state=false`** (default recipe, seed 0). The per-turn text drops the
      ball position and relative goal and keeps only "Move:", so both must be read from the image. Tests how much of the
      ~30% comes from the image vs the relative-goal text.
      Step-0 eval **0.0%**, **99.8% invalid** (default recipe: ~8-12%, ~55%). Without the numbers in the turn text the
      pretrained model never answers in JSON: it writes prose ("The correct path is: - Move right to the green circle ...")
      and then repeats "The correct move is:". The coordinate text is what anchors the JSON format, so the queue's
      early-abort rule (< 6 groups with signal in steps 1-3) will probably stop this run.
      **Aborted at step 5** by that rule: 0 groups with signal, rollout invalid 97%, eval 0%. Reading ball and goal from the
      image cannot be tested this way without a format warm-up (SFT/BC, still awaiting approval).
* **Scripted baselines (CPU, no model; 192 episodes per eval set, same env settings).** Overall is the mean of the three sets.

  | Policy | Training layouts | Held-out 11x11 | Held-out 15x15 | Overall |
  |---|---|---|---|---|
  | random move in [-3, 3]^2 | 9.4 | 11.5 | 7.3 | **9.4** |
  | echo the relative goal (straight line, clipped to 3) | 18.8 | 21.4 | 14.6 | **18.2** |
  | echo, but a random 3-cell axis move on 30% of turns | 38.0 | 34.9 | 39.6 | **37.5** |

  * The untrained model's ~8-12% step-0 eval matches random moves.
  * Trained greedy policies (~29-32%) sit between echo and echo+jitter. Sampled rollouts late in long runs (32-42%) are at
    the echo+jitter level.
  * A likely reading (not verified on transcripts): RL mostly teaches "echo the offset" plus some wall escapes. T=1
    sampling supplies extra random escapes, which greedy decoding lacks. That would explain why sampled rollouts keep
    rising while greedy eval plateaus at ~30%. A sampled eval (awaiting approval) would test this directly.
  * Script: scratchpad `echo_baseline.py` (not committed).
    * Next on GPU1 (from 22:20): **training `temperature=1.2`** (default recipe, seed 0; eval stays greedy). Do hotter
      rollouts propose more wall escapes for GRPO to reinforce, or just more invalid replies?
      Step-0 eval (greedy, so unaffected by the training temperature) **9.9%**, 57% invalid.
      Steps 1-5: rollout invalid **93% -> 89%** (default T=1.0: 86% -> 78%), only 2-7 groups with signal (default: 6-15),
      rollouts 0.1-0.3%, mean log-prob -4.5 (hot sampling). Yet step-5 greedy eval is **30.7%** (training 39.1, held-out
      11x11 25.0, 15x15 28.1), 3% invalid (default at step 5: 14.6%; it first reached ~30% at step 20). Single eval; could be
      noise or a lucky early update. The queue's collapse rule (rollout invalid > 85% two steps running) would have killed
      this run at step 7, so its monitor was stopped (22:28). That also killed the run, which was restarted by hand from
      the step-5 checkpoint with no auto-abort.
      Resumed 22:29. Steps 6-10: rollout invalid **88% -> 50%** (default: 75% -> 64%), groups with signal 8 -> 44, rollouts
      0.3% -> 6.6%, mean log-prob -4.7 -> -2.6. Step-10 eval **27.6%** (training 32.8, held-out 11x11 23.4, 15x15 26.6),
      0% invalid (default at step 10: 19.3%). Two evals ahead of the default, so the early jump looks real.
      Steps 11-15: rollouts **11% -> 33%** at T=1.2, invalid 39% -> 14%, 54-62 groups with signal, log-prob -2.4 -> -1.5.
      Step-15 eval **26.0%** (training 31.2, held-out 11x11 21.9, 15x15 25.0), 0% invalid (default: 25.0%). The greedy eval
      lead is gone; sampled rollouts are far ahead of the default at this step.
      Steps 16-20: rollouts 30-38%, invalid 11% -> 3%, mean log-prob **-1.2 -> -0.31** (sharpening fast; default sits
      at ~-1.0), groups with signal 62 -> 52. Step-20 eval **25.5%** (training 31.2, held-out 11x11 20.3, 15x15 25.0),
      0% invalid (default: 32.3%). Greedy eval has drifted down since step 5 while T=1.2 rollouts sit at ~35%.
      Steps 21-25: rollouts 29-37%, invalid 2% -> 0.6%, mean log-prob **-0.21 -> -0.09**, groups with signal **49 -> 29**.
      Step-25 eval **25.5%**, with per-set scores identical to step 20 (31.2 / 20.3 / 25.0), so the greedy policy has frozen
      (default at step 25: 31.8%). The hotter sampling made the policy collapse onto near-deterministic replies, which
      cancels out the temperature, so the GRPO signal is drying up.
      Steps 26-30: rollouts 27-39%, log-prob flat at -0.09, 32-45 groups with signal. Step-30 eval **25.5%**: same per-set
      scores for the third eval running. **Stopped at step 30** (02:33) since the greedy policy hadn't changed in 10 steps.
      **Verdict: T=1.2 speeds up the first 5-10 steps (30.7% at step 5 vs 14.6%), then collapses the policy onto
      near-deterministic replies and freezes greedy eval ~5 points below the default** (25.5 vs ~30%). One seed.
    * Next on GPU1 (from 02:35, launched by hand without auto-abort): **T=1.2 + `kl_coef=0.05`**. Does a KL anchor keep the
      fast start and prevent the sharpening?
      Step-0 eval **11.5%**, 54% invalid (normal).
      Steps 1-5: rollout invalid 94% -> 87%, only 1-6 groups with signal, KL 0.001 -> 0.05. Step-5 eval **30.2%** (training
      37.5, held-out 11x11 29.7, 15x15 23.4), 6% invalid: **the fast start repeats** (no-KL T=1.2: 30.7%; T=1.0: 14.6%).
      The two runs share seed 0 and the KL term is still small, so this is not an independent replication.
      Steps 6-10: KL 0.07 -> 0.12, mean log-prob **held at about -4** (no-KL T=1.2: -4.7 -> -2.6), rollout invalid
      83% -> 75% (no-KL: 88% -> 50%), 8-17 groups with signal, rollouts ~1%. Step-10 eval **28.1%** (training 32.8,
      held-out 11x11 25.0, 15x15 26.6), 0% invalid (no-KL T=1.2: 27.6%). KL is stopping the sharpening; the cost is
      fewer valid rollouts.
      Steps 11-15: rollout invalid ~72-74%, 19-26 groups with signal, rollouts ~1%, KL ~0.1, log-prob -3.7 to -4.1.
      Step-15 eval **24.0%** (training 31.2, held-out 11x11 17.2, 15x15 23.4), 0% invalid (no-KL T=1.2: 26.0%;
      T=1.0: 25.0%). Learning has stalled: the KL anchor keeps the policy
      hot, so 3/4 of T=1.2 rollouts stay invalid and terminated.
      Steps 16-20: rollout invalid 71% -> 55%, 28-42 groups with signal, rollouts 2-5%, KL 0.12 -> 0.19, log-prob -3.5 ->
      -2.7. Step-20 eval **19.3%** (training 18.8, held-out 11x11 18.8, 15x15 20.3), 0% invalid (no-KL T=1.2: 25.5%; T=1.0:
      32.3%). Greedy eval has fallen at every eval since step 5 (30.2 -> 28.1 -> 24.0 -> 19.3) while rollouts recover.
      Steps 21-25: rollout invalid 61% -> 74% -> 58%, 30-40 groups with signal, rollouts 2-5%, KL 0.15-0.19. Step-25 eval
      **27.1%** (training 28.1, held-out 11x11 26.6, 15x15 26.6), 0% invalid: back up from 19.3%, so the step-20 low was
      partly noise (T=1.0 at step 25: 31.8%).
      Steps 26-30: rollout invalid 53% -> 70% (oscillating), 33-43 groups with signal, rollouts 3-6%, KL 0.13-0.20. Step-30
      eval **26.6%** (26.6 on all three sets), 0% invalid (T=1.0: 29.2%; no-KL T=1.2: 25.5%).
      Steps 31-35: rollout invalid 70% -> 58%, 24-41 groups with signal, rollouts 2-5%. Step-35 eval **26.6%** (training
      32.8, held-out 11x11 20.3, 15x15 26.6), 0% invalid (T=1.0: 28.6%).
      Step-40 eval **22.9%** (training 28.1, held-out 11x11 17.2, 15x15 23.4), 0% invalid; rollouts 2-3%, invalid 66-68%.
      Finished 04:18; the vLLM EngineDeadError in its log is teardown after a clean exit.
      **Verdict: T=1.2 + KL is worse than T=1.0.** Mean over steps 30-40: **25.3%** vs **29.7%** (training 29.2 vs 35.9,
      held-out 11x11 21.4 vs 27.6, 15x15 25.5 vs 25.5). KL does stop the sharpening seen without it (log-prob stays at
      about -3.5). The cost: 2/3 of T=1.2 rollouts stay invalid and terminated, so rollouts never pass ~6% and learning
      stalls. Seed 0, one run.
    * Next on GPU1 (from 04:19, via the queue): **training `temperature=0.8`** (default recipe, seed 0). The other side of
      the temperature axis: fewer junk replies, less exploration. Step-0 eval **9.9%** (greedy, as in every run).
      Steps 1-5: rollout invalid 81% -> 62% (T=1.0: 86% -> 78%), **14-38 groups with signal** (T=1.0: 6-15), rollouts
      1-4.5%, log-prob -0.9 -> -0.7. Step-5 eval **15.1%** (training 23.4, held-out 11x11 7.8, 15x15 14.1), 52% invalid
      (T=1.0: 14.6%). Much more signal per step, same greedy eval so far.
      Steps 6-9: rollouts **4% -> 32%**, invalid 56% -> 13%, 43-61 groups with signal, log-prob -0.7 -> -0.3. Fastest
      rollout rise of any run; T=1.0 rollouts were still ~2% at step 9.
      Step-10 eval **32.3%** (training 37.5, held-out 11x11 26.6, 15x15 32.8), 0% invalid (T=1.0 at step 10: 19.3%; it
      first reached ~32% at step 20). All 64 groups with signal, rollouts 32%, invalid 9%. **T=0.8 reaches the ~30% plateau in
      half the steps.** Steps are slower (~13 min vs ~5: more valid replies, so longer episodes). Watch for the T=1.2-style
      freeze: log-prob is already at -0.26.
      Steps 11-15: rollouts 28% -> 39% at T=0.8, invalid 7% -> 2%, 58-60 groups with signal, log-prob steady at ~-0.28.
      Step-15 eval **24.5%** (training 29.7, held-out 11x11 23.4, 15x15 20.3), 0% invalid (T=1.0: 25.0%). It fell back from
      32.3%; the step-10 peak may have been partly noise. Sampled rollouts (39%) now run 15 points above greedy eval.
      Steps 16-20: rollouts 33-43% at T=0.8, invalid 2-5%, log-prob -0.32 -> -0.42. Step-20 eval **21.4%** (training
      23.4, held-out 11x11 21.9, 15x15 18.8), 0% invalid (T=1.0: 32.3%).
      * Greedy eval is falling (32.3 -> 24.5 -> 21.4) while T=0.8 rollouts are the highest of any run at this step.
      * The sampled-vs-greedy gap is now ~15-20 points.
      * So the trained policy's success lives in its sampling distribution, and its argmax reply is worse. This is the
        clearest case yet for a sampled eval (awaiting approval).
      Steps 21-22: rollout invalid creeping up (5% -> 12%), rollouts 32-33%.
      Step-25 eval **23.4%** (training 20.3, held-out 11x11 23.4, 15x15 26.6), 0% invalid (T=1.0: 31.8%). Rollouts 31-39%,
      invalid 4-5%, log-prob -0.5. **Stopped after the step-25 eval** (09:06) to free GPU1 for the BC run.
      **T=0.8 seed 0, in short:** greedy eval was the fastest to ~30% (32.3% at step 10), then fell to 21-24% for steps 15-25,
      below T=1.0 (25-32%). Its T=0.8 rollouts are the best of any run (31-43%), so the sampled-vs-greedy gap reached
      ~15 points.
    * 500M **`compact` (absolute goal) + terminate** (GPU0, started 07:32): step-0 eval **1.0%**, **78% invalid**
      (compact_rel at step 0: 7.8-11.5%, ~52% invalid). The absolute-coordinate prompt starts with far fewer valid
      replies; the queue's step-3 signal check decides whether terminate can learn from it.
      Steps 1-5: rollout invalid flat at 85-86%, success ~0.1%, only ~3 groups with signal per step, grad norm
      **1.9-3.8** (noisy updates from a handful of groups). Step-5 eval **0.5%**, eval invalid up to **90%**. Getting
      worse, not better. Steps 6-9: rollout invalid 85% -> 82% (just under the queue's 85% stop rule), success
      0-0.2%, grad norm 1.1-2.9. Will stop it if the step-10 eval shows no gain.
      Step-10 eval **1.0%**, 88% invalid; rollouts 0.2%, 2 groups with signal. **Stopped by hand at step 10.**
      **Verdict:** the relative goal is what lets 500M start: with absolute coordinates only ~15% of replies are valid
      and almost none succeed, so terminate leaves nothing to learn from (same failure as simple + terminate).
    * Replacement on GPU0: best recipe with **group_size 32** (64 x 32 = 2048 episodes/step), the same budget as the
      128 x 16 run: more samples per start state vs more start states per update.
      Started 07:51. Step-0 eval 11.5% (54% invalid). Steps 1-5: rollout invalid 85% -> 72%, grad norm 0.99 -> 0.27,
      28 of 64 groups with signal at step 5, ~229 s/step (same as 128 x 16). Step-5 eval **9.9%** (training 14.1,
      held-out 11x11 10.9, 15x15 4.7), 77% invalid (128 x 16 at 5: 11.5%; 64 x 16 seeds: 14.6 / 5.7 / 6.8%).
      Steps 6-10: rollout invalid 71% -> 57%, 48 of 64 groups with signal, grad norm 0.17 -> 0.12. Step-10 eval
      **17.2%** (training 21.9, held-out 11x11 15.6, 15x15 14.1), 68% invalid (128 x 16 at 10: 21.4%; 64 x 16 seeds:
      19.3 / 10.9 / 18.2%).
      Steps 11-15: rollout invalid 57% -> 50%, 52 of 64 groups with signal, grad norm ~0.10. Step-15 eval **24.5%**
      (training 26.6, held-out 11x11 23.4, 15x15 23.4), 36% invalid (128 x 16 at 15: 26.6%; 64 x 16: 25.0 / 14.1 / 26.0%).
      Steps 16-20: rollout invalid 46% -> 37%, rollouts 7-11%, 58 of 64 groups with signal, grad norm 0.07-0.09.
      Step-20 eval **29.7%** (training 32.8, held-out 11x11 26.6, 15x15 29.7), 25% invalid (128 x 16 at 20: 31.8%;
      64 x 16: 32.3 / 25.0 / 27.1%).
      Steps 21-25: rollout invalid 36% -> 29%, rollouts 13-17%, grad norm ~0.065. Step-25 eval **27.1%** (training
      29.7, held-out 11x11 23.4, 15x15 28.1), 14% invalid (128 x 16 at 25: 32.8%; 64 x 16: 31.8 / 25.0 / 29.2%).
      With the same episode budget, it trails 128 x 16 on rollout success (13% vs 22%) and format (29% vs 22% invalid).
      Steps 26-30: rollout invalid 27% -> 23%, rollouts 16-22%, grad norm ~0.06. Step-30 eval **32.3%** (training 34.4,
      held-out 11x11 31.2, 15x15 31.2), 5% invalid (128 x 16 at 30: 30.2%). Evals of the two 2048-episode runs
      cross back and forth within noise; neither is clearly above the 64 x 16 seeds at the same step.
      Steps 31-35: rollout invalid 22% -> 16%, rollouts 19% -> 30%, grad norm 0.05-0.07. Step-35 eval **32.8%**
      (training 37.5, held-out 11x11 26.6, 15x15 34.4), 0.3% invalid (64 x 16 seeds at 35: 29.7 / 30.2 / 29.7%).
      RAM: trainer RSS grows with episode length in the 2048-episode runs (42 and 36 GB now, from ~30); ~88 GB
      available, guard at 30 GB.
      Finished at step 40: final eval **32.3%** (training 40.6, held-out 11x11 25.0, 15x15 31.2), 0.6% invalid; rollouts
      34%, invalid 12%, grad norm 0.05. Mean eval over steps 30-40: **32.5%** vs 29.7 / 29.0 / 30.6% for the three
      64 x 16 seeds. A 2-3 point edge, consistent across the last three evals but from one seed.
    * Next on GPU0: **seed 1 of 128 x 16** to check whether the 2048-episode edge replicates.
      Started 16:50. Step-0 eval 8.3% (59% invalid; seed-1 eval set, so compare with the 64 x 16 seed-1 run: 7.3%).
      Steps 1-5: rollout invalid 82% -> 75%, grad norm 0.77 -> 0.37, 49 groups with signal at step 5. Step-5 eval
      **9.9%**, 66% invalid (64 x 16 seed 1 at 5: 5.7%).
      Steps 6-10: rollout invalid 75% -> 57%, grad norm 0.37 -> 0.13. Step-10 eval **17.2%** (training 17.2, held-out
      11x11 14.1, 15x15 20.3), 59% invalid (64 x 16 seed 1 at 10: 10.9%).
      Steps 11-15: rollout invalid 55% -> 47%, rollouts 5-9%, 92 groups with signal, grad norm ~0.10. Step-15 eval
      **22.9%** (training 23.4, held-out 11x11 21.9, 15x15 23.4), 31% invalid (64 x 16 seed 1 at 15: 14.1%).
      Steps 16-20: rollout invalid 43% -> 33%, rollouts 9-14%, 110 groups with signal, grad norm 0.07-0.10. Step-20 eval
      **27.6%** (training 29.7, held-out 11x11 21.9, 15x15 31.2), 6% invalid (64 x 16 seed 1 at 20: 25.0%).
      Steps 21-25: rollout invalid 29% -> 22%, rollouts 20-23%, grad norm ~0.06. Step-25 eval **34.4%** (training
      35.9, held-out 11x11 31.2, 15x15 35.9), 0.7% invalid (64 x 16 seed 1 at 25: 25.0%; its best by step 40 was 30.2%).
      Steps 26-30: rollouts 24% -> 34% -> 27%, invalid 20% -> 9%, but grad norm **rose 0.055 -> 0.113**. Step-30 eval
      **27.1%** (training 25.0, held-out 11x11 25.0, 15x15 31.2), 0% invalid: a 7-point drop from step 25, more than the
      ~3-point eval noise (64 x 16 seed 1 at 30: 26.6%). Watching for the start of a collapse.
      Steps 31-35: grad norm back to 0.06-0.09, rollout invalid 5% -> 2%, **rollouts 33-42%** (best of any run), but
      step-35 greedy eval fell again to **25.0%** (training 25.0, held-out 11x11 23.4, 15x15 26.6), 0% invalid
      (64 x 16 seed 1 at 35: 30.2%). Sampled training success and greedy eval are diverging (42% vs 25%); invalid
      replies are not the cause. Unverified guess: greedy decoding gets stuck repeating a move that sampling escapes.
      Letting it finish at step 40.
      Finished at step 40: final eval **26.0%** (training 26.6, held-out 11x11 25.0, 15x15 26.6), 0% invalid; rollouts
      40%, invalid 1%, grad norm 0.05. Mean eval over steps 30-40: **26.0%** vs 29.0% for 64 x 16 seed 1.
    * **Verdict on 2048 episodes per step (mean greedy eval over steps 30-40, same-seed pairs):**
        * group_size 32 (64 x 32): 32.5 vs 29.7 (seed 0), 31.1 vs 29.0 (seed 1). **+2-3 points on both seeds**; seed 2
          is running. Small (about the eval noise) but consistent.
        * 128 x 16: 31.4 vs 29.7 (seed 0), **26.0 vs 29.0 (seed 1)**. **Does not replicate.** Seed 1 peaked at 34.4%
          (step 25), then fell while sampled rollouts rose to 40%.
        * Spend extra episodes on more samples per start state (group_size 32), not on more start states.
        * Open issue: late in both seed-1 2048-episode runs, greedy eval falls while sampled rollouts rise (26% vs 40% for
          128 x 16). Testing it directly needs a sampled eval (`eval_temperature`), not run yet.
    * Next on GPU0: **group_size 32 at lr=2.5e-6 for 60 steps** (seed 0; vs the lr=5e-6 seed-0 run, 32.5%). Does a
      slower lr avoid the late drop and keep climbing?
      Started 04:11. Step-0 eval **9.4%**, 59% invalid (lr=5e-6 run on the same weights and eval set: 11.5%, 54%;
      greedy evals of identical weights differ by a few points with vLLM batching).
      Steps 1-5: rollout invalid 86% -> 83% (lr=5e-6: 85% -> 72%), 19 groups with signal at step 5, grad norm
      0.86 -> 0.65. Step-5 eval **10.4%**, 53% invalid (lr=5e-6 at 5: 9.9%, 77%). Format learning is slower, as expected.
      Steps 6-10: rollout invalid 82% -> 72% (lr=5e-6: 70% -> 57%), 34 groups with signal, grad norm 0.48 -> 0.25.
      Step-10 eval **12.5%**, 62% invalid (lr=5e-6 at 10: 17.2%, 68%). About 3-5 steps behind the lr=5e-6 run.
      Steps 11-15: rollout invalid 71% -> 66% (lr=5e-6: 57% -> 50%), rollouts 2-4%, 39 groups with signal, grad norm
      ~0.2. Step-15 eval **13.0%**, 73% invalid (lr=5e-6 at 15: 24.5%, 36%). Now ~7 steps behind and the gap is widening.
      Steps 16-20: rollout invalid 64% -> 58%, rollouts 3-5%, grad norm ~0.14. Step-20 eval **14.1%**, 67% invalid
      (lr=5e-6 at 20: 29.7%, 25%). Its training curve tracks the lr=5e-6 run at about half the step count (rollout invalid
      58% at step 20 vs at step ~11), so 60 steps here ~= 30 steps at lr=5e-6, before the point where the late drop
      appeared (steps 35-40). It may need ~80 steps to answer the question.
      Steps 21-25: rollout invalid 56% -> 54%, rollouts 3-6%, grad norm ~0.11. Step-25 eval **16.7%** (training 23.4,
      held-out 11x11 10.9, 15x15 15.6), 59% invalid (lr=5e-6 at 12-13 steps' equivalent: ~20%).
      Steps 26-30: rollout invalid 52% -> 49%, rollouts 5-7%, grad norm ~0.1. Step-30 eval **20.3%** (training 23.4,
      held-out 11x11 14.1, 15x15 23.4), 49% invalid (lr=5e-6 at 15: 24.5%).
      Steps 31-35: rollout invalid 50% -> 45%, rollouts 6-9%. Step-35 eval **26.0%** (training 29.7, held-out 11x11
      23.4, 15x15 25.0), 37% invalid (lr=5e-6 at 17-18: ~27%).
      Steps 36-40: rollout invalid 46% -> 41%, rollouts 7-10%. Step-40 eval **26.6%** (training 31.2, held-out 11x11
      23.4, 15x15 25.0), 25% invalid (lr=5e-6 at 20: 29.7%).
      Steps 41-45: rollout invalid 41% -> 37%, rollouts 10-14%. Step-45 eval **27.6%** (training 34.4, held-out 11x11
      25.0, 15x15 23.4), 17% invalid (lr=5e-6 at 22-23: ~30%).
      Steps 46-50: rollout invalid 36% -> 32%, rollouts 11-16%. Step-50 eval **29.2%** (training 31.2, held-out 11x11
      26.6, 15x15 29.7), 13% invalid (lr=5e-6 at 25: 31.2%). Still rising, no drop yet.
      Steps 51-55: rollout invalid 30% -> 26%, rollouts 15-25%. Step-55 eval **28.1%** (training 31.2, held-out 11x11
      21.9, 15x15 31.2), 10% invalid.
      Step 60: eval **30.7%** (training 35.9, held-out 11x11 26.6, 15x15 29.7), 13% invalid; rollouts 21-25%, invalid
      22%. Mean over steps 50-60: **29.3%**, about where lr=5e-6 was at steps 25-30 (31.2 / 32.3%), so no gain per
      update and twice the steps. No late drop yet, but it hasn't reached the equivalent point (lr=5e-6 steps 35-40).
      **Extended to 90 steps** (resumes from the step-60 checkpoint; no code change): 90 here ~= 45 at lr=5e-6.
      Resumed 14:02. Steps 61-64: rollouts 20-25%, invalid 20-22%, grad norm ~0.05; ~20 min/step, so step 90 lands ~00:30.
      Step 65: eval **32.3%** (training 40.6, held-out 11x11 20.3, 15x15 35.9), 8% invalid; rollouts 24%, invalid 19%.
      Step 70: eval **30.2%** (training 43.8, held-out 11x11 17.2, 15x15 29.7), 8% invalid; rollouts 21-26%, invalid 16%.
      Training-layout success keeps rising (35.9 -> 40.6 -> 43.8) while held-out 11x11 falls (26.6 -> 20.3 -> 17.2): possible overfitting to the training layouts.
      Step 75: eval **28.6%** (training 31.2, held-out 11x11 23.4, 15x15 31.2), 6% invalid; rollouts 32%, invalid 12%.
      Training-layout eval fell back 43.8 -> 31.2, so the step 60-70 split was mostly per-set noise (64 episodes each),
      not clear overfitting. Rollouts still rising (21% -> 32% over steps 60-75) while greedy eval stays ~29-32%.
      Step 80: eval **32.3%** (training 42.2, held-out 11x11 25.0, 15x15 29.7), 5% invalid; rollouts 32-42%, invalid 10%.
      Step 85: eval **31.8%** (training 35.9, held-out 11x11 31.2, 15x15 28.1), 7% invalid; rollouts 31-37%, invalid 9%.
      Step 90: eval **33.3%** (training 39.1, held-out 11x11 29.7, 15x15 31.2), 5% invalid; rollouts 28-40%, invalid 7%.
      **Verdict:** lr=2.5e-6 reaches the same plateau as lr=5e-6 in about twice the updates, and goes no higher.
      * Mean over steps 60-90: **31.3%** (training 38.4, held-out 11x11 24.8, 15x15 30.8). lr=5e-6 gs32 seed 0, steps
        30-40: 32.5%.
      * **No late drop** in 90 steps: every eval from step 60 on is 28.6-33.3%.
      * Halving the lr removes neither the plateau nor the gap between sampled rollouts (28-42%) and greedy eval.
      * 64 x 32, seed 0, one run.
    * Next on GPU0 (from 03:01, by hand, no auto-abort): **T=1.2, seed 1** (default recipe otherwise). An independent
      replication of the T=1.2 fast start.
      Step-0 eval **7.3%**, 58% invalid (normal for seed 1).
      Steps 1-5: rollout invalid flat at ~94%, 3-5 groups with signal, log-prob ~-5. Step-5 eval **12.5%** (training
      12.5, held-out 11x11 14.1, 15x15 10.9), 54% invalid (T=1.0 seed 1: 5.7%). **The 30% jump at step 5 does not
      replicate** with seed 1, so the seed-0 jump (twice, with and without KL) looks seed-specific.
      Steps 6-10: rollout invalid still 91-94%, 0-4 groups with signal (one step with none), log-prob ~-5. Step-10 eval
      **5.7%**, 66% invalid (T=1.0 seed 1: 10.9%). Seed 1 at T=1.2 is barely bootstrapping the format: hot sampling
      leaves too few valid replies to learn from. It will be stopped at step 15 unless signal picks up.
      Steps 11-15: signal picked up: rollout invalid 89% -> 69%, 6 -> 18 groups with signal. Step-15 eval **25.0%**
      (training 28.1, held-out 11x11 20.3, 15x15 26.6), 0% invalid (T=1.0 seed 1: 14.1%). Kept running. The jump came
      10 steps later than for seed 0, but it came: once a few groups get the format right, greedy eval jumps fast at T=1.2.
      Steps 16-20: rollout invalid 66% -> 49%, 21-43 groups with signal, rollouts 1-4%, log-prob ~-3.3. Step-20 eval
      **25.5%** (training 29.7, held-out 11x11 21.9, 15x15 25.0), 0% invalid (T=1.0 seed 1: 25.0%). Level with T=1.0 now.
      Steps 21-25: rollout invalid 47% -> 37%, 45-60 groups with signal, rollouts 5-11%, log-prob ~-3.0 (no sharpening
      yet; seed 0 was at -0.2 by step 21). Step-25 eval **27.6%** (training 26.6, held-out 11x11 28.1, 15x15 28.1), 0%
      invalid (T=1.0 seed 1: 25.0%).
      **Crashed at step 27** (04:23):
      * Error: `ValueError: Image features and image tokens do not match, tokens: 1025, features: 1024` in the training
        forward pass.
      * Likely cause: the update splices vLLM's generated token ids (`gen_ids`) into the batch, so a sampled `<image>`
        special-token id reaches the model as an extra image slot. The text-level `scrub_media_tokens` fix (1ae1d2e) does
        not cover it.
      * Hotter sampling makes such tokens likelier.
      * Not fixed (code change; awaiting approval). The wrapper's next pass resumes from the step-26 checkpoint with fresh
        rollouts.
      * Resumed from step 26 (pass 2, 04:24).
      * **Fixed in 70be652**: `HFPolicy._build_exact` replaces any generated image/video placeholder id with padding and
        leaves it untrained. Reproduced without the fix ("tokens: 65, features: 64" on the tiny test VLM); slow
        regression test added.
      Steps 26-30: rollout invalid 34% -> 26%, 54-60 groups with signal, rollouts 9-17%, log-prob ~-3.0 (no sharpening).
      Step-30 eval **26.6%** (training 29.7, held-out 11x11 23.4, 15x15 26.6), 0% invalid (T=1.0 seed 1: 26.6%).
      Steps 31-35: rollouts 14-22%, invalid 25% -> 20%, 59-64 groups with signal, log-prob -3.3 -> -3.6. Step-35 eval
      **27.6%** (training 29.7, held-out 11x11 29.7, 15x15 23.4), 0% invalid (T=1.0 seed 1: 30.2%).
      Step-40 eval **29.2%** (training 29.7, held-out 11x11 28.1, 15x15 29.7), 0% invalid; rollouts 21-24%. Finished 06:48.
      Mean over steps 30-40: **27.8%** vs **29.0%** at T=1.0 seed 1 (training 29.7 vs 30.2, held-out 11x11 27.1 vs 26.6,
      15x15 26.6 vs 30.2).
      **Verdict on T=1.2 (2 seeds + KL): no gain over T=1.0.**
      * Seed 0 froze at 25.5% (vs 29.7%); seed 1 is level (27.8 vs 29.0%); T=1.2 + KL (seed 0) is 25.3%.
      * The early-jump timing differs by seed (step 5 for seed 0, step 15 for seed 1).
      * Costs: very sparse signal at the start, the sharpening collapse on seed 0, and one crash from a sampled `<image>` token.
    * Next on GPU0 (from 06:50, via the queue): **T=0.8, seed 1**, a replicate of the T=0.8 run.
      Step-0 eval **8.3%**, 58% invalid (normal for seed 1).
      Steps 1-5: rollout invalid 81% -> 66%, 16-31 groups with signal (T=1.0 seed 1: far fewer), rollouts 1-3%. Step-5
      eval **14.1%** (training 15.6, held-out 11x11 18.8, 15x15 7.8), 49% invalid (T=1.0 seed 1: 5.7%; T=1.2 seed 1: 12.5%).
      Steps 6-10: rollouts 4.5% -> 24%, invalid 62% -> 21%, 35-56 groups with signal, log-prob -0.76 -> -0.24 (the same
      fast rise as seed 0). Step-10 eval **22.4%** (training 25.0, held-out 11x11 21.9, 15x15 20.3), 0% invalid (T=1.0 seed
      1: 10.9%; T=1.2 seed 1: 5.7%).
      Steps 11-15: rollouts 25% -> 34%, invalid 16% -> 5%, log-prob ~-0.25. Step-15 eval **20.3%** (training 17.2, held-out
      11x11 26.6, 15x15 17.2), 0% invalid (T=1.0 seed 1: 14.1%).
      * Same shape as seed 0: greedy eval peaks at step 10 and dips at step 15 while T=0.8 rollouts keep rising, so the
        sampled-vs-greedy gap opens to ~14 points by step 15.
      Steps 16-20: rollouts 34-41%, invalid ~2%, log-prob -0.25 -> -0.30. Step-20 eval **14.1%** (training 10.9, held-out
      11x11 18.8, 15x15 12.5), 0% invalid (T=1.0 seed 1: 25.0%). Sampled-vs-greedy gap: **~24 points**. **Stopped after
      the step-20 eval** to free GPU0 for the traj_mean run.
      **T=0.8, both seeds:** the fastest start (greedy ~22-32% by step 10, rollouts > 30% by step 9). After that greedy
      eval drops below T=1.0 (seed 0: 21-24% at steps 15-25; seed 1: 14-20% at steps 15-20) while rollouts keep rising.
      T=0.8 trains the best *sampled* policy so far and a worse *greedy* one.
    * 256M `simple` (lr=5e-6, 128 tokens, noop) started 07:13: step-0 eval **1.6%**, eval invalid 82%.
      Steps 1-3: 1-2 groups with signal, rollout invalid 93%, success ~0%: **dropped after step 3**. 256M cannot
      bootstrap the `<action>` JSON format from scratch (500M could: 9-16 groups at steps 1-4).
    * 256M `compact` at **lr=1e-5** (started 07:52 on the free GPU0): compact at 5e-6 never collapsed but peaked at 8.8%;
      does its slower drift leave room for a faster lr (compact_rel at 1e-5 collapsed by ~step 20)?
      Step 0: eval 1.0% (94% invalid). Steps 1-3: groups with signal 17 -> 45, rollout invalid 78% -> 34%, success
      2% -> 10% (faster than compact at 5e-6: 19 -> 37 groups, invalid 77% -> 71%).
      Step-5 eval **6.2%** (training 7.8, held-out 11x11 4.7, 15x15 6.2), eval invalid 12% (compact at 5e-6, step 5:
      2.6%, 49% invalid); rollouts 14%, invalid 26%.
      Steps 6-8: rollouts 12-14%, invalid 16-25%, grad norm 0.67-0.98 (higher than at 5e-6).
      Step 9: rollout invalid jumps 22% -> 59%, success 14% -> 9%. Step-10 eval **6.2%** (6.2 on every set), eval
      invalid **39%** (compact at 5e-6, step 10: 5.7%, 2% invalid). Possible early drift; watching.
      Step 11: rollout invalid back down to 34%, grad norm 0.39: a wobble, not (yet) a collapse.
      Steps 12-15: rollouts 11-15%, invalid 7-23%, grad norm 0.39-0.55. Step-15 eval **4.7%** (training 6.2,
      held-out 11x11 3.1, 15x15 4.7), eval invalid 33%. Not collapsing, but no better than compact at 5e-6 (peak 8.8%).
      Steps 16-20: rollouts 11% -> 6%, invalid 44% -> 60%, groups with signal 46 -> 32-36, grad norm 0.66 -> 1.1-1.2.
      Step-20 eval **2.1%**, eval invalid 56%. Replies drift into pseudo-arithmetic (`x=0.88, y=0.40.
      (x+y=1.80+0.40=1.28)`, `Two-dimensionalietzschean: x-y=4.40-3.42= 1.28`), the same move repeated every turn.
      **Collapsed; stopped at step 20.** Verdict: doubling the lr for `compact` neither raised the peak (6.2% vs 8.8%)
      nor kept it stable; `compact` is best at 5e-6.
    * 500M compact_rel terminate lr=5e-6, **seed 1** (started ~12:40 on GPU0): replicate of the best run. seed also
      changes the 64 eval start states, so evals are on a different same-distribution set.
      Step 0: eval 7.3% (59% invalid; seed 0: 7.8%). Steps 1-5: groups with signal 10 -> 21 (seed 0: 6 -> 14 by step 4),
      rollout invalid 85% -> 78%, turns 1.4-1.6, grad norm 1.5 -> 0.46. Step-5 eval **5.7%**, 77% invalid (seed 0 at
      step 5: 14.6%, 63% invalid): a slower start than seed 0.
      Steps 6-10: groups with signal 22 -> 32, rollout invalid 75% -> 67%, turns 1.7 -> 2.0, grad norm 0.27-0.36.
      Step-10 eval **10.9%** (training 12.5, held-out 11x11 12.5, 15x15 7.8), 72% invalid (seed 0 at step 10: 19.3%).
      Still ~5 steps behind seed 0, but on the same upward path.
      Steps 11-15: groups 34-38, rollout invalid 65% -> 56%, turns 2.2 -> 2.7, grad norm 0.19-0.24. Step-15 eval **14.1%**
      (training 12.5, held-out 11x11 17.2, 15x15 12.5), 59% invalid (seed 0 at 15: 25.0%; at 10: 19.3%).
      Steps 16-20: groups 39 -> 54, rollout invalid 51% -> 44%, turns 3.1 -> 3.7, grad norm 0.11-0.16. Step-20 eval
      **25.0%** (training 31.2, held-out 11x11 23.4, 15x15 20.3), 33% invalid: matches seed 0 at step 15. Catching up.
      Steps 21-25: groups 49-59, rollout invalid 42% -> 35%, rollouts 8% -> 14%, grad norm 0.09-0.12. Step-25 eval
      **25.0%** (training 28.1, held-out 11x11 25.0, 15x15 21.9), eval invalid 13%.
      Steps 26-30: rollouts 12% -> 19%, invalid 34% -> 28%, grad norm 0.08-0.09. Step-30 eval **26.6%** (training 28.1,
      held-out 11x11 23.4, 15x15 28.1), eval invalid 4%. Past step 30 with no drift: the stability replicates.
      Steps 31-35: rollouts 19% -> 25%, invalid 27% -> 20%, grad norm 0.07. Step-35 eval **30.2%** (training 35.9,
      held-out 11x11 26.6, 15x15 28.1), eval invalid 2% (seed 0 at 35: 28.6%).
      Steps 36-40: rollouts 22-32%, invalid 20% -> 16%, grad norm 0.06. Final step-40 eval **30.2%** (training 26.6,
      held-out 11x11 29.7, 15x15 34.4), eval invalid 0.8%. **Finished without collapse.** Two seeds: final 31.2% / 30.2%,
      both stable to step 40; seed 1 learned the format ~5-10 steps later, then caught up.
    * 500M compact_rel terminate lr=5e-6, **seed 2, 80 steps** (GPU0, started 17:22): does it keep improving past ~30%
      or eventually collapse? Step 0: eval 11.5% (different eval set again). Steps 1-5: groups 13-20, rollout invalid
      86% -> 73%, grad norm 1.2 -> 0.37. Step-5 eval **6.8%**, 78% invalid: a slow start like seed 1 (5.7% at step 5).
      Steps 6-10: groups 24 -> 43, rollout invalid 71% -> 59%, grad norm 0.31 -> 0.16. Step-10 eval **18.2%** (training
      21.9, held-out 11x11 21.9, 15x15 10.9), 54% invalid (seed 0 at 10: 19.3%; seed 1: 10.9%).
      Steps 11-20: groups 38-53, rollout invalid 57% -> 46%, grad norm 0.11-0.16. Step-20 eval **27.1%** (training 28.1,
      held-out 11x11 32.8, 15x15 20.3), 17% invalid (seed 0 at 20: 32.3%; seed 1: 25.0%).
      Steps 21-25: groups 52-57, rollout invalid 41% -> 39%, grad norm 0.09-0.10. Step-25 eval **29.2%** (training
      26.6, held-out 11x11 40.6, 15x15 20.3), 5% invalid.
      Steps 26-30: rollout invalid 36% -> 31%, rollouts 13% -> 18%, grad norm 0.08-0.09. Step-30 eval **31.8%** (training
      31.2, held-out 11x11 40.6, 15x15 23.4), 3% invalid. Three seeds at step 30: 29.2 / 26.6 / 31.8%.
      Steps 31-35: rollout invalid 29% -> 26%, grad norm 0.07. Step-35 eval **29.7%** (training 37.5, held-out 11x11
      32.8, 15x15 18.8), 2% invalid.
      Steps 36-40: rollouts 19-24%, invalid 25% -> 20%, grad norm 0.06. Step-40 eval **30.2%** (training 32.8, held-out
      11x11 32.8, 15x15 25.0), 0.3% invalid. **Three seeds at step 40: 31.2 / 30.2 / 30.2%**, all without collapse.
      Continuing to step 80.
      Steps 41-45: rollouts 22-28%, invalid 20% -> 15%, grad norm 0.05-0.06. Step-45 eval **29.7%** (training 31.2,
      held-out 11x11 40.6, 15x15 17.2), 0.1% invalid. Holding at ~30%.
      Steps 46-49: rollouts 28% -> 33%, invalid 16% -> 12%, grad norm 0.05.
      Step-50 eval **28.6%**; rollouts 34%. Past step 50, still flat at ~30% greedy.
      Steps 51-55: rollouts 28-41%, invalid 11% -> 9%, grad norm 0.05. Step-55 eval **29.2%**, 0% invalid.
      Steps 56-60: rollouts 29-37%, invalid 9% -> 7%, grad norm 0.04-0.05, logp flat (-0.88 -> -0.84). Step-60 eval
      **25.5%** (training 23.4, held-out 11x11 31.2, 15x15 21.9), 0% invalid. A dip, but no drift signs (invalid, grad
      norm, logp all steady); one eval is 192 episodes (SE ~3 pts), so watch 65-80 before calling it a decline.
      Steps 61-65: rollouts 32-42%, invalid 7% -> 5%, grad norm 0.04-0.05. Step-65 eval **29.2%** (training 34.4,
      held-out 11x11 32.8, 15x15 20.3), 0% invalid: back on the ~29-30% plateau, so step 60 was noise.
      Steps 66-70: rollouts 34-44%, invalid 5%, grad norm 0.04-0.05. Step-70 eval **30.7%** (training 32.8, held-out
      11x11 39.1, 15x15 20.3), 0% invalid.
      Steps 71-75: rollouts 30-38%, invalid 5%, grad norm 0.04-0.05, logp flat at -0.79. Step-75 eval **30.2%**
      (training 34.4, held-out 11x11 37.5, 15x15 18.8), 0% invalid.
      Finished at step 80: final eval **28.1%** (training 31.2, held-out 11x11 34.4, 15x15 18.8), 0% invalid; rollouts
      30%, invalid 7%, grad norm 0.05. **Verdict:** no late collapse in 80 steps, but no further gain either: mean eval
      29.6% over steps 20-40 vs 28.9% over 45-80 (peak 31.8% at step 30). The recipe is stable at lr=5e-6 and the
      ~30% plateau is not a matter of training longer. 15x15 held-out stays weakest (~19-25%).
    * 256M `compact` (absolute goal, lr=5e-6, noop) started 20:46: step-0 eval 0.5% (95% invalid). Steps 1-4: groups
      with signal 19 -> 38, rollout invalid 77% -> 66%, success 2% -> 6%. Step-5 eval **2.6%** (training 1.6, held-out
      11x11 0.0, 15x15 6.2), eval invalid **49%**. Same model, lr and step with `compact_rel`: **16.2%**, 12% invalid.
      So compact_rel's edge survives the mask fix: giving the goal relative to the ball is most of the early learning.
      Greedy replies are stuck repeats (`x=0.88, y=-2.23.` 20 times) or number runs (`3.81, 2.24, 2.24, 2.24, ...`).
      Step 10: eval **5.7%** (training 7.8, held-out 11x11 4.7, 15x15 4.7), eval invalid 2%; rollouts 11%, invalid 65% ->
      40%, grad norm ~0.4 (compact_rel at step 10: 24.0%). Format is now learned; the moves lag far behind.
      Steps 11-14: rollouts 8-13%, invalid 30% -> 19%, grad norm ~0.3: steady, no drift yet.
      Step 15: eval **7.3%** (training 7.8, held-out 11x11 6.2, 15x15 7.8), 0% invalid; rollouts 15%.
      Steps 16-19: rollouts 14-17%, invalid 14% -> 8%, grad norm ~0.3, log-prob -0.79 -> -0.43 (growing confident).
      Step 20: eval **6.8%** (training 7.8, held-out 11x11 6.2, 15x15 6.2), 0% invalid: plateaued, but no collapse at
      step 20 (both compact_rel runs at this lr had collapsed by step 18-20). Slower learning, slower drift.
      Steps 21-25: rollouts 15-22%, invalid 7% -> 13% (slight rise), grad norm ~0.4. Step-25 eval **7.8%** (training
      9.4, held-out 11x11 7.8, 15x15 6.2), 0% invalid. Still no collapse; plateau at 7-8%.
      Step 30: eval **8.3%** (training 10.9, held-out 11x11 7.8, 15x15 6.2), 0.4% invalid; rollouts 16-21%, invalid
      11-16%, grad norm ~0.45. Slowly rising, stable at step 30 -- the first 256M run in v2 to pass step 20 intact.
      Step 35: eval **8.8%** (training 12.5, held-out 11x11 7.8, 15x15 6.2); rollouts 14-25%, invalid 6-7%, grad norm
      0.55-0.79 (rising a little).
      Steps 36-39: rollouts 22-24%, invalid 6-9%, grad norm 0.56-0.76, log-prob -0.79 -> -1.18 (less confident).
      Finished at step 40: final eval **7.8%** (training 10.9, held-out 11x11 4.7, 15x15 7.8), 0% invalid.
      Verdict: the only 256M v2 run to finish 40 steps without collapsing, but it peaked at 8.8% (step 35), against
      compact_rel's 24.0% at step 10. Absolute goal coordinates learn slower and drift slower.

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

**Next: what the Metaworld work suggests for mazes** (from `scripts/report_notes/metaworld.md` and `summary.md`, 2026-11-23)
* **Metaworld findings that apply here:**
  * BC on scripted experts first, then RL. "RL refines a good clone; it does not add generalization". BC fixed every
    model whose rollouts were too near-deterministic to explore.
  * Per-trajectory loss weighting (`loss_agg=traj_mean`) fixes most of the token-mean bias, where long failed episodes
    outweigh short successes. Every maze run so far used token_mean, and failures here are 20-turn episodes.
  * Rollout temperature 0.8-1.0 beats 1.2. This matches the maze results above.
  * lr 2e-6 holds a policy better than 1e-5. Here lr=2.5e-6 reached the same plateau as 5e-6, so this is not the lever.
  * Larger models do better on these same procgen mazes: Qwen3-VL-2B 0.48 / 0.31 / 0.33 at step 40 (training / held-out
    11x11 / 15x15) vs ~0.36 / 0.28 / 0.26 for SmolVLM2-500M.
* **From the other agent's results PDF** (`reports/vlmrl_results.pdf`, built 2026-11-22):
  * §2.2 and Fig. 2: with token_mean, SmolVLM2-2.2B collapses within 5 updates on mazes (replies shrink from 98 to 22 to 5
    characters, 98% invalid). `traj_mean` fixes it (UMaze 0.97); `traj_mean` + KL 0.05 also steadies medium (0.94).
  * Fig. 1: SmolVLM2-2.2B on these procgen mazes (`traj_mean`) is at 0.38 / 0.23 / 0.30 at step 40, about the same as
    500M here.
  * §8 diagnostics: the step-1 log-prob, its drift (sharpening), and reply length. Sharpening is what preceded the T=1.2
    seed-0 freeze above.
  * §5.5 / Fig. 9: RL from a clone holds it at lr 2e-6 and erodes it at 1e-5. Appendix B: full-FT clones use BC lr 1e-5.
    So the BC run below uses `bc_lr=1e-5` and keeps the maze RL lr of 5e-6. If RL erodes the clone, 2e-6 is next.
* **The maze BFS oracle is now a BC expert** (c4d1a3e, `EnvBundle.expert`). Through the `compact_rel` protocol it solves
  100% of training / held-out 11x11 / held-out 15x15 episodes in 5.9 / 6.6 / 7.7 turns. Fast test added.
* **Runs (each changes one thing vs the default recipe, seed 0):**
  * GPU1: **`bc_steps=30` then RL** (`smolvlm_procgen_v2_terminate_500m_bc30_gpu1.yaml`). Starts after the T=0.8 seed-0
    run's step-25 eval; the step-0 eval is the clone.
  * GPU0: **`loss_agg=traj_mean`** (`smolvlm_procgen_v2_terminate_500m_trajmean_gpu0.yaml`). Starts after the T=0.8
    seed-1 run's step-20 eval.
  * **BC-30 result (GPU1, started 09:06):** 30 BC steps took about 10 min, with the expert at 1.00 every step. The fit
    went from log-prob -1.08 to -0.165 per token (not memorized).
    * **The clone scores 10.9%** at step 0, no better than the untrained model (8-12%). Every episode ran to max turns,
      but with **0% invalid replies** (untrained: ~55%).
    * The transcripts show it copies the oracle's style (moves of up to 2-3 cells to cell centres), not its routing. It
      oscillates between two cells (`{"x": -2.00, "y": 0.00}` / `{"x": 0.40, "y": -0.00}` for 10 turns).
    * The oracle's moves follow BFS paths around walls that are visible only in the image. 30 steps of cloning don't
      teach the 500M model to read them. This is the same vision bottleneck as the include_state=false run.
    * The clone still gives RL a format-clean start, so RL continues from it (40 steps).
    * **Sampled at T=1.0, the same clone succeeds 27% / 23%** (RL steps 1-2 rollouts, before RL has moved it much), against
      10.9% greedy. 0.2% invalid, 56-60 groups with signal.
    * That matches the oscillation in the transcripts. Greedy decoding repeats the same pair of moves forever; sampling
      breaks the loop.
    * Together with the T=0.8 runs, this is now the strongest evidence that greedy eval understates these policies.
    * RL steps 1-5: rollouts 23% -> 33%, invalid ~0.2%, 56-60 groups with signal, log-prob ~-0.18. Step-5 eval **16.1%**
      (training 14.1, held-out 11x11 15.6, 15x15 18.8), 0% invalid (RL-only at step 5: 14.6%). Greedy eval lags rollouts by
      ~17 points. ~16 min/step.
    * Steps 6-9: rollouts 30-38%, groups with signal 59 -> 53, log-prob -0.16 -> -0.13 (slowly sharpening).
    * Step-10 eval **18.8%** (training 23.4, held-out 11x11 17.2, 15x15 15.6), 0% invalid (RL-only at step 10: 19.3%).
      Rollouts 38%. Greedy eval tracks RL-only while sampled rollouts are ~2x higher. BC has not lifted greedy performance.
    * Steps 11-15: rollouts 39-45% (highest of any run so far), 0% invalid, log-prob ~-0.16. Step-15 eval **28.6%** (training
      34.4, held-out 11x11 21.9, 15x15 29.7), 0% invalid (RL-only at step 15: 25.0%).
    * Steps 16-20: rollouts 45-58%, 0% invalid, log-prob -0.15 to -0.22.
    * **Step-20 eval 44.3%** (training 46.9, **held-out 11x11 43.8**, 15x15 42.2), 0% invalid. RL-only at step 20: 32.3%.
      * This is the best eval of any SmolVLM run by ~10 points.
      * It is above the scripted "echo + 30% random escapes" policy (37.5%), so it is no longer explained by echoing the
        offset.
      * On held-out layouts it beats the other agent's Qwen3-VL-2B at step 40 (0.31 / 0.33 on 11x11 / 15x15).
      * Single eval, one seed. The clone itself scored 10.9%, so RL from the clone is doing the work. Cloning seems to have
        given RL a policy whose sampled moves follow the oracle's routing, so successes can be reinforced.
    * Step-25 eval **40.1%** (training 48.4, held-out 11x11 32.8, 15x15 39.1). Step-30 eval **39.1%** (training 45.3,
      held-out 11x11 31.2, 15x15 40.6). Both 0% invalid. Steps 26-30: rollouts 51-59%, 54-63 groups with signal.
      * Step 20's 44.3% was partly a high draw. The run is holding near 40%, still ~10 points above RL-only at the same
        steps (31.8 / 29.2%) and above the scripted 37.5%.
    * Step-35 eval **38.5%** (training 50.0, held-out 11x11 31.2, 15x15 34.4), 0% invalid. Rollouts 53-60%, log-prob
      -0.13 (sharper than RL-only's ~-0.85). The gap between sampled rollouts (~57%) and greedy eval (~39%) is ~18 points.
    * Step-40 eval **38.5%** (training 43.8, held-out 11x11 32.8, 15x15 39.1), 0% invalid. Finished.
    * **Verdict (seed 0): BC-30 + RL is the best SmolVLM recipe so far.** Mean over steps 30-40 is **38.7%**, vs 29.7%
      for RL alone and 33.7% for `traj_mean`. It is just above the scripted 37.5%. Held-out 15x15 is ~38%, held-out 11x11
      ~32%. It peaked at 44.3% (step 20), then held flat near 39% while sampled rollouts stayed at 45-60%.
    * Seed 1 is running on GPU0 to confirm. Next on GPU1: BC-30 + `traj_mean`
      (`smolvlm_procgen_v2_terminate_500m_bc30_trajmean_gpu1.yaml`).
  * **traj_mean (GPU0, started 09:53):** step-0 eval **9.4%**, 57% invalid (normal).
    Steps 1-5: rollout invalid 85% -> 70% (token_mean: 86% -> 78%), 9-26 groups with signal. Step-5 eval **14.6%**
    (training 20.3, held-out 11x11 9.4, 15x15 14.1), 75% invalid (token_mean at step 5: 14.6%).
    Steps 6-10: rollout invalid 68% -> 60%, 28-44 groups with signal, rollouts 2-6%. Step-10 eval **18.8%** (training 26.6,
    held-out 11x11 14.1, 15x15 15.6), 17% invalid (token_mean: 19.3%). Same pace so far.
    Steps 11-15: rollout invalid 59% -> 53%, 39-49 groups with signal, rollouts 4-8%. Step-15 eval **23.4%** (training
    29.7, held-out 11x11 17.2, 15x15 23.4), 16% invalid (token_mean: 25.0%).
    Steps 16-20: rollout invalid 51% -> 38%, rollouts 6-13%, log-prob ~-1.1 (steady, no sharpening). Step-20 eval **27.1%**
    (training 35.9, held-out 11x11 20.3, 15x15 25.0), 7% invalid (token_mean: 32.3%).
    Steps 21-25: rollout invalid 34% -> 25%, rollouts 14-19%, log-prob ~-1.0. Step-25 eval **27.6%** (training 31.2,
    held-out 11x11 20.3, 15x15 31.2), 1% invalid (token_mean: 31.8%).
    Steps 26-30: rollouts 20% -> 27%, invalid 23% -> 14%, 56-63 groups with signal. Step-30 eval **30.2%** (training 39.1,
    held-out 11x11 25.0, 15x15 26.6), 0.5% invalid (token_mean: 29.2%).
    Steps 31-35: rollouts 28-33%, invalid 13% -> 7%, 58-64 groups with signal, log-prob -0.91 -> -0.85. Step-35 eval
    **35.9%** (training 39.1, held-out 11x11 32.8, 15x15 35.9), 0% invalid (token_mean: 28.6%). This is the highest single
    greedy eval of any 64 x 16 run (previous best 34.4%, 128 x 16 seed 1 at step 25), and it is strong on both held-out sets.
    Steps 36-40: rollouts 38-42%, 0% invalid. Step-40 eval **34.9%** (training 43.8, held-out 11x11 31.2, 15x15 29.7),
    0% invalid. Finished.
    * **Verdict: a modest gain over token_mean, one seed.** Mean over steps 30-40 is **33.7%** vs 29.7% (token_mean, seed 0),
      about +4 points. Each eval has ~+-3 points of noise, so this is suggestive, not settled.
    * traj_mean is slower early (27% vs 32% at step 20), then keeps climbing where token_mean plateaus. Log-prob stayed
      steady (~-0.85), with no sharpening.
    * It is well below BC + RL (44.3% at step 20). GPU0 now runs a seed-1 replicate of BC + RL
      (`smolvlm_procgen_v2_terminate_500m_bc30_seed1_gpu0.yaml`). BC + traj_mean is the next candidate.
  * **BC-30 seed 1 (GPU0, started 15:30):** the clone's step-0 eval is **13.5%** (training 12.5, held-out 11x11 14.1,
    15x15 14.1), 0% invalid. The seed-0 clone scored 10.9%, so the clone reproduces.
    Steps 1-4: rollouts 28-36%, 0.1-0.2% invalid, 58-61 groups with signal (seed 0 at steps 1-3: 23-28%).
    Step-5 eval **20.3%** (training 21.9, held-out 11x11 21.9, 15x15 17.2), 0% invalid (seed 0 at step 5: 16.1%).
    Steps 6-9: rollouts 30-39%, log-prob -0.19 -> -0.16.
    Step-10 eval **21.9%** (training 17.2, held-out 11x11 23.4, 15x15 25.0), 0% invalid (seed 0 at step 10: 18.8%).
    Steps 11-15: rollouts 38-48%. Step-15 eval **28.1%** (28.1 on all three sets), 0% invalid (seed 0 at step 15: 28.6%).
    Steps 16-20: rollouts 38-54%. Step-20 eval **41.1%** (training 46.9, held-out 11x11 39.1, 15x15 37.5), 0% invalid
    (seed 0 at step 20: 44.3%). **The BC-30 jump replicates:** both seeds are at 41-44% by step 20 (RL-only: 32.3%).
    Steps 21-25: rollouts 48-58%, log-prob -0.16 -> -0.13. Step-25 eval **39.6%** (training 46.9, held-out 11x11 34.4,
    15x15 37.5), 0% invalid (seed 0 at step 25: 40.1%). Same flattening near 40% as seed 0.
    Steps 26-30: rollouts 46-55%, log-prob -0.12 -> -0.10. Step-30 eval **37.0%** (training 43.8, held-out 11x11 28.1,
    15x15 39.1), 0% invalid (seed 0 at step 30: 39.1%).
    Steps 31-35: rollouts 48-55%, log-prob -0.10 -> -0.08 (still sharpening). Step-35 eval **34.9%** (training 37.5,
    held-out 11x11 35.9, 15x15 31.2), 0% invalid (seed 0 at step 35: 38.5%). Greedy eval is drifting down from the step-20 peak.
    Step-40 eval **37.5%** (training 39.1, held-out 11x11 34.4, 15x15 39.1), 0% invalid. Finished. Log-prob ended at -0.08.
    * **Verdict (2 seeds): BC-30 + RL replicates.** Mean over steps 30-40 is **36.5%** for seed 1 and 38.7% for seed 0, so
      **37.6%** on average. RL alone scored 29.7% (seed 0) and 29.0% (seed 1), so cloning adds about 8 points.
    * The ceiling also replicates. Both seeds peak at step 20 (44.3 / 41.1%), then settle at 35-39% while log-prob sharpens
      (-0.2 -> -0.08) and sampled rollouts stay at 47-58%. The extra RL steps make greedy decoding more confident, not better.
    * Next on GPU0: BC-30 then RL at lr 2.5e-6 (`smolvlm_procgen_v2_terminate_500m_bc30_lr2p5e6_gpu0.yaml`). The aim is
      to test whether a gentler RL step avoids the post-step-20 sharpening and plateau.
  * **BC-30 + lr 2.5e-6 (GPU0, started 01:29, seed 0):** the clone's step-0 eval is **10.9%** (training 17.2, held-out
    11x11 4.7, 15x15 10.9), 0% invalid. The BC phase is unchanged (bc_lr=1e-5), so this is the same clone as before
    (10.9 / 13.5 / 16.7% across three earlier copies).
    RL steps 1-4: rollouts 21-29%, log-prob ~-0.24 (BC-30 at lr 5e-6, seed 0, steps 1-3: 23-28%).
    Step-5 eval **15.1%** (training 20.3, held-out 11x11 7.8, 15x15 17.2), 0% invalid (BC-30 at lr 5e-6 at step 5: 16.1 / 20.3%).
    Steps 6-10: rollouts 28-33%, log-prob ~-0.22. Step-10 eval **15.6%** (training 14.1, held-out 11x11 15.6, 15x15 17.2),
    0% invalid (lr 5e-6 at step 10: 18.8 / 21.9%). Slower, as expected at half the step size.
    Steps 11-15: rollouts 35-42%, log-prob -0.21 -> -0.18. Step-15 eval **18.2%** (training 18.8, held-out 11x11 15.6,
    15x15 20.3), 0% invalid (lr 5e-6 at step 15: 28.6 / 28.1%). About 10 points behind.
    Steps 16-20: rollouts 37-44%, log-prob ~-0.17. Step-20 eval **22.9%** (training 28.1, held-out 11x11 18.8, 15x15 21.9),
    0% invalid (lr 5e-6 at step 20: 44.3 / 41.1%). Rollouts are ~10 points behind too; no sign yet of a higher ceiling.
    Steps 21-25: rollouts 45-56%, log-prob ~-0.16. Step-25 eval **33.8%** (training 45.3, held-out 11x11 29.7,
    15x15 26.6), 0% invalid (lr 5e-6 at step 25: 40.1 / 39.6%). Gap down to ~6 points; catching up, not overtaking.
    Steps 26-30: rollouts 46-56%, log-prob ~-0.16. Step-30 eval **39.1%** (training 45.3, held-out 11x11 35.9,
    15x15 35.9), 0% invalid (lr 5e-6 at step 30: 39.1 / 37.0%). Caught up; now at the same plateau.
    Steps 31-35: rollouts 50-61%, log-prob -0.15 -> -0.12. Step-35 eval **39.6%** (training 45.3, held-out 11x11 37.5,
    15x15 35.9), 0% invalid (lr 5e-6 at step 35: 38.5 / 34.9%).
    Steps 36-40: rollouts 52-57%, log-prob -0.11 -> -0.13. Step-40 eval **39.6%** (same per-set counts as step 35),
    0% invalid. **Mean over steps 30-40: 39.4%** vs 37.6% for lr 5e-6 (two seeds): within eval noise (~±3 points).
    Verdict: halving lr reaches the same plateau ~10 steps later; it does not raise it.
  * **BC-30 + traj_mean (GPU1, started 19:19, seed 0):** the clone's step-0 eval is **16.7%** (training 20.3, held-out 11x11
    10.9, 15x15 18.8), 0% invalid. `loss_agg` does not touch the BC phase, so this is the same recipe as the seed-0 clone
    (10.9%). The 6-point spread across nominally identical clones is the run-to-run noise (vLLM sampling is not bit-exact).
    Steps 1-4: rollouts 27-42%, 56-61 groups with signal, log-prob -0.23 -> -0.19.
    Step-5 eval **21.4%** (training 26.6, held-out 11x11 15.6, 15x15 21.9), 0% invalid (BC-30 token_mean at step 5: 16.1 / 20.3%).
    Steps 6-10: rollouts 35-47%. Step-10 eval **24.0%** (training 31.2, held-out 11x11 20.3, 15x15 20.3), 0% invalid
    (BC-30 token_mean at step 10: 18.8 / 21.9%).
    Steps 11-15: rollouts 49-51%, log-prob ~-0.19 (steady). Step-15 eval **34.4%** (training 42.2, held-out 11x11 29.7,
    15x15 31.2), 0% invalid (BC-30 token_mean at step 15: 28.6 / 28.1%).
    Steps 16-20: rollouts 51-60%, log-prob -0.17 -> -0.16. Step-20 eval **37.5%** (training 45.3, held-out 11x11 32.8,
    15x15 34.4), 0% invalid (BC-30 token_mean at step 20: 44.3 / 41.1%).
    Steps 21-25: rollouts 52-65%, log-prob -0.16 -> -0.18 (not sharpening). Step-25 eval **36.5%** (training 43.8,
    held-out 11x11 31.2, 15x15 34.4), 0% invalid (BC-30 token_mean at step 25: 40.1 / 39.6%).
    Steps 26-30: rollouts 53-61%, log-prob -0.17 -> -0.13 (now sharpening too). Step-30 eval **36.5%** (training 42.2,
    held-out 11x11 32.8, 15x15 34.4), 0% invalid (BC-30 token_mean at step 30: 39.1 / 37.0%).
    Steps 31-35: rollouts 51-58%. Step-35 eval **37.0%** (training 43.8, held-out 11x11 32.8, 15x15 34.4), 0% invalid
    (BC-30 token_mean at step 35: 38.5 / 34.9%).
    Step-40 eval **34.4%** (training 43.8, held-out 11x11 29.7, 15x15 29.7), 0% invalid. Finished.
    * **Verdict: `traj_mean` adds nothing on top of BC.** Mean over steps 30-40 is **35.9%**, vs 37.6% for BC-30 with
      token_mean (2 seeds: 38.7 / 36.5%). That is within noise, and slightly below.
    * It changed the shape of the run: no step-20 spike (37.5% vs 44.3 / 41.1%), log-prob stayed softer (-0.13 to -0.18
      vs -0.08 to -0.11), and rollouts were the highest of any run (52-65%). Greedy eval still plateaued at 35-37%.
    * So the post-step-20 sharpening is not what caps greedy eval. With sharpening held back, greedy eval sits at the same
      level.
    * (The vLLM "EngineDeadError" lines at the end of this run and of BC-30 seed 1 come from shutdown after the run had
      exited 0.)
    * Next on GPU1: BC-30 with DAgger on its last 15 steps (`smolvlm_procgen_v2_terminate_500m_bc30_dagger15_gpu1.yaml`).
      The aim is a better clone: the maze oracle can label any state the clone reaches.
  * **BC-30 + DAgger-15 (GPU1, started 05:11, seed 0): DAgger made the clone worse.** The clone's step-0 eval is
    **3.6%** (training 6.2, held-out 11x11 3.1, 15x15 1.6), 0% invalid. Plain clones scored 10.9-16.7%.
    * During the DAgger steps (16-30), the clone's own sampled success fell from 33% to 14-22%.
    * The greedy transcripts show why. The clone sends a small move into a wall (e.g. `{"x": 0.40, "y": 0.00}`), the
      ball doesn't move, and with one frame of context and greedy decoding the same observation gives the same reply,
      for all 20 turns.
    * The oracle's labels in the states a weak clone reaches (pressed against walls, off-centre) are small corrective
      moves. Cloning them taught small, timid moves, which makes the wall-loop more likely. This matches the Metaworld
      DAgger result in `rollout.py`'s docstring.
    * Left running to see whether RL recovers from this clone.
    * RL steps 1-4: rollouts 10-19% (plain BC-30 at steps 1-3: 23-28%).
    * Step-5 eval **4.7%** (4.7 on all three sets), rollouts 14% (plain BC-30 at step 5: 16.1 / 20.3%). Stopping after the
      step-10 eval. Next on GPU1: BC-60 (`smolvlm_procgen_v2_terminate_500m_bc60_gpu1.yaml`).
    * Steps 6-10: rollouts 14-18%. Step-10 eval **6.8%** (training 7.8, held-out 11x11 4.7, 15x15 7.8), 0% invalid,
      vs 15-20% for plain BC30 at step 10. Stopped by hand after this eval (failed run, kept on disk); BC-60 started on GPU1.
  - **BC-60 then RL** (`bc_steps: [60]`, otherwise BC30; GPU1): clone eval **15.6%** (training 21.9, held-out 11x11 14.1,
    15x15 10.9), 0% invalid, BC log-prob -0.10. Inside the 10.9-16.7% spread of the BC30 clones: twice the cloning steps
    does not give a better clone.
    Steps 1-5: rollouts 28-34%, log-prob -0.15 -> -0.12 (sharper than BC30's ~-0.2 at the same step). Step-5 eval **18.8%**
    (training 17.2, held-out 11x11 21.9, 15x15 17.2), 0% invalid; BC30 seeds at step 5: 20.3 / ~16-20%. Same track so far.
    Steps 6-10: rollouts 33-40%, log-prob -0.09 -> -0.08 (already as sharp as BC30 at step 40). Step-10 eval **22.9%**
    (training 26.6, held-out 11x11 23.4, 15x15 18.8), 0% invalid; BC30 at step 10: 15-22%. On track, slightly ahead.
    Steps 11-15: rollouts 42-45%, log-prob ~-0.09. Step-15 eval **33.9%** (training 39.1, held-out 11x11 42.2,
    15x15 20.3), 0% invalid; BC30 at step 15: 28.6 / 28.1%. Ahead by ~5 points, but the larger 15x15 mazes lag.
    Steps 16-20: rollouts 41-55%, log-prob -0.10 -> -0.14. Step-20 eval **43.8%** (training 48.4, held-out 11x11 45.3,
    15x15 37.5), 0% invalid; BC30 at step 20: 44.3 / 41.1%. Same peak as BC30, reached at the same step.
    Steps 21-25: rollouts 52-62%, log-prob ~-0.15. Step-25 eval **43.2%** (training 48.4, held-out 11x11 42.2,
    15x15 39.1), 0% invalid; BC30 at step 25: 40.1 / 39.6%. Holding the peak so far (BC30 had started to slide).
  - **Queued on GPU0: BC30 + KL 0.05 to the clone** (`kl_ref: clone`, new option in cb32f9c; the old `kl_coef` pulled
    toward the pretrained model, which would undo the clone). Started on GPU0 at 11:38.
    * **First attempt invalid (bug):** cloning runs through the same update, so with `kl_coef > 0` it loaded and cached
      the reference (the pretrained model) during cloning. RL step 1 showed `train/kl` 0.73 (vs ~0.001 at step 1 of the
      earlier KL-to-pretrained runs), and the clone itself was KL-penalised (clone eval 14.6%, inside the usual range).
      Fixed in cebb7ca (no KL during cloning; cached reference dropped) with a regression test. Stopped after one RL
      step; dir kept as `..._kl_ref=clone_badref`. Rerun started on GPU0 at 12:20.
    * Rerun: clone eval 13.5%; `train/kl` 0.0006 / 0.002 at RL steps 1-2 (reference is now the clone). Rollouts 29 / 23%.
    * Steps 1-5: rollouts 23-38%, log-prob -0.22 -> -0.19, KL 0.0006 -> 0.008. Step-5 eval **16.1%** (training 17.2,
      held-out 11x11 15.6, 15x15 15.6), 0% invalid; BC30 at step 5: 16-20%. Same track so far.
    * (A second frame is not the loop fix: `max_frames=2` on 256M, above, swapped wall-loops for copying its own previous
      reply, and collapsed at the same step.)
    * The loop also explains the sampled-vs-greedy gap in every run: under greedy decoding, any move that leaves the state
      unchanged repeats until the episode ends.
  * BC run: RL steps take ~10 min, since the clone's valid replies run full 20-turn episodes. Rollouts 23-28% at steps 1-3.
  * The two T=0.8 runs are cut short (40 -> 25 and 20 steps) to free the GPUs. Their pattern is already clear on both
    seeds: a faster start, then greedy eval dips while rollouts climb.
