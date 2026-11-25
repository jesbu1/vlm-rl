Metaworld (Farama, Sawyer arm), same trainer and text protocol; wandb project `tri/vlm-rl-metaworld`.

**What the model sees and does** (one turn = one image, no history)
* **Input:** the task instruction, a 224×224 corner-camera frame, and `Gripper: open. Last move: dx=+2.0, dy=…, dz=…`.
  No coordinates are given; the goal is visible only in the image.
* **Reply:** `{"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}`, a hand move in cm (at most 10 per turn).
* **Episode:** 15 turns, ending early on Metaworld's `success`.
* **Execution:** a waypoint controller moves the hand to the reply's target over up to 40 simulator steps, at
  `clip(gain · distance in cm, −1, 1)` per step. Gain 0.5 since 2026-11-22; the earlier 1.0 overshot (below).

**Reward: preferences from Metaworld's ground truth (`reward=pref_mw`)**

Each GRPO group is 8 rollouts from the same start (same goal position and seed).
1. **Score every rollout** by its final state: `s = 10·success + log(0.001 + r_final)`. Here `r_final` is Metaworld's
   dense reward (0–10) at the last step. An episode ended by an invalid reply gets `s = −20` (ranked last).
2. **Compare every pair:** `P(i beats j) = sigmoid((s_i − s_j) / 0.1)`, a tie (0.5) if `|s_i − s_j| < 0.02`.
   On the log scale that means "within 2%", so 0.026 vs 0.033 (hand 26 vs 24 cm from the puck) still counts.
3. **Win rate:** each rollout's reward is its mean `P(i beats j)` over the other 7, in [0, 1].
4. **GRPO:** advantages are z-scored within the group; a group with no spread (all tied) is dropped.

*Example (4 rollouts):*

| rollout | r_final | s | reward | advantage |
|---|---|---|---|---|
| invalid reply | — | −20 | 0 | −1.34 |
| no success | 0.03 | −3.5 | 1/3 | −0.45 |
| no success | 0.07 | −2.6 | 2/3 | +0.45 |
| success | 9.8 | 12.3 | 1 | +1.34 |

* **Why this works:** with temperature 0.1 on log rewards this is essentially a *ranking* by final progress (success
  first), soft only between near-equal outcomes. Every group has a signal even when nothing succeeds. A sparse success
  reward gave none (Metaworld success starts at 0%).
* **Loss:** each trajectory weighs the same (`loss_agg=traj_mean`), its turns trained on the exact sampled tokens.
* **Object-first variant (`reward=pref_mw_obj`), for tasks with an object.** Rollout *a* beats *b* at the first rule
  that separates them: (1) valid reply vs invalid; (2) success; (3) the object ended **≥ 1 cm closer to its goal**
  (Metaworld's `obj_to_target`); (4) `pref_mw`'s dense-reward rule. Rule 3 is needed on push: there the dense reward
  at the final state is highest for *gripping the puck and holding still* (≈ 5.5), while pushing it partway and
  losing contact scores < 2. Under `pref_mw`, RL taught exactly that (below).

**Reach** (eval = 32 greedy episodes on held-out / training goal positions; zero-shot is 0 for every model)

| model | eval, held-out / train | how |
|---|---|---|
| **Qwen3.5-4B** (LoRA on MLPs, lr 1e-5) | 0 (step 15) → **0.94 / 0.91** (95) → 0.78 / 0.88 (200) | RL only. Held-out plateaus at 0.5–0.78 after step 95 while training positions reach 0.97 |
| **Qwen3-VL-4B** (LoRA) | 0.28 / 0.28 after BC → 0.66 / 0.81 (30) → **0.59–0.78 / 0.88–1.00** (70–100) | BC then RL. RL alone: 0 (always dx = dy = 0) |
| Qwen3-VL-2B (full FT, lr 2e-6) | 0 → 0.22 / 0.22 (40) → 0.38 / 0.53 (100) | RL only, T = 1.6 (stuck at T = 1.2) |
| Gemma 4 E2B (full FT, lr 1e-6) | 0.09 / 0.09 after BC → 0.28–0.44 / 0.41–0.56 (85–100) | BC then RL, slow but rising. RL only at T = 1.6: 0.1–0.2 |

* **Exploration decides whether RL alone can start.** The preference reward needs rollouts in a group to differ. At
  T = 1.2 the mean log-prob of sampled tokens at step 1 was −0.6 for Qwen3.5-4B but −0.04 to −0.07 for the other
  three: near-deterministic, so groups tie. Higher temperature helped Qwen3-VL-2B and Gemma; Qwen3-VL-4B never moved
  sideways.
* **Behaviour cloning fixes the start.** Metaworld's scripted experts are run through the same text protocol, their
  moves written as the model's replies; the model is fine-tuned on the successful episodes (`bc_steps=30`, 64 episodes
  per step), then RL runs from the clone. RL then improves on it: Qwen3-VL-4B 0.28 → 0.75 in 70 steps.

**Push and ML10**
* **Push, RL only: 0** in every run, also from the reach adapter (the policy repeats one fixed move).
* **What held push back was mostly our controller.** At gain 1.0 it ran at full speed until 1 cm from each target,
  overshot, and lifted the hand while it dragged the puck, so the puck slipped out. The scripted expert succeeded
  only 0.63 through our protocol; at gain 0.5 it succeeds **0.97**, in 5.9 instead of 9.4 turns. ML10's expert goes
  from 0.85 to 0.91 on its training tasks and from 0.42 to 0.72 on its held-out tasks.
* **At gain 0.5 the push clone doubles** (30 BC steps; training / held-out positions):

  | clone | gain 1.0 | gain 0.5 |
  |---|---|---|
  | Qwen3.5-4B push | 0.22 / 0.03 | **0.47 / 0.25** |
  | Qwen3.5-4B push, DART 1 cm | — | 0.44 / 0.22 |
  | Qwen3-VL-4B push | 0.06 / 0.00 | 0.09 / 0.22 |
  | Qwen3.5-4B ML10 (training / held-out tasks) | 0.12 / 0.12 | 0.10 / 0.00; **0.38 / 0.00 with 100 BC steps** |

* **RL now improves push:** from the 30-step gain-0.5 clone, training rollouts rise from 0.15 to ~0.37 (steps 26–35)
  and eval reaches 0.59 / 0.31 (step 35); at gain 1.0 the rollouts never left ~0.1.
* **More cloning: push training positions nearly solved, held-out positions not.** After 100 BC steps Qwen3.5-4B
  scores 0.94 / 0.28 and Qwen3-VL-4B 0.91 / 0.28 (log-prob of the expert's replies −0.003: memorized). Held-out stays
  at ~0.25–0.28 from 30 to 100 steps: ML1 trains on only 50 puck / goal positions.
* **Ablation, outside the ML1 protocol: 450 training positions** (`env_kwargs.train_seeds=10`: the training
  positions of benchmark seeds 0–9, minus those within 1 cm of a held-out one; seed 1's training positions *are* seed
  0's held-out positions). Eval is unchanged. Note the denser coverage: the nearest training position to a held-out
  one is 2.2 cm away (median) instead of 3.8 cm. Live.
* **ML10 needs more cloning:** 100 steps gives 0.38 on its training tasks (log-prob −0.18, not yet memorized);
  200 steps live. Its 5 held-out tasks stay at ~0: new tasks from 10 training tasks is a different problem.
* **What we learned at gain 1.0** (clone 0.22 / 0.03, training / held-out positions):
  * **RL with `pref_mw` undid the clone** (rollouts 0.12 → 0.01): it learned to grip the puck and hold it, the
    dense reward's local optimum. **`pref_mw_obj` keeps the clone** (rollouts ~0.1 for 50 steps), with no clear gain.
  * **DAgger made the clone worse** (0.22 → 0.00; ML10 0.12 → 0.04). The experts are state machines with sharp
    thresholds on state the model cannot see (more than 2 cm from the puck means "go back up 20 cm"), so in the
    clone's own states nearly identical images get opposite labels.
  * **DART helped held-out push** (expert driven with 1 cm noise, clean labels cloned): 19 / 96 held-out successes
    over three evals vs 4 / 64 for the plain clone. On ML10 it hurt (0.12 → 0.02).
  * **Lower rollout temperature** gave ~1.5× more successful rollouts (T = 0.8–1.0: 0.12 vs T = 1.2: 0.08).
* **Eval noise:** greedy decoding in vLLM is not exactly reproducible, and 15 turns amplify any difference: the same
  clone scored 0.12, 0.16 and 0.31 on held-out positions in three runs. Differences below ~0.15 on one 32-episode
  eval are noise; the 128 training rollouts per step are the steadier signal.
* **ML10, Qwen3.5-4B (gain 1.0):** RL only 0.12 on its training tasks; from the reach adapter 0.17, and 0.02 on the
  held-out tasks; BC then RL 0.04–0.15.
* **The goal is ML10 / ML45 with as little engineering as possible.** The generic recipe, with no per-task code
  (`configs/sweeps/metaworld_recipe.yaml`): controller gain 0.5, BC on Metaworld's scripted experts (steps scaled with the
  number of tasks), training variations from 10 benchmark seeds (`train_seeds=10`: on push this took the clone from
  0.94 / 0.28 to 0.62 / 0.69, training / held-out positions), then RL with `pref_mw_obj` at T = 1.0. ML45's experts
  succeed 0.82 on its training tasks and 0.97 on its held-out tasks through our protocol. Live: ML10 (200 BC steps),
  ML45 (400 BC steps; Qwen3.5-4B and Qwen3-VL-4B).
* **RL from a good clone has not improved it so far.** Pooled evals at RL steps 5–15 (clone at step 0):

  | RL setting | push, held-out positions (clone 0.62–0.69) | ML10, training tasks (clone 0.52–0.54) |
  |---|---|---|
  | lr 1e-5 | 0.54 | 0.45 |
  | lr 2e-6 | 0.50 | 0.57 |
  | lr 1e-5, `adv_min=0` (only above-average rollouts reinforced) | 0.48 | 0.46 (0.34 at steps 20–30) |

  Nothing preserves held-out push; positive-only advantages hurt ML10 (above-average *failures* get reinforced). lr 2e-6
  is the only setting slightly above the ML10 clone (~2 standard errors). So the recipe is BC-first: the clone's quality,
  set by how much expert data it saw, is what moves ML10; the recipe's RL phase will also be run at lr 2e-6.
* **Recipe, ML10 clone** (BC 200 steps on 500 variations per task): 0.54 on the training tasks, 0.02 held-out; RL from it
  at lr 1e-5 and lr 2e-6 is live. With more steps, lr 2e-6 holds clones better (held-out push 0.56 vs 0.44 at lr 1e-5,
  steps 20–30; clone 0.62).
* **Over 50–60 steps at lr 2e-6, RL slowly improves the training distribution**: ML10 rollouts 0.49 → 0.64 (steps 1–10 →
  41–50), push 0.53 → 0.66; greedy evals move less (ML10 training tasks 0.54 → 0.56–0.59) and held-out push stays slightly
  below the clone (0.51–0.57 vs 0.62). RL refines a good clone; it does not add generalization.
* **ML45: first transfer to held-out tasks.** The recipe's ML45 clone (Qwen3.5-4B, 400 BC steps) scores 0.57 on the 45
  training tasks (expert ceiling 0.82) and **0.14 on the 5 held-out tasks**, zero-shot (7/50; ML10's clones: 0.00–0.02).
  RL from it at lr 1e-5 and 2e-6 is live; the Qwen3-VL-4B ML45 clone is still training.
* **ML10 recipe RL improves the clone** at both learning rates: training tasks 0.68 (lr 1e-5) / 0.66 (lr 2e-6) pooled over
  steps 30–50, against 0.57 for the clone; held-out tasks stay ~0.02. **ML45 with Qwen3-VL-4B:** 0.57 training, 0.04
  held-out (Qwen3.5-4B: 0.57 / 0.10–0.14 over six evals), so Qwen3.5-4B transfers better.
* **RL does not move ML45** over 40 steps (training tasks 0.53–0.61, held-out 0.08–0.14, either lr), and ML10's RL gain
  flattens at 0.60–0.71. More cloning is the lever: live are 800 BC steps on ML45 and 400 on ML10 (from the recipe clones),
  then RL at lr 2e-6.
* **More cloning beats RL on ML10:** the 400-step recipe clone scores 0.69 on the training tasks (200 steps: 0.54), as much
  as the best RL run; over 100 RL steps, lr 1e-5 peaked at 0.71 then fell to 0.54 while lr 2e-6 held 0.64. ML45 RL at lr 2e-6:
  training tasks 0.57 → 0.63, held-out flat (0.08–0.14). 800-step ML45 clone training.
* **800-step ML45 clone: 0.73 on the training tasks** (400 steps: 0.57; expert ceiling 0.82) but 0.06 on the held-out ones
  (400 steps: 0.12 pooled). More cloning keeps improving the training tasks on both benchmarks; held-out transfer does not
  follow.
* **Cloning longer trades transfer for fit (ML45):** 400 / 800 BC steps give 0.57 / 0.70 on the training tasks and
  0.12 / 0.056 on the held-out tasks (pooled over 5–6 evals; p ≈ 0.01). RL leaves both where cloning put them.
* **ML10 cloning saturates:** 0.69 at 400 BC steps, 0.67 at 800 (fit −0.19 → −0.11; experts 0.91). Per-task evals of the
  saved clones are next, to see which tasks stay unsolved.
* **Reasoning, native thinking:** Qwen3.5-4B with thinking on (4096 tokens, ML45 eval only) scores 0 / 0; 87% of replies
  are invalid because the reasoning loops and never closes its thinking block (the 400-step clone with thinking: 0.02 / 0).
  Next: short reasoning in the reply (`compact_rel_reason`, 256 tokens), evals running.
* **Per task** (eval-only, 270 / 100 episodes): ML45 held-out transfer is only door-lock (0.44 at 400 BC steps) and
  door-unlock (0.20); bin-picking, box-close and hand-insert are 0. ML10's held-out drawer-open and door-close, the
  opposites of its training tasks drawer-close and door-open, stay at ~0.05: the policy does the motion it learned for the
  object, not the one the instruction asks for. Some ML45 training-task zeros are the expert's (handle-pull-side,
  lever-pull, stick-push).
* **Short reasoning** (`compact_rel_reason`): every reply valid; the base model describes scenes correctly but moves
  poorly (0.01 / 0), and the answer-only clone loses its skill under the new prompt (0.12 / 0.06 vs 0.57 / 0.12). Live:
  **cloning with reasons** (`bc_rationale`: the base model writes a reason for each expert move, the clone learns
  "reason, then move"), then RL with reasoning, on ML10 and ML45.
* **Cloning with reasons, first attempt: worse** (ML45: 0.34 / 0.04 vs 0.57 / 0.12). The base model, asked why each
  expert move was right, often said it was wrong (the experts' moves exceed the prompt's 10 cm), and the clone learned
  "the move is incorrect" before each move. Second attempt: one short scene sentence, never judging the move. Also running:
  RL only with reasoning from the base model on ML10.
* **ML10 clone with reasons (first attempt): held-out tasks 0.07 vs 0.02** (answer-only; 7/100 vs 2/100). door-close (the
  opposite of training task door-open) 0.04 → 0.17, lever-pull and sweep-into 0 → 0.06. Training tasks 0.51 vs 0.61:
  door-open 0.24 → 0.55 and reach 0.30 → 0.45, but pick-place 0.73 → 0.13, push 0.94 → 0.33. RL-only reasoning (from the
  base model): 0.01 at step 10 (drawer-close 0.21).
* **RL-learned reasoning transfers.** RL only, from the base model with short reasoning (no cloning): at step 30 ML10's
  drawer-close is 0.93 and the held-out **door-close 0.52** (answer-only clone 0.04, clone with reasons 0.17); held-out
  tasks 0.12. Its reasons tie the instruction to the motion ("needs to be pushed shut → move right"). Live on ML45 too.
* **Correction: the RL-reasoning transfer did not hold.** At step 40 held-out door-close is 0.04 (0.52 at step 30) and
  held-out tasks 0.01, while training tasks improve (0.10; drawer-close 1.00). The reasoning still reads the instruction,
  but the moves drifted toward the training tasks' (forward and down). ML45 RL-only reasoning: nothing yet at step 10.
* **RL-only reasoning stalls:** ML10 at 0.10 on the training tasks from step 40 to 60 (held-out 0); ML45 gets no signal
  from the base model (rollouts 0.00–0.01 over 23 steps, stopped).
* **Cloning with reasons, second attempt (ML45):** one-sentence reasons restore the skill (training tasks 0.47 vs 0.49
  answer-only) but not transfer (held-out 0.05 vs 0.12). ML10's second attempt still training.
* **ML10, second reasoning clone:** 0.62 / 0.03 (answer-only 0.61 / 0.02); cloned reasoning does not reliably help transfer.
* **GRPO vs REINFORCE++ (live):** `adv_mode=reinforce_pp` (group-mean baseline, then one std over the whole batch, instead of
  GRPO's per-group std). Twins from the 400-step ML10 and ML45 clones, and RL-only reasoning on ML10.
* **GRPO vs REINFORCE++: no difference.** Twins from the 400-step clones (lr 2e-6), pooled over steps 10–40: ML10 training
  tasks 0.64 vs 0.63, held-out 0.01 vs 0.03; ML45 0.53 vs 0.52, held-out 0.13 vs 0.12 (REINFORCE++ vs GRPO). Neither erodes
  the clone at this lr. RL-only reasoning on ML10: REINFORCE++ slightly slower (0.03 vs 0.07 at step 30).
* **The clones read the instruction.** With every task given another task's instruction, the 4B clones' training-task
  success falls 0.49 → 0.11 (ML45) and 0.61 → 0.28 (ML10; sweep 1.00 → 0, window-open 0.82 → 0). Held-out failures are
  missing motions (e.g. pulling a drawer, never demonstrated in ML10), not ignored language. Qwen3.5-9B with the same
  recipe is cloning (ML45 239/400, ML10 211/400).
