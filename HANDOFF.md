# Handoff: VLM RL on Metaworld (ML10 / ML45)

State as of 2026-11-26. Read this first, then [`reports/vlmrl_results.pdf`](reports/vlmrl_results.pdf), which has every
result with figures and the reasoning behind each decision. The [docs site](docs/index.md) (`mkdocs serve`) explains the
code.

## TL;DR

- **Goal:** get a VLM policy to work on Metaworld **ML10 / ML45** with as little engineering as possible: no per-task
  code, only levers that apply to every task.
- **What works:** cloning Metaworld's scripted experts (behaviour cloning, "BC") through our text interface, on many
  task variations, then optional RL. With Qwen3.5-4B (LoRA) this gives **0.73 on ML45's 45 training tasks** (the experts
  score 0.82 through the same interface) and **0.69 on ML10's 10 training tasks**.
- **What doesn't yet:** **held-out tasks.** The best ML45 clone scores 0.12 on its 5 held-out tasks, and only on tasks that
  look like training tasks (door-lock, door-unlock). ML10's held-out tasks are ~0 for every cloned policy. The clones do
  read the instruction: with another task's instruction their training-task success falls by half or more. They fail
  held-out tasks because those need motions no training task demonstrated (ML10 never shows pulling a drawer open).
- **Best lead: state the goal once and keep the text history.** The first turn gives the goal relative to the hand in
  text, and every earlier turn's text stays in context (only the latest image). On ML10 this raised the held-out tasks
  from 0.02 to **0.22** (door-close 0.84, sweep-into 0.18) and the training tasks from 0.61 to 0.70. The policy
  dead-reckons: it replies with the remaining offset each turn, and "move the hand to the goal" carries over to
  door-close. **The stated goal does the work:** the same history without the goal in text gives 0.01 on held-out tasks
  (training 0.66). ML45 did not gain (0.07 held-out vs 0.12).
- **Finding the goal itself does not transfer.** Trained to state the goal before each move (`compact_rel_locate`, the
  estimate cloned from the true offset), the policy finds training-task goals within 1–3 cm but misses held-out ones by
  26–31 cm, and held-out success stays at 0.01–0.03. Within a training task the goal moves only 0–7 cm, so the policy
  memorizes each task's goal instead of learning to find a marker.
- **Is cloning necessary? So far yes.** RL alone learns one or two ML10 tasks (drawer-close, a little reach; ~0.10 on
  the training tasks), with 16 or 32 groups per step, and nothing on ML45.
- **Other lead:** **reasoning learned by RL.** RL from the *base* model (no cloning), with one or two
  sentences of reasoning before each move, learned drawer-close in 30 steps and solved the held-out door-close at 0.52
  (answer-only clones: 0.04). It did **not** hold: at step 40 door-close was back to 0.04 while the training tasks kept
  improving. The reasoning still read the instruction correctly, but the moves drifted toward the training tasks'.
  Making such transfer stick is open.
- **Protocol: held-out tasks are zero-shot.** No adaptation phase on the held-out tasks (no RL or cloning on them),
  by the project owner's decision; every held-out number here and in the PDF is zero-shot.

## The system in one paragraph

Each turn the model sees the task instruction, one 224×224 camera frame and a short state line (`Gripper: open. Last move:
dx=+2.0, dy=…`), and replies with a hand move in cm as JSON (`{"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}`). A
waypoint controller executes it. Earlier turns are not in the context (`max_frames=1`). Training is GRPO: groups of 8
rollouts from the same start, rewards z-scored within the group, REINFORCE on the exact sampled token ids, rollouts
served by vLLM (LoRA hot-loaded each step). The reward is a preference ranking from Metaworld's ground truth
(`pref_mw_obj`: a valid reply beats an invalid one, then success, then the object ending ≥ 1 cm closer to its goal, then
the dense reward). Behaviour cloning drives Metaworld's scripted expert (`metaworld.policies`) through the same text
interface and fine-tunes on its replies before RL.

## Repo map

| Path | What |
|---|---|
| `vlmrl/trainer.py` | `TrainConfig` (every knob), the GRPO loop, `behaviour_clone` (BC / DAgger / DART / `bc_rationale`), eval, per-task eval metrics |
| `vlmrl/rollout.py` | multi-turn rollouts (`rollout_pooled`), `expert_rollout` (scripted expert through the text interface) |
| `vlmrl/envs/metaworld.py` | the Metaworld env (batched, controller, rendering, `train_seeds`), prompt styles (`compact_rel`, `compact_rel_reason`), `oracle_action` (the scripted expert as one of our actions) |
| `vlmrl/rewards.py` | `pref_mw_obj`, `pref_mw` and the preference machinery |
| `vlmrl/policy.py`, `vlmrl/vllm_client.py` | the HF trainer policy (LoRA / full) and the vLLM rollout client |
| `configs/experiment/metaworld_ml{1,10,45}.yaml` | benchmark presets (eval sets and sizes) |
| `configs/sweeps/metaworld_recipe.yaml` | **the recipe** (all current defaults, with the evidence for each in its header) |
| `configs/policy/qwen35_4b_lora_mlp.yaml` | the policy preset used for almost everything (Qwen3.5-4B, LoRA rank 64 on the MLPs) |
| `scripts/batch_submit.py`, `scripts/job_entry.py`, `scripts/sweep.py` | cloud submission, the container entry, multi-run lanes |
| `scripts/make_results_pdf.py`, `reports/` | the results PDF (figures drawn from wandb) |
| `scripts/make_smolvlm_procgen_report.py`, `scripts/report_notes/*.md` | the wandb report |
| `tests/` | `pytest -q`; Metaworld rendering tests skip without a renderer |

## Setup on your own machines

**Hardware.** One run needs **2 GPUs**: the trainer on one, its vLLM rollout server on the other. Ours were 48 GB GPUs
(L40S); A6000 / A100 / H100 class works the same. On that pair one cloning step (64 expert episodes) takes ~1.4 min
and one RL step (128 episodes × 15 turns) ~6–8 min, so 400 cloning steps are ~9–10 h and 100 RL steps another ~10–13 h. A machine with 2k GPUs runs k experiments side by side (`--lanes k`).

**wandb.** Use your own: `wandb login` (or `WANDB_API_KEY=...`), plus `WANDB_BASE_URL=...` only for a self-hosted
server. Runs go to your default entity; set `WANDB_ENTITY=<your team>` to pick another. The environment wins over the
configs, so sweeps that still name our entity log to yours. `WANDB_MODE=offline` disables logging; everything is also
written to each run's directory (`metrics.jsonl`, eval transcripts and videos).

**Code and tests (any machine, CPU is fine):**

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[pointmaze,offline,lora,wandb,dev]"
.venv/bin/python -m pytest -q          # a few minutes; Metaworld and GPU tests skip
```

**Training: use the Docker image** (recommended). It pins both environments: the trainer, with Metaworld 3.1.1 on
mujoco 3.3 and flash-linear-attention for Qwen3.5; and vLLM, in its own venv at `$VLLM_BIN`. Getting the same stack outside
Docker is possible ([installation](docs/getting-started/installation.md)) but is where most setup time goes.

```bash
git commit ...                          # the image is built from the committed tree only
PUSH=0 docker/build.sh vlmrl:$(git rev-parse --short=12 HEAD)
docker run --rm vlmrl:<tag> python -m pytest -q -p no:cacheprovider tests/test_sweep.py tests/test_policy.py tests/test_metaworld.py
docker run --rm -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa vlmrl:<tag> python -m pytest -q tests/test_metaworld.py
```

The second test command runs the rendering tests on CPU. On a GPU host, `--gpus all` with the default `MUJOCO_GL=egl`
works too.

## Running things

**The recipe on ML10 / ML45** (cloning, then RL), on 2 local GPUs:

```bash
docker run --rm --gpus '"device=0,1"' --shm-size 32g -e WANDB_API_KEY -e WANDB_ENTITY \
    -v $PWD/out:/out -v $HOME/.cache/huggingface:/root/.cache/huggingface vlmrl:<tag> \
    python scripts/sweep.py configs/sweeps/metaworld_recipe.yaml --lanes 1 --out_root /out/runs --sync_root /out/sync \
    --only 'metaworld_ml10' --set bc_steps=400 --set lr=2e-6 --group mytest-ml10
```

- `--only` is a regex over run names. Use `'metaworld_ml1$'` for ML1: `'metaworld_ml1'` also matches ml10.
- Keys missing from a preset need `+key=value`, and keys inside `eval_sets` need `++` (Hydra struct mode).
- Re-running the same command resumes unfinished runs from `resume/state.pt`.
- Outputs land in `out/runs/<group>/<run>/`:
  - `metrics.jsonl`, every step;
  - `episodes/`, eval transcripts and videos;
  - `ckpt_bc/`, the clone, saved right after cloning;
  - `ckpt_step<N>/`, periodic checkpoints.
- `out/sync/` holds the per-step weights handed to vLLM; it can be deleted.

**Re-use a saved clone:** `--set "init_adapter='/out/runs/<group>/<run>/ckpt_bc'"` (a local dir or an `s3://` URI). Add
`--set bc_steps=N` to clone further from it, or leave cloning off to go straight to RL. **Our checkpoints stay in our
cloud storage and are not shared**: re-create a clone with the commands here (~9–10 h for 400 cloning steps).

**Eval only** (e.g. a bigger per-task eval of a clone): `--set steps=0`, the `init_adapter` above, and
`--set ++eval_sets.train_tasks.episodes=270 --set ++eval_sets.heldout_tasks.episodes=100`. Per-task success is logged as
`eval/<set>/task/<env>/success`.

**Reasoning variants:**
- Short reasoning in every reply: `env_kwargs.prompt_style=compact_rel_reason`, `max_new_tokens=256`.
- RL-only reasoning (the most promising thread): that, plus no `bc_steps`, `lr=1e-5`, `temperature=1.0`.
- Cloning with reasons: that, plus `bc_rationale=true` (the base model writes a reason for each expert move).
- Native thinking (`chat_template_kwargs.enable_thinking=true`) is not usable as is (below).
- Goal stated in text once, with the text history kept: `env_kwargs.prompt_style=compact_rel_goal --set max_frames=null
  --set max_images=1`. Only the latest image is in context; every earlier turn's text stays.
- Goal read from the image: `env_kwargs.prompt_style=compact_rel_locate`. Each reply states the goal offset, then the
  move; cloning trains the estimate on the true offset, and evals log its error (`eval/<set>/goal_err_cm`).
- Fewer training tasks: `+env_kwargs.train_tasks=[door-open,dial-turn,...]` (training and the training-task eval use
  only these; held-out tasks unchanged).
- Zero-shot probe of any base model, vLLM only (no trainer copy, so 27B+ fits): `scripts/probe_zero_shot.py`.
- Diagnostics: `+env_kwargs.swap_instructions=true` (each task gets another task's instruction) and
  `+env_kwargs.goal_marker_scale=0` (hides the goal markers; 1.0 is Metaworld's size).

**Several runs at once:** a sweep's grid (or several `--only` patterns) on `--lanes k` with 2k GPUs; lane i uses GPUs 2i
(trainer) and 2i+1 (vLLM). `--gpus_per_lane 1` shares one GPU between both, for small models only.

**Cloud (optional).** The same image runs on AWS Batch or SageMaker; see [docs/training/batch.md](docs/training/batch.md)
and [docs/training/sagemaker.md](docs/training/sagemaker.md). Account settings go in a git-ignored
`configs/sagemaker/local_batch.yaml`. Its `env:` block, e.g. `WANDB_BASE_URL` / `WANDB_ENTITY`, is passed to every job,
and so are `secret_env` (e.g. `WANDB_API_KEY`), copied from your shell at submit time.

## What we learned (keep these defaults)

| Setting | Value | Why (details in the PDF) |
|---|---|---|
| controller `gain` | **0.5** | at 1.0 the hand overshot and lifted while dragging; the scripted expert succeeded only 0.63 on push (0.97 at 0.5) |
| cloning steps | scale with the number of tasks | ML10: 0.10 / 0.38 / 0.52 / 0.69 on training tasks at 30 / 100 / 200 / 400 steps (saturates by 800) |
| `train_seeds` | **10** | variations from 10 benchmark seeds; on ML1 push the clone went from 0.94 / 0.28 (training / held-out positions) to 0.62 / 0.69; ML10/45 held-out tasks are other task classes, so nothing leaks |
| reward | `pref_mw_obj` | `pref_mw` (dense reward only) taught push to grip the puck and hold still |
| RL `lr` from a clone | **2e-6** | 1e-5 erodes held-out positions and decays after ~80 steps; 2e-6 keeps the clone |
| `temperature` | 1.0 from a clone, 1.2 from scratch | a clone already explores; from scratch, low-entropy models never vary their moves |
| `loss_agg` | `traj_mean` | token-mean collapsed small models (failed episodes carry most tokens) |
| `adv_min` | unset | positive-only advantages reinforce above-average *failures* and hurt ML10 |
| `adv_mode` | `zscore` (GRPO) | `reinforce_pp` (REINFORCE++-baseline: one std over the batch) performed the same from clones at lr 2e-6 (ML10 0.64 vs 0.63, ML45 0.53 vs 0.52 on training tasks) |
| DAgger | don't | the experts are threshold state machines; their labels in the clone's own states contradict each other |
| native thinking | don't | Qwen3.5-4B's thinking loops and never closes within 4096 tokens (87% invalid) |

## Gotchas

- **Evals are noisy.** vLLM greedy decoding isn't bit-exact across runs; the same clone scored 0.12, 0.16 and 0.31 on
  held-out push positions. One 32-episode eval is ±0.15. Pool evals, use the larger per-task evals, and watch the 128
  training rollouts per step.
- **Metaworld ML1 seed overlap.** Seed 1's training variations *are* seed 0's test variations. `train_seeds` drops
  anything within 1 cm of a held-out variation; keep that if you add data.
- **Cloning longer trades transfer for fit.** ML45 at 400 / 800 cloning steps: 0.57 / 0.73 on training tasks, but
  0.12 / 0.056 on held-out tasks. The checkpoint that transfers best isn't the one that scores best on training tasks.
- **A few tasks are partially observable for the VLM.** The goal is visible in the image for almost every task
  (Metaworld's hidden goal only affects the state vector, which the VLM never sees), but sweep's goal is never in view
  and dial-turn, plate-slide-side, stick-push and stick-pull's are tiny or sometimes outside the crop.
  `env_kwargs.goal_marker_scale=2.0` fixes most of them; see "Observability" in
  [docs/environments/metaworld.md](docs/environments/metaworld.md). All runs so far used markers ×1.
- **Some ML45 training tasks are expert-limited.** The scripted expert fails handle-pull-side, lever-pull and stick-push
  through our interface. Cloning cannot fix these.
- **Metaworld 3.1.1 needs mujoco 3.3** (incompatible with 4). The image pins it.
- **Concurrent contributors.** Another agent commits to `main` (the SmolVLM maze study, `takeaways:` commits). Always
  `git pull --rebase` before pushing; never force-push. The repo is public: keep account IDs, bucket names and keys out of
  committed files (they belong in `configs/sagemaker/local*.yaml`).

## State at handoff (2026-11-24)

Our runs live in our wandb and cloud storage, which you will not have access to. **Everything they showed is in
[`reports/vlmrl_results.pdf`](reports/vlmrl_results.pdf), and the run histories behind every figure are cached in
`reports/data/runs.json`** (per-step metrics of ~60 runs), so you can re-plot or re-analyse them without access.

Still running when this was written:

| Run (our wandb group) | What it tests | What to look for |
|---|---|---|
| `metaworld-rlreason-q35-ml10` | RL only with short reasoning, from the base model, 150 steps | held-out door-close was 0.52 at step 30, 0.04 at step 40, 0 at 50–60; training tasks stalled at 0.10. The same on ML45 got no signal at all and was stopped |
| `metaworld-adv-reinforce_pp-ml10-clone400` / `metaworld-adv-zscore-ml10-clone400` (and `-ml45-`) | REINFORCE++ vs GRPO advantages from the same 400-step clone, identical settings | no difference through step 40 (both keep the clone and drift slightly up); final numbers at step 60 |
| `metaworld-adv-reinforce_pp-ml10-rlreason` | REINFORCE++ for RL-only reasoning on ML10 | slightly slower than GRPO so far (0.03 vs 0.07 at step 30) |

| `metaworld-recipe-q35-9b-ml45`, `-ml10` | Qwen3.5-9B (`policy=qwen35_9b_lora_mlp`) with the same recipe | clones: ML45 0.51 / 0.08 (4B 0.49 / 0.12), ML10 0.67 / 0.02 (4B 0.61 / 0.02): slightly better fit, no transfer gain from scale |
| `metaworld-swap-ml45-clone400`, `metaworld-swap-ml10-clone400` (done) | the 4B clones evaluated with swapped instructions (`env_kwargs.swap_instructions=true`) | training tasks fell 0.49 → 0.11 (ML45) and 0.61 → 0.28 (ML10): the clones read the instruction |

| `metaworld-goalhist-q35-ml10`, `-ml45` | goal offset in text at the start + text history (only the latest image), 400 cloning steps then RL at lr 2e-6 | (done) ML10: held-out 0.22 over steps 0–60 (one-frame clone 0.02; door-close 0.81, sweep-into 0.20), training 0.70 at the clone (0.61), 0.71 with RL. ML45: held-out 0.03 at the clone, 0.12 by RL step 60 (one-frame clone 0.12), training 0.61 → 0.67 (0.49) |
| `metaworld-locate-q35-ml10`, `metaworld-locatehist-q35-ml10` | `compact_rel_locate`: each reply states the goal offset, then the move; cloning trains the estimate on the true offset; nothing about the goal is given. One frame / text history. 400 cloning steps, then RL at lr 2e-6 | clones: training 0.65 / 0.69 (one frame / history), held-out 0.03 / 0.01; median goal error 3.4 / 1.3 cm on training tasks, 31 / 26 cm on held-out tasks. RL running (no change through step 20) |
| `metaworld-histonly-q35-ml10` | the same text history **without** the goal in text | (done) clone: training 0.66 (reach 0.85), held-out 0.01 (door-close 0/20); after 60 RL steps 0.74 / 0.01. The history alone transfers nothing |
| `metaworld-rlonly-q35-ml10-g32` | RL only (no cloning) on ML10 with 32 groups per step (~3 per task, twice the usual), temperature 1.2, lr 1e-5, 300 steps | through step 170: training tasks 0.08–0.10 from step 40 on (drawer-close 1.0, reach 0.1–0.25), held-out 0–0.01, the same plateau as with 16 groups. Watch whether it moves by step 300 |
| `metaworld-nomarker-ml45-clone400` (done) | the 4B ML45 clone evaluated with the goal markers hidden | training tasks 0.49 → 0.40: point-goal and opening/pulling tasks need the marker, closing/pressing and held-out tasks do not |

Finished just before handoff: cloning with one-sentence reasons (`metaworld-reasonbc2-q35-ml10`, `-ml45`) matched the answer-only clones on training tasks (ML10 0.62, ML45 0.47) but not better on held-out tasks (0.03, 0.05).

The project owner can add their final numbers to the PDF.

Numbers to reproduce on your machines (Qwen3.5-4B, LoRA on the MLPs):

| Setup | Training tasks | Held-out tasks |
|---|---|---|
| ML10, 400 cloning steps, `train_seeds=10` | 0.69 | 0.00–0.02 |
| ML45, 400 cloning steps | 0.57 | 0.12 (door-lock 0.44, door-unlock 0.20) |
| ML45, 800 cloning steps | 0.73 | 0.06 |
| ML10, goal once in text + text history, 400 cloning steps | 0.70 | 0.22 (door-close 0.81, sweep-into 0.20) |
| ML10, text history without the goal, 400 cloning steps | 0.66 | 0.01 |
| ML10, policy states the goal itself (`compact_rel_locate`), one frame / text history | 0.65 / 0.69 | 0.03 / 0.01 (goal error 31 / 26 cm; training tasks 3.4 / 1.3 cm) |
| ML10, RL only with reasoning, step 30 / 40 | 0.07 / 0.10 (drawer-close 0.93 / 1.00) | 0.12 / 0.01 (door-close 0.52 / 0.04) |

## Suggested next steps

1. **Read "What would improve held-out tasks?" in the PDF first.** Diagnosis: imitation learns to replay, task by task
   (motions and perception), as π₀.₅ does in PEAR's LIBERO-Pro comparison (0.75% on unseen tasks vs PEAR's 57.4%).
   The route with evidence of zero-shot task transfer is PEAR's: RL only, from a base model that can reason, with long
   reasoning. Running at handoff: Qwen3.5-9B RL-only reasoning on ML10 (`metaworld-rlreason-q35-9b-ml10`), zero-shot
   thinking probes of Qwen3.5-9B / 27B / 35B-A3B (`vlmrl-probe-think-*-ml10`, results in each job's S3 prefix as
   `metrics.json`), and clones on 10 / 20 random and 10 look-alike ML45 training tasks (`metaworld-ntasks-ml45-*`:
   does transfer follow the number of tasks or their coverage of the held-out motions?).
2. **Make the RL-reasoning transfer stick.** It appeared (door-close 0.52 at step 30) and vanished by step 40 as RL
   specialized the motion to the training tasks. It is cheap to rerun: no cloning, ~30–60 RL steps on ML10 (~4–8 h;
   `compact_rel_reason`, `lr=1e-5`, `temperature=1.0`, evals every 5 steps with the larger eval sets). Try a lower lr, a
   broader task mix (ML45), and selecting checkpoints on held-out performance.
3. **Combine skill and reasoning.** Start RL *with* `compact_rel_reason` from the **answer-only** clone (motor skill from
   cloning), not from scratch, and check whether held-out tasks rise while training tasks stay near 0.69. The first
   cloning-with-reasons attempt lost precise manipulation (pick-place 0.73 → 0.13), so watch both.
4. **ML45 with RL-learned reasoning.** 45 training tasks gave cloning its only held-out transfer; see whether RL
   reasoning adds to it.
5. **Bigger, pooled evals** for every claim (`++eval_sets.*.episodes`, eval every 10 steps).

## Keeping the results current

- **PDF:** text in `reports/vlmrl_results.tex`, runs in the `RUNS` table of `scripts/make_results_pdf.py`.
  - A `"group:<wandb group>"` entry picks that group's newest run, so a run can be registered before it starts.
  - `uv run scripts/make_results_pdf.py` rebuilds from the cache, offline. This needs pdflatex; matplotlib comes from
    `uv`'s throwaway environment.
  - `--fetch --entity <you> --metaworld_project <project>` refreshes from your wandb. It reads the runs it can, keeps the
    cached copies of the rest (ours), and so mixes yours with ours.
- **wandb report:** `scripts/make_smolvlm_procgen_report.py` builds our report from our runs (groups and run IDs in its
  tables). With another entity it creates a new report (`--entity <you> --project <project>`); swap in your own groups
  first. The PDF is the summary to keep up to date.
