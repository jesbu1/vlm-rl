# Handoff: VLM RL on Metaworld (ML10 / ML45)

State as of 2026-11-24. Read this first, then [`reports/vlmrl_results.pdf`](reports/vlmrl_results.pdf), which has every
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
- **Most interesting lead:** **reasoning learned by RL.** RL from the *base* model (no cloning), with one or two
  sentences of reasoning before each move, learned drawer-close in 30 steps and solved the held-out door-close at 0.52
  (answer-only clones: 0.04). It did **not** hold: at step 40 door-close was back to 0.04 while the training tasks kept
  improving. The reasoning still read the instruction correctly, but the moves drifted toward the training tasks'.
  Making such transfer stick is open.
- **Decision pending with the project owner:** whether to report held-out tasks after a short adaptation phase (a few RL
  steps per held-out task, the benchmark's own meta-RL protocol) or zero-shot only, as now.

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

| `metaworld-recipe-q35-9b-ml45`, `-ml10` | Qwen3.5-9B (`policy=qwen35_9b_lora_mlp`) with the same recipe | does 2.3× the parameters transfer better than the 4B (ML45 0.12, ML10 0.02 held-out)? |
| `metaworld-swap-ml45-clone400`, `metaworld-swap-ml10-clone400` (done) | the 4B clones evaluated with swapped instructions (`env_kwargs.swap_instructions=true`) | training tasks fell 0.49 → 0.11 (ML45) and 0.61 → 0.28 (ML10): the clones read the instruction |

Finished just before handoff: cloning with one-sentence reasons (`metaworld-reasonbc2-q35-ml10`, `-ml45`) matched the answer-only clones on training tasks (ML10 0.62, ML45 0.47) but not better on held-out tasks (0.03, 0.05).

The project owner can add their final numbers to the PDF.

Numbers to reproduce on your machines (Qwen3.5-4B, LoRA on the MLPs):

| Setup | Training tasks | Held-out tasks |
|---|---|---|
| ML10, 400 cloning steps, `train_seeds=10` | 0.69 | 0.00–0.02 |
| ML45, 400 cloning steps | 0.57 | 0.12 (door-lock 0.44, door-unlock 0.20) |
| ML45, 800 cloning steps | 0.73 | 0.06 |
| ML10, RL only with reasoning, step 30 / 40 | 0.07 / 0.10 (drawer-close 0.93 / 1.00) | 0.12 / 0.01 (door-close 0.52 / 0.04) |

## Suggested next steps

1. **Make the RL-reasoning transfer stick.** It appeared (door-close 0.52 at step 30) and vanished by step 40 as RL
   specialized the motion to the training tasks. It is cheap to rerun: no cloning, ~30–60 RL steps on ML10 (~4–8 h;
   `compact_rel_reason`, `lr=1e-5`, `temperature=1.0`, evals every 5 steps with the larger eval sets). Try a lower lr, a
   broader task mix (ML45), and selecting checkpoints on held-out performance.
2. **Combine skill and reasoning.** Start RL *with* `compact_rel_reason` from the **answer-only** clone (motor skill from
   cloning), not from scratch, and check whether held-out tasks rise while training tasks stay near 0.69. The first
   cloning-with-reasons attempt lost precise manipulation (pick-place 0.73 → 0.13), so watch both.
3. **ML45 with RL-learned reasoning.** 45 training tasks gave cloning its only held-out transfer; see whether RL
   reasoning adds to it.
4. **Held-out adaptation** (pending the owner's decision): one ML1 run per held-out task (`env_kwargs.benchmark='ML1:door-lock-v3'`,
   …) from a saved clone, a few RL steps each; report success against episodes used.
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
