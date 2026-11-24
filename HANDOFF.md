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
  look like training tasks (door-lock, door-unlock). ML10's held-out tasks are ~0 for every cloned policy. The instruction
  does not steer the motion: ML10's clone cannot do drawer-open / door-close, the reverses of its training tasks.
- **Most promising thread:** **reasoning learned by RL.** RL from the *base* model (no cloning), with one or two
  sentences of reasoning before each move, learned drawer-close (0.93) in 30 steps and solved the held-out door-close at
  **0.52** (answer-only clones: 0.04). That is a single eval so far; confirm it first (see "State at handoff").
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

## Setup

**Local (CPU), for code and tests.**

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[pointmaze,offline,lora,wandb,dev]"
.venv/bin/python -m pytest -q          # ~5 min; tests that need Metaworld or a GPU skip
```

Metaworld is not needed locally. It runs in the image, and the image's tests cover it.

**Image.** Everything on the cluster runs one Docker image built from the *committed* tree and tagged with the 12-character
commit hash. Commit, then build on an x86 machine with Docker, test inside the image, and push:

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <registry>
IMG=$(.venv/bin/python scripts/batch_submit.py --image)            # <registry>/<repo>:<commit>
PUSH=0 docker/build.sh $IMG
docker run --rm $IMG python -m pytest -q -p no:cacheprovider tests/test_sweep.py tests/test_policy.py tests/test_metaworld.py
docker run --rm -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa $IMG python -m pytest -q tests/test_metaworld.py   # rendering tests on CPU
docker push $IMG
```

**Cloud (AWS Batch).** Follow [docs/training/batch.md](docs/training/batch.md). Account-specific settings (queue, S3
output prefix, role, tags) go in `configs/sagemaker/local_batch.yaml`, which is git-ignored; ask the project owner for
theirs. Each run uses one lane of 2 GPUs (trainer + vLLM), so `--gpus 2 --gpus_per_lane 2` puts one run on each job.

**wandb.** Runs log to a self-hosted wandb server: set `WANDB_BASE_URL` to it and `WANDB_API_KEY` to your key, both when
submitting (the job inherits the key) and when building the report or the PDF. Projects: `tri/vlm-rl-metaworld`
(everything here) and `tri/vlm-rl-pointmaze` (the maze work).

## Running things

**The recipe on ML10 / ML45** (cloning, then RL; ~5 h for ML10 with 400 cloning steps, ~10 h for ML45):

```bash
export WANDB_API_KEY=...       # and WANDB_BASE_URL
python scripts/batch_submit.py configs/sweeps/metaworld_recipe.yaml --tag <image tag> \
    --gpus 2 --gpus_per_lane 2 --vcpus 24 --memory_mib 170000 \
    --only 'metaworld_ml45' --set bc_steps=400 --set lr=2e-6 \
    --group metaworld-mytest-ml45 --name vlmrl-mytest-ml45
```

- `--only` is a regex over run names (`'metaworld_ml1$'` for ML1; `'metaworld_ml1'` also matches ml10).
- Keys missing from a preset need `+key=value`; keys inside `eval_sets` need `++` (Hydra struct mode).
- Outputs sync to `<s3_output>/<job name>/<group>/<run>/`: `metrics.jsonl` (every step), `episodes/` (eval transcripts and
  videos), `ckpt_bc/` (the clone, right after cloning), `ckpt_step<N>/`, `resume/state.pt` (a resubmitted job resumes).

**Re-use a saved clone:** `--set "init_adapter='<s3_output>/<job>/<group>/<run>/ckpt_bc'"`. Add `--set bc_steps=N`
to clone further from it, or leave cloning off to go straight to RL.

**Eval only** (e.g. a bigger per-task eval of a clone): `--set steps=0`, the `init_adapter` above, and
`--set ++eval_sets.train_tasks.episodes=270 --set ++eval_sets.heldout_tasks.episodes=100`. Per-task success is logged as
`eval/<set>/task/<env>/success`.

**Reasoning variants:**
- Short reasoning in every reply: `env_kwargs.prompt_style=compact_rel_reason`, `max_new_tokens=256`.
- RL-only reasoning: that, plus no `bc_steps`, `lr=1e-5`, `temperature=1.0`.
- Cloning with reasons: that, plus `bc_rationale=true` (the base model writes a reason for each expert move).
- Native thinking (`chat_template_kwargs.enable_thinking=true`) is not usable as is (below).

**Check on runs:**

```bash
aws s3 cp <s3_output>/<job>/<group>/<run>/metrics.jsonl - | tail -3
```

Or use the wandb group. Batch shows cancelled jobs as `FAILED`.

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
- **Some ML45 training tasks are expert-limited.** The scripted expert fails handle-pull-side, lever-pull and stick-push
  through our interface. Cloning cannot fix these.
- **Metaworld 3.1.1 needs mujoco 3.3** (incompatible with 4). The image pins it.
- **Concurrent contributors.** Another agent commits to `main` (the SmolVLM maze study, `takeaways:` commits). Always
  `git pull --rebase` before pushing; never force-push. The repo is public: keep account IDs, bucket names and keys out of
  committed files (they belong in `configs/sagemaker/local*.yaml`).

## State at handoff (2026-11-24)

Running in `tri/vlm-rl-metaworld` (wandb group = what to open):

| wandb group | What it tests | What to look for |
|---|---|---|
| `metaworld-rlreason-q35-ml10` | RL only with short reasoning, from the base model, 150 steps | **does held-out door-close (0.52 at step 30) hold at steps 40+?** Training tasks were only 0.07 |
| `metaworld-rlreason-q35-ml45` | the same on ML45 | any held-out task beyond door-lock / door-unlock |
| `metaworld-reasonbc2-q35-ml10`, `-ml45` | cloning with one-sentence reasons (second attempt), then RL with reasoning | held-out vs the answer-only clones (ML10 0.02, door-close 0.04; ML45 0.12) |

Best saved clones (`ckpt_bc` under `<s3_output>/<job>/<group>/<run>/`):

| Clone | job / group | Training / held-out tasks |
|---|---|---|
| ML45, 400 steps (best transfer) | `vlmrl-recipe-q35-ml45` / `metaworld-recipe-q35-ml45` | 0.57 / 0.12 |
| ML45, 800 steps (best fit) | `vlmrl-recipe-q35-ml45-bc800` / `metaworld-recipe-q35-ml45-bc800` | 0.73 / 0.06 |
| ML10, 400 steps | `vlmrl-recipe-q35-ml10-bc400` / `metaworld-recipe-q35-ml10-bc400` | 0.69 / 0.00–0.02 |

## Suggested next steps

1. **Confirm the RL-reasoning transfer.** Pool the door-close evals of `metaworld-rlreason-q35-ml10` over steps 30–60. If
   it holds, this is the headline result.
2. **Combine skill and reasoning.** Start RL *with* `compact_rel_reason` from the **answer-only** clone (motor skill from
   cloning), not from scratch, and check whether held-out tasks rise while training tasks stay near 0.69. The first
   cloning-with-reasons attempt lost precise manipulation (pick-place 0.73 → 0.13), so watch both.
3. **ML45 with RL-learned reasoning.** 45 training tasks gave cloning its only held-out transfer; see whether RL
   reasoning adds to it.
4. **Held-out adaptation** (pending the owner's decision): one ML1 run per held-out task (`env_kwargs.benchmark='ML1:door-lock-v3'`,
   …) from a saved clone, a few RL steps each; report success against episodes used.
5. **Bigger, pooled evals** for every claim (`++eval_sets.*.episodes`, eval every 10 steps).

## Keeping the results current

- **PDF:** text in `reports/vlmrl_results.tex`, runs in the `RUNS` table of `scripts/make_results_pdf.py` (`"group:<wandb
  group>"` picks a group's newest run, so a run can be registered before it starts). Rebuild with
  `uv run scripts/make_results_pdf.py --fetch` (needs pdflatex; matplotlib comes from `uv`'s throwaway environment).
- **wandb report:** notes in `scripts/report_notes/*.md`, panels in `scripts/make_smolvlm_procgen_report.py`; rebuild with
  `.venv/bin/python scripts/make_smolvlm_procgen_report.py`. It updates the existing report in place (never delete it).
