# Metaworld

[Metaworld](https://github.com/Farama-Foundation/Metaworld) (Farama) tabletop manipulation with a Sawyer arm, through
the same PEAR text protocol as PointMaze: `vlmrl/envs/metaworld.py`, `env=metaworld` (`task=metaworld`).

```bash
uv pip install -e ".[pointmaze,metaworld,dev]"     # pins mujoco 3.3.0 (see Install)
python scripts/train.py task=metaworld env_kwargs.benchmark=ML10 ...
```

## Benchmarks and splits

| `env_kwargs.benchmark` | training split | held-out split (`eval_split: test`) |
|---|---|---|
| `ML1:<task>` (e.g. `ML1:push-v3`) | 50 goal / object positions of one task | 50 *different* positions of the same task |
| `ML10` | 10 tasks (50 variations each) | 5 different tasks |
| `ML45` | 45 tasks | 5 different tasks |

* Start states are `{benchmark, split, env_name, task, seed}`: a task variation index plus a reset seed, so every
  episode is exactly reproducible (`scripts`-free reset-to-task, as with PointMaze's reset-to-state).
* `TaskSampler` draws a task uniformly from the split, then a variation and a seed.
* The held-out split is Metaworld's own; with [eval sets](pointmaze.md#eval-sets-in-and-out-of-distribution) one run
  reports training and held-out tasks side by side (`experiment=metaworld_zeroshot` evaluates ML1, ML10 and ML45 at once).

## The robot, in PEAR terms

```json
{"hand": {"x": 0.46, "y": 60.14, "z": 19.51}, "gripper": {"opening": 1.0}}
```

* `hand` is the end effector in **centimeters** (z up); `gripper.opening` is 1 open, ~0.3 closed.
* An action is the **change** of every quantity. The hand moves toward the target with Metaworld's own controls (at
  most 1 cm per simulator step) until it is **reached** (0.5 cm), **blocked** (no progress over 6 steps), or the
  turn's budget (`max_substeps`, 40) is used, the PEAR EXECUTION rule. Moves are capped at `max_delta` (10 cm).
* A negative `opening` change closes the gripper, a positive one opens it; the command is held until changed, and a
  changed command gets at least 12 simulator steps (the fingers are slow).
* Episodes end on Metaworld's `info["success"]`, after `max_turns` (15), or at the benchmark's 500-step path limit.
  `reward=success` is sparse; the summed dense Metaworld reward is in `info["mw_reward"]`.

## Observations and rendering

* Frames are rendered at **640×480** and **center-cropped to 224×224** (`render_width`, `render_height`,
  `image_size`, `crop: center`). The crop zooms on the workspace. Across task variations and along expert trajectories
  (reach, push, pick-place, shelf-place, box-close, basketball) goals, objects and the gripper stay in view. The
  basketball dunk's top position touches the upper edge. `crop: resize` scales the short side to 224 first: the whole
  scene, but objects are only a few pixels.
* The `corner` camera is used by default. It is rolled 180° (its "up" axis points down), so the env rotates the frame 180°. Runs before 2026-11-19 09:00 only flipped rows, which mirrored every frame left-right (`test_corner_view_is_rotated_not_mirrored` checks the rotation against the camera's projection). In the image, +x moves the hand right, +y left (both slightly away from the camera), +z up.
* Rendering is headless MuJoCo: `MUJOCO_GL=egl` on GPU machines (the image sets
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`); `MUJOCO_GL=osmesa` works on CPU (slower).
* The goal position is hidden in ML benchmarks (Metaworld's `_partially_observable`): the VLM sees it only in the image.

## Prompts

| `prompt_style` | each turn | reply |
|---|---|---|
| `pear` (default) | state JSON + image; task instruction in the system prompt's TASK section | fenced JSON change of `hand.x/y/z` and `gripper.opening` |
| `compact` | image + `Hand: x=..., y=..., z=.... Gripper: open. Move:` | `{"x": 2.0, "y": -1.0, "z": 0.0, "grip": "close"}` |
| `compact_rel` | image + `Gripper: open. Last move: dx=+2.0, dy=-1.0, dz=+0.0. Move:`; no absolute coordinates anywhere, axis directions in the image in the system prompt; the goal is only in the image | `{"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}` |

With `compact`, Qwen3-VL-2B copies the hand's absolute position as its move; `compact_rel` gives it nothing absolute to copy.

Metaworld has no language annotations: `metaworld.py::task_text` returns one instruction per task (`TASK_TEXT`, all 50, checked by hand).

## Scripted experts as an upper bound

`metaworld.py::oracle_action` turns Metaworld's scripted expert for each task into one of our actions: the target it
steers toward (recorded from its `move(hand, to_xyz)` call) and the sign of its grab effort. Driven **through the text
protocol** (format -> parse -> execute), 15 turns.

**Controller gain** (`env_kwargs.gain`, default 0.5). Each simulator step commands `clip(gain · (target − hand) [cm],
−1, 1)`. Runs before 2026-11-22 used 1.0, which runs at full speed until 1 cm from the waypoint, overshoots, and lifts
the hand while it drags an object (the puck slips out). Expert success, 8 episodes per task (30 for push):

| | gain 1.0 | gain 0.5 |
|---|---|---|
| ML1 push, training positions | 0.63 | **0.97** (in 5.9 turns instead of 9.4) |
| ML10 training tasks | 0.85 | **0.91** (push 0.62 → 0.88, pick-place 0.75 → 1.0, basketball 0.62 → 0.88; door-open 0.88 → 0.62) |
| ML10 held-out tasks | 0.42 | **0.72** (shelf-place 0 → 0.75, sweep-into 0 → 0.62) |

At gain 1.0, 2-3 episodes per task (an earlier, smaller sample):

| benchmark | split | expert success | weakest tasks |
|---|---|---|---|
| ML1 push | held-out positions | 1.00 | |
| ML1 pick-place | held-out positions | 0.50 | |
| ML10 | training tasks | 0.75 | peg-insert-side 0.0; basketball, door-open, pick-place 0.5 |
| ML10 | held-out tasks | 0.60 | shelf-place 0.0, sweep-into 0.5 |
| ML45 | held-out tasks | 0.70 | bin-picking 0.0 |

**Behaviour cloning before RL:** with `bc_steps=N` the trainer first runs N updates on the experts' transcripts
(`rollout.expert_rollout`: each turn the expert's action is written as the model's reply, so the transcripts have
exactly the RL rollouts' format). It uses `bc_episodes` episodes per step, the successful ones only by default, at
`bc_lr`. Then RL starts from the cloned policy; the step-0 eval measures the clone. The motivation: push and the less
exploratory models never sample the motion they need (no sideways moves, or one fixed move every turn), so RL from
scratch gets no signal. The clone is saved as `ckpt_bc` (reusable with `init_adapter`).

**DAgger** (`bc_dagger_steps=K`): the last K of the `bc_steps` roll out the current clone instead of the expert; the
expert labels every state the clone reaches, and those labels are trained on (failed episodes included). A pure clone
of push succeeds 0.22 where its expert succeeds 0.61: it keeps sending the drop move after the hand is already on the
table, a state the expert never visits. **DAgger made it worse** (push 0.22 → 0.00, ML10 0.12 → 0.04 after 20
steps; log-prob of the labels −0.09 → −0.50). The scripted experts are state machines with sharp thresholds on
privileged state (push: more than 2 cm from the puck in x/y means "go back up 20 cm"), so in the clone's states nearly
identical images get opposite labels.

**DART** (`bc_noise=σ`): the expert drives, with Gaussian noise of σ cm added to each executed move; the transcript
keeps the expert's clean reply. The states stay near the expert's path (consistent labels) but include small slips.

The experts are tuned for closed-loop control at every simulator step. Through coarse waypoints, a binary gripper and 15
turns, precise grasp-and-place tasks lose the most. Treat these as the ceiling of the protocol, not of a learned policy.

**Cloning with reasons** (`bc_rationale=true`, with `prompt_style=compact_rel_reason`): the experts' replies carry no
reasoning, so plain cloning trains answer-only replies. With `bc_rationale`, every expert reply is first prefixed
with one or two sentences written by the rollout model when shown the observation and the expert's move
(STaR-style rationalization); training uses the normal prompt, so the clone states where the gripper, object and
goal are and what the task needs before each move.

**Goal in text, with history** (`env_kwargs.prompt_style=compact_rel_goal`, plus `max_frames=null max_images=1`): the
first observation states the goal relative to the hand (`Goal: dx=+12.0, dy=-3.5, dz=+4.0 from the hand.`, the same
information the goal marker draws), and the whole text history stays in context while only the latest image is
kept, so the offset stays usable and the policy can see what its earlier moves did.

**Reasoning variant** (`env_kwargs.prompt_style=compact_rel_reason`): like `compact_rel`, but each reply first gives one
or two sentences (where the gripper, the object and the goal are, what to do next) and then the JSON move; the last
`{...}` in the reply is parsed, so axis names in the reasoning are never read as the move. Use `max_new_tokens` ≈ 256.
Native thinking (`chat_template_kwargs.enable_thinking=true`) is the other option; Qwen3.5-4B's thinking often runs
past 2048 tokens.

## Observability: is it an MDP?

Metaworld's ML1 / ML10 / ML45 hide the goal only in the 39-number **state vector** (`_partially_observable` zeros its last
three entries). The VLM never sees that vector: it sees the camera crop, the gripper state and its last move. Goals are
drawn as markers (sites named `*goal*`, opaque in every task's XML) and are placed at the true target on reset (47 of 50
tasks; door-lock / door-unlock only from the first step) and after every step. Whether the VLM faces an MDP is therefore
whether the goal is visible in the 224×224 crop. Measured by hiding the markers and counting changed pixels (3 variations
per task, the image after one step):

| Group | Tasks | Changed pixels (centre crop, markers ×1) |
|---|---|---|
| clearly visible goal | reach(-wall), push(-back/-wall), pick-place(-wall), shelf-place, soccer, basketball, box-close, coffee-*, lever-pull, pick-out-of-hole, plate-slide(-back, -back-side), window-*, door-open/close, drawer-open, sweep-into, peg-unplug-side, handle-*, faucet-* | 25–450 |
| goal given by the scene (no marker) | assembly, disassemble, bin-picking, button-press(-*) | — |
| marker hidden inside/behind an object that shows the goal | drawer-close, hammer, peg-insert-side (the hole), hand-insert (the hole), door-lock / door-unlock (the lock) | 0 |
| **goal varies but is hardly visible** | dial-turn (17), plate-slide-side (7–19), stick-push (6–13), stick-pull (0–53: outside the crop in some variations) | small |
| **never visible** | sweep (goal at the table edge, varies ±2.9 cm) | 0 |

`env_kwargs.goal_marker_scale=2.0` (draws the goal markers twice as large) makes the small ones clearly visible (dial-turn
293+, plate-slide-side 72+, stick-push 109+ pixels) without shrinking the objects; `crop=resize` (the whole frame, scaled)
is the only setting that keeps stick-pull's goal in view in every variation, but it shrinks everything about 4×. Sweep's
marker is not visible in any setting. Runs so far used markers ×1 and the centre crop. A single frame with no history is
also not strictly Markov (depth from one view, no velocities), a milder gap than a hidden goal.

## Install

Metaworld 3.1.1 is **incompatible with mujoco 4** (its mocap-weld setup writes 11 values into mujoco 4's 12-value
constraint data), so the `metaworld` extra pins `mujoco==3.3.0`; PointMaze works on 3.3.0 too (all tests pass). When
downgrading an existing environment, remove mujoco 4's split packages first:

```bash
uv pip uninstall mujoco mujoco-core mujoco-studio mujoco-studio-wasm
uv pip install -e ".[pointmaze,metaworld]"
```

The Docker image installs mujoco 3.3.0 + Metaworld and the EGL / OSMesa libraries.

## Presets and runs

* `task=metaworld`: ML1 push, held-out eval, PEAR prompt, 15 turns.
* `experiment=metaworld_zeroshot`: eval only (`steps=0`) on ML1 push train / test, ML10 train / test and ML45 test.
* `configs/sweeps/metaworld_zeroshot.yaml`: Qwen3-VL-2B, Gemma 4 E2B and Qwen3-VL-4B x {pear, compact}.

Tests: `tests/test_metaworld.py` (splits, instructions, exact resets, hand / gripper execution, prompt round trips,
the scripted expert through the protocol, 640x480 -> 224 rendering); skipped when Metaworld is not installed.
