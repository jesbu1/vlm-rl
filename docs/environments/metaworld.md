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
protocol** (format -> parse -> execute), 15 turns, 2-3 episodes per task:

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
scratch gets no signal.

The experts are tuned for closed-loop control at every simulator step. Through coarse waypoints, a binary gripper and 15
turns, precise grasp-and-place tasks lose the most. Treat these as the ceiling of the protocol, not of a learned policy.

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
