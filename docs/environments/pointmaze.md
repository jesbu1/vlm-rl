# PointMaze

`env=pointmaze` (`task=pointmaze`). Source: [`vlmrl/envs/pointmaze.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/envs/pointmaze.py).

![large maze](../assets/large_frame.png){ width="280" } ![oracle on medium](../assets/medium_oracle.gif){ width="280" }

## Dynamics

gymnasium-robotics `PointMazeEnv`: a MuJoCo point mass with force actions in \([-1,1]^2\),
\(dt = 0.01\), and success when \(\lVert pos - goal\rVert \le 0.45\). This is the same physics that generated
the D4RL/Minari `pointmaze` datasets. `PointMazeSimVecEnv` keeps a pool of these and supports
**exact reset-to-state** (`{"qpos", "qvel", "goal"}` via `set_state` plus a goal-site update). Episodes
end on success (`continuing_task=False`).

## The robot, in PEAR terms

One part, `ball`, with settable quantities `x, y` (maze units; one cell = 1; x right, y up, (0, 0) at
the image centre). The action is the **change** `{"ball": {"x": dx, "y": dy}}`, clipped to length
`max_delta` (1.0). See [Prompts](prompts.md) for the full text.

## Execution: until reached and at rest

Following PEAR's EXECUTION section (`settle=true`, default), a PD controller (kp = 10, kd = 1) drives raw
forces toward `pos + delta` **until**:

| `info["stop"]` | condition |
|---|---|
| `reached` | ‖target − pos‖ < 0.05 **and** ‖v‖ < 0.1 |
| `blocked` | moved < 0.01 over the last 20 physics steps (e.g. pushing into a wall) |
| `success` | entered the goal radius (episode ends) |
| `steps` | hit the 200-substep cap |

A free 1-cell move settles in ≈42 substeps. With `settle=false`, every action runs exactly
`action_repeat` (40) substeps. The executed force sequence is returned as `info["low_level_actions"]`,
which is useful as world-model training data. `action_mode=force` (holding raw forces) exists for the
`simple` prompt style only.

## Observations

`render_topdown(maze_map, agent_xy, goal_xy, size)` is a **pure function of low-dimensional state**:
dark walls, light floor, a faint cell grid, a blue agent, and a green goal disc with its success-radius
ring. Offline datasets have no images, but they can be rendered identically, which you need in
order to train a WAM on Minari data. `Obs.extras` carries `goal` and `dist`.

## Options (`env_kwargs.*`)

| key | default | meaning |
|---|---|---|
| `maze` | `umaze` | `open`, `umaze`, `medium`, `large`, `open_diverse_gr`, `medium_diverse_gr`, `large_diverse_gr` |
| `max_turns` | 10 (task preset) | step limit per episode |
| `image_size` | 336 | rendered frame size (px) |
| `include_state` | true | state JSON in each turn and goal coordinates in the TASK; false = image-only |
| `prompt_style` | `pear` | `pear` (paper protocol) or `simple` (`<action>{"dx","dy"}</action>`) |
| `settle` | true | PEAR execution semantics (above) |
| `action_mode` | `delta` | `force` only with `prompt_style=simple` |
| `max_delta` | 1.0 | maximum move per action |
| `start_states` | `random` | `random`, a Minari id, or a `.npz` path |
| `eval_start_states` | `random` | same, for the fixed eval set |
| `min_goal_dist` | 1.5 | minimum start–goal distance |
| `split` | `none` | train / held-out split of random starts: `none`, `pair`, `goal` (below) |
| `holdout_frac` | 0.25 | fraction of start–goal cell pairs (`pair`) or goal cells (`goal`) held out of training |
| `split_seed` | 0 | which pairs / goal cells are held out |
| `eval_split` | `heldout` | part the eval sampler draws from when `split` ≠ `none`: `heldout`, `train`, `all` |

## Start states

- **`random`**: uniform over free cells (± 0.25 noise), zero velocity, at least `min_goal_dist` apart.
- **`D4RL/pointmaze/umaze-v2`** (any Minari id): random timesteps of offline trajectories, with their
  real velocities and the dataset's goal. `episode` and `t` are kept in the state dict so a WAM can
  look up the matching reference frames. Verified against the real UMaze dataset (13,210 episodes, 1M steps).
- **`path/to/file.npz`** with arrays `qpos [N,2]`, `qvel [N,2]`, `goal [N,2]`. Extra arrays pass through.

## Train / held-out split

By default, training and eval draw random starts from the same distribution: eval episodes are new
samples, but any start–goal combination can also show up in training. `split` partitions the
(start cell, goal cell) pairs (`vlmrl/envs/pointmaze.py::split_pairs`, deterministic given `split_seed`):

- **`pair`** holds out `holdout_frac` of the pairs. Every cell appears in training, but some
  start–goal combinations never do.
- **`goal`** holds out `holdout_frac` of the goal cells. Training never asks the ball to reach them.

The train sampler draws only from the train part. The eval sampler draws from `eval_split`. Position
noise stays inside the cell, so a held-out pair is never produced by the train part. Sizes at
`min_goal_dist=1.5`, `holdout_frac=0.25`:

| maze | `pair`: train / held-out pairs | `goal`: train / held-out pairs |
|---|---|---|
| umaze | 20 / 6 | 17 / 9 (2 of 7 goal cells) |
| medium | 411 / 137 | 424 / 124 |
| large | 1442 / 480 | 1420 / 502 |

## Eval sets (in and out of distribution)

`eval_sets` names several eval environments. Each one is a set of `env_kwargs` overrides plus an optional
`episodes` count (default `eval_episodes`), for example `experiment=pointmaze_generalization`:

```yaml
env_kwargs: {maze: umaze, split: goal}           # training: UMaze, 2 of 7 goal cells held out
eval_sets:
  id:         {eval_split: train}                # in distribution: new episodes, goal cells seen in training
  ood_goals:  {eval_split: heldout}              # held-out goal cells
  ood_medium: {maze: medium, eval_split: all, max_turns: 20}   # unseen layout
```

From the command line: `+eval_sets.ood_medium.maze=medium +eval_sets.ood_medium.eval_split=all`.

- Each set gets its own env, prompt and fixed start states, drawn once from its eval sampler.
- **Pooled:** all sets run in one rollout loop (`vlmrl/rollout.py::rollout_pooled`). Each round, the
  unfinished episodes of every set go into a single generation call, so vLLM batches across sets. Each set
  then steps its own env, which is batched over that set's active episodes.
- **Logging:** `eval/<name>/{success,turns,invalid_frac,end_*}` per set, and `eval/success` etc. as the
  unweighted mean over sets (overall success = mean of the per-set success rates). Media per set:
  `eval/<name>/episode` (video), `eval/<name>/transcript`, `eval/<name>/samples`.
- Without `eval_sets`, there is one eval set built from `env_kwargs`, logged as `eval/*` as before.

## Oracle and difficulty

`oracle_delta(maze_map, pos, goal)` runs BFS over cells and aims at the next cell centre. Through the
text protocol, over 30 rounds:

| maze | oracle success / mean rounds | random-waypoint success |
|---|---|---|
| umaze | 1.00 / 3.8 | 0.25 |
| medium | 1.00 / 6.1 | 0.12 |
| large | 1.00 / 8.9 | 0.00 |

UMaze saturates quickly under RL (see the [experiment log](../training/experiments.md)). `medium` and
`large` are where differences between models and rewards show up.
