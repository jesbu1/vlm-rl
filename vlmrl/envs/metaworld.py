"""Metaworld (Farama) manipulation for VLM-RL: ML1 / ML10 / ML45 with their built-in train / test splits.

The robot is a Sawyer arm controlled in end-effector space. In PEAR terms it has two parts:

    {"hand": {"x": ..., "y": ..., "z": ...}, "gripper": {"opening": ...}}     # cm, opening in [0, 1]

An action is the *change* of those quantities. The hand then moves toward the target (Metaworld's own action space:
up to 1 cm per simulator step) until it is reached and at rest, blocked, or the per-turn budget is used, holding the
gripper command (close if the opening change is negative, open if positive). Episodes end on Metaworld's
``info["success"]``, the turn limit, or the benchmark's 500-step path limit.

Benchmarks (``benchmark=``): ``ML1:<task>`` (one task; train / test = disjoint goal / object positions),
``ML10`` (10 training tasks, 5 held-out tasks), ``ML45`` (45 / 5). Start states name a split, a task, a task
variation and a seed, so episodes are exactly reproducible and eval sets can use the held-out split.
Needs ``pip install -e .[metaworld]`` (pins mujoco 3.3.0: Metaworld 3.1 is incompatible with mujoco 4).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from vlmrl.envs.base import EnvBundle
from vlmrl.types import Obs, StepResult

os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering on GPU machines (falls back below)

# One-line instructions (Metaworld has no language annotations).
TASK_TEXT = {
    "assembly": "Pick up the nut and place it over the peg.",
    "basketball": "Pick up the ball and dunk it into the basket.",
    "bin-picking": "Pick up the cube in the left bin and place it in the right bin.",
    "box-close": "Pick up the lid and place it on the box to close it.",
    "button-press-topdown": "Press the button down from the top.",
    "button-press-topdown-wall": "Press the button down from the top; there is a wall next to it.",
    "button-press": "Press the button in from the front.",
    "button-press-wall": "Press the button in from the front; there is a short wall in front of it.",
    "coffee-button": "Press the button on the coffee machine.",
    "coffee-pull": "Pull the coffee mug out of the coffee machine to the goal.",
    "coffee-push": "Push the coffee mug under the coffee machine.",
    "dial-turn": "Turn the dial a full rotation counterclockwise.",
    "disassemble": "Pick the nut up and off the peg.",
    "door-close": "Close the door by pushing it shut.",
    "door-lock": "Lock the door by turning the lock knob down.",
    "door-open": "Open the door by pulling its handle.",
    "door-unlock": "Unlock the door by turning the lock knob up.",
    "drawer-close": "Push the drawer closed.",
    "drawer-open": "Pull the drawer open by its handle.",
    "faucet-close": "Turn the faucet handle counterclockwise to close it.",
    "faucet-open": "Turn the faucet handle clockwise to open it.",
    "hammer": "Pick up the hammer and hit the nail with it.",
    "hand-insert": "Move the object into the hole in the table.",
    "handle-press-side": "Press the handle down.",
    "handle-press": "Press the handle down.",
    "handle-pull-side": "Pull the handle up.",
    "handle-pull": "Pull the handle up.",
    "lever-pull": "Pull the lever up.",
    "peg-insert-side": "Pick up the peg and insert it into the hole.",
    "peg-unplug-side": "Pull the peg out of the hole.",
    "pick-out-of-hole": "Pick the object up out of the hole.",
    "pick-place": "Pick up the puck and place it at the goal.",
    "pick-place-wall": "Pick up the puck and place it at the goal behind the wall.",
    "plate-slide-back-side": "Slide the plate back towards the center.",
    "plate-slide-back": "Slide the plate back towards the center.",
    "plate-slide-side": "Slide the plate into the goal.",
    "plate-slide": "Slide the plate into the goal.",
    "push-back": "Pull the puck backward to the goal.",
    "push": "Push the puck to the goal.",
    "push-wall": "Push the puck around the wall to the goal.",
    "reach": "Move the hand to the goal position.",
    "reach-wall": "Move the hand to the goal position behind the wall.",
    "shelf-place": "Pick up the puck and place it on the shelf.",
    "soccer": "Move the ball into the goal.",
    "stick-pull": "Grab the stick and use it to pull the box to the goal.",
    "stick-push": "Grab the stick and use it to push the box to the goal.",
    "sweep-into": "Sweep the puck to the goal.",
    "sweep": "Sweep the puck to the goal.",
    "window-close": "Close the window.",
    "window-open": "Open the window.",
}


def task_text(env_name: str, swap: bool = False) -> str:
    """The instruction for a task. ``swap``: the next task's instruction in alphabetical order instead (a diagnostic: if
    a trained policy succeeds as often with another task's instruction, it is not reading the instruction)."""
    name = re.sub(r"-v\d+$", "", env_name)
    if swap:   # the next task (alphabetically) whose instruction differs: -side variants share theirs
        names = sorted(TASK_TEXT)
        i = names.index(name)
        name = next(names[(i + k) % len(names)] for k in range(1, len(names))
                    if TASK_TEXT[names[(i + k) % len(names)]] != TASK_TEXT[name])
    return TASK_TEXT[name]


@lru_cache(maxsize=None)
def benchmark(name: str, seed: int = 0):
    """``ML1:<task>`` / ``ML10`` / ``ML45`` (cached: building a benchmark samples all its task variations)."""
    import metaworld
    if name.startswith("ML1:"):
        return metaworld.ML1(name.split(":", 1)[1], seed=seed)
    if name in ("ML10", "ML45"):
        return getattr(metaworld, name)(seed=seed)
    raise ValueError(f"unknown Metaworld benchmark {name!r}: ML1:<task>, ML10 or ML45")


def split_tasks(bench, split: str) -> dict[str, list]:
    """{env_name: [task variations]} for the 'train' or 'test' split."""
    assert split in ("train", "test"), split
    tasks = bench.train_tasks if split == "train" else bench.test_tasks
    out: dict[str, list] = {}
    for t in tasks:
        out.setdefault(t.env_name, []).append(t)
    return out


def _rand_vec(task) -> np.ndarray:
    """A task variation's sampled object / goal positions (Metaworld pickles them into ``Task.data``)."""
    import pickle
    return np.asarray(pickle.loads(task.data)["rand_vec"], dtype=float)


class TaskSampler:
    """Uniform over the split's tasks, then over that task's variations; each start state gets its own seed.

    ``n_seeds > 1`` (training only): pool the split's variations of benchmark seeds ``seed .. seed + n_seeds - 1``, for
    more object / goal positions than the benchmark's 50 per task. Other seeds' *training* variations can be this
    seed's *test* variations (ML1: seed 1's 50 training positions are exactly seed 0's 50 held-out positions), so any
    variation within ``exclude_cm`` of a held-out one (same task, positions compared) is dropped.

    ``tasks``: only these of the split's tasks (names with or without ``-v3``), e.g. 10 of ML45's 45 training tasks to
    measure how held-out transfer grows with the number of training tasks."""

    def __init__(self, bench_name: str, split: str, seed: int = 0, n_seeds: int = 1, exclude_cm: float = 1.0,
                 tasks: Sequence[str] | None = None):
        self.bench_name, self.split, self.seed = bench_name, split, seed
        self.tasks = split_tasks(benchmark(bench_name, seed), split)
        if tasks:
            want = {t if t.endswith("-v3") else f"{t}-v3" for t in tasks}
            unknown = want - set(self.tasks)
            assert not unknown, f"not {split} tasks of {bench_name}: {sorted(unknown)}"
            self.tasks = {k: v for k, v in self.tasks.items() if k in want}
        self.env_names = sorted(self.tasks)
        self.pool: dict[str, list[tuple[int, int]]] | None = None        # env_name -> [(bench seed, variation)]
        if n_seeds > 1:
            assert split == "train", "extra benchmark seeds are for training only"
            held = split_tasks(benchmark(bench_name, seed), "test")
            self.pool, self.dropped = {}, 0
            for s in range(seed, seed + n_seeds):
                for name, tasks in split_tasks(benchmark(bench_name, s), split).items():
                    if name not in self.tasks:
                        continue
                    ref = [_rand_vec(t) for t in held.get(name, [])]      # ML10/ML45: held-out tasks differ
                    for i, t in enumerate(tasks):
                        v = _rand_vec(t)
                        if ref and min(np.linalg.norm(r - v) for r in ref) * 100.0 < exclude_cm:
                            self.dropped += 1
                            continue
                        self.pool.setdefault(name, []).append((s, i))
            self.env_names = sorted(self.pool)

    def sample(self, n, rng):
        out = []
        for _ in range(n):
            name = self.env_names[int(rng.integers(len(self.env_names)))]
            st = {"benchmark": self.bench_name, "split": self.split, "env_name": name}
            if self.pool is None:
                st["task"] = int(rng.integers(len(self.tasks[name])))
            else:
                st["bench_seed"], st["task"] = self.pool[name][int(rng.integers(len(self.pool[name])))]
            st["seed"] = int(rng.integers(2**31 - 1))
            out.append(st)
        return out


class MetaWorldVecEnv:
    """Batched Metaworld with exact reset-to-task. One simulator per (split, task) is pooled and reused."""

    def __init__(self, bench_name: str, seed: int = 0, image_size: int = 224, camera: str = "corner",
                 max_delta: float = 10.0, max_substeps: int = 40, pos_tol: float = 0.5, blocked_window: int = 6,
                 blocked_eps: float = 0.05, gain: float = 0.5, grip_steps: int = 12,
                 render_size: tuple[int, int] = (640, 480), crop: str = "center", goal_marker_scale: float = 1.0,
                 action_frame: str = "world"):
        """Lengths in cm. ``max_delta``: longest hand move per action; ``max_substeps``: simulator steps per turn.
        Frames are rendered at ``render_size`` (width, height) and cropped to ``image_size`` square:
        ``crop="center"`` (default) takes the central ``image_size`` pixels of the full-resolution frame: a zoom on
        the workspace in which goals, objects and the gripper stay in view across task variations and expert
        trajectories (checked on reach / push / pick-place / shelf-place / box-close / basketball);
        ``crop="resize"`` scales the short side to ``image_size`` first (whole scene, but objects are a few pixels).

        ``action_frame``: the axes of moves, of the reported last move and of the goal offset. ``"world"``: Metaworld's
        x / y / z (from the corner camera both x and y point diagonally away, one to the right, one to the left).
        ``"camera"``: +x = right in the image, +y = away from the camera along the table, +z = up, from the camera's
        pose; a reasoning model can say "move right and back" without decoding diagonal axes."""
        assert crop in ("resize", "center"), crop
        assert action_frame in ("world", "camera"), action_frame
        self.action_frame = action_frame
        self._frames: dict[int, np.ndarray] = {}
        self.render_size, self.crop = tuple(render_size), crop
        self.bench_name, self.bench_seed = bench_name, seed
        self.image_size, self.camera = image_size, camera
        self.max_delta, self.max_substeps, self.pos_tol = max_delta, max_substeps, pos_tol
        self.blocked_window, self.blocked_eps, self.gain = blocked_window, blocked_eps, gain
        self.grip_steps = grip_steps   # a changed gripper command gets at least this many steps (fingers are slow)
        # goal markers (sites named *goal*) drawn this many times larger: at 1.0 several are a few pixels in the 224 crop
        # (dial-turn 17, plate-slide-side 7-19, stick-push 6-13 changed pixels); at 2.0 they are 70-300
        self.goal_marker_scale = goal_marker_scale
        self._pool: dict[tuple, list] = {}
        self._renderers: dict[str, object] = {}
        self._envs: list = []
        self._keys: list[tuple] = []
        self._obs: list[np.ndarray] = []
        self._grip: list[float] = []
        self._progress: list[float] = []   # Metaworld's dense reward (0-10) at each simulator's current state
        self._obj_dist: list[float] = []   # info["obj_to_target"] (m): the task object's distance to its goal (reach: hand)

    def _make(self, split: str, env_name: str):
        b = benchmark(self.bench_name, self.bench_seed)
        cls = (b.train_classes if split == "train" else b.test_classes)[env_name]
        w, h = self.render_size
        env = cls(render_mode="rgb_array", camera_name=self.camera, width=w, height=h)
        if self.goal_marker_scale != 1.0:   # once per simulator (they are pooled and reused): the size is a model constant
            gids = [i for i in range(env.model.nsite) if "goal" in (env.model.site(i).name or "").lower()]
            env.model.site_size[gids] *= self.goal_marker_scale
        return env

    def _crop(self, img: np.ndarray) -> np.ndarray:
        s = self.image_size
        if self.crop == "resize":
            from PIL import Image
            h, w = img.shape[:2]
            k = s / min(h, w)
            img = np.asarray(Image.fromarray(img).resize((max(s, round(w * k)), max(s, round(h * k))), Image.BICUBIC))
        h, w = img.shape[:2]
        top, left = (h - s) // 2, (w - s) // 2
        return img[top:top + s, left:left + s]

    def raw(self, k: int):
        return self._envs[k]

    def full_obs(self, k: int) -> np.ndarray:
        """Metaworld observation with the goal visible (for the scripted expert; the VLM never sees it)."""
        env = self._envs[k]
        was = env._partially_observable
        env._partially_observable = False
        o = env._get_obs()
        env._partially_observable = was
        return o

    def _render(self, k: int) -> np.ndarray:
        """Render simulator k with one shared renderer per task class. Each simulator's own `env.render()` would open
        its own GL context (~128 MiB of GPU memory each, so ~25 GB for a 128-episode step + eval). Simulators of one
        class share the model structure (meshes, textures: what the GL context holds); the scene is built from the
        simulator's own model and data, so per-task goal sites are drawn where they are."""
        env, name = self._envs[k], self._keys[k][1]
        r = self._renderers.get(name)
        if r is None:
            import mujoco
            w, h = self.render_size
            r = self._renderers[name] = mujoco.Renderer(env.model, height=h, width=w)
        r._model = env.model   # update_scene() reads the model from here
        r.update_scene(env.data, camera=self.camera)
        return r.render()

    def _view(self, k: int, extras: dict | None = None) -> Obs:
        env, o = self._envs[k], self._obs[k]
        img = self._render(k)
        if env.camera_name == "corner":          # Metaworld's corner camera is rolled 180 degrees (its "up" axis points
            img = img[::-1, ::-1]                # down): rotate, not flip rows only, which would mirror the scene
        img = self._crop(img)
        state = np.r_[o[:3] * 100.0, o[3]]       # hand xyz in cm, gripper opening in [0, 1]
        # the task's goal relative to the hand, in cm, in the dx/dy/dz frame of the moves (shown only by
        # prompt_style=compact_rel_goal, once, at the start: the same information the goal marker draws in the image)
        goal = getattr(env, "_target_pos", None)
        offset = ([round(float(v), 1) for v in self.to_frame(k, (np.asarray(goal, dtype=float)[:3] - o[:3]) * 100.0)]
                  if goal is not None else None)
        return Obs(image=np.ascontiguousarray(img), state=state,
                   extras={"env_name": self._keys[k][1], "path": int(env.curr_path_length),
                           "mw_progress": self._progress[k], "mw_obj_dist": self._obj_dist[k], "goal_offset": offset,
                           **(extras or {})})

    def reset(self, start_states):
        for env, key in zip(self._envs, self._keys):
            self._pool.setdefault(key, []).append(env)
        self._envs, self._keys, self._obs, self._grip, self._progress, self._obj_dist = [], [], [], [], [], []
        for s in start_states:
            key = (s["split"], s["env_name"])
            pool = self._pool.get(key)
            env = pool.pop() if pool else self._make(*key)
            b = benchmark(self.bench_name, int(s.get("bench_seed", self.bench_seed)))     # cached per seed
            env.set_task(split_tasks(b, s["split"])[s["env_name"]][s["task"]])
            o, _ = env.reset(seed=int(s["seed"]))
            self._envs.append(env)
            self._keys.append(key)
            self._obs.append(o)
            self._grip.append(-1.0)               # start with the gripper commanded open
            r, info = env.evaluate_state(o, np.array([0, 0, 0, -1.0], dtype=np.float32))
            self._progress.append(float(r))
            self._obj_dist.append(float(info["obj_to_target"]))
        return [self._view(k) for k in range(len(start_states))]

    def step(self, ids, actions):
        out = []
        for k, a in zip(ids, actions):
            env = self._envs[k]
            a = np.asarray(a, dtype=np.float64)
            delta = self.frame(k) @ a[:3]                    # move in the action frame -> world (identity for "world")
            n = np.linalg.norm(delta)
            if n > self.max_delta:
                delta *= self.max_delta / n
            grip_changed = False
            if abs(a[3]) > 1e-3:                  # opening change < 0 -> close (+1 effort), > 0 -> open (-1)
                new = 1.0 if a[3] < 0 else -1.0
                grip_changed, self._grip[k] = new != self._grip[k], new
            min_steps = self.grip_steps if grip_changed else 3
            start = self._obs[k][:3] * 100.0
            target = start + delta
            success, stop, reward_sum, hist, low = False, "steps", 0.0, [], []
            for i in range(self.max_substeps):
                if env.curr_path_length >= env.max_path_length:
                    stop = "path_limit"
                    break
                hand = self._obs[k][:3] * 100.0
                u = np.r_[np.clip((target - hand) * self.gain, -1.0, 1.0), self._grip[k]]
                o, r, _, _, info = env.step(u.astype(np.float32))
                self._obs[k] = o
                self._progress[k] = float(r)
                self._obj_dist[k] = float(info["obj_to_target"])
                reward_sum += float(r)
                low.append(u)
                hist.append(o[:3] * 100.0)
                if info.get("success"):
                    success, stop = True, "success"
                    break
                if np.linalg.norm(target - o[:3] * 100.0) < self.pos_tol and i >= min_steps:
                    stop = "reached"
                    break
                if (i >= max(self.blocked_window, min_steps)
                        and np.linalg.norm(hist[-1] - hist[-1 - self.blocked_window]) < self.blocked_eps):
                    stop = "blocked"
                    break
            truncated = not success and env.curr_path_length >= env.max_path_length
            moved = self.to_frame(k, self._obs[k][:3] * 100.0 - start)
            out.append(StepResult(obs=self._view(k, {"moved": moved}), reward=float(success), terminated=success, truncated=truncated,
                                  info={"success": success, "stop": stop, "mw_reward": reward_sum,
                                        "low_level_actions": np.stack(low) if low else np.zeros((0, 4))}))
        return out

    def frame(self, k: int) -> np.ndarray:
        """3x3 matrix whose columns are the action frame's axes in world coordinates (orthonormal)."""
        if self.action_frame == "world":
            return np.eye(3)
        env = self._envs[k]
        key = id(env)
        if key not in self._frames:
            import mujoco
            cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
            mujoco.mj_forward(env.model, env.data)
            R = np.asarray(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)   # columns: camera x, y, z in world
            right = -R[:, 0] if self.camera == "corner" else R[:, 0]          # the corner view is rotated 180 degrees
            away = -R[:, 2]                                                     # the camera looks along its -z
            away = np.r_[away[:2], 0.0] / np.linalg.norm(away[:2])
            right = np.r_[right[:2], 0.0]
            right -= away * (right @ away)
            right /= np.linalg.norm(right)
            self._frames[key] = np.stack([right, away, np.array([0.0, 0.0, 1.0])], axis=1)
        return self._frames[key]

    def to_frame(self, k: int, v) -> np.ndarray:
        """A world vector (3,) in the action frame."""
        return self.frame(k).T @ np.asarray(v, dtype=float)

    def close(self):
        for e in self._envs + [e for pool in self._pool.values() for e in pool]:
            e.close()
        for r in self._renderers.values():
            r.close()
        self._envs, self._pool, self._renderers = [], {}, {}


def oracle_action(vec: MetaWorldVecEnv, k: int) -> np.ndarray:
    """Metaworld's scripted expert as one of our actions: the target position it steers toward (cm, relative to the
    hand) and the sign of its grab effort (close / open). An upper bound for the text protocol.

    Every expert computes ``move(hand, to_xyz=target, p=...)`` inside ``get_action``; we record ``to_xyz`` by
    temporarily wrapping the ``move`` its module calls, which works for all 50 experts without per-task code."""
    import sys

    from metaworld.policies import ENV_POLICY_MAP
    pol = ENV_POLICY_MAP[vec._keys[k][1]]()
    mod = sys.modules[type(pol).__module__]
    o = vec.full_obs(k)
    seen = []
    orig = mod.move

    def recording_move(from_xyz, to_xyz, p):
        seen.append(np.asarray(to_xyz, dtype=np.float64))
        return orig(from_xyz, to_xyz, p)
    mod.move = recording_move
    try:
        act = pol.get_action(o)
    finally:
        mod.move = orig
    to_xyz = seen[-1] if seen else o[:3] + np.asarray(act[:3]) * vec_action_scale(vec, k)
    return np.r_[vec.to_frame(k, (to_xyz - o[:3]) * 100.0), -1.0 if float(act[3]) > 0 else 1.0]


def vec_action_scale(vec: MetaWorldVecEnv, k: int) -> float:
    return float(getattr(vec._envs[k], "action_scale", 0.01))


# ----------------------------------------------------------------------------- agent interface

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_COMPACT = {q: re.compile(rf'"?{q}"?\s*[:=]\s*({_NUM})', re.IGNORECASE) for q in ("x", "y", "z")}
_COMPACT_REL = {q: re.compile(rf'"?d{q}"?\s*[:=]\s*({_NUM})', re.IGNORECASE) for q in ("x", "y", "z")}
_GOAL_EST = re.compile(rf"goal\W*dx\s*[:=]\s*({_NUM})\W+dy\s*[:=]\s*({_NUM})\W+dz\s*[:=]\s*({_NUM})", re.IGNORECASE)
_COMPACT_GRIP = re.compile(r'"?(?:grip|gripper)"?\s*[:=]\s*"?(open|close|closed|' + _NUM + r')"?', re.IGNORECASE)


class MetaWorldTextInterface:
    """``prompt_style="pear"``: PEAR's protocol (state JSON + image; reply = fenced JSON *change* of every
    quantity). ``"compact"``: one short line per turn, reply ``{"x": dx, "y": dy, "z": dz, "grip": "open|close"}``
    (for small models). ``"compact_rel"``: ``compact`` without any absolute coordinate, so there is nothing to
    copy: the reply is ``{"dx", "dy", "dz", "grip"}``, the system prompt says which way each axis moves the hand in the
    image, and each turn reports how far the hand actually moved last turn. The goal is only in the image (unlike
    PointMaze's ``compact_rel``, no goal offset is given). ``"compact_rel_reason"``: ``compact_rel`` with one or two
    sentences of reasoning first (where the gripper, the object and the goal are, what to do next), then the JSON move;
    the last ``{...}`` in the reply is the action, so axis names inside the reasoning are never read as the move.
    ``"compact_rel_locate"``: ``compact_rel`` where each reply first states where the task's goal is relative to the
    hand (``Goal: dx=+12.0, dy=-3.5, dz=+4.0.``), then the JSON move. Nothing about the goal is given: the policy reads
    it from the image. Behaviour cloning supervises the estimate with the true offset (``expert_reply``), the
    simulator state the scripted experts already use; at test time only the image is there."""

    SCHEMA = {"hand": ["x", "y", "z"], "gripper": ["opening"]}

    def __init__(self, max_delta: float = 10.0, prompt_style: str = "pear", include_state: bool = True,
                 swap_instructions: bool = False, action_frame: str = "world", scene_notes: bool = False):
        assert prompt_style in ("pear", "compact", "compact_rel", "compact_rel_reason", "compact_rel_goal",
                                "compact_rel_locate"), prompt_style
        self.max_delta, self.prompt_style, self.include_state = max_delta, prompt_style, include_state
        self.rel = prompt_style in ("compact_rel", "compact_rel_reason", "compact_rel_goal", "compact_rel_locate")
        self.swap_instructions = swap_instructions
        self.action_frame = action_frame          # "camera": moves are right / away from the camera / up (MetaWorldVecEnv)
        self.scene_notes = scene_notes            # explain the scene's conventions (gripper, goal marker, grasping)
        # the absolute-position styles (pear, compact) report world coordinates, which camera-frame moves would contradict
        assert action_frame == "world" or self.rel, f"action_frame={action_frame!r} needs a compact_rel* prompt style"

    def environment_section(self) -> str:
        return (
            "Robot and frame:\n"
            "- The robot is a Sawyer arm with a two-finger gripper, working on a table.\n"
            "- hand is the gripper's position: x and y are horizontal, z points up; positions are in centimeters.\n"
            "- gripper.opening is 1 when fully open and about 0.3 when closed; a negative change closes the gripper "
            "(it keeps squeezing until a positive change opens it again).\n"
            f"- Each action moves the hand by at most {self.max_delta:g} cm in total; longer changes are shortened.\n"
            "Cameras:\n"
            "- corner: a fixed camera looking at the table, the robot and the objects.\n"
            "Ending:\n"
            "- The episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet.")

    WORLD_AXES = ("in the image, +dx moves the hand to the right, +dy to the left, both slightly away from the camera; "
                  "+dz moves it up.")
    CAMERA_AXES = ("they are relative to the hand's current position: +dx moves the hand to the right in the image, "
                   "+dy moves it away from the camera (toward the back of the table), +dz moves it up; -dx is left, -dy "
                   "toward the camera, -dz down. A longer move is shortened to the limit in the same direction.")

    SCENE_NOTES = ("Scene: the gripper is the two-finger tool at the end of the arm. When the task has a goal position, a "
                   "small colored sphere floating in the image marks it (some scenes show a second one, for a related "
                   "task); a sphere is only a marker and cannot be touched or moved. The "
                   "objects to act on (a puck, a block, a handle, a button, a door, ...) are on the table. To pick "
                   "something up: open the gripper, move above it, lower until the fingers are around it, close the "
                   "gripper, then lift. ")

    def system_prompt(self, start_state):
        text = self._system_prompt(start_state)
        if self.action_frame == "camera" and self.rel:
            assert self.WORLD_AXES in text, "prompt style without the axis sentence"
            text = text.replace(self.WORLD_AXES, self.CAMERA_AXES)
        if self.scene_notes and self.rel:
            task = f"Task: {task_text(start_state['env_name'], self.swap_instructions)} "
            assert task in text, "prompt style without the task line"
            text = text.replace(task, task + self.SCENE_NOTES, 1)
        return text

    def _system_prompt(self, start_state):
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.system_prompt(self.environment_section(), task_text(start_state["env_name"], self.swap_instructions),
                                          length_unit="centimeters")
        if self.prompt_style == "compact_rel_reason":
            return ("You control a robot arm with a gripper above a table, seen from a fixed camera. "
                    f"Task: {task_text(start_state['env_name'], self.swap_instructions)} "
                    "Each turn, first write one or two short sentences: where the gripper, the object and the goal "
                    "are in the image, and what the hand should do next. Then, on the last line, give how far to move "
                    'the hand, in centimeters, as JSON, for example {"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}. '
                    "These are changes, not positions: in the image, +dx moves the hand to the right, +dy to the left, "
                    f"both slightly away from the camera; +dz moves it up. At most {self.max_delta:g} cm per turn. "
                    '"grip": "close" closes the gripper, "open" opens it.')
        if self.prompt_style == "compact_rel_locate":
            return ("You control a robot arm with a gripper above a table, seen from a fixed camera. "
                    f"Task: {task_text(start_state['env_name'], self.swap_instructions)} "
                    "Each turn, first say where the task's goal is relative to the hand, in centimeters and in the same "
                    "dx/dy/dz directions as the moves, then how far to move the hand, as JSON, for example "
                    'Goal: dx=+12.0, dy=-3.5, dz=+4.0. {"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"} '
                    "The moves are changes, not positions: in the image, +dx moves the hand to the right, +dy to the "
                    f"left, both slightly away from the camera; +dz moves it up. At most {self.max_delta:g} cm per turn. "
                    '"grip": "close" closes the gripper, "open" opens it.')
        if self.prompt_style == "compact_rel_goal":
            return ("You control a robot arm with a gripper above a table, seen from a fixed camera. "
                    f"Task: {task_text(start_state['env_name'], self.swap_instructions)} "
                    "At the start you are told where the goal is relative to the hand, in the same dx/dy/dz directions "
                    "as the moves; the earlier turns stay in the conversation. Each turn, reply with only how far to move "
                    'the hand, in centimeters, as JSON, for example {"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}. '
                    "These are changes, not positions: in the image, +dx moves the hand to the right, +dy to the left, "
                    f"both slightly away from the camera; +dz moves it up. At most {self.max_delta:g} cm per turn. "
                    '"grip": "close" closes the gripper, "open" opens it.')
        if self.prompt_style == "compact_rel":
            return ("You control a robot arm with a gripper above a table, seen from a fixed camera. "
                    f"Task: {task_text(start_state['env_name'], self.swap_instructions)} "
                    "Each turn, reply with only how far to move the hand, in centimeters, as JSON, for example "
                    '{"dx": 3.0, "dy": -2.0, "dz": -1.0, "grip": "open"}. These are changes, not positions: '
                    "in the image, +dx moves the hand to the right, +dy to the left, both slightly away from the "
                    f"camera; +dz moves it up. At most {self.max_delta:g} cm per turn. "
                    '"grip": "close" closes the gripper, "open" opens it.')
        return ("You control a robot arm with a gripper above a table, seen from a fixed camera. "
                f"Task: {task_text(start_state['env_name'], self.swap_instructions)} "
                "Positions are in centimeters; z points up. Each turn, reply with only the move as JSON, for example "
                f'{{"x": 2.0, "y": -1.0, "z": 0.0, "grip": "close"}}. The hand moves by that amount (at most '
                f'{self.max_delta:g} cm per turn); "grip" is "open" or "close".')

    def observation_message(self, obs, turn, last, parse_error):
        from PIL import Image
        img = {"type": "image", "image": Image.fromarray(obs.image)}
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            parts = []
            if parse_error:
                parts.append("Your previous response contained no valid action, so the robot did not move.")
            if self.include_state:
                parts.append(protocol.format_state(self.SCHEMA, obs.state))
            return ([{"type": "text", "text": "\n".join(parts)}] if parts else []) + [img]
        h = obs.state
        if self.rel:
            text = "No valid move last turn. " if parse_error else ""
            g = obs.extras.get("goal_offset")
            if self.prompt_style == "compact_rel_goal" and turn == 0 and g is not None:
                text += f"Goal: dx={g[0]:+.1f}, dy={g[1]:+.1f}, dz={g[2]:+.1f} from the hand. "
            if self.include_state:
                text += f"Gripper: {'open' if h[3] > 0.6 else 'closed'}. "
                m = obs.extras.get("moved")
                if m is not None:
                    text += f"Last move: dx={m[0]:+.1f}, dy={m[1]:+.1f}, dz={m[2]:+.1f}. "
            return [img, {"type": "text", "text": text + "Move:"}]
        text = ("No valid move last turn. " if parse_error else "") + (
            f"Hand: x={h[0]:.1f}, y={h[1]:.1f}, z={h[2]:.1f}. Gripper: {'open' if h[3] > 0.6 else 'closed'}. Move:"
            if self.include_state else "Move:")
        return [img, {"type": "text", "text": text}]

    def parse_action(self, text):
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.parse_action(text, self.SCHEMA)
        pats = _COMPACT_REL if self.rel else _COMPACT
        if self.prompt_style in ("compact_rel_reason", "compact_rel_locate"):
            objs = re.findall(r"\{[^{}]*\}", text)
            if not objs:
                return None
            text = objs[-1]                 # the move is the last JSON object; the reasoning before it is free text
        vals = [m.group(1) if (m := pats[q].search(text)) else None for q in ("x", "y", "z")]
        g = _COMPACT_GRIP.search(text)
        if all(v is None for v in vals) and not g:
            return None
        vals = [v if v is not None else "0" for v in vals]   # a missing axis is no move along it ({"grip": "open"})
        grip = 0.0
        if g:
            w = g.group(1).lower()
            grip = -1.0 if w.startswith("clos") else 1.0 if w == "open" else float(w)
        a = np.array([*map(float, vals), grip])
        return a if np.all(np.isfinite(a)) else None

    def format_action(self, a) -> str:
        a = np.asarray(a, dtype=float)
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.example_action(self.SCHEMA, a)
        grip = "close" if a[3] < 0 else "open"
        if self.rel:
            return f'{{"dx": {a[0]:.1f}, "dy": {a[1]:.1f}, "dz": {a[2]:.1f}, "grip": "{grip}"}}'
        return f'{{"x": {a[0]:.1f}, "y": {a[1]:.1f}, "z": {a[2]:.1f}, "grip": "{grip}"}}'

    def expert_reply(self, a, obs) -> str:
        """The scripted expert's reply as training text. ``compact_rel_locate`` prefixes the true goal offset of the
        observation, so cloning also teaches where the goal is; other styles are just the move."""
        if self.prompt_style != "compact_rel_locate":
            return self.format_action(a)
        g = (obs.extras or {}).get("goal_offset") if obs is not None else None
        goal = f"Goal: dx={g[0]:+.1f}, dy={g[1]:+.1f}, dz={g[2]:+.1f}." if g is not None else "Goal: unknown."
        return f"{goal} {self.format_action(a)}"

    @property
    def states_goal(self) -> bool:
        """Replies state the goal offset (read by ``goal_estimate``; the trainer logs its error)."""
        return self.prompt_style == "compact_rel_locate"

    def goal_estimate(self, text: str):
        """The goal offset a ``compact_rel_locate`` reply states (cm, dx/dy/dz), or None."""
        m = _GOAL_EST.search(text or "")
        return np.array([float(v) for v in m.groups()]) if m else None

    def noop_action(self):
        return np.zeros(4)


# ----------------------------------------------------------------------------- factory

@dataclass
class MetaWorldConfig:
    benchmark: str = "ML1:reach-v3"   # ML1:<task> | ML10 | ML45
    eval_split: str = "test"          # test = held-out goals (ML1) / held-out tasks (ML10, ML45) | train
    prompt_style: str = "pear"        # pear | compact | compact_rel (| _reason | _goal | _locate)
    include_state: bool = True
    swap_instructions: bool = False   # diagnostic: every task gets the next task's instruction (see task_text)
    image_size: int = 224              # square crop fed to the VLM
    render_width: int = 640            # rendered frame, before cropping
    render_height: int = 480
    crop: str = "center"               # center = central pixels of the 640x480 frame | resize = short side first
    camera: str = "corner"
    max_delta: float = 10.0
    max_substeps: int = 40
    # axes of moves, last move and goal offset: world (Metaworld x/y/z, diagonal in the corner view) | camera (right in
    # the image / away from the camera / up, with a plainer description of them in the prompt)
    action_frame: str = "world"
    # a paragraph on the scene's conventions (the gripper, the goal marker, how grasping works), for policies that reason
    scene_notes: bool = False
    # waypoint controller: per-step command = clip(gain * (target - hand) [cm], -1, 1). 0.5 slows down from 2 cm out;
    # 1.0 (the default before 2026-11-22) ran at full speed to 1 cm, overshot and lifted the hand while dragging an
    # object: scripted-expert success push 0.63 -> 0.97, ML10 training tasks 0.85 -> 0.91, held-out tasks 0.42 -> 0.72
    gain: float = 0.5
    # goal markers drawn this many times larger (see docs/environments/metaworld.md, Observability); 1.0 = Metaworld's
    goal_marker_scale: float = 1.0
    # training positions from this many benchmark seeds (1 = the benchmark's own split; > 1 is an ablation outside the
    # ML1/ML10 protocol: more positions, held-out ones excluded, eval unchanged)
    train_seeds: int = 1
    # train on only these of the benchmark's training tasks (and evaluate eval_split=train on them); held-out tasks are
    # unchanged. None = all
    train_tasks: list[str] | None = None
    max_turns: int = 15
    seed: int = 0


def make_metaworld(cfg: MetaWorldConfig | None = None, **kw) -> EnvBundle:
    cfg = cfg or MetaWorldConfig(**kw)
    return EnvBundle(
        make_vec_env=lambda: MetaWorldVecEnv(cfg.benchmark, cfg.seed, cfg.image_size, cfg.camera, cfg.max_delta,
                                             cfg.max_substeps, render_size=(cfg.render_width, cfg.render_height),
                                             crop=cfg.crop, gain=cfg.gain, goal_marker_scale=cfg.goal_marker_scale,
                                             action_frame=cfg.action_frame),
        interface=MetaWorldTextInterface(cfg.max_delta, cfg.prompt_style, cfg.include_state, cfg.swap_instructions,
                                         cfg.action_frame, cfg.scene_notes),
        train_sampler=TaskSampler(cfg.benchmark, "train", cfg.seed, n_seeds=cfg.train_seeds, tasks=cfg.train_tasks),
        eval_sampler=TaskSampler(cfg.benchmark, cfg.eval_split, cfg.seed,
                                 tasks=cfg.train_tasks if cfg.eval_split == "train" else None),
        max_turns=cfg.max_turns,
        expert=oracle_action,
    )
