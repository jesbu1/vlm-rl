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
    "bin-picking": "Pick up the cube in the right bin and place it in the left bin.",
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


def task_text(env_name: str) -> str:
    return TASK_TEXT[re.sub(r"-v\d+$", "", env_name)]


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


class TaskSampler:
    """Uniform over the split's tasks, then over that task's variations; each start state gets its own seed."""

    def __init__(self, bench_name: str, split: str, seed: int = 0):
        self.bench_name, self.split, self.seed = bench_name, split, seed
        self.tasks = split_tasks(benchmark(bench_name, seed), split)
        self.env_names = sorted(self.tasks)

    def sample(self, n, rng):
        out = []
        for _ in range(n):
            name = self.env_names[int(rng.integers(len(self.env_names)))]
            out.append({"benchmark": self.bench_name, "split": self.split, "env_name": name,
                        "task": int(rng.integers(len(self.tasks[name]))), "seed": int(rng.integers(2**31 - 1))})
        return out


class MetaWorldVecEnv:
    """Batched Metaworld with exact reset-to-task. One simulator per (split, task) is pooled and reused."""

    def __init__(self, bench_name: str, seed: int = 0, image_size: int = 224, camera: str = "corner",
                 max_delta: float = 10.0, max_substeps: int = 40, pos_tol: float = 0.5, blocked_window: int = 6,
                 blocked_eps: float = 0.05, gain: float = 1.0, grip_steps: int = 12,
                 render_size: tuple[int, int] = (640, 480), crop: str = "center"):
        """Lengths in cm. ``max_delta``: longest hand move per action; ``max_substeps``: simulator steps per turn.
        Frames are rendered at ``render_size`` (width, height) and cropped to ``image_size`` square:
        ``crop="center"`` (default) takes the central ``image_size`` pixels of the full-resolution frame: a zoom on
        the workspace in which goals, objects and the gripper stay in view across task variations and expert
        trajectories (checked on reach / push / pick-place / shelf-place / box-close / basketball);
        ``crop="resize"`` scales the short side to ``image_size`` first (whole scene, but objects are a few pixels)."""
        assert crop in ("resize", "center"), crop
        self.render_size, self.crop = tuple(render_size), crop
        self.bench_name, self.bench_seed = bench_name, seed
        self.image_size, self.camera = image_size, camera
        self.max_delta, self.max_substeps, self.pos_tol = max_delta, max_substeps, pos_tol
        self.blocked_window, self.blocked_eps, self.gain = blocked_window, blocked_eps, gain
        self.grip_steps = grip_steps   # a changed gripper command gets at least this many steps (fingers are slow)
        self._pool: dict[tuple, list] = {}
        self._envs: list = []
        self._keys: list[tuple] = []
        self._obs: list[np.ndarray] = []
        self._grip: list[float] = []

    def _make(self, split: str, env_name: str):
        b = benchmark(self.bench_name, self.bench_seed)
        cls = (b.train_classes if split == "train" else b.test_classes)[env_name]
        w, h = self.render_size
        return cls(render_mode="rgb_array", camera_name=self.camera, width=w, height=h)

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

    def _view(self, k: int, extras: dict | None = None) -> Obs:
        env, o = self._envs[k], self._obs[k]
        img = env.render()
        if env.camera_name == "corner":          # Metaworld's corner camera renders upside down
            img = img[::-1]
        img = self._crop(img)
        state = np.r_[o[:3] * 100.0, o[3]]       # hand xyz in cm, gripper opening in [0, 1]
        return Obs(image=np.ascontiguousarray(img), state=state,
                   extras={"env_name": self._keys[k][1], "path": int(env.curr_path_length), **(extras or {})})

    def reset(self, start_states):
        for env, key in zip(self._envs, self._keys):
            self._pool.setdefault(key, []).append(env)
        self._envs, self._keys, self._obs, self._grip = [], [], [], []
        b = benchmark(self.bench_name, self.bench_seed)
        for s in start_states:
            key = (s["split"], s["env_name"])
            pool = self._pool.get(key)
            env = pool.pop() if pool else self._make(*key)
            env.set_task(split_tasks(b, s["split"])[s["env_name"]][s["task"]])
            o, _ = env.reset(seed=int(s["seed"]))
            self._envs.append(env)
            self._keys.append(key)
            self._obs.append(o)
            self._grip.append(-1.0)               # start with the gripper commanded open
        return [self._view(k) for k in range(len(start_states))]

    def step(self, ids, actions):
        out = []
        for k, a in zip(ids, actions):
            env = self._envs[k]
            a = np.asarray(a, dtype=np.float64)
            delta = a[:3].copy()
            n = np.linalg.norm(delta)
            if n > self.max_delta:
                delta *= self.max_delta / n
            grip_changed = False
            if abs(a[3]) > 1e-3:                  # opening change < 0 -> close (+1 effort), > 0 -> open (-1)
                new = 1.0 if a[3] < 0 else -1.0
                grip_changed, self._grip[k] = new != self._grip[k], new
            min_steps = self.grip_steps if grip_changed else 3
            target = self._obs[k][:3] * 100.0 + delta
            success, stop, reward_sum, hist, low = False, "steps", 0.0, [], []
            for i in range(self.max_substeps):
                if env.curr_path_length >= env.max_path_length:
                    stop = "path_limit"
                    break
                hand = self._obs[k][:3] * 100.0
                u = np.r_[np.clip((target - hand) * self.gain, -1.0, 1.0), self._grip[k]]
                o, r, _, _, info = env.step(u.astype(np.float32))
                self._obs[k] = o
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
            out.append(StepResult(obs=self._view(k), reward=float(success), terminated=success, truncated=truncated,
                                  info={"success": success, "stop": stop, "mw_reward": reward_sum,
                                        "low_level_actions": np.stack(low) if low else np.zeros((0, 4))}))
        return out

    def close(self):
        for e in self._envs + [e for pool in self._pool.values() for e in pool]:
            e.close()
        self._envs, self._pool = [], {}


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
    return np.r_[(to_xyz - o[:3]) * 100.0, -1.0 if float(act[3]) > 0 else 1.0]


def vec_action_scale(vec: MetaWorldVecEnv, k: int) -> float:
    return float(getattr(vec._envs[k], "action_scale", 0.01))


# ----------------------------------------------------------------------------- agent interface

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_COMPACT = {q: re.compile(rf'"?{q}"?\s*[:=]\s*({_NUM})', re.IGNORECASE) for q in ("x", "y", "z")}
_COMPACT_GRIP = re.compile(r'"?(?:grip|gripper)"?\s*[:=]\s*"?(open|close|closed|' + _NUM + r')"?', re.IGNORECASE)


class MetaWorldTextInterface:
    """``prompt_style="pear"``: PEAR's protocol (state JSON + image; reply = fenced JSON *change* of every
    quantity). ``"compact"``: one short line per turn, reply ``{"x": dx, "y": dy, "z": dz, "grip": "open|close"}``
    (for small models)."""

    SCHEMA = {"hand": ["x", "y", "z"], "gripper": ["opening"]}

    def __init__(self, max_delta: float = 10.0, prompt_style: str = "pear", include_state: bool = True):
        assert prompt_style in ("pear", "compact"), prompt_style
        self.max_delta, self.prompt_style, self.include_state = max_delta, prompt_style, include_state

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

    def system_prompt(self, start_state):
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.system_prompt(self.environment_section(), task_text(start_state["env_name"]),
                                          length_unit="centimeters")
        return ("You control a robot arm with a gripper above a table, seen from a fixed camera. "
                f"Task: {task_text(start_state['env_name'])} "
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
        text = ("No valid move last turn. " if parse_error else "") + (
            f"Hand: x={h[0]:.1f}, y={h[1]:.1f}, z={h[2]:.1f}. Gripper: {'open' if h[3] > 0.6 else 'closed'}. Move:"
            if self.include_state else "Move:")
        return [img, {"type": "text", "text": text}]

    def parse_action(self, text):
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.parse_action(text, self.SCHEMA)
        vals = [m.group(1) if (m := _COMPACT[q].search(text)) else None for q in ("x", "y", "z")]
        if any(v is None for v in vals):
            return None
        g = _COMPACT_GRIP.search(text)
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
        return f'{{"x": {a[0]:.1f}, "y": {a[1]:.1f}, "z": {a[2]:.1f}, "grip": "{grip}"}}'

    def noop_action(self):
        return np.zeros(4)


# ----------------------------------------------------------------------------- factory

@dataclass
class MetaWorldConfig:
    benchmark: str = "ML1:reach-v3"   # ML1:<task> | ML10 | ML45
    eval_split: str = "test"          # test = held-out goals (ML1) / held-out tasks (ML10, ML45) | train
    prompt_style: str = "pear"
    include_state: bool = True
    image_size: int = 224              # square crop fed to the VLM
    render_width: int = 640            # rendered frame, before cropping
    render_height: int = 480
    crop: str = "center"               # center = central pixels of the 640x480 frame | resize = short side first
    camera: str = "corner"
    max_delta: float = 10.0
    max_substeps: int = 40
    max_turns: int = 15
    seed: int = 0


def make_metaworld(cfg: MetaWorldConfig | None = None, **kw) -> EnvBundle:
    cfg = cfg or MetaWorldConfig(**kw)
    return EnvBundle(
        make_vec_env=lambda: MetaWorldVecEnv(cfg.benchmark, cfg.seed, cfg.image_size, cfg.camera, cfg.max_delta,
                                             cfg.max_substeps, render_size=(cfg.render_width, cfg.render_height),
                                             crop=cfg.crop),
        interface=MetaWorldTextInterface(cfg.max_delta, cfg.prompt_style, cfg.include_state),
        train_sampler=TaskSampler(cfg.benchmark, "train", cfg.seed),
        eval_sampler=TaskSampler(cfg.benchmark, cfg.eval_split, cfg.seed),
        max_turns=cfg.max_turns,
    )
