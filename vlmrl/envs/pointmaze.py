"""PointMaze for VLM-RL.

Dynamics come from gymnasium-robotics' MuJoCo PointMaze (the same physics used to generate the
D4RL/Minari ``pointmaze`` offline datasets), so a world model trained on those datasets and this
simulator are interchangeable behind ``VecEnv``.

Images come from ``render_topdown``: a pure function of (maze_map, agent xy, goal xy). Because it
needs only low-dimensional state, it can render *offline datasets* into frames identically to the
online env -- handy for training a WAM on Minari data and then swapping it in here.

Action modes (what the VLM writes in ``<action>{...}</action>``):
* ``delta`` (default): {"dx": float, "dy": float} -- a relative waypoint in maze units (one cell =
  1.0). An inner PD controller tracks it for ``action_repeat`` physics steps with raw forces. This
  mirrors how D4RL PointMaze data was collected (waypoint controller) and is far easier for a VLM
  than raw forces on a momentum-driven point mass.
* ``force``: {"fx": float, "fy": float} in [-1, 1], held for ``action_repeat`` physics steps.

Either way the executed low-level force sequence is returned in ``info["low_level_actions"]``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from vlmrl.envs.base import EnvBundle
from vlmrl.types import Obs, StepResult

GOAL_RADIUS = 0.45  # gymnasium-robotics success threshold


_GEN_RE = re.compile(r"gen(\d+)x(\d+)s(\d+)")
_GEN_FAMILY_RE = re.compile(r"gen(\d+)x(\d+)")


def generate_maze(rows: int, cols: int, seed: int, loop_frac: float = 0.15) -> list[list]:
    """Procedural maze: a recursive-backtracker spanning tree on the odd-indexed cells of a ``rows`` x ``cols``
    grid (walls = 1 on the border and between corridors), then a fraction ``loop_frac`` of the remaining interior
    walls between two corridors is knocked out so there are loops (several routes, fewer pure dead ends).
    Deterministic in ``seed``; every free cell is reachable. ``rows``/``cols`` must be odd and >= 5."""
    assert rows % 2 and cols % 2 and rows >= 5 and cols >= 5, "maze sizes must be odd and >= 5"
    rng = np.random.default_rng(seed)
    m = np.ones((rows, cols), dtype=np.int64)
    start = (2 * int(rng.integers((rows - 1) // 2)) + 1, 2 * int(rng.integers((cols - 1) // 2)) + 1)
    m[start] = 0
    stack = [start]
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj, di // 2, dj // 2) for di, dj in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 0 < i + di < rows - 1 and 0 < j + dj < cols - 1 and m[i + di, j + dj] == 1]
        if not nbrs:
            stack.pop()
            continue
        ni, nj, si, sj = nbrs[int(rng.integers(len(nbrs)))]
        m[i + si, j + sj] = m[ni, nj] = 0
        stack.append((ni, nj))
    # interior walls separating two corridors (horizontally or vertically) -> candidates for loops
    cand = [(i, j) for i in range(1, rows - 1) for j in range(1, cols - 1) if m[i, j] == 1 and
            ((i % 2 == 1 and j % 2 == 0 and m[i, j - 1] == 0 and m[i, j + 1] == 0) or
             (i % 2 == 0 and j % 2 == 1 and m[i - 1, j] == 0 and m[i + 1, j] == 0))]
    for k in rng.permutation(len(cand))[:int(round(loop_frac * len(cand)))]:
        m[cand[k]] = 0
    return m.tolist()


def maze_family(spec: str, n: int, offset: int = 0) -> list[str]:
    """``"gen11x11"`` -> ``["gen11x11s<offset>", ..., "gen11x11s<offset+n-1>"]``."""
    if not _GEN_FAMILY_RE.fullmatch(spec):
        raise ValueError(f"not a procedural maze family: {spec!r} (expected gen<rows>x<cols>)")
    return [f"{spec}s{offset + i}" for i in range(n)]


def get_maze_map(name: str) -> list[list]:
    """A gymnasium-robotics map (``umaze``, ``medium``, ``large``, ...) or a procedural one, ``gen<rows>x<cols>s<seed>``."""
    g = _GEN_RE.fullmatch(name)
    if g:
        return generate_maze(int(g[1]), int(g[2]), int(g[3]))
    from gymnasium_robotics.envs.maze import maps
    table = {
        "open": maps.OPEN, "umaze": maps.U_MAZE, "medium": maps.MEDIUM_MAZE, "large": maps.LARGE_MAZE,
        "open_diverse_gr": maps.OPEN_DIVERSE_GR, "medium_diverse_gr": maps.MEDIUM_MAZE_DIVERSE_GR,
        "large_diverse_gr": maps.LARGE_MAZE_DIVERSE_GR,
    }
    if name not in table:
        raise ValueError(f"unknown maze {name!r}; options: {sorted(table)} or gen<rows>x<cols>s<seed>")
    return table[name]


def _walls(maze_map) -> np.ndarray:
    return np.array([[c == 1 for c in row] for row in maze_map], dtype=bool)


# ----------------------------------------------------------------------------- rendering

def xy_to_pixel(xy, maze_map, size: int) -> tuple[float, float]:
    """Maze coords (x right, y up, origin at maze centre, 1 unit per cell) -> pixel (col, row)."""
    h, w = len(maze_map), len(maze_map[0])
    cell = size / max(h, w)
    ox, oy = (size - w * cell) / 2, (size - h * cell) / 2
    return ox + (xy[0] + w / 2) * cell, oy + (h / 2 - xy[1]) * cell


def render_topdown(maze_map, agent_xy, goal_xy, size: int = 336, trail: Sequence | None = None,
                   draw_grid: bool = True) -> np.ndarray:
    """Clean top-down frame: walls dark, floor light, agent blue disc, goal green disc + ring."""
    walls = _walls(maze_map)
    h, w = walls.shape
    cell = size / max(h, w)
    img = Image.new("RGB", (size, size), (40, 40, 40))
    d = ImageDraw.Draw(img)
    for i in range(h):
        for j in range(w):
            x0, y0 = xy_to_pixel(((j - w / 2), (h / 2 - i)), maze_map, size)
            color = (60, 60, 60) if walls[i, j] else (235, 235, 225)
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color)
    if draw_grid:  # faint lines at integer coordinates (= cell borders) help the VLM read distances
        for j in range(w + 1):
            x0, _ = xy_to_pixel((j - w / 2, 0), maze_map, size)
            d.line([x0, 0, x0, size], fill=(200, 200, 190), width=1)
        for i in range(h + 1):
            _, y0 = xy_to_pixel((0, h / 2 - i), maze_map, size)
            d.line([0, y0, size, y0], fill=(200, 200, 190), width=1)
    if trail is not None and len(trail) > 1:
        d.line([xy_to_pixel(p, maze_map, size) for p in trail], fill=(120, 150, 230), width=2)
    gx, gy = xy_to_pixel(goal_xy, maze_map, size)
    rr = GOAL_RADIUS * cell
    d.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], outline=(20, 160, 60), width=2)
    r = 0.18 * cell
    d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(20, 170, 60))
    ax, ay = xy_to_pixel(agent_xy, maze_map, size)
    d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(30, 80, 220))
    return np.asarray(img)


# ----------------------------------------------------------------------------- dynamics

class PointMazeSimVecEnv:
    """Batched MuJoCo PointMaze with exact reset-to-state (qpos, qvel, goal).

    A start state may name its maze (``"maze": "gen11x11s7"``); otherwise the env's default ``maze`` is used.
    There is one MuJoCo simulator per maze, shared by all slots on that maze: each slot keeps its own
    (qpos, qvel, goal), loaded into the simulator before its step and saved after. Building a MuJoCo env
    takes ~0.1 s, so a maze-split task (a different maze per episode, ~1000 episodes per step) builds each
    maze once instead of one env per slot. At most ``max_sims`` idle simulators are kept (least recently used
    are closed)."""

    def __init__(self, maze: str = "umaze", action_mode: str = "delta", action_repeat: int = 40,
                 kp: float = 10.0, kd: float = 1.0, max_delta: float = 1.0, image_size: int = 336,
                 draw_trail: bool = False, seed: int = 0, settle: bool = True, max_substeps: int = 200,
                 pos_tol: float = 0.05, vel_tol: float = 0.1, blocked_window: int = 20, blocked_eps: float = 0.01,
                 max_sims: int = 256):
        """``settle=True`` (PEAR's EXECUTION semantics, delta mode): run the controller until the target
        is reached and the ball is at rest, or until it is blocked (moved < ``blocked_eps`` over the last
        ``blocked_window`` physics steps), capped at ``max_substeps``. ``settle=False``: exactly
        ``action_repeat`` physics steps per action (always the case in force mode)."""
        assert action_mode in ("delta", "force")
        self.maze_name, self.maze_map = maze, get_maze_map(maze)
        self.action_mode, self.action_repeat = action_mode, action_repeat
        self.kp, self.kd, self.max_delta = kp, kd, max_delta
        self.settle = settle and action_mode == "delta"
        self.max_substeps, self.pos_tol, self.vel_tol = max_substeps, pos_tol, vel_tol
        self.blocked_window, self.blocked_eps = blocked_window, blocked_eps
        self.image_size, self.draw_trail = image_size, draw_trail
        self.seed = seed
        self.max_sims = max_sims
        self._sims: dict[str, Any] = {}         # maze name -> MuJoCo env (insertion order = LRU order)
        self._slot_maze: list[str] = []
        self._maps: dict[str, list] = {maze: self.maze_map}
        self._n_made = 0
        self._qpos: list[np.ndarray] = []       # per-slot state, loaded into the maze's simulator to step
        self._qvel: list[np.ndarray] = []
        self._goals: list[np.ndarray] = []
        self._trails: list[list] = []

    def maze_map_of(self, k: int) -> list[list]:
        return self._maps[self._slot_maze[k]]

    def _make(self, maze: str):
        from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
        if maze not in self._maps:
            self._maps[maze] = get_maze_map(maze)
        env = PointMazeEnv(maze_map=self._maps[maze], continuing_task=False, reward_type="sparse")
        env.reset(seed=self.seed + self._n_made)
        self._n_made += 1
        return env

    def _sim(self, maze: str):
        env = self._sims.pop(maze, None) or self._make(maze)
        self._sims[maze] = env                  # most recently used last
        return env

    def _load(self, k: int):
        """Put slot k's state (and goal) into its maze's simulator and return the env."""
        env = self._sim(self._slot_maze[k])
        env.goal = self._goals[k].copy()
        env.update_target_site_pos()
        env.point_env.set_state(self._qpos[k], self._qvel[k])
        return env

    def _obs(self, k: int) -> Obs:
        pos, vel, goal = self._qpos[k][:2].copy(), self._qvel[k][:2].copy(), self._goals[k]
        img = render_topdown(self.maze_map_of(k), pos, goal, self.image_size,
                             trail=self._trails[k] if self.draw_trail else None)
        return Obs(image=img, state=np.concatenate([pos, vel]),
                   extras={"goal": goal.copy(), "dist": float(np.linalg.norm(pos - goal)), "geo_dist": self._geo_dist(k, pos)})

    def _geo_dist(self, k: int, pos) -> float:
        """Shortest-path length in cells (4-connected, through free cells) from the ball's cell to the goal's."""
        mm = self.maze_map_of(k)
        r, c = _cell_of(pos, mm)
        f = self._geo[k]
        return float(f[r, c]) if 0 <= r < f.shape[0] and 0 <= c < f.shape[1] and np.isfinite(f[r, c]) else float(f.size)

    def reset(self, start_states):
        self._slot_maze = [str(s.get("maze") or self.maze_name) for s in start_states]
        for maze in dict.fromkeys(self._slot_maze):
            self._sim(maze)
        in_use = set(self._slot_maze)
        for maze in [m for m in self._sims if m not in in_use][:max(0, len(self._sims) - self.max_sims)]:
            self._sims.pop(maze).close()        # evict least recently used idle simulators
        self._qpos = [np.asarray(s["qpos"], np.float64).copy() for s in start_states]
        self._qvel = [np.asarray(s.get("qvel", [0, 0]), np.float64).copy() for s in start_states]
        self._goals = [np.asarray(s["goal"], np.float64).copy() for s in start_states]
        self._geo = [geodesic_field(self.maze_map_of(k), g) for k, g in enumerate(self._goals)]
        self._trails = [[q[:2].copy()] for q in self._qpos]
        self._n = len(start_states)
        return [self._obs(k) for k in range(self._n)]

    def step(self, ids, actions):
        out = []
        for k, a in zip(ids, actions):
            env = self._load(k)
            d = env.point_env.data
            a = np.asarray(a, dtype=np.float64)
            if self.action_mode == "delta":
                delta = a.copy()
                n = np.linalg.norm(delta)
                if n > self.max_delta:
                    delta *= self.max_delta / n
                target = d.qpos[:2] + delta
            low, success, stop = [], False, "steps"
            hist = [d.qpos[:2].copy()]
            for i in range(self.max_substeps if self.settle else self.action_repeat):
                if self.action_mode == "delta":
                    u = np.clip(self.kp * (target - d.qpos[:2]) - self.kd * d.qvel[:2], -1, 1)
                else:
                    u = np.clip(a, -1, 1)
                _, _, term, _, info = env.step(u)
                low.append(u)
                pos = d.qpos[:2].copy()
                self._trails[k].append(pos)
                hist.append(pos)
                if term or info["success"]:
                    success, stop = True, "success"
                    break
                if self.settle:
                    if (np.linalg.norm(target - pos) < self.pos_tol and np.linalg.norm(d.qvel[:2]) < self.vel_tol):
                        stop = "reached"
                        break
                    if i >= self.blocked_window and np.linalg.norm(pos - hist[-1 - self.blocked_window]) < self.blocked_eps:
                        stop = "blocked"
                        break
            self._qpos[k], self._qvel[k] = d.qpos.copy(), d.qvel.copy()
            ob = self._obs(k)
            out.append(StepResult(
                obs=ob, reward=float(success), terminated=success,
                info={"success": success, "dist": ob.extras["dist"], "geo_dist": ob.extras["geo_dist"],
                      "low_level_actions": np.stack(low), "stop": stop}))
        return out

    def close(self):
        for e in self._sims.values():
            e.close()
        self._sims, self._slot_maze = {}, []


# ----------------------------------------------------------------------------- agent interface

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_COMPACT_X = re.compile(r'"?\bd?x"?\s*[:=]\s*(' + _NUM + ")", re.IGNORECASE)   # compact style: first x / y value
_COMPACT_Y = re.compile(r'"?\bd?y"?\s*[:=]\s*(' + _NUM + ")", re.IGNORECASE)


class PointMazeTextInterface:
    """Agent-facing protocol for PointMaze.

    ``prompt_style="pear"`` (default) uses PEAR's protocol verbatim (``vlmrl.envs.protocol``): the robot
    is one part ``ball`` with quantities ``x, y``; each turn starts with the state JSON, then the
    image; the action is the *change* ``{"ball": {"x": dx, "y": dy}}`` in a final fenced json block.
    ``prompt_style="simple"`` is a shorter custom prompt with ``<action>{"dx":..,"dy":..}</action>``.
    ``prompt_style="compact"`` is a few-sentence prompt for small models (<1B): each turn is the image +
    "Ball: x=.., y=... Goal: x=.., y=... Move:", and the reply is ``{"x": dx, "y": dy}``, parsed leniently
    (the first x and y values anywhere in the reply).
    ``prompt_style="compact_rel"`` is ``compact`` with the goal given relative to the ball
    ("Goal: x=+2.53, y=-1.90 from the ball."), so the straight-line move is in the prompt.
    ``think`` adds / drops the "Think briefly, then ..." cue (default: on for simple, off for pear / compact).
    """

    SCHEMA = {"ball": ["x", "y"]}

    def __init__(self, maze_map, action_mode: str = "delta", include_state: bool = True,
                 max_delta: float = 1.0, image_size: int = 336, prompt_style: str = "pear",
                 think: bool | None = None):
        assert prompt_style in ("pear", "simple", "compact", "compact_rel")
        if prompt_style in ("pear", "compact", "compact_rel"):
            assert action_mode == "delta", f"{prompt_style} actions are changes of position (delta mode)"
        self.maze_map, self.action_mode = maze_map, action_mode
        self.include_state, self.max_delta = include_state, max_delta
        self.keys = ("dx", "dy") if action_mode == "delta" else ("fx", "fy")
        self.image_size, self.prompt_style = image_size, prompt_style
        self.think = prompt_style == "simple" if think is None else think

    @property
    def _compact(self) -> bool:
        return self.prompt_style in ("compact", "compact_rel")

    # ---------------------------------------------------------------- prompts
    def _map_for(self, start_state) -> list[list]:
        """The maze of this episode (start states of maze-split tasks carry a ``maze`` name)."""
        name = (start_state or {}).get("maze")
        if not name:
            return self.maze_map
        cache = self.__dict__.setdefault("_maps", {})
        if name not in cache:
            cache[name] = get_maze_map(str(name))
        return cache[name]

    def environment_section(self, maze_map=None) -> str:
        maze_map = maze_map or self.maze_map
        h, w = len(maze_map), len(maze_map[0])
        return (
            "Robot and frame:\n"
            f"- The robot is a ball (the BLUE dot) rolling on the floor of a 2D maze {w} cells wide and {h} cells tall.\n"
            "- x points to the right of the image and y points up; (0, 0) is the centre of the image. Each cell has length 1.\n"
            f"- Each action moves the ball by at most {self.max_delta} in total; longer changes are shortened.\n"
            "- Dark cells are walls and block the ball; light cells are free floor. Thin grid lines mark the cell borders.\n"
            "Cameras:\n"
            "- top: fixed directly above the maze, showing the whole maze.\n"
            "Ending:\n"
            "- The episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet."
        )

    def instruction(self, start_state) -> str:
        text = "Move the ball into the GREEN circle."
        if self.include_state and "goal" in start_state:
            g = np.asarray(start_state["goal"], dtype=float)
            text += f" The circle is centred at x={g[0]:.2f}, y={g[1]:.2f}."
        return text

    def system_prompt(self, start_state):
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            text = protocol.system_prompt(self.environment_section(self._map_for(start_state)), self.instruction(start_state),
                                          length_unit="maze cells")
            if self.think:
                text = text.replace("End your response with", "Think briefly, then end your response with", 1)
            return text
        mm = self._map_for(start_state)
        h, w = len(mm), len(mm[0])
        if self._compact:
            return (
                "You move a blue ball through a maze to the green circle. The image shows the maze from above: "
                f"dark cells are walls, light cells are floor. The maze is {w} by {h} cells; x points right, "
                "y points up, 1 unit = 1 cell. Each turn, reply with only the move as JSON, for example "
                '{"x": 1.0, "y": -0.5}. ' f"The ball moves by that amount (at most {self.max_delta:g} per turn) "
                "and stops at walls." + (" Think briefly first." if self.think else "")
            )
        if self.action_mode == "delta":
            act = (f'a relative waypoint {{"dx": <float>, "dy": <float>}} in maze units; the robot moves by '
                   f"roughly that offset this round (clipped to length {self.max_delta}). "
                   "Positive dx = right, positive dy = up.")
        else:
            act = ('a force {"fx": <float>, "fy": <float>} with each component in [-1, 1], held for one '
                   "round. The ball has momentum: forces change velocity, not position.")
        return (
            "You control a small ball (BLUE dot) in a 2D maze, seen from directly above. "
            "Your goal is to move the ball into the GREEN circle.\n"
            f"The maze is {w} cells wide and {h} cells tall. Dark cells are walls; the ball cannot pass "
            "through them. Light cells are free floor. Thin grid lines mark cell borders; each cell is "
            "1.0 unit wide. Coordinates: x increases to the right, y increases upward, (0, 0) is the "
            "centre of the image.\n"
            f"Each round you see the current view and choose {act}\n"
            "Plan a path through free cells around walls; move through cell centres to avoid getting stuck "
            "on corners. " + ("Think briefly, then end" if self.think else "End") +
            " your reply with exactly one action tag, e.g.\n"
            f'<action>{{"{self.keys[0]}": 0.5, "{self.keys[1]}": -1.0}}</action>'
        )

    def observation_message(self, obs, turn, last, parse_error):
        img = {"type": "image", "image": Image.fromarray(obs.image)}
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            parts = []
            if parse_error:
                parts.append("Your previous response contained no valid action, so the robot did not move.")
            if self.include_state and obs.state is not None:
                parts.append(protocol.format_state(self.SCHEMA, obs.state[:2]))
            # PEAR: the state JSON comes first, followed by the camera images
            return ([{"type": "text", "text": "\n".join(parts)}] if parts else []) + [img]
        if self._compact:
            text = "No valid move last turn. " if parse_error else ""
            if self.include_state and obs.state is not None:
                p, g = obs.state[:2], obs.extras["goal"]
                text += f"Ball: x={p[0]:.2f}, y={p[1]:.2f}. "
                if self.prompt_style == "compact_rel":
                    text += f"Goal: x={g[0] - p[0]:+.2f}, y={g[1] - p[1]:+.2f} from the ball. "
                else:
                    text += f"Goal: x={g[0]:.2f}, y={g[1]:.2f}. "
            return [img, {"type": "text", "text": text + "Move:"}]
        text = []
        if parse_error:
            text.append("Your previous reply had no valid <action> tag, so the ball did nothing.")
        text.append(f"Round {turn + 1}.")
        if self.include_state and obs.state is not None:
            p, v, g = obs.state[:2], obs.state[2:4], obs.extras["goal"]
            text.append(f"Ball position: ({p[0]:.2f}, {p[1]:.2f}), velocity: ({v[0]:.2f}, {v[1]:.2f}). "
                        f"Goal position: ({g[0]:.2f}, {g[1]:.2f}).")
        text.append("Choose your next action.")
        return [img, {"type": "text", "text": " ".join(text)}]

    # ---------------------------------------------------------------- actions
    def parse_action(self, text):
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.parse_action(text, self.SCHEMA)
        if self._compact:
            mx, my = _COMPACT_X.search(text), _COMPACT_Y.search(text)
            if not (mx and my):
                return None
            a = np.array([float(mx.group(1)), float(my.group(1))])
            return a if np.all(np.isfinite(a)) else None
        m = _ACTION_RE.findall(text)
        if not m:
            return None
        body = m[-1].strip()
        vals = None
        try:
            obj = json.loads(body)
            vals = [float(obj[k]) for k in self.keys]
        except Exception:
            nums = re.findall(_NUM, body)   # lenient fallback: "<action>0.5, -1</action>"
            if len(nums) >= 2:
                vals = [float(nums[0]), float(nums[1])]
        if vals is None or not all(np.isfinite(vals)):
            return None
        a = np.array(vals, dtype=np.float64)
        return np.clip(a, -1, 1) if self.action_mode == "force" else a

    def format_action(self, a) -> str:
        """Inverse of parse_action (for scripted policies / SFT data)."""
        if self.prompt_style == "pear":
            from vlmrl.envs import protocol
            return protocol.example_action(self.SCHEMA, a)
        if self._compact:
            return f'{{"x": {a[0]:.2f}, "y": {a[1]:.2f}}}'
        return f'<action>{{"{self.keys[0]}": {a[0]:.3f}, "{self.keys[1]}": {a[1]:.3f}}}</action>'

    def noop_action(self):
        return np.zeros(2)


# ----------------------------------------------------------------------------- oracle

def _cell_of(xy, maze_map):
    h, w = len(maze_map), len(maze_map[0])
    return int(np.floor(h / 2 - xy[1])), int(np.floor(xy[0] + w / 2))


def _centre_of(cell, maze_map):
    h, w = len(maze_map), len(maze_map[0])
    return np.array([cell[1] + 0.5 - w / 2, h / 2 - cell[0] - 0.5])


def geodesic_field(maze_map, goal) -> np.ndarray:
    """BFS distance in cells from the goal's cell to every free cell (inf for walls / unreachable)."""
    from collections import deque
    walls = _walls(maze_map)
    f = np.full(walls.shape, np.inf)
    g = _cell_of(goal, maze_map)
    f[g] = 0
    q = deque([g])
    while q:
        c = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1] and not walls[n] and f[n] == np.inf:
                f[n] = f[c] + 1
                q.append(n)
    return f


def oracle_delta(maze_map, pos, goal, max_delta: float = 1.0) -> np.ndarray:
    """BFS over free cells; aim at the next cell centre on the shortest path (or the goal itself)."""
    from collections import deque
    pos = np.asarray(pos, dtype=float)
    walls = _walls(maze_map)
    s, g = _cell_of(pos, maze_map), _cell_of(goal, maze_map)
    prev, q = {s: None}, deque([s])
    while q:
        c = q.popleft()
        if c == g:
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1] and not walls[n] and n not in prev:
                prev[n] = c
                q.append(n)
    path, c = [], g
    while c is not None and c in prev:
        path.append(c)
        c = prev[c]
    path = path[::-1]
    if len(path) <= 1:
        target = np.asarray(goal)
    else:
        # go as far as possible along the straight run of path cells in the current row / column
        j = 1
        while (j + 1 < len(path) and (path[j + 1][0] == path[0][0] or path[j + 1][1] == path[0][1])
               and np.linalg.norm(_centre_of(path[j + 1], maze_map) - pos) <= max_delta):
            j += 1
        target = np.asarray(goal) if j == len(path) - 1 else _centre_of(path[j], maze_map)
    delta = target - np.asarray(pos)
    n = np.linalg.norm(delta)
    return delta * (max_delta / n) if n > max_delta else delta


# ----------------------------------------------------------------------------- start states

SPLITS = ("none", "pair", "goal")


def split_pairs(maze_map, min_dist: float = 1.0, split: str = "pair", holdout_frac: float = 0.25,
                seed: int = 0) -> tuple[list, list]:
    """Deterministic train / held-out partition of (start cell, goal cell) index pairs over free cells.

    ``split="pair"`` holds out a random ``holdout_frac`` of the pairs (every cell is seen in training, some
    start-goal combinations are not). ``split="goal"`` holds out a random ``holdout_frac`` of the goal
    cells: training never asks the ball to reach them. Only pairs whose cell centres are >= ``min_dist``
    apart are used. Indices refer to ``np.argwhere(~walls)`` order (= ``RandomStartSampler.centres``)."""
    assert split in ("pair", "goal"), split
    free = np.argwhere(~_walls(maze_map))
    h, w = len(maze_map), len(maze_map[0])
    centres = np.stack([free[:, 1] + 0.5 - w / 2, h / 2 - free[:, 0] - 0.5], 1)
    pairs = [(a, b) for a in range(len(free)) for b in range(len(free))
             if a != b and np.linalg.norm(centres[a] - centres[b]) >= min_dist]
    rng = np.random.default_rng(seed)
    if split == "pair":
        perm = rng.permutation(len(pairs))
        hold = set(perm[:max(1, round(holdout_frac * len(pairs)))].tolist())
        held = [p for i, p in enumerate(pairs) if i in hold]
    else:
        goals = sorted({b for _, b in pairs})
        hold = set(rng.permutation(goals)[:max(1, round(holdout_frac * len(goals)))].tolist())
        held = [p for p in pairs if p[1] in hold]
    train = [p for p in pairs if p not in held]
    assert train and held, f"split={split} holdout_frac={holdout_frac} leaves an empty part"
    return train, held


class RandomStartSampler:
    """Random start cell + goal cell (uniform over free cells, with position noise), zero velocity.

    With ``split`` != "none", (start cell, goal cell) pairs are drawn uniformly from one ``part`` of
    :func:`split_pairs` ("train" | "heldout" | "all"). Position noise stays inside the cell, so a held-out
    pair is never produced by the train part."""

    def __init__(self, maze_map, min_dist: float = 1.0, noise: float = 0.25, split: str = "none",
                 part: str = "all", holdout_frac: float = 0.25, split_seed: int = 0):
        assert split in SPLITS and part in ("train", "heldout", "all"), (split, part)
        free = np.argwhere(~_walls(maze_map))
        h, w = len(maze_map), len(maze_map[0])
        self.centres = np.stack([free[:, 1] + 0.5 - w / 2, h / 2 - free[:, 0] - 0.5], 1)
        self.min_dist, self.noise = min_dist, noise
        self.pairs = None
        if split != "none":
            train, held = split_pairs(maze_map, min_dist, split, holdout_frac, split_seed)
            self.pairs = {"train": train, "heldout": held, "all": train + held}[part]

    def sample(self, n, rng):
        if self.pairs is not None:
            return [self._sample_pair(rng) for _ in range(n)]
        out = []
        for _ in range(n):
            while True:
                a, b = rng.choice(len(self.centres), 2, replace=False)
                p = self.centres[a] + rng.uniform(-self.noise, self.noise, 2)
                g = self.centres[b] + rng.uniform(-self.noise, self.noise, 2)
                if np.linalg.norm(p - g) >= self.min_dist:
                    break
            out.append({"qpos": p, "qvel": np.zeros(2), "goal": g})
        return out

    def _sample_pair(self, rng):
        a, b = self.pairs[rng.integers(len(self.pairs))]
        for _ in range(100):  # centres are >= min_dist apart, so this only re-draws the noise
            p = self.centres[a] + rng.uniform(-self.noise, self.noise, 2)
            g = self.centres[b] + rng.uniform(-self.noise, self.noise, 2)
            if np.linalg.norm(p - g) >= self.min_dist:
                break
        else:
            p, g = self.centres[a].copy(), self.centres[b].copy()
        return {"qpos": p, "qvel": np.zeros(2), "goal": g}


class MazeSetSampler:
    """A maze drawn uniformly from ``mazes`` per episode, then a random start/goal in it (RandomStartSampler).
    The state carries the maze name, which the VecEnv and the text interface follow."""

    def __init__(self, mazes: Sequence[str], min_dist: float = 1.0, noise: float = 0.25):
        assert mazes, "empty maze set"
        self.mazes, self.min_dist, self.noise = list(mazes), min_dist, noise
        self._samplers: dict[str, RandomStartSampler] = {}

    def sample(self, n, rng):
        out = []
        for _ in range(n):
            m = self.mazes[int(rng.integers(len(self.mazes)))]
            if m not in self._samplers:
                self._samplers[m] = RandomStartSampler(get_maze_map(m), self.min_dist, self.noise)
            s = self._samplers[m].sample(1, rng)[0]
            s["maze"] = m
            out.append(s)
        return out


class OfflineStartSampler:
    """Draw start states (qpos, qvel, goal) from offline trajectories.

    Accepts either a Minari dataset id (e.g. ``D4RL/pointmaze/umaze-v2``) or an ``.npz`` with arrays
    ``qpos [N,2]``, ``qvel [N,2]``, ``goal [N,2]`` (any extra arrays are passed through in the state
    dict, e.g. ``episode``/``t`` indices or a WAM latent key). This is the hook for "use offline data
    as reference starting states".
    """

    def __init__(self, source: str, min_goal_dist: float = 1.0, max_states: int | None = 200_000,
                 download: bool = True):
        if source.endswith(".npz"):
            data = dict(np.load(source))
        else:
            data = self._from_minari(source, max_states, download)
        far = np.linalg.norm(data["qpos"] - data["goal"], axis=1) >= min_goal_dist
        self.data = {k: v[far] for k, v in data.items()}
        self.n = int(far.sum())
        assert self.n > 0, "no offline states satisfy min_goal_dist"

    @staticmethod
    def _from_minari(dataset_id, max_states, download):
        import minari
        ds = minari.load_dataset(dataset_id, download=download)
        qpos, qvel, goal, ep, t = [], [], [], [], []
        total = 0
        for e in ds.iterate_episodes():
            o = e.observations
            L = len(o["observation"]) - 1
            qpos.append(o["observation"][:L, :2]); qvel.append(o["observation"][:L, 2:4])
            goal.append(o["desired_goal"][:L]); ep.append(np.full(L, e.id)); t.append(np.arange(L))
            total += L
            if max_states and total >= max_states:
                break
        return {"qpos": np.concatenate(qpos), "qvel": np.concatenate(qvel), "goal": np.concatenate(goal),
                "episode": np.concatenate(ep), "t": np.concatenate(t)}

    def sample(self, n, rng):
        idx = rng.integers(0, self.n, n)
        return [{k: v[i] for k, v in self.data.items()} for i in idx]


# ----------------------------------------------------------------------------- factory

TEST_SEED_OFFSET = 10_000   # held-out procedural mazes use seeds >= this; training mazes use seeds < it


@dataclass
class PointMazeConfig:
    maze: str = "umaze"               # a named / generated maze, or a procedural family "gen<rows>x<cols>" (split)
    action_mode: str = "delta"
    action_repeat: int = 40
    max_delta: float = 1.0
    image_size: int = 336
    include_state: bool = True
    prompt_style: str = "pear"        # "pear" (paper Appendix H protocol) | "simple" | "compact" (small models) | "compact_rel" (goal relative to ball)
    think: bool | None = None         # "Think briefly" cue; None = style default (simple: on, pear: off)
    settle: bool = True               # PEAR execution: act until target reached & at rest, or blocked
    max_turns: int = 12
    start_states: str = "random"      # "random" | minari dataset id | path.npz
    eval_start_states: str = "random"
    min_goal_dist: float = 1.5
    # train / eval split of random starts within one maze (see split_pairs): "none" | "pair" | "goal"
    split: str = "none"
    holdout_frac: float = 0.25
    split_seed: int = 0
    # which part the eval sampler draws from: heldout | train | all. Within one maze (split != "none"): the
    # (start, goal) pairs of split_pairs. For a procedural family (maze=gen<rows>x<cols>): the *layouts*, training
    # seeds [0, train_mazes) vs held-out seeds [TEST_SEED_OFFSET, TEST_SEED_OFFSET + eval_mazes).
    eval_split: str = "heldout"
    train_mazes: int = 64
    eval_mazes: int = 16


def make_pointmaze(cfg: PointMazeConfig | None = None, **kw) -> EnvBundle:
    cfg = cfg or PointMazeConfig(**kw)
    if _GEN_FAMILY_RE.fullmatch(cfg.maze):   # maze split: a different procedural maze per episode
        assert cfg.start_states == cfg.eval_start_states == "random", "procedural mazes have no offline data"
        assert cfg.split == "none", "a procedural family already splits by layout; split= is for a single maze"
        assert cfg.eval_split in ("heldout", "train", "all"), cfg.eval_split
        assert cfg.train_mazes <= TEST_SEED_OFFSET, "training seeds would overlap the held-out ones"
        train = maze_family(cfg.maze, cfg.train_mazes)
        held = maze_family(cfg.maze, cfg.eval_mazes, TEST_SEED_OFFSET)
        test = {"heldout": held, "train": train, "all": train + held}[cfg.eval_split]
        default_maze = train[0]
        train_sampler = MazeSetSampler(train, min_dist=cfg.min_goal_dist)
        eval_sampler = MazeSetSampler(test, min_dist=cfg.min_goal_dist)
    else:
        default_maze = cfg.maze
        train_sampler = eval_sampler = None
    maze_map = get_maze_map(default_maze)

    def sampler(spec, part):
        if spec == "random":
            return RandomStartSampler(maze_map, min_dist=cfg.min_goal_dist, split=cfg.split, part=part,
                                      holdout_frac=cfg.holdout_frac, split_seed=cfg.split_seed)
        if cfg.split != "none":
            raise NotImplementedError("split applies to random start states only")
        return OfflineStartSampler(spec, min_goal_dist=cfg.min_goal_dist)

    return EnvBundle(
        make_vec_env=lambda: PointMazeSimVecEnv(default_maze, cfg.action_mode, cfg.action_repeat,
                                                max_delta=cfg.max_delta, image_size=cfg.image_size,
                                                settle=cfg.settle),
        interface=PointMazeTextInterface(maze_map, cfg.action_mode, cfg.include_state, cfg.max_delta,
                                         cfg.image_size, cfg.prompt_style, cfg.think),
        train_sampler=train_sampler or sampler(cfg.start_states, "train"),
        eval_sampler=eval_sampler or sampler(cfg.eval_start_states, cfg.eval_split),
        max_turns=cfg.max_turns,
    )
