"""Toy "world model" for PointMaze, showing how a WAM + preference rewards plug in.

The toy WAM is a kinematic stand-in (position += clipped delta, walls ignored) that renders frames
with the same top-down renderer. Replace ``ToyKinematicWAM`` with your trained WAM and
``ToyProgressPreference`` with its preference head; the rest of the pipeline is unchanged.

    python scripts/train.py --env examples.toy_wam:make_bundle --reward examples.toy_wam:make_reward \
        --env_kwargs '{"start_states": "D4RL/pointmaze/umaze-v2"}' ...
"""
from __future__ import annotations

import numpy as np

from vlmrl.envs.base import EnvBundle
from vlmrl.envs.pointmaze import (OfflineStartSampler, PointMazeTextInterface, RandomStartSampler,
                                  get_maze_map, render_topdown)
from vlmrl.envs.world_model import WorldModelVecEnv
from vlmrl.rewards import ScoreFnPreference, preference_reward


class ToyKinematicWAM:
    def __init__(self, maze: str = "umaze", image_size: int = 336, max_delta: float = 1.0):
        self.maze_map, self.size, self.max_delta = get_maze_map(maze), image_size, max_delta

    def init(self, start_states):
        # a real WAM would encode a reference frame (e.g. rendered from the offline state) into a latent
        return [{"pos": np.asarray(s["qpos"], float)[:2].copy(), "goal": np.asarray(s["goal"], float)}
                for s in start_states]

    def render(self, states):
        return [render_topdown(self.maze_map, s["pos"], s["goal"], self.size) for s in states]

    def step(self, states, actions):
        nxt, dist = [], []
        for s, a in zip(states, actions):
            n = np.linalg.norm(a)
            a = a * (self.max_delta / n) if n > self.max_delta else a
            ns = {"pos": s["pos"] + a, "goal": s["goal"]}
            nxt.append(ns)
            dist.append(np.linalg.norm(ns["pos"] - ns["goal"]))
        dist = np.array(dist)
        # a WAM could predict done/success; here we expose distance only and let preferences do the work
        return nxt, self.render(nxt), {"dist": dist, "done": (dist < 0.45).astype(float),
                                       "success": dist < 0.45}


class ToyProgressPreference:
    """Prefers the imagined trajectory whose final frame is closer to the goal. A WAM preference head
    would instead look at (frames, actions) of both trajectories and output P(a > b)."""

    def __init__(self, tau: float = 0.2):
        self.inner = ScoreFnPreference(lambda t: -t.turns[-1].result.info["dist"] if t.turns else 0.0, tau)

    def compare(self, pairs):
        return self.inner.compare(pairs)


def make_bundle(maze: str = "umaze", start_states: str = "random", max_turns: int = 8,
                image_size: int = 336, include_state: bool = False, min_goal_dist: float = 1.5):
    mm = get_maze_map(maze)
    sampler = (RandomStartSampler(mm, min_goal_dist) if start_states == "random"
               else OfflineStartSampler(start_states, min_goal_dist))
    return EnvBundle(
        make_vec_env=lambda: WorldModelVecEnv(ToyKinematicWAM(maze, image_size)),
        interface=PointMazeTextInterface(mm, "delta", include_state=include_state),
        train_sampler=sampler, eval_sampler=sampler, max_turns=max_turns)


def make_reward():
    return preference_reward(ToyProgressPreference(), mode="winrate")
