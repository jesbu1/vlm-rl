"""Adapter: any learned world(-action) model -> ``VecEnv``.

A WAM only has to implement two batched calls:

    init(start_states)          -> model states  (encode offline reference frames / latents)
    step(states, actions)       -> next states, frames, extras

``WorldModelVecEnv`` handles slot bookkeeping, so the trainer cannot tell a WAM from the simulator.
Rewards normally come from a ``PreferenceModel`` (see ``vlmrl.rewards.preference_reward``) that
compares imagined trajectories within a GRPO group -- the env itself can return reward 0.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

import numpy as np

from vlmrl.types import Obs, StepResult


class WorldActionModel(Protocol):
    def init(self, start_states: Sequence[dict[str, Any]]) -> list[Any]: ...

    def step(self, states: list[Any], actions: np.ndarray) -> tuple[list[Any], list[np.ndarray], dict[str, np.ndarray]]:
        """Batched transition. ``extras`` may contain per-item arrays such as ``reward``, ``done``
        (probability or bool), ``success`` or ``state`` (a low-dim estimate for the prompt)."""
        ...

    def render(self, states: list[Any]) -> list[np.ndarray]: ...


class WorldModelVecEnv:
    def __init__(self, wam: WorldActionModel, carry_keys: Sequence[str] = ("goal",),
                 done_threshold: float = 0.5, obs_fn: Callable[..., Obs] | None = None):
        """``carry_keys``: start-state fields copied into every ``Obs.extras`` (e.g. the goal, which a
        text interface may verbalise). ``obs_fn(frame, extras_i, start_state)`` overrides Obs building."""
        self.wam, self.carry_keys, self.done_threshold = wam, tuple(carry_keys), done_threshold
        self.obs_fn = obs_fn
        self._states: list[Any] = []
        self._starts: list[dict] = []

    def _obs(self, k, frame, ex):
        if self.obs_fn:
            return self.obs_fn(frame, ex, self._starts[k])
        extras = {c: self._starts[k][c] for c in self.carry_keys if c in self._starts[k]}
        state = ex.get("state")
        return Obs(image=frame, state=None if state is None else np.asarray(state), extras=extras)

    def reset(self, start_states):
        self._starts = list(start_states)
        self._states = self.wam.init(self._starts)
        frames = self.wam.render(self._states)
        return [self._obs(k, f, {}) for k, f in enumerate(frames)]

    def step(self, ids, actions):
        ids = list(ids)
        nxt, frames, extras = self.wam.step([self._states[k] for k in ids], np.stack(actions))
        out = []
        for j, k in enumerate(ids):
            self._states[k] = nxt[j]
            ex = {key: v[j] for key, v in extras.items()}
            done = bool(float(ex.get("done", 0.0)) >= self.done_threshold)
            info = {key: v for key, v in ex.items() if key != "state"}
            info["success"] = bool(ex.get("success", False))
            out.append(StepResult(obs=self._obs(k, frames[j], ex), reward=float(ex.get("reward", 0.0)),
                                  terminated=done, info=info))
        return out

    def close(self):
        pass
