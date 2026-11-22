"""Environment abstractions.

An "environment" for VLM-RL is split into three swappable pieces:

* ``VecEnv``          -- the *dynamics*: batched reset-from-state + step. A MuJoCo simulator and a
                         learned world-action-model (WAM) both implement this.
* ``TextInterface``   -- the *agent-facing protocol*: system prompt, how an ``Obs`` becomes a chat
                         message, and how generated text is parsed into an action. Shared by the
                         sim and the WAM so the policy sees an identical task.
* ``StartStateSampler`` -- where episodes start: random resets, or states drawn from offline data
                         (the "reference starting states" a WAM would roll out from).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from vlmrl.types import Obs, StepResult


@runtime_checkable
class VecEnv(Protocol):
    """Batch of independent episodes addressed by slot index.

    ``reset`` (re)creates ``len(start_states)`` slots. ``step`` advances only the given slots so the
    rollout loop can keep stepping unfinished episodes while finished ones idle. A WAM implements
    ``step`` as one batched forward pass over ``ids``.
    """

    def reset(self, start_states: Sequence[dict[str, Any]]) -> list[Obs]: ...

    def step(self, ids: Sequence[int], actions: Sequence[np.ndarray]) -> list[StepResult]: ...

    def close(self) -> None: ...


class TextInterface(Protocol):
    def system_prompt(self, start_state: dict[str, Any]) -> str: ...

    def observation_message(self, obs: Obs, turn: int, last: StepResult | None,
                            parse_error: bool) -> list[dict]:
        """Return HF chat ``content`` parts, e.g. [{"type":"image","image":PIL}, {"type":"text",...}]."""
        ...

    def parse_action(self, text: str) -> np.ndarray | None: ...

    def noop_action(self) -> np.ndarray: ...


class StartStateSampler(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]: ...


@dataclass
class EnvBundle:
    """Everything the trainer needs from a task. Returned by env factories in the registry."""
    make_vec_env: Any                  # () -> VecEnv  (called once; slots are created by reset)
    interface: TextInterface
    train_sampler: StartStateSampler
    eval_sampler: StartStateSampler
    max_turns: int
    expert: Any = None                 # optional scripted expert (vec_env, slot) -> action, for behaviour cloning
