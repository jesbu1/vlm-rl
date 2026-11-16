"""Core data types shared by environments, rollouts, rewards and the trainer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Obs:
    """What an environment (simulator *or* learned world model) emits each step.

    `image` is an HxWx3 uint8 frame (or None for text-only envs). `state` is an optional
    low-dimensional vector the text interface may choose to verbalise. `extras` holds anything
    env-specific (e.g. the goal) that the interface needs to build the prompt.
    """
    image: np.ndarray | None = None
    state: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    obs: Obs
    reward: float = 0.0
    terminated: bool = False   # task ended (success / failure) -> no more turns
    truncated: bool = False    # env-side time limit
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """One agent round: the observation it saw, what it said, what was executed."""
    obs: Obs
    response: str
    action: np.ndarray | None          # parsed action (None => parse failure, a no-op was executed)
    result: StepResult


@dataclass
class Trajectory:
    """A full multi-turn episode.

    `messages` is the chat transcript in HF multimodal format (PIL images inline). It is the
    single source of truth for what the policy saw, and is re-tokenised for training.
    """
    group_id: int
    start_state: dict[str, Any]
    messages: list[dict] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    done: bool = False
    # filled in by reward fn / algorithm
    score: float = 0.0
    advantage: float = 0.0

    @property
    def env_return(self) -> float:
        return float(sum(t.result.reward for t in self.turns))

    @property
    def success(self) -> bool:
        return any(bool(t.result.info.get("success", False)) for t in self.turns)

    @property
    def num_invalid(self) -> int:
        return sum(t.action is None for t in self.turns)

    @property
    def frames(self) -> list[np.ndarray]:
        fr = [t.obs.image for t in self.turns]
        if self.turns:
            fr.append(self.turns[-1].result.obs.image)
        return [f for f in fr if f is not None]
