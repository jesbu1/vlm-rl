"""Shared fixtures + test tiers.

Tiers
  default      fast unit/integration tests, CPU only, no downloads (~seconds)
  --run-slow   also run tests that load a real (tiny) HF VLM; downloads SmolVLM-256M if missing
  --run-gpu    also run tests that need CUDA (larger model smoke tests)
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # so `examples.*` is importable

from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze, oracle_delta  # noqa: E402
from vlmrl.policy import Sample  # noqa: E402
from vlmrl.types import Obs, StepResult, Trajectory, Turn  # noqa: E402

TINY_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="run tests that load a real tiny HF VLM")
    parser.addoption("--run-gpu", action="store_true", help="run tests that need a CUDA GPU")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="needs --run-slow"))
        if "gpu" in item.keywords and not (config.getoption("--run-gpu") and torch.cuda.is_available()):
            item.add_marker(pytest.mark.skip(reason="needs --run-gpu and CUDA"))


# ----------------------------------------------------------------------------- env fixtures

@pytest.fixture
def umaze():
    return make_pointmaze(PointMazeConfig(maze="umaze", max_turns=6, image_size=64))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_traj(dist: float = 1.0, success: bool = False, n_turns: int = 1, invalid: int = 0,
              group_id: int = 0) -> Trajectory:
    """Synthetic trajectory with controllable outcome (no env needed)."""
    t = Trajectory(group_id=group_id, start_state={})
    for i in range(n_turns):
        last = i == n_turns - 1
        act = None if i < invalid else np.zeros(2)
        t.turns.append(Turn(Obs(), f"turn{i}", act, StepResult(
            Obs(), float(success and last), success and last,
            info={"dist": dist if last else dist + 1, "success": success and last})))
    return t


# ----------------------------------------------------------------------------- scripted "policies"

_FLOAT = re.compile(r"[-+]?\d+\.\d+")
_GOAL = re.compile(r"centred at x=([-+]?\d+(?:\.\d+)?), y=([-+]?\d+(?:\.\d+)?)")


def _fmt(d, style):
    if style == "pear":
        return f'Going.\n```json\n{{"ball": {{"x": {d[0]:.3f}, "y": {d[1]:.3f}}}}}\n```'
    return f'Going. <action>{{"dx": {d[0]:.3f}, "dy": {d[1]:.3f}}}</action>'


def _pos_goal(conv):
    """Read ball position + goal from a PointMaze conversation (either prompt style)."""
    texts = [c["text"] for c in conv[-1]["content"] if c["type"] == "text"]
    m = _GOAL.search(conv[0]["content"][0]["text"])
    if m:  # pear: state JSON in the last user message, goal in the system prompt
        st = json.loads(texts[-1].splitlines()[-1])
        return np.array([st["ball"]["x"], st["ball"]["y"]]), np.array([float(m.group(1)), float(m.group(2))])
    nums = [float(x) for x in _FLOAT.findall(texts[-1])]
    return np.array(nums[0:2]), np.array(nums[4:6])


def oracle_generate(maze_map, style: str = "pear"):
    """Text policy that reads position/goal from the prompt and answers with the BFS oracle."""
    def gen(convs):
        out = []
        for c in convs:
            pos, goal = _pos_goal(c)
            out.append(_fmt(oracle_delta(maze_map, pos, goal), style))
        return out
    return gen


def random_generate(seed: int = 0, p_invalid: float = 0.0, style: str = "pear"):
    r = np.random.default_rng(seed)

    def gen(convs):
        out = []
        for _ in convs:
            if r.random() < p_invalid:
                out.append("I am not sure.")
            else:
                out.append(_fmt(r.uniform(-1, 1, 2), style))
        return out
    return gen


def const_generate(d=(0.0, 0.0), style: str = "pear"):
    return lambda convs: [_fmt(d, style)] * len(convs)


class FakePolicy:
    """Drop-in for HFPolicy with a 1-parameter-vector "model": lets the full trainer loop run in
    milliseconds. logp of every response token is ``theta[group % 4]`` so gradients are checkable."""

    device = "cpu"

    def __init__(self, generate_fn):
        self.model = torch.nn.Module()
        self.model.theta = torch.nn.Parameter(torch.zeros(4))
        self._gen = generate_fn
        self.saved = []

    def generate(self, convs, max_new_tokens=0, temperature=1.0, top_p=1.0, batch_size=0):
        return self._gen(convs)

    def build_batch(self, samples: list[Sample]):
        T = 8
        mask = torch.zeros(len(samples), T)
        mask[:, 4:] = 1
        return {"loss_mask": mask, "advantages": torch.tensor([s.advantage for s in samples]),
                "group": torch.tensor([s.meta["group"] % 4 for s in samples])}

    def token_logprobs(self, batch, topk=0, gather_ids=None):
        logp = -torch.nn.functional.softplus(self.model.theta[batch["group"]])[:, None].expand(-1, 4)
        mask = batch["loss_mask"][:, 4:]
        if not topk and gather_ids is None:
            return logp, mask
        st = {}
        if topk:  # fake 2-token vocabulary: p(target) and 1 - p(target)
            p = logp.detach().exp()
            st["topk_ids"] = torch.zeros(*logp.shape, 2, dtype=torch.long)
            st["topk_ids"][..., 1] = 1
            st["topk_probs"] = torch.stack([p, 1 - p], -1)
        if gather_ids is not None:
            lp1 = torch.log1p(-logp.exp().clamp(max=1 - 1e-6))
            st["logp_at_ids"] = torch.stack([logp, lp1], -1)
        return logp, mask, st

    def ref_token_logprobs(self, batch):
        """Reference policy = the initial theta (zeros)."""
        return -torch.nn.functional.softplus(torch.zeros(4)[batch["group"]])[:, None].expand(-1, 4)

    def save(self, path):
        self.saved.append(path)


@pytest.fixture(scope="session")
def tiny_vlm(request):
    if not request.config.getoption("--run-slow"):
        pytest.skip("needs --run-slow")
    from vlmrl.policy import HFPolicy
    return HFPolicy(TINY_VLM, device="cpu", processor_kwargs={"do_image_splitting": False},
                    gradient_checkpointing=False)
