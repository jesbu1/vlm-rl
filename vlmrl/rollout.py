"""Batched multi-turn rollouts and conversion of trajectories into training samples."""
from __future__ import annotations

from typing import Callable, Sequence

from vlmrl.envs.base import TextInterface, VecEnv
from vlmrl.policy import Sample
from vlmrl.types import Trajectory, Turn

Generate = Callable[[list[list[dict]]], list[str]]


def context_view(messages: list[dict], max_images: int | None) -> list[dict]:
    """What the policy actually conditions on. With ``max_images`` set, only the most recent images
    are kept; older ones become a short text placeholder (text history is kept)."""
    if max_images is None:
        return messages
    n_img = sum(1 for m in messages for c in m["content"] if c.get("type") == "image")
    drop = max(0, n_img - max_images)
    out = []
    for m in messages:
        content = []
        for c in m["content"]:
            if c.get("type") == "image" and drop > 0:
                content.append({"type": "text", "text": "[earlier image omitted]"})
                drop -= 1
            else:
                content.append(c)
        out.append({"role": m["role"], "content": content})
    return out


def rollout(generate: Generate, env: VecEnv, interface: TextInterface, start_states: Sequence[dict],
            group_ids: Sequence[int], max_turns: int, max_images: int | None = None) -> list[Trajectory]:
    """Run ``len(start_states)`` episodes in lock-step. Each round, every unfinished episode gets one
    generation (batched) and one env step (batched over the still-active slots)."""
    obs = env.reset(start_states)
    trajs = []
    for k, s in enumerate(start_states):
        msgs = [{"role": "system", "content": [{"type": "text", "text": interface.system_prompt(s)}]},
                {"role": "user", "content": interface.observation_message(obs[k], 0, None, False)}]
        trajs.append(Trajectory(group_id=group_ids[k], start_state=s, messages=msgs))
    active = list(range(len(trajs)))
    for t in range(max_turns):
        if not active:
            break
        texts = generate([context_view(trajs[k].messages, max_images) for k in active])
        parsed = [interface.parse_action(x) for x in texts]
        actions = [a if a is not None else interface.noop_action() for a in parsed]
        results = env.step(active, actions)
        still = []
        for k, text, a, res in zip(active, texts, parsed, results):
            tr = trajs[k]
            tr.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            tr.turns.append(Turn(obs=obs[k], response=text, action=a, result=res))
            obs[k] = res.obs
            if res.terminated or res.truncated or t == max_turns - 1:
                tr.done = True
            else:
                tr.messages.append({"role": "user", "content": interface.observation_message(
                    res.obs, t + 1, res, parse_error=a is None)})
                still.append(k)
        active = still
    return trajs


def to_samples(trajs: Sequence[Trajectory], max_images: int | None) -> list[Sample]:
    """Full context -> one sequence per episode with every assistant turn in the loss (PEAR-style).
    Truncated image history -> one sequence per turn, each with exactly the context that turn saw."""
    samples = []
    for tr in trajs:
        a_idx = [i for i, m in enumerate(tr.messages) if m["role"] == "assistant"]
        if max_images is None:
            samples.append(Sample(tr.messages, a_idx, tr.advantage, {"group": tr.group_id}))
        else:
            for i in a_idx:
                view = context_view(tr.messages[:i], max_images) + [tr.messages[i]]
                samples.append(Sample(view, [i], tr.advantage, {"group": tr.group_id}))
    return samples
