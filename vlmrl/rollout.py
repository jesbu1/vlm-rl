"""Batched multi-turn rollouts and conversion of trajectories into training samples."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from vlmrl.envs.base import TextInterface, VecEnv
from vlmrl.policy import Sample
from vlmrl.types import StepResult, Trajectory, Turn

Generate = Callable[[list[list[dict]]], list[str]]


def context_view(messages: list[dict], max_images: int | None, max_frames: int | None = None) -> list[dict]:
    """What the policy actually conditions on. With ``max_frames`` set, only the system prompt and the
    last ``max_frames`` rounds (observation message + the replies after it) are kept; older rounds are
    dropped entirely, text included. With ``max_images`` set, only the most recent images are kept;
    older ones become a short text placeholder (text history is kept)."""
    if max_frames is not None:
        users = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(users) > max_frames:
            cut = users[-max_frames]
            messages = [m for i, m in enumerate(messages) if i >= cut or (i < users[0])]
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
            group_ids: Sequence[int], max_turns: int, max_images: int | None = None,
            on_invalid: str = "terminate", max_context_tokens: int | None = None,
            last_tokens: Callable[[], list[int]] | None = None,
            max_frames: int | None = None) -> list[Trajectory]:
    """Run ``len(start_states)`` episodes in lock-step. Each round, every unfinished episode gets one
    generation (batched) and one env step (batched over the still-active slots).

    Episode end (PEAR, paper App. A): the env reports success/termination, the step limit
    (``max_turns``) is hit, the context reaches ``max_context_tokens``, or -- with
    ``on_invalid="terminate"`` -- a response contains no valid action. ``on_invalid="noop"`` instead
    executes ``interface.noop_action()`` and tells the model in the next observation.
    ``last_tokens()`` returns the context length (prompt + reply tokens) of each item of the most
    recent ``generate`` call; it is needed only for ``max_context_tokens``.
    """
    assert on_invalid in ("terminate", "noop")
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
        texts = generate([context_view(trajs[k].messages, max_images, max_frames) for k in active])
        ntok = last_tokens() if (max_context_tokens and last_tokens) else [0] * len(active)
        parsed = [interface.parse_action(x) for x in texts]
        stepping = [j for j, a in enumerate(parsed) if a is not None or on_invalid == "noop"]
        results = env.step([active[j] for j in stepping],
                           [parsed[j] if parsed[j] is not None else interface.noop_action() for j in stepping]
                           ) if stepping else []
        res_of = dict(zip(stepping, results))
        still = []
        for j, (k, text, a) in enumerate(zip(active, texts, parsed)):
            tr = trajs[k]
            tr.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            if j not in res_of:                      # invalid action -> episode over (PEAR)
                # state is unchanged: carry the latest step info + scalar obs extras (e.g. "dist")
                info = dict(tr.turns[-1].result.info) if tr.turns else {}
                info.update({key: v for key, v in obs[k].extras.items() if np.isscalar(v)})
                info.update(success=False, invalid=True)
                tr.turns.append(Turn(obs=obs[k], response=text, action=None,
                                     result=StepResult(obs=obs[k], terminated=True, info=info)))
                tr.done, tr.end_reason = True, "invalid"
                continue
            res = res_of[j]
            tr.turns.append(Turn(obs=obs[k], response=text, action=a, result=res))
            obs[k] = res.obs
            if res.terminated or res.truncated:
                tr.done = True
                tr.end_reason = "success" if res.info.get("success") else ("terminated" if res.terminated else "truncated")
            elif t == max_turns - 1:
                tr.done, tr.end_reason = True, "max_turns"
            elif max_context_tokens and ntok[j] >= max_context_tokens:
                tr.done, tr.end_reason = True, "context"
            else:
                tr.messages.append({"role": "user", "content": interface.observation_message(
                    res.obs, t + 1, res, parse_error=a is None)})
                still.append(k)
        active = still
    return trajs


def to_samples(trajs: Sequence[Trajectory], max_images: int | None, max_frames: int | None = None) -> list[Sample]:
    """Full context -> one sequence per episode with every assistant turn in the loss (PEAR-style).
    Truncated image/frame history -> one sequence per turn, each with exactly the context that turn saw."""
    samples = []
    for tr in trajs:
        a_idx = [i for i, m in enumerate(tr.messages) if m["role"] == "assistant"]
        if max_images is None and max_frames is None:
            samples.append(Sample(tr.messages, a_idx, tr.advantage, {"group": tr.group_id}))
        else:
            for i in a_idx:
                view = context_view(tr.messages[:i], max_images, max_frames) + [tr.messages[i]]
                samples.append(Sample(view, [len(view) - 1], tr.advantage, {"group": tr.group_id}))
    return samples
