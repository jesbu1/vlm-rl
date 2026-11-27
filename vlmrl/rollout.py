"""Batched multi-turn rollouts and conversion of trajectories into training samples."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from vlmrl.envs.base import TextInterface, VecEnv
from vlmrl.policy import Sample
from vlmrl.types import StepResult, Trajectory, Turn

Generate = Callable[[list[list[dict]]], list[str]]

# Image/video placeholder strings of the supported VLM processors. A sampled reply that contains one verbatim
# (SmolVLM once wrote "<image>") would be counted as an image slot when the conversation is re-tokenised, and the
# processor raises "number of <image> tokens ... should be the same as the number of images".
_MEDIA_TOKENS = re.compile(
    r"<image>|<video>|<fake_token_around_image>|<global-img>|<row_\d+_col_\d+>|<\|(?:image|video)_pad\|>"
    r"|<\|vision_(?:start|end)\|>|<\|image\|>|<image_soft_token>|<(?:start|end)_of_image>")


def scrub_media_tokens(text: str) -> str:
    """Remove media placeholder strings from a model reply before it enters the conversation history."""
    return _MEDIA_TOKENS.sub("", text)


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


@dataclass
class RolloutTask:
    """One batch of episodes on one env: its own dynamics, prompt interface and turn limit."""
    env: VecEnv
    interface: TextInterface
    start_states: Sequence[dict]
    group_ids: Sequence[int]
    max_turns: int


def rollout(generate: Generate, env: VecEnv, interface: TextInterface, start_states: Sequence[dict],
            group_ids: Sequence[int], max_turns: int, max_images: int | None = None,
            on_invalid: str = "terminate", max_context_tokens: int | None = None,
            last_tokens: Callable[[], list[int]] | None = None,
            max_frames: int | None = None, last_logprobs: Callable[[], list] | None = None,
            thinking: bool = False, last_gen_ids: Callable[[], list] | None = None) -> list[Trajectory]:
    """Run ``len(start_states)`` episodes in lock-step. Each round, every unfinished episode gets one
    generation (batched) and one env step (batched over the still-active slots).

    Episode end (PEAR, paper App. A): the env reports success/termination, the step limit
    (``max_turns``) is hit, the context reaches ``max_context_tokens``, or -- with
    ``on_invalid="terminate"`` -- a response contains no valid action. ``on_invalid="noop"`` instead
    executes ``interface.noop_action()`` and tells the model in the next observation.
    ``last_tokens()`` returns the context length (prompt + reply tokens) of each item of the most
    recent ``generate`` call; it is needed only for ``max_context_tokens``. ``last_logprobs()`` returns, per item,
    (sum of the sampled tokens' log-probs, number of tokens) as the rollout engine computed them (stored on each
    ``Turn``; the trainer compares them with its own log-probs of the same replies).
    """
    return rollout_pooled(generate, [RolloutTask(env, interface, start_states, group_ids, max_turns)],
                          max_images, on_invalid, max_context_tokens, last_tokens, max_frames, last_logprobs,
                          thinking, last_gen_ids)[0]


def final_answer(text: str, thinking: bool = False) -> str | None:
    """The part of a reply the action is parsed from: after the last ``</think>`` (reasoning models write
    ``reasoning...</think>answer``; the generation prompt opens the ``<think>``). With ``thinking=True`` (the
    template was asked to think) a reply that never closes its reasoning, e.g. cut off at ``max_new_tokens``, has no
    answer: None, an invalid action, instead of a number picked out of half-finished reasoning."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return None if thinking else text


def _gen_ids(last_gen_ids, n):
    ids = list(last_gen_ids() or []) if last_gen_ids else []
    return ids if len(ids) == n else [None] * n


def _gen_logp(last_logprobs, n):
    """(sum of sampled-token log-probs, n tokens) per item of the last generate call, or (None, 0) if unknown."""
    lp = list(last_logprobs() or []) if last_logprobs else []
    return lp if len(lp) == n else [(None, 0)] * n


def rollout_pooled(generate: Generate, tasks: Sequence[RolloutTask], max_images: int | None = None,
                   on_invalid: str = "terminate", max_context_tokens: int | None = None,
                   last_tokens: Callable[[], list[int]] | None = None,
                   max_frames: int | None = None,
                   last_logprobs: Callable[[], list] | None = None,
                   thinking: bool = False, last_gen_ids: Callable[[], list] | None = None) -> list[list[Trajectory]]:
    """``rollout`` over several tasks at once (e.g. eval sets on different mazes): each round, the
    unfinished episodes of *all* tasks share one ``generate`` call, then every task steps its own env on
    its active slots. Returns one trajectory list per task, in task order."""
    assert on_invalid in ("terminate", "noop")
    obs, trajs = [], []
    for task in tasks:
        o = task.env.reset(task.start_states)
        obs.append(o)
        trajs.append([Trajectory(group_id=task.group_ids[k], start_state=s, messages=[
            {"role": "system", "content": [{"type": "text", "text": task.interface.system_prompt(s)}]},
            {"role": "user", "content": task.interface.observation_message(o[k], 0, None, False)}])
            for k, s in enumerate(task.start_states)])
    active = [(i, k) for i, tr in enumerate(trajs) for k in range(len(tr))]  # (task, slot)
    for t in range(max((task.max_turns for task in tasks), default=0)):
        if not active:
            break
        texts = generate([context_view(trajs[i][k].messages, max_images, max_frames) for i, k in active])
        ntok = last_tokens() if (max_context_tokens and last_tokens) else [0] * len(active)
        glp = _gen_logp(last_logprobs, len(active))
        gids = _gen_ids(last_gen_ids, len(active))
        answers = [final_answer(x, thinking) for x in texts]
        parsed = [tasks[i].interface.parse_action(x) if x is not None else None for (i, _), x in zip(active, answers)]
        res_of = {}
        for i, task in enumerate(tasks):
            stepping = [j for j, (ti, _) in enumerate(active) if ti == i and (parsed[j] is not None or on_invalid == "noop")]
            if stepping:
                results = task.env.step([active[j][1] for j in stepping],
                                        [parsed[j] if parsed[j] is not None else task.interface.noop_action()
                                         for j in stepping])
                res_of.update(zip(stepping, results))
        still = []
        for j, ((i, k), text, a) in enumerate(zip(active, texts, parsed)):
            tr = trajs[i][k]
            tr.messages.append({"role": "assistant", "content": [{"type": "text", "text": scrub_media_tokens(text)}]})
            if j not in res_of:                      # invalid action -> episode over (PEAR)
                # state is unchanged: carry the latest step info + scalar obs extras (e.g. "dist")
                info = dict(tr.turns[-1].result.info) if tr.turns else {}
                info.update({key: v for key, v in obs[i][k].extras.items() if np.isscalar(v)})
                info.update(success=False, invalid=True)
                tr.turns.append(Turn(obs=obs[i][k], response=text, action=None,
                                     result=StepResult(obs=obs[i][k], terminated=True, info=info),
                                     gen_logp=glp[j][0], gen_ntok=glp[j][1], gen_ids=gids[j]))
                tr.done, tr.end_reason = True, "invalid"
                continue
            res = res_of[j]
            tr.turns.append(Turn(obs=obs[i][k], response=text, action=a, result=res,
                                 gen_logp=glp[j][0], gen_ntok=glp[j][1], gen_ids=gids[j]))
            obs[i][k] = res.obs
            if res.terminated or res.truncated:
                tr.done = True
                tr.end_reason = "success" if res.info.get("success") else ("terminated" if res.terminated else "truncated")
            elif t == tasks[i].max_turns - 1:
                tr.done, tr.end_reason = True, "max_turns"
            elif max_context_tokens and ntok[j] >= max_context_tokens:
                tr.done, tr.end_reason = True, "context"
            else:
                tr.messages.append({"role": "user", "content": tasks[i].interface.observation_message(
                    res.obs, t + 1, res, parse_error=a is None)})
                still.append((i, k))
        active = still
    return trajs


def expert_rollout(env: VecEnv, interface: TextInterface, expert: Callable, start_states: Sequence[dict],
                   group_ids: Sequence[int], max_turns: int, generate: Generate | None = None,
                   max_images: int | None = None, max_frames: int | None = None,
                   thinking: bool = False, noise: float = 0.0,
                   rng: np.random.Generator | None = None) -> list[Trajectory]:
    """Episodes driven by a scripted expert through the text protocol: each turn the expert's action is written as
    the reply the model would give (``interface.expert_reply`` if the interface has one, else ``format_action``) and
    parsed back before it is executed, so the transcripts have exactly the structure of ``rollout``'s and can be
    trained on (behaviour cloning).

    With ``generate`` (DAgger): the *policy's* reply is executed (the expert's when the policy's is invalid), but the
    transcript records the expert's reply for the state the policy reached. Training on it teaches the clone to
    recover from its own mistakes, which pure cloning never shows it. The next observation's "last move" is the
    policy's; with ``max_frames=1`` no earlier reply is in the context, so the recorded labels are never seen as
    history.

    With ``noise`` (DART): the executed move is the expert's plus Gaussian noise of that standard deviation on the
    move components (the first three; Metaworld's fourth, the gripper, is left alone), while the transcript keeps
    the expert's clean reply. The episodes stay near the expert's path, where its labels are consistent, but show
    how to recover from small slips (with DAgger, the expert's labels in the states a weak clone reaches were
    inconsistent, and cloning them made the policy worse)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    obs = env.reset(start_states)
    trajs = [Trajectory(group_id=g, start_state=s, messages=[
        {"role": "system", "content": [{"type": "text", "text": interface.system_prompt(s)}]},
        {"role": "user", "content": interface.observation_message(obs[k], 0, None, False)}])
        for k, (s, g) in enumerate(zip(start_states, group_ids))]
    active = list(range(len(start_states)))
    for t in range(max_turns):
        if not active:
            break
        reply = getattr(interface, "expert_reply", None)       # e.g. a stated goal before the move (Metaworld _locate)
        texts = [reply(expert(env, k), obs[k]) if reply else interface.format_action(expert(env, k)) for k in active]
        actions = [interface.parse_action(x) for x in texts]
        if generate is not None:
            own = generate([context_view(trajs[k].messages, max_images, max_frames) for k in active])
            for j, x in enumerate(own):
                ans = final_answer(x, thinking)
                a = interface.parse_action(ans) if ans is not None else None
                if a is not None:
                    actions[j] = a
        elif noise > 0:
            actions = [np.r_[np.asarray(a[:3], dtype=float) + rng.normal(0.0, noise, min(3, len(a))),
                             np.asarray(a[3:], dtype=float)] for a in actions]   # never the gripper (Metaworld a[3])
        results = env.step(active, actions)
        still = []
        for k, text, a, res in zip(active, texts, actions, results):
            tr = trajs[k]
            tr.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            tr.turns.append(Turn(obs=obs[k], response=text, action=a, result=res))
            obs[k] = res.obs
            if res.terminated or res.truncated:
                tr.done = True
                tr.end_reason = "success" if res.info.get("success") else ("terminated" if res.terminated else "truncated")
            elif t == max_turns - 1:
                tr.done, tr.end_reason = True, "max_turns"
            else:
                tr.messages.append({"role": "user", "content": interface.observation_message(res.obs, t + 1, res, False)})
                still.append(k)
        active = still
    return trajs


def to_samples(trajs: Sequence[Trajectory], max_images: int | None, max_frames: int | None = None) -> list[Sample]:
    """Full context -> one sequence per episode with every assistant turn in the loss (PEAR-style).
    Truncated image/frame history -> one sequence per turn, each with exactly the context that turn saw."""
    samples = []
    for ti, tr in enumerate(trajs):
        a_idx = [i for i, m in enumerate(tr.messages) if m["role"] == "assistant"]
        if max_images is None and max_frames is None:
            samples.append(Sample(tr.messages, a_idx, tr.advantage, {"group": tr.group_id, "traj": ti}))
        else:
            for j, i in enumerate(a_idx):        # the j-th assistant message is the j-th turn's reply
                view = context_view(tr.messages[:i], max_images, max_frames) + [tr.messages[i]]
                gen = tr.turns[j].gen_ids if j < len(tr.turns) else None
                samples.append(Sample(view, [len(view) - 1], tr.advantage, {"group": tr.group_id, "traj": ti}, gen_ids=gen))
    return samples
