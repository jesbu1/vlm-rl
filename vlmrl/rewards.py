"""Group-level reward functions.

GRPO compares the G rollouts that share a start state. A reward function therefore sees whole
*groups* and returns one scalar score per trajectory. That makes scalar rewards (success, return,
shaped distance) and *relative* signals (pairwise preferences from a WAM / VLM judge / human) the
same kind of object: ``groups -> scores``.

    RewardFn = Callable[[list[list[Trajectory]]], list[list[float]]]
"""
from __future__ import annotations

import itertools
from typing import Callable, Protocol, Sequence

import numpy as np

from vlmrl.types import Trajectory

Groups = list[list[Trajectory]]
RewardFn = Callable[[Groups], list[list[float]]]


# ----------------------------------------------------------------------------- scalar rewards

def per_traj(fn: Callable[[Trajectory], float]) -> RewardFn:
    """Lift a per-trajectory scalar into a group reward."""
    return lambda groups: [[float(fn(t)) for t in g] for g in groups]


def success_reward() -> RewardFn:
    """PEAR's signal: 1 if the task succeeded, else 0."""
    return per_traj(lambda t: float(t.success))


def return_reward() -> RewardFn:
    return per_traj(lambda t: t.env_return)


def info_reward(key: str, reduce: str = "last", scale: float = 1.0) -> RewardFn:
    """Read a per-step ``info[key]`` (e.g. ``dist`` for PointMaze -> use scale=-1)."""
    red = {"last": lambda xs: xs[-1], "sum": sum, "mean": np.mean, "min": min, "max": max}[reduce]
    return per_traj(lambda t: scale * float(red([s.result.info[key] for s in t.turns])) if t.turns else 0.0)


def format_penalty(coef: float = 1.0) -> RewardFn:
    """-coef * fraction of turns whose output failed to parse."""
    return per_traj(lambda t: -coef * t.num_invalid / max(1, len(t.turns)))


def efficiency_bonus(coef: float = 0.1, max_turns: int = 1) -> RewardFn:
    """Small bonus for finishing in fewer turns (only applied to successful episodes)."""
    return per_traj(lambda t: coef * (1 - len(t.turns) / max_turns) if t.success else 0.0)


def combine(*terms: tuple[RewardFn, float]) -> RewardFn:
    def fn(groups):
        total = [[0.0] * len(g) for g in groups]
        for rf, w in terms:
            for gi, row in enumerate(rf(groups)):
                for i, v in enumerate(row):
                    total[gi][i] += w * v
        return total
    return fn


# ----------------------------------------------------------------------------- preferences

class PreferenceModel(Protocol):
    """P(a preferred over b). Batched so a WAM / judge can score all pairs in one pass."""

    def compare(self, pairs: Sequence[tuple[Trajectory, Trajectory]]) -> np.ndarray: ...


def bradley_terry(n: int, pairs: list[tuple[int, int]], probs: np.ndarray, iters: int = 50,
                  eps: float = 1e-3) -> np.ndarray:
    """Fit BT strengths from soft pairwise outcomes with the MM algorithm; return log-strengths."""
    W = np.full((n, n), eps)  # W[i, j] = (soft) wins of i over j
    for (i, j), p in zip(pairs, probs):
        W[i, j] += p
        W[j, i] += 1 - p
    np.fill_diagonal(W, 0)
    N = W + W.T
    s = np.ones(n)
    for _ in range(iters):
        denom = (N / (s[:, None] + s[None, :])).sum(1)
        s = W.sum(1) / np.maximum(denom, 1e-12)
        s /= np.exp(np.log(s).mean())
    return np.log(s)


def preference_reward(model: PreferenceModel, mode: str = "winrate", max_pairs: int | None = None,
                      seed: int = 0) -> RewardFn:
    """Turn within-group pairwise preferences into per-trajectory scores.

    * ``winrate``: mean P(i beats j) over the other members (Borda/Copeland-style), in [0, 1].
    * ``bt``: Bradley-Terry log-strengths (only differences matter; advantage centering removes the offset).

    ``max_pairs`` subsamples pairs per group (G=8 -> 28 pairs; cheap for a WAM, maybe not for a big judge).
    Both orders (a,b) and (b,a) are queried and averaged to cancel position bias.
    """
    rng = np.random.default_rng(seed)

    def fn(groups):
        jobs = []  # (group, i, j)
        for gi, g in enumerate(groups):
            pairs = list(itertools.combinations(range(len(g)), 2))
            if max_pairs and len(pairs) > max_pairs:
                pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
            jobs += [(gi, i, j) for i, j in pairs]
        fwd = model.compare([(groups[g][i], groups[g][j]) for g, i, j in jobs])
        bwd = model.compare([(groups[g][j], groups[g][i]) for g, i, j in jobs])
        p = 0.5 * (np.asarray(fwd) + 1 - np.asarray(bwd))
        out = []
        for gi, g in enumerate(groups):
            idx = [k for k, (gg, _, _) in enumerate(jobs) if gg == gi]
            prs = [(jobs[k][1], jobs[k][2]) for k in idx]
            pk = p[idx]
            if mode == "bt":
                out.append(bradley_terry(len(g), prs, pk).tolist())
            else:
                wins, cnt = np.zeros(len(g)), np.zeros(len(g))
                for (i, j), q in zip(prs, pk):
                    wins[i] += q; wins[j] += 1 - q; cnt[i] += 1; cnt[j] += 1
                out.append((wins / np.maximum(cnt, 1)).tolist())
        return out

    return fn


class ScoreFnPreference:
    """Preference model induced by a scalar score with a logistic link (temperature ``tau``).

    Useful as (a) a test oracle for the preference pipeline, and (b) a template for a WAM preference
    head: replace ``score`` with the WAM's progress/value estimate on imagined trajectories.
    """

    def __init__(self, score: Callable[[Trajectory], float], tau: float = 0.1, margin: float = 0.0):
        self.score, self.tau, self.margin = score, tau, margin

    def compare(self, pairs):
        """``margin``: score differences below it are ties (p = 0.5), so a group of practically equal outcomes has no
        spread and is filtered, instead of z-scoring tiny differences into full-size advantages."""
        d = np.array([self.score(a) - self.score(b) for a, b in pairs])
        d = np.where(np.abs(d) < self.margin, 0.0, d)
        return 1 / (1 + np.exp(-d / self.tau))


def _final_extra(t: Trajectory, key: str, default: float = 0.0) -> float:
    """A scalar the env reports in every observation (``Obs.extras``), read at the episode's final state."""
    return float(t.turns[-1].result.obs.extras.get(key, default)) if t.turns else default


def mw_final_score(t: Trajectory) -> float:
    """Metaworld ground truth: success first, then the dense reward at the final state (0-10), on a log scale.

    Far from the object Metaworld's reward is small but still ordered (push: 0.026 at 26 cm from the puck, 0.072 at
    13 cm, 2.0 at 6 cm). The log makes differences relative, so a tie margin means "within x%" at every distance; an
    absolute margin turned every early push comparison into a tie. log(1e-3 + r) spans < 10, so +10 for success
    always wins.

    An episode that ended on an invalid reply ranks below every valid one (ties among themselves). Scored by its
    final state, it would often *beat* the valid episodes: early on, moving the hand usually lowers Metaworld's reward,
    so stopping at the start wins, and the policy learned to answer invalidly (Qwen3.5: 14% -> 99% invalid in 3
    updates)."""
    if t.end_reason == "invalid":
        return -20.0
    return 10.0 * t.success + float(np.log(1e-3 + _final_extra(t, "mw_progress")))


class LexPreference:
    """Lexicographic preference: the first key whose values differ by at least its margin decides (1 or 0); within
    every margin is a tie (0.5). Use it when one criterion must dominate the next whatever their scales."""

    def __init__(self, keys: Sequence[tuple[Callable[[Trajectory], float], float]]):
        self.keys = list(keys)

    def compare(self, pairs):
        out = []
        for a, b in pairs:
            p = 0.5
            for key, margin in self.keys:
                d = key(a) - key(b)
                if abs(d) >= margin:
                    p = 1.0 if d > 0 else 0.0
                    break
            out.append(p)
        return np.array(out)


def mw_object_first() -> LexPreference:
    """Metaworld ground truth, object progress before the dense reward. Rollout a beats b if, in order:
    1. a ended on a valid reply and b on an invalid one;
    2. a succeeded and b did not;
    3. a's task object ended at least 1 cm closer to the goal (``info["obj_to_target"]``; reach: the hand);
    4. a's final dense reward is higher (log scale, within 2% is a tie), as in ``pref_mw``.

    Why step 3: on push, Metaworld's dense reward at the final state is highest for *holding* the puck where it is
    (about 1 + 2 * 2 + 5 * 0.1 = 5.5 with the hand on it), while pushing it partway and losing contact scores
    2 * grasp < 2. Ranking by that alone taught the cloned policy to grip the puck and hold still (success 0.12 -> 0.01
    in 15 updates). Group members share the start, so a smaller final distance is more progress."""
    return LexPreference([
        (lambda t: float(t.end_reason != "invalid"), 0.5),
        (lambda t: float(t.success), 0.5),
        (lambda t: -100.0 * _final_extra(t, "mw_obj_dist"), 1.0),
        (lambda t: float(np.log(1e-3 + _final_extra(t, "mw_progress"))), 0.02),
    ])


# ----------------------------------------------------------------------------- registry

def build_reward(spec: str, max_turns: int = 1) -> RewardFn:
    """Build from a compact spec: ``"success"``, ``"success+0.1*format"``, ``"pref_oracle_dist"``,
    or ``"module.path:factory"`` (a zero-arg callable returning a RewardFn)."""
    if ":" in spec:
        import importlib
        mod, fn = spec.split(":")
        return getattr(importlib.import_module(mod), fn)()
    named = {
        "success": success_reward,
        "return": return_reward,
        "format": lambda: format_penalty(1.0),
        "neg_dist": lambda: info_reward("dist", "last", -1.0),
        # PointMaze: final shortest-path distance in cells (walls respected), so moving toward the goal through a
        # wall or into a dead end is not rewarded the way Euclidean distance rewards it
        "neg_geo_dist": lambda: info_reward("geo_dist", "last", -1.0),
        "efficiency": lambda: efficiency_bonus(1.0, max_turns),
        # preference pipeline driven by a privileged-state oracle: prefers lower final distance, then success
        "pref_oracle_dist": lambda: preference_reward(ScoreFnPreference(
            lambda t: (2.0 * t.success - (t.turns[-1].result.info["dist"] if t.turns else 0.0)), tau=0.2)),
        # Metaworld: preferences from the ground-truth dense reward. Within a group, a success beats a failure, then
        # the higher Metaworld reward (0-10) at the final state wins (ties: within 2%); win rates become the scores, so
        # every group with any progress difference has a signal (sparse success alone gives none on push early on)
        "pref_mw": lambda: preference_reward(ScoreFnPreference(mw_final_score, tau=0.1, margin=0.02)),
        "mw_progress": lambda: per_traj(lambda t: _final_extra(t, "mw_progress")),
        # Metaworld, object first: valid, then success, then the object >= 1 cm closer to the goal, then pref_mw's
        # dense-reward rule (see mw_object_first: on push the dense reward alone favours holding the puck still)
        "pref_mw_obj": lambda: preference_reward(mw_object_first()),
        "mw_obj_dist": lambda: per_traj(lambda t: -_final_extra(t, "mw_obj_dist")),
    }
    terms = []
    for part in spec.split("+"):
        w, name = (part.split("*") if "*" in part else (1.0, part))
        terms.append((named[name.strip()](), float(w)))
    return terms[0][0] if len(terms) == 1 and terms[0][1] == 1.0 else combine(*terms)
