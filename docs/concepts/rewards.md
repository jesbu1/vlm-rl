# Rewards & preferences

A reward function maps **groups** of trajectories to one score per trajectory:

```python
RewardFn = Callable[[list[list[Trajectory]]], list[list[float]]]
```

GRPO only compares the G rollouts that share a start state, so absolute scalars and *relative*
judgments (preferences) are the same kind of object.

## Specs

`reward=` takes a compact spec, or `module:factory`, a zero-argument callable that returns a `RewardFn`:

| spec | score |
|---|---|
| `success` | 1 if any step reported `info["success"]` (PEAR) |
| `return` | sum of env rewards |
| `neg_dist` | −final distance to the goal (PointMaze; carried through invalid-action endings) |
| `neg_geo_dist` | −final shortest-path distance in cells (PointMaze `info["geo_dist"]`; walls respected, unlike `neg_dist`) |
| `format` | −fraction of turns with an unparseable action |
| `efficiency` | `1 − turns/max_turns` for successful episodes, else 0 |
| `pref_oracle_dist` | preference pipeline with a privileged oracle judge (success first, then lower final distance) |
| `pref_mw` | Metaworld: preferences from ground truth (success first, then Metaworld's dense reward at the final state, `mw_progress`, compared on a log scale; within 2% is a tie; an episode that ended on an invalid reply ranks below every valid one), win rates as scores |
| `mw_progress` | Metaworld's dense reward (0-10) at the final state, as a scalar |
| `success+0.5*neg_dist` | weighted sum (quote on the CLI: `reward='success+0.5*neg_dist'`) |
| `examples.toy_wam:make_reward` | a custom factory |

Building blocks in `vlmrl.rewards`: `per_traj(fn)`, `success_reward()`, `return_reward()`,
`info_reward(key, reduce, scale)`, `format_penalty(coef)`, `efficiency_bonus(coef, max_turns)`,
`combine((fn, w), ...)`, `preference_reward(model, mode, max_pairs)`.

## Preferences

```python
class PreferenceModel(Protocol):
    def compare(self, pairs: Sequence[tuple[Trajectory, Trajectory]]) -> np.ndarray: ...   # P(a ≻ b)
```

`preference_reward(model, mode="winrate" | "bt", max_pairs=None)`:

1. For each group, take all \(\binom{G}{2}\) pairs (or a random `max_pairs` of them).
2. Query **both orders**, \((a,b)\) and \((b,a)\), in one batched call each, and average
   \(p = \tfrac12[P(a\succ b) + 1 - P(b\succ a)]\). This cancels position bias: a judge that always prefers
   the first item gives everyone 0.5.
3. Convert the pairwise outcomes to scores:
    - `winrate`: \(s_i = \operatorname{mean}_j P(i \succ j) \in [0,1]\) (Borda-style; the group mean is always ½).
    - `bt`: Bradley–Terry log-strengths fitted by the MM algorithm on soft outcomes.

Advantages are then computed exactly as for scalar rewards.

**Why groups make preferences easy.** A group is G rollouts from **the same reference state**, so
every comparison is between alternatives from identical conditions. These are the comparisons a
learned judge (e.g. a WAM's preference head) can make most reliably. No cross-state calibration is
needed, because advantages are normalized within each group.

`ScoreFnPreference(score, tau)` turns any scalar into a logistic preference model. It's used for the
oracle and as a template for a WAM head. Mix scalar and preference terms with
`combine((success_reward(), 1.0), (preference_reward(pm), 0.5))`.
