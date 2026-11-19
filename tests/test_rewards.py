import numpy as np
import pytest

from conftest import make_traj
from vlmrl.rewards import (ScoreFnPreference, bradley_terry, build_reward, combine, efficiency_bonus,
                           format_penalty, info_reward, preference_reward, return_reward, success_reward)


def test_success_and_return():
    g = [[make_traj(success=True), make_traj(success=False)]]
    assert success_reward()(g) == [[1.0, 0.0]]
    assert return_reward()(g) == [[1.0, 0.0]]


def test_info_reward_reductions():
    t = make_traj(dist=0.5, n_turns=3)  # dists: 1.5, 1.5, 0.5
    assert info_reward("dist", "last", -1.0)([[t]]) == [[-0.5]]
    assert info_reward("dist", "sum")([[t]]) == [[3.5]]
    assert info_reward("dist", "min")([[t]]) == [[0.5]]
    assert info_reward("dist")([[make_traj(n_turns=0)]]) == [[0.0]]


def test_format_and_efficiency():
    assert format_penalty(2.0)([[make_traj(n_turns=4, invalid=1)]]) == [[-0.5]]
    e = efficiency_bonus(1.0, max_turns=10)
    assert e([[make_traj(success=True, n_turns=2), make_traj(success=False, n_turns=2)]]) == [[0.8, 0.0]]


def test_combine_weights():
    g = [[make_traj(success=True, n_turns=2, invalid=1)]]
    assert np.allclose(combine((success_reward(), 2.0), (format_penalty(1.0), 0.5))(g), [[2.0 - 0.25]])


def test_build_reward_specs():
    g = [[make_traj(dist=1.0, success=True), make_traj(dist=2.0)]]
    assert build_reward("success")(g) == [[1.0, 0.0]]
    assert np.allclose(build_reward("success+0.5*neg_dist")(g), [[0.5, -1.0]])
    assert np.allclose(build_reward("2*success+format")(g), [[2.0, 0.0]])
    s = build_reward("pref_oracle_dist")(g)[0]
    assert s[0] > s[1]
    with pytest.raises(KeyError):
        build_reward("nonexistent")


def test_build_reward_from_module_path():
    s = build_reward("examples.toy_wam:make_reward")([[make_traj(dist=0.1), make_traj(dist=3.0)]])[0]
    assert s[0] > s[1]


DIST_PREF = ScoreFnPreference(lambda t: -t.turns[-1].result.info["dist"], tau=0.3)


@pytest.mark.parametrize("mode", ["winrate", "bt"])
def test_preference_ranks(mode):
    g = [make_traj(dist=d) for d in (3.0, 0.5, 2.0, 1.0)]
    s = preference_reward(DIST_PREF, mode)([g])[0]
    assert list(np.argsort(s)) == [0, 2, 3, 1]


def test_winrate_range_and_multiple_groups():
    groups = [[make_traj(dist=d) for d in (1, 2, 3)], [make_traj(dist=d) for d in (5, 4)]]
    s = preference_reward(DIST_PREF)(groups)
    assert len(s) == 2 and len(s[0]) == 3 and len(s[1]) == 2
    assert all(0 <= v <= 1 for row in s for v in row)
    assert s[1][1] > s[1][0]
    assert np.isclose(np.mean(s[0]), 0.5)  # winrates in a group always average to 1/2


class PositionBiased:
    """Always prefers whichever trajectory is shown first."""
    def compare(self, pairs):
        return np.ones(len(pairs))


def test_both_orders_cancel_position_bias():
    s = preference_reward(PositionBiased())([[make_traj(), make_traj(), make_traj()]])[0]
    assert np.allclose(s, 0.5)


def test_max_pairs_subsamples():
    calls = []

    class Counting:
        def compare(self, pairs):
            calls.append(len(pairs))
            return np.full(len(pairs), 0.5)
    preference_reward(Counting(), max_pairs=5)([[make_traj() for _ in range(8)]])
    assert calls == [5, 5]  # 28 pairs -> 5, queried in both orders


def test_bradley_terry_recovers_order():
    s = bradley_terry(3, [(0, 1), (1, 2), (0, 2)], np.array([0.9, 0.9, 0.9]))
    assert s[0] > s[1] > s[2]
    assert np.isclose(np.mean(s), 0, atol=1e-6)


def test_score_fn_preference_is_logistic():
    pm = ScoreFnPreference(lambda t: t.turns[-1].result.info["dist"], tau=1.0)
    p = pm.compare([(make_traj(dist=1), make_traj(dist=1)), (make_traj(dist=2), make_traj(dist=1))])
    assert np.isclose(p[0], 0.5) and np.isclose(p[1], 1 / (1 + np.exp(-1)))


def _mw(progress, success=False):
    t = make_traj(success=success)
    t.turns[-1].result.obs.extras = {"mw_progress": progress}
    return t


def test_pref_mw_ranks_by_success_then_final_metaworld_reward():
    g = [_mw(1.0), _mw(9.5, success=True), _mw(4.0), _mw(0.2)]
    s = build_reward("pref_mw")([g])[0]
    assert list(np.argsort(s)) == [3, 0, 2, 1]
    assert np.allclose(build_reward("mw_progress")([g])[0], [1.0, 9.5, 4.0, 0.2])


def test_pref_mw_ties_have_no_spread():
    """Practically equal outcomes (within 2%) are ties: the group has no signal and DAPO drops it."""
    s = build_reward("pref_mw")([[_mw(0.053), _mw(0.0531), _mw(0.0535)]])[0]
    assert np.std(s) == 0


def test_pref_mw_orders_small_rewards_far_from_the_object():
    """Push far from the puck: 0.026 (26 cm), 0.033 (24 cm), 0.072 (13 cm) must still be ranked, not tied."""
    s = build_reward("pref_mw")([[_mw(0.033), _mw(0.072), _mw(0.026)]])[0]
    assert list(np.argsort(s)) == [2, 0, 1] and np.std(s) > 0.1
