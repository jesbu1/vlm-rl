import numpy as np

from conftest import random_generate
from examples.toy_wam import ToyKinematicWAM, make_bundle, make_reward
from vlmrl.envs.base import VecEnv
from vlmrl.envs.world_model import WorldModelVecEnv
from vlmrl.rollout import rollout
from vlmrl.types import Obs


class CounterWAM:
    """State = integer counter; done when counter >= 3. Records batch sizes."""

    def __init__(self):
        self.batches = []

    def init(self, start_states):
        return [int(s.get("c", 0)) for s in start_states]

    def render(self, states):
        return [np.full((4, 4, 3), s, np.uint8) for s in states]

    def step(self, states, actions):
        self.batches.append(len(states))
        nxt = [s + int(a[0]) for s, a in zip(states, actions)]
        return nxt, self.render(nxt), {"done": np.array([n >= 3 for n in nxt], float),
                                       "reward": np.array(nxt, float), "state": np.array([[n] for n in nxt])}


def test_protocol_conformance():
    assert isinstance(WorldModelVecEnv(CounterWAM()), VecEnv)


def test_slot_bookkeeping_and_batching():
    wam = CounterWAM()
    env = WorldModelVecEnv(wam, carry_keys=("goal",))
    obs = env.reset([{"c": 0, "goal": 9}, {"c": 2, "goal": 8}, {"c": 1}])
    assert [o.image[0, 0, 0] for o in obs] == [0, 2, 1]
    assert obs[0].extras == {"goal": 9} and obs[2].extras == {}
    res = env.step([0, 1], [np.array([1]), np.array([1])])
    assert wam.batches == [2]  # one batched call for the active slots only
    assert [r.obs.image[0, 0, 0] for r in res] == [1, 3]
    assert [r.terminated for r in res] == [False, True]
    assert res[1].reward == 3.0 and np.array_equal(res[1].obs.state, [3])
    assert "state" not in res[1].info and res[1].info["success"] is False
    assert env._states == [1, 3, 1]  # slot 2 untouched


def test_done_threshold_and_obs_fn():
    env = WorldModelVecEnv(CounterWAM(), done_threshold=2.0,
                           obs_fn=lambda f, ex, s: Obs(image=f, extras={"tag": s["tag"]}))
    env.reset([{"c": 5, "tag": "x"}])
    r = env.step([0], [np.array([0])])[0]
    assert not r.terminated and r.obs.extras == {"tag": "x"}


def test_toy_wam_moves_and_clips():
    wam = ToyKinematicWAM("open", image_size=32, max_delta=1.0)
    st = wam.init([{"qpos": np.array([0.0, 0.0]), "goal": np.array([2.0, 0.0])}])
    nxt, frames, ex = wam.step(st, np.array([[5.0, 0.0]]))
    assert np.allclose(nxt[0]["pos"], [1, 0]) and frames[0].shape == (32, 32, 3)
    assert np.isclose(ex["dist"][0], 1.0) and not ex["success"][0]


def test_toy_wam_bundle_rollout_with_preferences(rng):
    b = make_bundle(max_turns=4, image_size=32)
    env = b.make_vec_env()
    start = b.train_sampler.sample(1, rng)[0]
    trajs = rollout(random_generate(3), env, b.interface, [start] * 6, [0] * 6, b.max_turns)
    scores = make_reward()([trajs])[0]
    assert len(scores) == 6 and np.std(scores) > 0
    # the closest-to-goal trajectory gets the best preference score
    final = [t.turns[-1].result.info["dist"] for t in trajs]
    assert int(np.argmax(scores)) == int(np.argmin(final))
