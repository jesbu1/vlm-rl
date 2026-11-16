import numpy as np

from conftest import oracle_generate, random_generate
from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze
from vlmrl.rollout import context_view, rollout, to_samples


def _roles(t):
    return [m["role"] for m in t.messages]


def test_oracle_rollout_succeeds_and_structure(rng):
    b = make_pointmaze(PointMazeConfig(maze="medium", max_turns=25, image_size=48))
    env = b.make_vec_env()
    starts = b.train_sampler.sample(5, rng)
    trajs = rollout(oracle_generate(env.maze_map), env, b.interface, starts, [7] * 5, b.max_turns)
    for t in trajs:
        assert t.success and t.done and t.group_id == 7
        r = _roles(t)
        assert r[0] == "system" and r[1::2] == ["user"] * len(r[1::2]) and r[2::2] == ["assistant"] * len(r[2::2])
        assert r[-1] == "assistant"                  # finished episodes end on the agent's last action
        assert sum(x == "assistant" for x in r) == len(t.turns)
        assert len(t.frames) == len(t.turns) + 1
        assert t.turns[-1].result.terminated


def test_max_turns_truncates(umaze, rng):
    env = umaze.make_vec_env()
    noop = lambda convs: ['<action>{"dx": 0, "dy": 0}</action>'] * len(convs)
    trajs = rollout(noop, env, umaze.interface, umaze.train_sampler.sample(3, rng), [0, 0, 0], 4)
    assert all(len(t.turns) == 4 and t.done and not t.success for t in trajs)


def test_parse_failure_is_noop_and_reported(umaze, rng):
    env = umaze.make_vec_env()
    start = umaze.train_sampler.sample(1, rng)
    bad = lambda convs: ["I refuse to use tags"] * len(convs)
    t = rollout(bad, env, umaze.interface, start, [0], 3)[0]
    assert t.num_invalid == 3 and all(x.action is None for x in t.turns)
    assert np.allclose(t.turns[-1].result.obs.state[:2], start[0]["qpos"])  # never moved
    assert "no valid <action>" in t.messages[3]["content"][-1]["text"]


def test_finished_episodes_stop_generating(umaze, rng):
    """Episodes that finish early must not be passed to generate again."""
    env = umaze.make_vec_env()
    sizes = []
    orc = oracle_generate(env.maze_map)

    def gen(convs):
        sizes.append(len(convs))
        return orc(convs)
    starts = umaze.train_sampler.sample(6, rng)
    rollout(gen, env, umaze.interface, starts, list(range(6)), 20)
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == 6 and sizes[-1] >= 1


def test_context_view_keeps_last_k_images(umaze, rng):
    env = umaze.make_vec_env()
    t = rollout(random_generate(), env, umaze.interface, umaze.train_sampler.sample(1, rng), [0], 5)[0]
    n_img = lambda ms: sum(c["type"] == "image" for m in ms for c in m["content"])
    assert context_view(t.messages, None) is t.messages
    v = context_view(t.messages, 2)
    assert n_img(v) == 2 and n_img(t.messages) > 2               # original untouched
    assert sum("[earlier image omitted]" in c.get("text", "") for m in v for c in m["content"]) == n_img(t.messages) - 2
    # the kept images are the most recent ones
    last_imgs = [c["image"] for m in t.messages for c in m["content"] if c["type"] == "image"][-2:]
    kept = [c["image"] for m in v for c in m["content"] if c["type"] == "image"]
    assert kept == last_imgs


def test_rollout_context_matches_training_samples(umaze, rng):
    """With truncated history, each per-turn training sample must equal what generate() saw."""
    env = umaze.make_vec_env()
    seen = []
    base = random_generate(1)

    def gen(convs):
        seen.extend(convs)
        return base(convs)
    t = rollout(gen, env, umaze.interface, umaze.train_sampler.sample(1, rng), [0], 4, max_images=2)[0]
    t.advantage = 0.7
    samples = to_samples([t], 2)
    assert len(samples) == len(t.turns) == len(seen)
    for s, ctx in zip(samples, seen):
        assert s.messages[:-1] == ctx and s.messages[-1]["role"] == "assistant"
        assert s.train_msg_idx == [len(s.messages) - 1]
        assert s.advantage == 0.7


def test_full_context_single_sample(umaze, rng):
    env = umaze.make_vec_env()
    t = rollout(random_generate(), env, umaze.interface, umaze.train_sampler.sample(1, rng), [3], 4)[0]
    (s,) = to_samples([t], None)
    assert s.messages is t.messages and s.meta["group"] == 3
    assert [t.messages[i]["role"] for i in s.train_msg_idx] == ["assistant"] * len(t.turns)
