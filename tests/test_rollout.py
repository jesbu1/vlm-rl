import numpy as np

from conftest import const_generate, oracle_generate, random_generate
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
    trajs = rollout(const_generate(), env, umaze.interface, umaze.train_sampler.sample(3, rng), [0, 0, 0], 4)
    assert all(len(t.turns) == 4 and t.done and not t.success and t.end_reason == "max_turns" for t in trajs)


def test_parse_failure_is_noop_and_reported(umaze, rng):
    env = umaze.make_vec_env()
    start = umaze.train_sampler.sample(1, rng)
    bad = lambda convs: ["I refuse to use tags"] * len(convs)
    t = rollout(bad, env, umaze.interface, start, [0], 3, on_invalid="noop")[0]
    assert t.num_invalid == 3 and all(x.action is None for x in t.turns)
    assert np.allclose(t.turns[-1].result.obs.state[:2], start[0]["qpos"])  # never moved
    assert "no valid action" in t.messages[3]["content"][0]["text"]
    assert t.end_reason == "max_turns"


def test_invalid_action_terminates_episode_pear(umaze, rng):
    """PEAR: an episode ends when a response contains no valid action."""
    env = umaze.make_vec_env()
    rounds = iter([[umaze.interface.format_action([0.2, 0.0])] * 2, ["oops, no json"] * 2])
    starts = umaze.train_sampler.sample(2, rng)
    trajs = rollout(lambda c: next(rounds), env, umaze.interface, starts, [0, 0], 6)
    for t in trajs:
        assert len(t.turns) == 2 and t.end_reason == "invalid" and not t.success and t.num_invalid == 1
        assert t.messages[-1]["role"] == "assistant"
        assert "dist" in t.turns[-1].result.info  # carried so distance-based rewards still work
    # first-turn invalid: info comes from the observation extras
    env2 = umaze.make_vec_env()
    t = rollout(lambda c: ["nope"] * len(c), env2, umaze.interface, starts[:1], [0], 3)[0]
    assert t.end_reason == "invalid" and np.isclose(t.turns[0].result.info["dist"],
                                                    np.linalg.norm(starts[0]["qpos"] - starts[0]["goal"]))


def test_all_invalid_round_does_not_call_step_with_empty_batch(rng):
    from examples.toy_wam import make_bundle
    b = make_bundle(max_turns=3, image_size=32)
    t = rollout(lambda c: ["x"] * len(c), b.make_vec_env(), b.interface, b.train_sampler.sample(2, rng), [0, 0], 3)
    assert all(x.end_reason == "invalid" for x in t)


def test_context_token_budget_ends_episode(umaze, rng):
    env = umaze.make_vec_env()
    calls = {"n": 0}

    def gen(convs):
        calls["n"] += 1
        return const_generate((0.1, 0.0))(convs)
    budget = lambda: [1000 * calls["n"]] * 2           # context grows 1000 tokens per round
    trajs = rollout(gen, env, umaze.interface, umaze.train_sampler.sample(2, rng), [0, 0], 10,
                    max_context_tokens=2500, last_tokens=budget)
    assert all(len(t.turns) == 3 and t.end_reason == "context" for t in trajs)


def test_success_end_reason(rng):
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=15, image_size=32))
    env = b.make_vec_env()
    t = rollout(oracle_generate(env.maze_map), env, b.interface, b.train_sampler.sample(2, rng), [0, 0], 15)
    assert all(x.end_reason == "success" for x in t)


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


def test_context_view_max_frames_keeps_system_and_last_rounds(umaze, rng):
    env = umaze.make_vec_env()
    t = rollout(random_generate(), env, umaze.interface, umaze.train_sampler.sample(1, rng), [0], 5)[0]
    assert len(t.turns) >= 3
    ctx = t.messages[:-1]                                      # what turn T saw: ends with its observation
    v = context_view(ctx, None, 1)
    assert [m["role"] for m in v] == ["system", "user"] and v[0] is ctx[0] and v[1] is ctx[-1]
    v2 = context_view(ctx, None, 2)
    assert [m["role"] for m in v2] == ["system", "user", "assistant", "user"] and v2[1:] == ctx[-3:]
    assert context_view(ctx, None, 100) == ctx
    # frames and images compose: 2 rounds kept, only the newest image
    v3 = context_view(ctx, 1, 2)
    assert sum(c["type"] == "image" for m in v3 for c in m["content"]) == 1


def test_rollout_context_matches_training_samples_max_frames(umaze, rng):
    """max_frames=1: generate() sees [system, current obs] each turn, and per-turn samples match exactly."""
    env = umaze.make_vec_env()
    seen = []
    base = random_generate(1)

    def gen(convs):
        seen.extend(list(c) for c in convs)  # the untruncated first-turn view is the live message list
        return base(convs)
    t = rollout(gen, env, umaze.interface, umaze.train_sampler.sample(1, rng), [0], 4, max_frames=1)[0]
    assert all([m["role"] for m in ctx] == ["system", "user"] for ctx in seen)
    samples = to_samples([t], None, 1)
    assert len(samples) == len(t.turns) == len(seen)
    for s, ctx in zip(samples, seen):
        assert s.messages[:-1] == ctx and s.messages[-1]["role"] == "assistant" and s.train_msg_idx == [2]


def test_media_placeholders_in_replies_are_scrubbed(umaze, rng):
    """A reply that spells out an image placeholder must not reach the history (it would be counted as an
    image slot when the conversation is re-tokenised for training); the raw reply is kept on the turn."""
    from vlmrl.rollout import scrub_media_tokens
    assert scrub_media_tokens('<image>{"x": 1} <row_1_col_2><|image_pad|>') == '{"x": 1} '
    reply = "move <image> <x>1.0</x><y>0.0</y>"
    t = rollout(lambda c: [reply] * len(c), umaze.make_vec_env(), umaze.interface,
                umaze.train_sampler.sample(1, rng), [0], 2, on_invalid="noop")[0]
    replies = [m["content"][0]["text"] for m in t.messages if m["role"] == "assistant"]
    assert replies and all("<image>" not in r for r in replies)
    assert t.turns[0].response == reply


def test_rollout_engine_logprobs_are_kept_per_turn(umaze, rng):
    """``last_logprobs`` (from the vLLM generator) lands on each Turn; unknown -> None (HF generator)."""
    gen = const_generate((0.3, 0.0))
    seen = []

    def generate(convs):
        seen.append(len(convs))
        return gen(convs)
    starts = umaze.train_sampler.sample(2, rng)
    env = umaze.make_vec_env()
    trajs = rollout(generate, env, umaze.interface, starts, [0, 0], 3,
                    last_logprobs=lambda: [(-1.25, 5)] * seen[-1])
    assert all(t.gen_logp == -1.25 and t.gen_ntok == 5 for tr in trajs for t in tr.turns)
    trajs = rollout(gen, umaze.make_vec_env(), umaze.interface, starts, [0, 0], 2)
    assert all(t.gen_logp is None for tr in trajs for t in tr.turns)
