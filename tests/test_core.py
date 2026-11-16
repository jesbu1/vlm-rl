"""Fast unit tests (no model) + one tiny-model integration test (skipped if SmolVLM isn't cached).

    pytest -q tests
"""
import os

import numpy as np
import pytest
import torch

from vlmrl import algo
from vlmrl.envs.pointmaze import (PointMazeConfig, PointMazeTextInterface, get_maze_map, make_pointmaze,
                                  oracle_delta)
from vlmrl.envs.world_model import WorldModelVecEnv
from vlmrl.rewards import ScoreFnPreference, bradley_terry, build_reward, preference_reward
from vlmrl.rollout import context_view, rollout, to_samples
from vlmrl.types import Trajectory


def test_parse_action():
    itf = PointMazeTextInterface(get_maze_map("umaze"))
    assert np.allclose(itf.parse_action('think <action>{"dx": 0.5, "dy": -1}</action>'), [0.5, -1])
    assert np.allclose(itf.parse_action('<action>{"dx":1,"dy":1}</action> <action>{"dx": 0.1, "dy": 0.2}</action>'), [0.1, 0.2])
    assert np.allclose(itf.parse_action("<action>0.3, -0.4</action>"), [0.3, -0.4])
    assert itf.parse_action("no tag here") is None
    assert itf.parse_action("<action>{\"dx\": nan}</action>") is None


def test_advantages_and_filter():
    a = algo.group_advantages([[1, 0, 0, 1]], "center")[0]
    assert np.allclose(a, [0.5, -0.5, -0.5, 0.5])
    r = algo.group_advantages([[1, 0, 0, 0]], "rloo")[0]
    assert np.isclose(r[0], 1.0) and np.isclose(r[1], -1 / 3)
    assert not algo.keep_group([1, 1, 1]) and algo.keep_group([1, 0, 1])


def test_loss_sum_gradient_matches_reinforce():
    logp = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    adv = torch.full_like(logp, 2.0)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, _ = algo.policy_loss_sum(logp, adv, mask)
    loss.backward()
    assert torch.allclose(logp.grad, torch.tensor([[-2.0, -2.0, 0.0]]))
    # PPO path at ratio=1 has the same gradient
    lp2 = logp.detach().clone().requires_grad_(True)
    loss2, _ = algo.policy_loss_sum(lp2, adv, mask, old_logp=lp2.detach())
    loss2.backward()
    assert torch.allclose(lp2.grad, logp.grad)


def _fake_traj(dist, success=False):
    from vlmrl.types import Obs, StepResult, Turn
    t = Trajectory(group_id=0, start_state={})
    t.turns = [Turn(Obs(), "", np.zeros(2), StepResult(Obs(), float(success), success, info={"dist": dist, "success": success}))]
    return t


def test_preferences_rank_correctly():
    g = [_fake_traj(d) for d in (3.0, 0.5, 2.0, 1.0)]
    pm = ScoreFnPreference(lambda t: -t.turns[-1].result.info["dist"], tau=0.3)
    for mode in ("winrate", "bt"):
        s = preference_reward(pm, mode)([g])[0]
        assert list(np.argsort(s)) == [0, 2, 3, 1], (mode, s)
    s = bradley_terry(3, [(0, 1), (1, 2), (0, 2)], np.array([0.9, 0.9, 0.9]))
    assert s[0] > s[1] > s[2]


def test_reward_spec():
    g = [[_fake_traj(1.0, True), _fake_traj(2.0, False)]]
    assert build_reward("success")(g) == [[1.0, 0.0]]
    assert np.allclose(build_reward("success+0.5*neg_dist")(g), [[0.5, -1.0]])


def scripted_generate(env_maze_map):
    def gen(convs):
        out = []
        for c in convs:
            txt = c[-1]["content"][-1]["text"]
            nums = [float(x) for x in __import__("re").findall(r"[-+]?\d+\.\d+", txt)]
            pos, goal = np.array(nums[0:2]), np.array(nums[4:6])
            d = oracle_delta(env_maze_map, pos, goal)
            out.append(f'<action>{{"dx": {d[0]:.3f}, "dy": {d[1]:.3f}}}</action>')
        return out
    return gen


def test_rollout_with_oracle_text_policy():
    b = make_pointmaze(PointMazeConfig(maze="medium", max_turns=25, image_size=96))
    env = b.make_vec_env()
    starts = b.train_sampler.sample(6, np.random.default_rng(0))
    trajs = rollout(scripted_generate(env.maze_map), env, b.interface, starts, list(range(6)), b.max_turns)
    assert all(t.success for t in trajs)
    # message structure: system, (user, assistant)*
    for t in trajs:
        roles = [m["role"] for m in t.messages]
        assert roles[0] == "system" and roles[1::2] == ["user"] * len(roles[1::2])
        assert sum(r == "assistant" for r in roles) == len(t.turns)


def test_context_view_and_per_turn_samples():
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=4, image_size=64))
    env = b.make_vec_env()
    gen = lambda convs: ["<action>{\"dx\": 0, \"dy\": 0}</action>"] * len(convs)  # never succeeds
    trajs = rollout(gen, env, b.interface, b.train_sampler.sample(1, np.random.default_rng(0)), [0], 4, max_images=2)
    view = context_view(trajs[0].messages, 2)
    n_img = sum(c["type"] == "image" for m in view for c in m["content"])
    assert n_img == 2
    assert len(to_samples(trajs, None)) == 1 and len(to_samples(trajs, 2)) == 4


def test_world_model_env_and_toy_wam():
    from examples.toy_wam import ToyKinematicWAM, make_bundle, make_reward
    b = make_bundle(max_turns=6, image_size=64)
    env = b.make_vec_env()
    assert isinstance(env, WorldModelVecEnv)
    starts = b.train_sampler.sample(4, np.random.default_rng(0))
    gen = lambda convs: ['<action>{"dx": 0.5, "dy": 0.0}</action>', '<action>{"dx": -0.5, "dy": 0.0}</action>'] * (len(convs) // 2) + ['x'] * (len(convs) % 2)
    rep = [starts[0]] * 4
    trajs = rollout(gen, env, b.interface, rep, [0] * 4, b.max_turns)
    scores = make_reward()([trajs])[0]
    assert len(scores) == 4 and np.std(scores) > 0


SMOL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _have(model):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have(SMOL), reason="tiny VLM not cached")
def test_mask_and_update_with_tiny_vlm():
    from vlmrl.policy import HFPolicy
    from vlmrl.trainer import TrainConfig, Trainer
    pol = HFPolicy(SMOL, device="cpu", processor_kwargs={"do_image_splitting": False}, gradient_checkpointing=False)
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=3, image_size=64))
    env = b.make_vec_env()
    texts = iter(["first <action>{\"dx\": 1, \"dy\": 0}</action>", "second reply", "third <action>0,1</action>"] * 2)
    gen = lambda convs: [next(texts) for _ in convs]
    trajs = rollout(gen, env, b.interface, b.train_sampler.sample(2, np.random.default_rng(0)), [0, 0], 3)
    trajs[0].advantage, trajs[1].advantage = 1.0, -1.0
    batch = pol.build_batch(to_samples(trajs, None))
    # masked tokens decode to exactly the assistant replies (+ the template's end-of-turn tokens)
    ids = batch["input_ids"][0][batch["loss_mask"][0].bool()]
    dec = pol.tokenizer.decode(ids, skip_special_tokens=True)
    for want in ("first", "second reply", "third"):
        assert want in dec
    assert "Round" not in dec and "control a small ball" not in dec
    logp, mask = pol.token_logprobs(batch)
    assert logp.shape == mask.shape and torch.isfinite(logp).all()
    # one full trainer update changes the weights
    cfg = TrainConfig(model=SMOL, out_dir="/tmp/vlmrl_test", lr=1e-4, eval_every=0)
    tr = Trainer(cfg, policy=pol, bundle=b)
    before = next(p for p in pol.model.parameters() if p.requires_grad).detach().clone()
    m = tr.update([trajs])
    after = next(p for p in pol.model.parameters() if p.requires_grad).detach()
    assert np.isfinite(m["train/loss"]) and not torch.equal(before, after)
