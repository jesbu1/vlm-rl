"""Train / held-out maze splits, named eval sets, and pooled (cross-set batched) evaluation."""
import json
import re

import numpy as np
import pytest

from conftest import FakePolicy, oracle_generate, random_generate
from vlmrl.envs.pointmaze import (PointMazeConfig, RandomStartSampler, _cell_of, get_maze_map, make_pointmaze,
                                  split_pairs)
from vlmrl.rollout import RolloutTask, rollout, rollout_pooled
from vlmrl.trainer import TrainConfig, Trainer

_SIZE = re.compile(r"maze (\d+) cells wide and (\d+) cells tall")


def multi_maze_oracle(*mazes):
    """Oracle that picks the maze map from the system prompt, so one policy can play several mazes."""
    by_size = {}
    for name in mazes:
        mm = get_maze_map(name)
        by_size[(len(mm[0]), len(mm))] = oracle_generate(mm)

    def gen(convs):
        return [by_size[tuple(map(int, _SIZE.search(c[0]["content"][0]["text"]).groups()))]([c])[0] for c in convs]
    return gen


def _cells(states, mm):
    return {(_cell_of(s["qpos"], mm), _cell_of(s["goal"], mm)) for s in states}


@pytest.mark.parametrize("maze", ["umaze", "medium"])
@pytest.mark.parametrize("split", ["pair", "goal"])
def test_split_pairs_partition(maze, split):
    mm = get_maze_map(maze)
    train, held = split_pairs(mm, 1.5, split, 0.25, seed=0)
    assert train and held and not set(train) & set(held)
    assert split_pairs(mm, 1.5, split, 0.25, seed=0) == (train, held)          # deterministic
    assert split_pairs(mm, 1.5, split, 0.25, seed=1) != (train, held)
    if split == "goal":  # held-out goal cells are never goals in training
        assert not {b for _, b in train} & {b for _, b in held}


@pytest.mark.parametrize("split", ["pair", "goal"])
def test_samplers_draw_only_from_their_part(split):
    mm = get_maze_map("medium")
    rng = np.random.default_rng(0)
    s = {part: RandomStartSampler(mm, 1.5, split=split, part=part) for part in ("train", "heldout")}
    cells = {part: _cells(smp.sample(400, rng), mm) for part, smp in s.items()}
    free = np.argwhere(~(np.asarray(mm) == 1))
    idx = lambda pairs: {(tuple(free[a]), tuple(free[b])) for a, b in pairs}  # noqa: E731
    train, held = split_pairs(mm, 1.5, split)
    assert cells["train"] <= idx(train) and cells["heldout"] <= idx(held)
    assert not cells["train"] & cells["heldout"]
    assert all(np.linalg.norm(x["qpos"] - x["goal"]) >= 1.5 for x in s["heldout"].sample(200, rng))


def test_make_pointmaze_split_wires_train_and_eval_samplers():
    b = make_pointmaze(PointMazeConfig(maze="umaze", split="goal", image_size=32))
    mm = get_maze_map("umaze")
    rng = np.random.default_rng(0)
    train_goals = {g for _, g in _cells(b.train_sampler.sample(300, rng), mm)}
    eval_goals = {g for _, g in _cells(b.eval_sampler.sample(300, rng), mm)}
    assert train_goals and eval_goals and not train_goals & eval_goals
    seen = make_pointmaze(PointMazeConfig(maze="umaze", split="goal", eval_split="train", image_size=32))
    assert {g for _, g in _cells(seen.eval_sampler.sample(300, rng), mm)} <= train_goals


def test_no_split_keeps_the_original_sampler_stream():
    """split=none must reproduce the pre-split sampler exactly (running sweeps keep their eval starts)."""
    mm = get_maze_map("umaze")
    a = RandomStartSampler(mm, 1.5).sample(20, np.random.default_rng(3))
    b = make_pointmaze(PointMazeConfig(maze="umaze")).eval_sampler.sample(20, np.random.default_rng(3))
    assert all(np.array_equal(x["qpos"], y["qpos"]) and np.array_equal(x["goal"], y["goal"]) for x, y in zip(a, b))


def test_rollout_pooled_batches_tasks_and_matches_separate_rollouts():
    u = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=15, image_size=32))
    m = make_pointmaze(PointMazeConfig(maze="medium", max_turns=25, image_size=32))
    rng = np.random.default_rng(0)
    su, sm = u.eval_sampler.sample(3, rng), m.eval_sampler.sample(4, rng)
    oracle = multi_maze_oracle("umaze", "medium")
    calls = []

    def gen(convs):
        calls.append(len(convs))
        return oracle(convs)
    pooled = rollout_pooled(gen, [RolloutTask(u.make_vec_env(), u.interface, su, [0, 1, 2], 15),
                                  RolloutTask(m.make_vec_env(), m.interface, sm, [0, 1, 2, 3], 25)])
    assert calls[0] == 7                       # first round: every episode of both tasks in one call
    assert len(calls) <= 25 and [len(p) for p in pooled] == [3, 4]
    assert all(t.success for p in pooled for t in p)
    alone = rollout(oracle, m.make_vec_env(), m.interface, sm, [0, 1, 2, 3], 25)
    assert [len(t.turns) for t in alone] == [len(t.turns) for t in pooled[1]]
    assert [t.messages[-1]["content"] for t in alone] == [t.messages[-1]["content"] for t in pooled[1]]


def test_rollout_pooled_respects_per_task_max_turns():
    b = make_pointmaze(PointMazeConfig(maze="umaze", image_size=32))
    s = b.eval_sampler.sample(2, np.random.default_rng(0))
    stay = lambda convs: ['```json\n{"ball": {"x": 0, "y": 0}}\n```'] * len(convs)  # noqa: E731
    short, long_ = rollout_pooled(stay, [RolloutTask(b.make_vec_env(), b.interface, s, [0, 1], 2),
                                         RolloutTask(b.make_vec_env(), b.interface, s, [0, 1], 5)])
    assert [len(t.turns) for t in short] == [2, 2] and [len(t.turns) for t in long_] == [5, 5]
    assert all(t.end_reason == "max_turns" for t in short + long_)


def _eval_sets_cfg(tmp_path, **kw):
    return TrainConfig(out_dir=str(tmp_path), eval_episodes=4, log_samples_every=0, save_every=0,
                       env_kwargs={"maze": "umaze", "split": "goal", "image_size": 32, "max_turns": 15},
                       eval_sets={"id": {"eval_split": "train"}, "ood_goals": {"eval_split": "heldout"},
                                  "ood_medium": {"maze": "medium", "eval_split": "all", "max_turns": 25,
                                                 "episodes": 3}}, **kw)


def test_trainer_eval_sets_metrics_and_pooling(tmp_path):
    cfg = _eval_sets_cfg(tmp_path)
    tr = Trainer(cfg, policy=FakePolicy(multi_maze_oracle("umaze", "medium")))
    assert [s["name"] for s in tr.eval_sets] == ["eval/id", "eval/ood_goals", "eval/ood_medium"]
    assert [len(s["starts"]) for s in tr.eval_sets] == [4, 4, 3]
    assert tr.eval_sets[2]["env"].maze_name == "medium"
    mm = get_maze_map("umaze")
    train_goals = {g for _, g in _cells(tr.bundle.train_sampler.sample(300, np.random.default_rng(0)), mm)}
    assert {g for _, g in _cells(tr.eval_sets[0]["starts"], mm)} <= train_goals
    assert not {g for _, g in _cells(tr.eval_sets[1]["starts"], mm)} & train_goals
    calls = []
    gen = tr.policy._gen
    tr.policy._gen = lambda convs: (calls.append(len(convs)), gen(convs))[1]
    m, by_set = tr.evaluate()
    assert calls[0] == 11                      # all 3 sets share each generation round
    assert set(by_set) == {"eval/id", "eval/ood_goals", "eval/ood_medium"}
    for name in ("id", "ood_goals", "ood_medium"):
        assert m[f"eval/{name}/success"] == 1.0 and f"eval/{name}/end_invalid" in m
    assert m["eval/success"] == 1.0


def test_overall_eval_success_is_mean_over_sets(tmp_path):
    cfg = _eval_sets_cfg(tmp_path)
    oracle = oracle_generate(get_maze_map("umaze"))
    stay = '```json\n{"ball": {"x": 0, "y": 0}}\n```'
    solve_umaze_only = lambda convs: [oracle([c])[0] if "5 cells wide" in c[0]["content"][0]["text"] else stay  # noqa: E731
                                      for c in convs]
    tr = Trainer(cfg, policy=FakePolicy(solve_umaze_only))
    m, _ = tr.evaluate()
    per = [m[f"eval/{n}/success"] for n in ("id", "ood_goals", "ood_medium")]
    assert per[0] == per[1] == 1.0 and per[2] < 1.0
    assert m["eval/success"] == pytest.approx(np.mean(per))
    assert m["eval/turns"] == pytest.approx(np.mean([m[f"eval/{n}/turns"] for n in ("id", "ood_goals", "ood_medium")]))


def test_train_loop_logs_each_eval_set(tmp_path):
    cfg = _eval_sets_cfg(tmp_path, steps=1, groups_per_step=1, group_size=2, eval_every=1, filter_groups=False)
    Trainer(cfg, policy=FakePolicy(random_generate(0))).train()
    rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
    for r in rows:
        assert {"eval/success", "eval/id/success", "eval/ood_goals/success", "eval/ood_medium/success"} <= set(r)
    ep = tmp_path / "episodes"
    for name in ("id", "ood_goals", "ood_medium"):
        assert (ep / f"eval_{name}_step0.mp4").exists() and (ep / f"eval_{name}_step1.txt").exists()
    assert "8 cells wide" in (ep / "eval_ood_medium_step0.txt").read_text()


def test_wandb_offline_logs_per_set_media(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = _eval_sets_cfg(tmp_path, steps=0, eval_every=1, wandb_project="vlmrl-test")
    tr = Trainer(cfg, policy=FakePolicy(random_generate(0)))
    tr.train()
    files = [str(p) for p in (tmp_path / "wandb").glob("offline-run-*/files/media/**/*")]
    for name in ("id", "ood_goals", "ood_medium"):
        assert any(f"eval/{name}/samples" in f for f in files)
        assert any(f"eval/{name}/episode" in f and f.endswith(".mp4") for f in files)


def test_hydra_eval_sets_override():
    from vlmrl.config import compose
    c = compose(["experiment=pointmaze_q3vl4b_vllm", "env_kwargs.split=pair",
                 "+eval_sets.ood_medium.maze=medium", "+eval_sets.heldout.eval_split=heldout"])
    assert c.env_kwargs["split"] == "pair"
    assert c.eval_sets == {"ood_medium": {"maze": "medium"}, "heldout": {"eval_split": "heldout"}}


def test_report_summary_lists_eval_sets():
    pytest.importorskip("wandb")   # make_wandb_report imports wandb at module level; CI installs [pointmaze,dev] only
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "report", os.path.join(os.path.dirname(__file__), "..", "scripts", "make_wandb_report.py"))
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)

    class Run:
        summary = {"eval/success": 0.5, "eval/id/success": 1.0, "eval/ood_goals/success": 0.0, "eval/id/turns": 3}
    assert rep.eval_set_names(Run()) == ["id", "ood_goals"]
    row = {"model": "m", "reward": "success", "state": "finished", "url": "u", "steps": 1, "planned": 1,
           "eval0": 0.0, "eval_last": 0.5, "eval_last_step": 1, "eval_best": 0.5, "rollout_tail": 0.4,
           "min_per_step": 1.0, "eval_sets": {"id": 1.0, "ood_goals": 0.0}}
    table = rep.summary_markdown([row]).splitlines()
    assert "final `id`" in table[0] and "final `ood_goals`" in table[0]
    assert table[1].count("---") == 11 and "| 1.000 | 0.000 |" in table[2]
