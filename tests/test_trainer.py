"""Trainer tests: config/CLI parsing, collection + DAPO filter, update math, full loop (FakePolicy)."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import ROOT, TINY_VLM, FakePolicy, oracle_generate, random_generate
from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze
from vlmrl.trainer import TrainConfig, Trainer, make_env_bundle


def _bundle(max_turns=4):
    return make_pointmaze(PointMazeConfig(maze="umaze", max_turns=max_turns, image_size=32))


def _trainer(tmp_path, gen, reward_fn=None, **kw):
    cfg = TrainConfig(out_dir=str(tmp_path), groups_per_step=2, group_size=4, eval_every=0,
                      log_samples_every=0, save_every=0, lr=0.1, **kw)
    return Trainer(cfg, policy=FakePolicy(gen), bundle=_bundle(), reward_fn=reward_fn)


def test_make_env_bundle_registry_and_import_path():
    assert make_env_bundle("pointmaze", {"maze": "umaze"}).max_turns == 12
    b = make_env_bundle("examples.toy_wam:make_bundle", {"max_turns": 3})
    assert b.max_turns == 3


def test_collect_filters_degenerate_groups(tmp_path):
    # all-invalid policy -> every score identical -> every group filtered, resampled max rounds
    tr = _trainer(tmp_path, lambda c: ["nope"] * len(c), max_resample_rounds=3)
    kept, all_trajs, rounds = tr.collect()
    assert kept == [] and rounds == 3 and len(all_trajs) == 3 * 2 * 4


def test_collect_without_filter_keeps_everything(tmp_path):
    tr = _trainer(tmp_path, lambda c: ["nope"] * len(c), filter_groups=False)
    kept, all_trajs, rounds = tr.collect()
    assert len(kept) == 2 and rounds == 1 and all(t.advantage == 0 for g in kept for t in g)


def test_collect_groups_share_start_state_and_have_centred_advantages(tmp_path):
    tr = _trainer(tmp_path, random_generate(0), reward_fn=lambda gs: [[float(i) for i in range(len(g))] for g in gs])
    kept, _, rounds = tr.collect()
    assert len(kept) == 2 and rounds == 1
    for g in kept:
        assert all(np.array_equal(t.start_state["qpos"], g[0].start_state["qpos"]) for t in g)
        assert len({t.group_id for t in g}) == 1
        assert np.isclose(sum(t.advantage for t in g), 0)
        assert [t.score for t in g] == [0, 1, 2, 3]


def test_update_direction(tmp_path):
    """Positive advantage must increase logp of its tokens; negative must decrease."""
    tr = _trainer(tmp_path, random_generate(0), filter_groups=False)
    kept, _, _ = tr.collect()
    g = kept[0]
    for i, t in enumerate(g):
        t.group_id, t.advantage = 0, (1.0 if i < 2 else -1.0)
    g[0].group_id = g[1].group_id = 1  # theta[1] gets positive adv, theta[0] negative
    m = tr.update([g])
    th = tr.policy.model.theta.detach()
    # logp = -softplus(theta): increasing logp means decreasing theta
    assert th[1] < 0 < th[0]
    assert m["train/sequences"] == 4 and m["train/tokens"] == 16 and np.isfinite(m["train/grad_norm"])


def test_update_skips_zero_advantage(tmp_path):
    tr = _trainer(tmp_path, random_generate(0), filter_groups=False)
    kept, _, _ = tr.collect()
    for t in kept[0]:
        t.advantage = 0.0
    assert tr.update([kept[0]]) == {"train/skipped": 1.0}


def test_ppo_epochs_path(tmp_path):
    tr = _trainer(tmp_path, random_generate(0), ppo_epochs=3,
                  reward_fn=lambda gs: [[float(i) for i in range(len(g))] for g in gs])
    kept, _, _ = tr.collect()
    m = tr.update(kept)
    assert "train/clipfrac" in m and 0 <= m["train/clipfrac"] <= 1


def test_full_train_loop_logs_evaluates_saves(tmp_path):
    cfg = TrainConfig(out_dir=str(tmp_path), steps=2, groups_per_step=2, group_size=3, eval_every=1,
                      eval_episodes=3, log_samples_every=1, save_every=2, lr=0.01, reward="success+0.5*neg_dist")
    pol = FakePolicy(random_generate(0, p_invalid=0.2))
    Trainer(cfg, policy=pol, bundle=_bundle()).train()
    lines = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    assert [l["step"] for l in lines] == [0, 1, 2]
    for k in ("rollout/success", "rollout/invalid_frac", "time/rollout", "eval/success"):
        assert k in lines[1]
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "episodes" / "train_step1.gif").exists()
    assert "--- assistant" in (tmp_path / "episodes" / "eval_step0.txt").read_text()
    assert pol.saved == [f"{tmp_path}/ckpt_step2"]


def test_eval_with_oracle_policy_is_perfect(tmp_path):
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=15, image_size=32))
    cfg = TrainConfig(out_dir=str(tmp_path), eval_episodes=8)
    tr = Trainer(cfg, policy=FakePolicy(oracle_generate(b.make_vec_env().maze_map)), bundle=b)
    m, trajs = tr.evaluate()
    assert m["eval/success"] == 1.0 and m["eval/invalid_frac"] == 0.0


def test_cli_parsing(tmp_path, monkeypatch):
    sys.path.insert(0, f"{ROOT}/scripts")
    import train as train_cli
    cfgf = tmp_path / "c.json"
    cfgf.write_text(json.dumps({"_comment": "ignored", "lr": 3e-5, "steps": 7, "env_kwargs": {"maze": "large"}}))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfgf), "--steps", "9", "--lora_rank", "8",
                                      "--filter_groups", "false", "--max_images", "none",
                                      "--processor_kwargs", '{"max_pixels": 100}', "--attn_implementation", "sdpa"])
    c = train_cli.parse()
    assert (c.lr, c.steps, c.lora_rank, c.filter_groups, c.max_images) == (3e-5, 9, 8, False, None)
    assert c.env_kwargs == {"maze": "large"} and c.processor_kwargs == {"max_pixels": 100}
    assert c.attn_implementation == "sdpa"


@pytest.mark.parametrize("path", ["configs/pointmaze_qwen3vl4b_lora.json", "configs/pear_recipe_full_ft.json",
                                  "configs/pointmaze_toy_wam_prefs.json"])
def test_shipped_configs_are_valid(path):
    cfg = {k: v for k, v in json.load(open(f"{ROOT}/{path}")).items() if not k.startswith("_")}
    TrainConfig(**cfg)


@pytest.mark.slow
def test_cli_end_to_end_tiny_vlm(tmp_path):
    """Real HF model through the real CLI: 1 step, tiny everything."""
    cmd = [sys.executable, "scripts/train.py", "--model", TINY_VLM, "--processor_kwargs", '{"do_image_splitting": false}',
           "--env_kwargs", '{"maze": "umaze", "max_turns": 2, "image_size": 64}', "--steps", "1",
           "--groups_per_step", "1", "--group_size", "2", "--max_new_tokens", "8", "--eval_every", "1",
           "--eval_episodes", "2", "--filter_groups", "false", "--reward", "neg_dist", "--out_dir", str(tmp_path),
           "--gradient_checkpointing", "false", "--dtype", "float32"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    lines = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    assert lines[-1]["step"] == 1


@pytest.mark.slow
def test_tiny_vlm_update_changes_weights(tiny_vlm, tmp_path, rng):
    b = _bundle(3)
    cfg = TrainConfig(model=TINY_VLM, out_dir=str(tmp_path), lr=1e-4, eval_every=0, micro_batch_size=2)
    tr = Trainer(cfg, policy=tiny_vlm, bundle=b)
    from vlmrl.rollout import rollout
    texts = iter(['go <action>{"dx": 1, "dy": 0}</action>', "hmm", "<action>0,1</action>"] * 2)
    trajs = rollout(lambda c: [next(texts) for _ in c], tr.env, b.interface,
                    b.train_sampler.sample(2, rng), [0, 0], 3)
    trajs[0].advantage, trajs[1].advantage = 1.0, -1.0
    p = next(p for p in tiny_vlm.model.parameters() if p.requires_grad)
    before = p.detach().clone()
    m = tr.update([trajs])
    assert np.isfinite(m["train/loss"]) and not torch.equal(before, p.detach())


def test_generator_sync_called_after_each_update(tmp_path):
    cfg = TrainConfig(out_dir=str(tmp_path), steps=2, groups_per_step=1, group_size=3, eval_every=0,
                      log_samples_every=0, save_every=0, lr=0.01, filter_groups=False, reward="neg_dist")
    tr = Trainer(cfg, policy=FakePolicy(random_generate(0)), bundle=_bundle())

    class Recorder:
        synced = []
        generate = staticmethod(lambda convs, *a: random_generate(1)(convs))

        def sync(self, policy, step):
            self.synced.append(step)
    tr.generator = Recorder()
    tr.train()
    assert Recorder.synced == [1, 2]


def test_unknown_rollout_backend(tmp_path):
    with pytest.raises(ValueError):
        Trainer(TrainConfig(out_dir=str(tmp_path), rollout_backend="nope"), policy=FakePolicy(random_generate()),
                bundle=_bundle())


def test_wandb_offline_logging(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = TrainConfig(out_dir=str(tmp_path), steps=1, groups_per_step=1, group_size=3, eval_every=1,
                      eval_episodes=2, log_samples_every=1, save_every=0, filter_groups=False, reward="neg_dist",
                      wandb_project="vlmrl-test", wandb_run_name="unit", wandb_group="g", wandb_tags="a,b")
    tr = Trainer(cfg, policy=FakePolicy(random_generate(0)), bundle=_bundle())
    run = tr.wandb
    assert run.name == "unit" and run.group == "g" and set(run.tags) == {"a", "b"}
    tr.train()
    hist = list((tmp_path / "wandb").glob("offline-run-*/files/media/videos/**/*.gif"))
    assert hist, "episode GIFs should be logged as wandb media"


def test_sweep_expansion(tmp_path):
    sys.path.insert(0, f"{ROOT}/scripts")
    import sweep
    runs = sweep.expand({"base": "configs/pointmaze_qwen3vl4b_vllm.json", "group": "g", "project": "p",
                         "vllm_port": 9000, "grid": {"reward": ["success", "neg_dist"], "model": ["a/m1", "b/m2"]},
                         "overrides": {"steps": 3}})
    assert len(runs) == 4
    assert [r["model"] for r in runs] == ["a/m1", "a/m1", "b/m2", "b/m2"]   # model-major
    assert runs[0]["wandb_run_name"] == "model=m1__reward=success" and runs[0]["out_dir"] == "runs/g/model=m1__reward=success"
    assert all(r["steps"] == 3 and r["wandb_group"] == "g" and r["vllm_url"] == "http://localhost:9000" for r in runs)
    for r in runs:
        TrainConfig(**r)
    shipped = json.load(open(f"{ROOT}/configs/sweeps/pointmaze_small.json"))
    for r in sweep.expand(shipped):
        TrainConfig(**r)


def _resume_cfg(tmp_path, steps, **kw):
    return TrainConfig(out_dir=str(tmp_path), steps=steps, groups_per_step=2, group_size=3, eval_every=2,
                       eval_episodes=2, log_samples_every=0, save_every=0, lr=0.05, filter_groups=False,
                       reward="neg_dist", **kw)


def test_resume_continues_after_last_checkpoint(tmp_path):
    t1 = Trainer(_resume_cfg(tmp_path, 2), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    t1.train()
    theta, rng_state = t1.policy.model.theta.detach().clone(), t1.rng.bit_generator.state
    assert torch.any(theta != 0)
    # a re-launch with more steps picks up at step 3 with weights, optimizer and sampler RNG restored
    t2 = Trainer(_resume_cfg(tmp_path, 4), policy=FakePolicy(random_generate(1)), bundle=_bundle())
    assert t2.start_step == 2 and torch.equal(t2.policy.model.theta.detach(), theta)
    assert t2.rng.bit_generator.state == rng_state and len(t2.opt.state_dict()["state"]) == 1
    t2.train()
    assert [json.loads(l)["step"] for l in open(tmp_path / "metrics.jsonl")] == [0, 1, 2, 3, 4]
    # resume=False starts over
    assert Trainer(_resume_cfg(tmp_path, 4, resume=False), policy=FakePolicy(random_generate(0)),
                   bundle=_bundle()).start_step == 0


def test_resume_relogs_pending_step_and_truncates_extra_lines(tmp_path):
    Trainer(_resume_cfg(tmp_path, 2), policy=FakePolicy(random_generate(0)), bundle=_bundle()).train()
    path = tmp_path / "metrics.jsonl"
    lines = path.read_text().splitlines()
    # crash after checkpointing step 2 but before logging it; plus a stray line from a later step
    path.write_text("\n".join(lines[:-1] + ['{"step": 3, "x": 1}']) + "\n")
    t = Trainer(_resume_cfg(tmp_path, 2), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    t.train()  # nothing left to do
    got = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["step"] for l in got] == [0, 1, 2] and got[-1] == json.loads(lines[-1])


def test_restart_without_checkpoint_starts_over(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 0, "eval/success": 0.5}\n')
    t = Trainer(_resume_cfg(tmp_path, 1), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    assert t.start_step == 0 and not (tmp_path / "metrics.jsonl").exists()
    t.train()
    assert [json.loads(l)["step"] for l in open(tmp_path / "metrics.jsonl")] == [0, 1]


def test_wandb_resume_reuses_run_id(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    kw = dict(wandb_project="vlmrl-test", wandb_run_name="unit")
    t1 = Trainer(_resume_cfg(tmp_path, 1, **kw), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    rid = t1.wandb.id
    t1.train()
    t2 = Trainer(_resume_cfg(tmp_path, 2, **kw), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    assert t2.wandb.id == rid and t2.start_step == 1
    t2.train()
