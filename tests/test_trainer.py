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
    tr = _trainer(tmp_path, random_generate(0), ppo_epochs=3, policy_loss="ppo_clip",
                  reward_fn=lambda gs: [[float(i) for i in range(len(g))] for g in gs])
    kept, _, _ = tr.collect()
    m = tr.update(kept)
    assert "train/clipfrac" in m and 0 <= m["train/clipfrac"] <= 1


def test_multi_epoch_reinforce_with_score_centering(tmp_path, monkeypatch):
    """Epochs >1 are off-policy w.r.t. the rollout policy: score centering must be computed from the
    snapshot (topk) and applied (gather_ids) -- and must not change epoch-1 (on-policy) gradients."""
    calls = []
    tr = _trainer(tmp_path, random_generate(0), ppo_epochs=2, policy_loss="reinforce", score_centering=True,
                  reward_fn=lambda gs: [[float(i) for i in range(len(g))] for g in gs])
    orig = tr.policy.token_logprobs

    def spy(batch, topk=0, gather_ids=None):
        calls.append(("topk" if topk else "") + ("gather" if gather_ids is not None else ""))
        return orig(batch, topk, gather_ids)
    monkeypatch.setattr(tr.policy, "token_logprobs", spy)
    kept, _, _ = tr.collect()
    m = tr.update(kept)
    nb = len(calls) // 3
    assert calls == ["topk"] * nb + ["gather"] * (2 * nb) and np.isfinite(m["train/loss"])
    assert "train/clipfrac" not in m


def test_score_centering_off_when_single_epoch(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, random_generate(0), reward_fn=lambda gs: [[float(i) for i in range(len(g))] for g in gs])
    seen = []
    orig = tr.policy.token_logprobs
    monkeypatch.setattr(tr.policy, "token_logprobs", lambda b, topk=0, gather_ids=None: (seen.append((topk, gather_ids)), orig(b))[1])
    tr.update(tr.collect()[0])
    assert all(t == 0 and g is None for t, g in seen)  # on-policy: correction is exactly zero, skipped


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


def test_hydra_defaults_and_overrides():
    from vlmrl.config import compose
    c = compose()
    assert c.model == "Qwen/Qwen3-VL-4B-Instruct" and c.env == "pointmaze" and c.rollout_backend == "hf"
    assert c.adv_mode == "zscore" and c.on_invalid == "terminate" and c.env_kwargs["prompt_style"] == "pear"
    c = compose(["policy=qwen3vl2b_lora", "steps=9", "filter_groups=false", "max_images=3",
                 "env_kwargs.maze=large", "reward='success+0.5*neg_dist'", "+processor_kwargs.max_pixels=100",
                 "attn_implementation=sdpa", "max_context_tokens=null"])
    assert (c.model, c.steps, c.filter_groups, c.max_images) == ("Qwen/Qwen3-VL-2B-Instruct", 9, False, 3)
    assert c.env_kwargs["maze"] == "large" and c.env_kwargs["max_turns"] == 10      # merged, not replaced
    assert c.reward == "success+0.5*neg_dist" and c.processor_kwargs == {"max_pixels": 100}
    assert c.attn_implementation == "sdpa" and c.max_context_tokens is None


def test_hydra_type_checking():
    from hydra.errors import ConfigCompositionException
    from vlmrl.config import compose
    with pytest.raises(ConfigCompositionException):
        compose(["steps=abc"])
    with pytest.raises(ConfigCompositionException):
        compose(["not_a_field=1"])


@pytest.mark.parametrize("exp", ["pointmaze_q3vl4b_vllm", "pear_recipe", "toy_wam_prefs", "smoke"])
def test_shipped_experiments_compose(exp):
    from vlmrl.config import compose
    c = compose([f"experiment={exp}"])
    assert c.out_dir.startswith("runs/")
    if exp == "pear_recipe":
        assert (c.groups_per_step, c.group_size, c.lr, c.lora_rank, c.adv_mode, c.policy_loss) == \
            (128, 8, 1e-6, 0, "zscore", "reinforce")
        assert c.env_kwargs["maze"] == "medium" and c.env_kwargs["prompt_style"] == "pear"
    if exp == "pointmaze_q3vl4b_vllm":
        assert c.rollout_backend == "vllm" and c.lora_rank == 64


@pytest.mark.parametrize("policy", ["qwen3vl4b_lora", "qwen3vl2b_lora", "qwen25vl3b_lora", "smolvlm256m",
                                    "qwen25vl7b_full"])
def test_policy_presets_compose(policy):
    from vlmrl.config import compose
    assert compose([f"policy={policy}"]).model


def test_train_cli_prints_config(tmp_path):
    r = subprocess.run([sys.executable, "scripts/train.py", "--cfg", "job", "experiment=smoke", "steps=3"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "steps: 3" in r.stdout and "SmolVLM-256M" in r.stdout


@pytest.mark.slow
def test_cli_end_to_end_tiny_vlm(tmp_path):
    """Real HF model through the real CLI: 1 step, tiny everything."""
    cmd = [sys.executable, "scripts/train.py", "experiment=smoke", f"out_dir={tmp_path}",
           "gradient_checkpointing=false", "dtype=float32"]
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
    import yaml
    sys.path.insert(0, f"{ROOT}/scripts")
    import sweep
    runs = sweep.expand({"group": "g", "project": "p", "vllm_port": 9000,
                         "overrides": ["experiment=pointmaze_q3vl4b_vllm", "steps=3"],
                         "grid": {"reward": ["success", "success+0.5*neg_dist"], "policy": ["qwen3vl4b_lora", "qwen3vl2b_lora"]}})
    assert len(runs) == 4
    assert [r["cfg"].model for r in runs] == ["Qwen/Qwen3-VL-4B-Instruct"] * 2 + ["Qwen/Qwen3-VL-2B-Instruct"] * 2
    r0 = next(r for r in runs if r["name"] == "reward=success__policy=qwen3vl4b_lora")
    assert r0["cfg"].out_dir == "runs/g/reward=success__policy=qwen3vl4b_lora" and r0["cfg"].wandb_group == "g"
    assert all(r["cfg"].steps == 3 and r["cfg"].vllm_url == "http://localhost:9000" and r["cfg"].wandb_project == "p"
               for r in runs)
    assert {r["cfg"].reward for r in runs} == {"success", "success+0.5*neg_dist"}
    shipped = yaml.safe_load(open(f"{ROOT}/configs/sweeps/pointmaze_small.yaml"))
    sh = sweep.expand(shipped)
    assert len(sh) == 9 and sh[-1]["cfg"].model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert all(r["cfg"].wandb_entity == "tri" and r["cfg"].wandb_tags == "pointmaze,medium,lora,sweep" for r in sh)
