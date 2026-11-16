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
    assert (tmp_path / "episodes" / "train_step1.mp4").exists()
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
                                    "smolvlm2_2b_lora", "qwen25vl7b_full",
                                    "gemma4_e2b_lora", "minicpmv46_lora", "smolvlm256m_lora"])
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
    vids = list((tmp_path / "wandb").glob("offline-run-*/files/media/videos/**/*.mp4"))
    assert vids, "episode videos should be logged as wandb media (mp4)"
    tables = list((tmp_path / "wandb").glob("offline-run-*/files/media/table/**/*.json"))
    assert any("train/samples" in str(p) for p in tables) and any("eval/samples" in str(p) for p in tables)
    row = json.loads(next(p for p in tables if "eval/samples" in str(p)).read_text())
    assert "transcript" in row["columns"] and "--- assistant" in row["data"][0][row["columns"].index("transcript")]


@pytest.mark.parametrize("size", [(64, 64), (33, 47)])
def test_write_video_mp4_is_h264_and_readable(tmp_path, size):
    import imageio.v2 as imageio
    from vlmrl.trainer import write_video
    frames = [np.full((*size, 3), v, np.uint8) for v in (0, 120, 250)]
    p = str(tmp_path / "ep.mp4")
    write_video(frames, p, fps=2)
    r = imageio.get_reader(p)
    meta = r.get_meta_data()
    got = [f for f in r]
    assert meta["codec"] == "h264" and meta["fps"] == 2
    assert len(got) == 4                                  # 3 frames + held last frame
    assert got[0].shape[:2] == (size[0] + size[0] % 2, size[1] + size[1] % 2)   # padded to even
    assert abs(int(got[-1].mean()) - 250) < 6


def test_write_video_gif(tmp_path):
    import imageio.v2 as imageio
    from vlmrl.trainer import write_video
    write_video([np.full((8, 8, 3), v, np.uint8) for v in (0, 200)], str(tmp_path / "ep.gif"))
    assert len(imageio.mimread(str(tmp_path / "ep.gif"))) == 2


def test_gif_video_format_option(tmp_path):
    cfg = TrainConfig(out_dir=str(tmp_path), steps=1, groups_per_step=1, group_size=2, eval_every=0,
                      log_samples_every=1, save_every=0, filter_groups=False, reward="neg_dist", video_format="gif")
    Trainer(cfg, policy=FakePolicy(random_generate(0)), bundle=_bundle()).train()
    assert (tmp_path / "episodes" / "train_step1.gif").exists()


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


def test_backfill_wandb_resends_rows_the_server_lacks(tmp_path):
    t = Trainer(_resume_cfg(tmp_path, 3), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    t.train()

    class FakeRun:
        step = 2  # server has steps 0..1 (e.g. step 2+ were buffered when the process was killed)
        logged = []

        def log(self, data, step=None, commit=None):
            self.logged.append((step, data))
    t.wandb = FakeRun()
    t.backfill_wandb()
    assert [s for s, _ in FakeRun.logged] == [2, 3] and "rollout/success" in FakeRun.logged[0][1]
    assert "step" not in FakeRun.logged[0][1]


def test_log_episodes_mixes_train_successes_and_failures(tmp_path):
    from conftest import make_traj
    t = Trainer(_resume_cfg(tmp_path, 1, samples_per_log=4), policy=FakePolicy(random_generate(0)), bundle=_bundle())
    trajs = [make_traj(success=False, group_id=i) for i in range(5)] + [make_traj(success=True, group_id=9)] * 3
    t.log_episodes("train", 7, trajs)
    files = sorted(p.name for p in (tmp_path / "episodes").glob("train_step7*.txt"))
    assert files == ["train_step7.txt", "train_step7_1.txt", "train_step7_2.txt", "train_step7_3.txt"]
    heads = [(tmp_path / "episodes" / f).read_text().split()[0] for f in files]
    assert heads == ["success=True", "success=True", "success=False", "success=False"]
    t.log_episodes("train", 8, [make_traj(success=True)] * 5)  # no failures: fill with wins
    assert len(list((tmp_path / "episodes").glob("train_step8*.txt"))) == 4
