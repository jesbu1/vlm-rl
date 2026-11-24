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


@pytest.mark.parametrize("exp", ["pointmaze_q3vl4b_vllm", "pointmaze_q3vl4b_vllm_legacy", "pear_recipe",
                                 "toy_wam_prefs", "smoke", "pointmaze_generalization"])
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
    if exp == "pointmaze_q3vl4b_vllm_legacy":  # the original `dev` run's protocol
        assert (c.env_kwargs["prompt_style"], c.env_kwargs["settle"], c.on_invalid, c.adv_mode, c.max_frames) == \
            ("simple", False, "noop", "center", None)
        assert c.env_kwargs["maze"] == "umaze" and c.rollout_backend == "vllm" and c.reward == "success"
    if exp == "pointmaze_generalization":
        assert c.env_kwargs["split"] == "goal" and set(c.eval_sets) == {"id", "ood_goals", "ood_medium"}
        assert c.eval_sets["ood_medium"]["maze"] == "medium" and c.max_frames == 1


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
    write_video(frames, p, fps=2, min_size=0)            # native size: only even-padding
    r = imageio.get_reader(p)
    meta = r.get_meta_data()
    got = [f for f in r]
    assert meta["codec"] == "h264" and meta["fps"] == 2
    assert len(got) == 4                                  # 3 frames + held last frame
    assert got[0].shape[:2] == (size[0] + size[0] % 2, size[1] + size[1] % 2)   # padded to even
    assert abs(int(got[-1].mean()) - 250) < 6


def test_write_video_upscales_and_beats_native_fidelity(tmp_path):
    """Flat maze renders: nearest-neighbour upscale + crf keeps edges/dots sharp under yuv420p."""
    import imageio.v2 as imageio
    from vlmrl.envs.pointmaze import get_maze_map, render_topdown
    from vlmrl.trainer import write_video
    mm = get_maze_map("medium")
    frames = [render_topdown(mm, (-2.4 + 0.5 * t, -2.4), (1.6, 1.4), 336) for t in range(4)]

    def psnr(path, k):
        got = [g for g in imageio.get_reader(path)][:len(frames)]
        back = [g[k // 2::k, k // 2::k] for g in got]       # sample block centres back to native res
        mse = np.mean([np.mean((a.astype(float) - b.astype(float)) ** 2) for a, b in zip(frames, back)])
        return 10 * np.log10(255 ** 2 / mse)
    hi, lo = str(tmp_path / "hi.mp4"), str(tmp_path / "lo.mp4")
    write_video(frames, hi)                               # defaults: 336 -> 672 (x2), crf 18
    write_video(frames, lo, min_size=0, crf=28)
    assert imageio.get_reader(hi).get_meta_data()["size"] == (672, 672)
    assert psnr(hi, 2) > 45 and psnr(hi, 2) > psnr(lo, 1) + 5


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


@pytest.mark.parametrize("epochs,loss", [(1, "reinforce"), (2, "ppo_clip")])
def test_update_is_invariant_to_micro_batch_size(tmp_path, epochs, loss):
    """Batches are built lazily and gradients rescaled by the global token count afterwards, so the
    update must not depend on how samples are split into micro-batches."""
    import copy
    reward = lambda gs: [[float(i) for i in range(len(g))] for g in gs]  # noqa: E731
    thetas = []
    for mb in (1, 3, 64):
        tr = _trainer(tmp_path / str(mb), random_generate(0), reward_fn=reward, micro_batch_size=mb,
                      ppo_epochs=epochs, policy_loss=loss, grad_clip=1e9)
        kept, _, _ = copy.deepcopy(_trainer(tmp_path / "src", random_generate(0), reward_fn=reward).collect())
        m = tr.update(kept)
        thetas.append(tr.policy.model.theta.detach().clone())
        assert m["train/sequences"] == 8
    assert all(torch.allclose(thetas[0], t, atol=1e-6) for t in thetas[1:])


def test_kl_penalty_is_inert_at_the_reference_and_pulls_back_toward_it(tmp_path):
    """kl_coef > 0: at the reference the KL term is zero and changes nothing; once theta has moved away (set by hand
    here), the update pulls it back toward the reference, while kl_coef = 0 leaves it where it is (the fake policy's
    REINFORCE gradient is zero: group-normalized advantages sum to 0 and share one parameter per group)."""
    import copy
    reward = lambda gs: [[float(i) for i in range(len(g))] for g in gs]  # noqa: E731
    kept, _, _ = copy.deepcopy(_trainer(tmp_path / "src", random_generate(0), reward_fn=reward).collect())
    out = {}
    for kl in (0.0, 0.5):
        tr = _trainer(tmp_path / str(kl), random_generate(0), reward_fn=reward, kl_coef=kl, grad_clip=1e9)
        m0 = tr.update(copy.deepcopy(kept))
        t0 = tr.policy.model.theta.detach().clone()
        with torch.no_grad():
            tr.policy.model.theta.fill_(2.0)
        m1 = tr.update(copy.deepcopy(kept))
        out[kl] = (m0, t0, m1, tr.policy.model.theta.detach().clone())
    (a0, ta0, a1, ta1), (b0, tb0, b1, tb1) = out[0.0], out[0.5]
    assert "train/kl" not in a0 and b0["train/kl"] == 0.0 and torch.allclose(ta0, tb0)
    assert b1["train/kl"] > 0
    assert torch.allclose(ta1, torch.full((4,), 2.0))                # no KL: stays put
    assert (tb1 < 2.0).any() and (tb1 <= 2.0).all()                  # KL: pulled back toward 0


def test_update_compares_rollout_engine_and_trainer_logprobs(tmp_path):
    """train/logp_gap = trainer mean log-prob - engine mean log-prob on the same trained replies; tok_ratio = trained
    tokens / sampled tokens. Absent when the engine reports none (HF rollouts)."""
    tr = _trainer(tmp_path, random_generate(0), filter_groups=False)
    kept, _, _ = tr.collect()
    g = kept[0]
    for i, t in enumerate(g):
        t.advantage = 1.0 if i < 2 else -1.0
    assert "train/logp_gap" not in tr.update([g])
    for t in g:
        for tu in t.turns:
            tu.gen_logp, tu.gen_ntok = -1.0, 2
    m = tr.update([g])
    n_gen = 2 * sum(len(t.turns) for t in g)
    assert m["rollout/vllm_logp"] == pytest.approx(-0.5)
    assert m["train/logp_gap"] == pytest.approx(m["train/mean_logp"] + 0.5)
    assert m["train/tok_ratio"] == pytest.approx(m["train/tokens"] / n_gen)


def test_lr_warmup_is_linear_in_optimizer_steps(tmp_path):
    tr = _trainer(tmp_path, random_generate(0), filter_groups=False, lr_warmup_steps=4)
    lrs = []
    for _ in range(6):
        kept, _, _ = tr.collect()
        g = kept[0]
        for i, t in enumerate(g):
            t.advantage = 1.0 if i < 2 else -1.0
        lrs.append(tr.update([g])["train/lr"])
    assert lrs == pytest.approx([0.025, 0.05, 0.075, 0.1, 0.1, 0.1])     # lr 0.1 reached at the 4th step
    tr2 = _trainer(tmp_path / "b", random_generate(0), filter_groups=False)
    kept, _, _ = tr2.collect()
    for i, t in enumerate(kept[0]):
        t.advantage = 1.0 if i < 2 else -1.0
    assert tr2.update([kept[0]])["train/lr"] == pytest.approx(0.1)        # off by default


def test_traj_mean_gives_every_trajectory_equal_total_weight(tmp_path):
    """loss_agg=traj_mean: sum over a trajectory's tokens of its rescaled advantage = A / n_traj * total tokens,
    whatever its length, so a long failure no longer outweighs a short success."""
    from vlmrl.policy import Sample
    tr = _trainer(tmp_path, random_generate(0), loss_agg="traj_mean")
    long_fail = [Sample([], [0], -1.0, {"traj": 0}, gen_ids=[1] * 30) for _ in range(4)]   # 4 turns x 30 tokens
    short_win = [Sample([], [0], 1.0, {"traj": 1}, gen_ids=[1] * 10)]                       # 1 turn x 10 tokens
    samples = long_fail + short_win
    tr._weight_by_trajectory(samples)
    w = {t: sum(s.advantage * len(s.gen_ids) for s in samples if s.meta["traj"] == t) for t in (0, 1)}
    assert w[0] == pytest.approx(-w[1]) and w[1] == pytest.approx(130 / 2)                  # balanced, token-mean scale
    with pytest.raises(ValueError):
        _trainer(tmp_path / "b", random_generate(0), loss_agg="nope")._weight_by_trajectory(samples)


def test_behaviour_clone_trains_on_expert_transcripts(tmp_path):
    import dataclasses
    b = dataclasses.replace(_bundle(), expert=lambda env, k: np.array([0.3, 0.0]))
    cfg = TrainConfig(out_dir=str(tmp_path), groups_per_step=2, group_size=4, eval_every=0, log_samples_every=0,
                      save_every=0, lr=0.1, bc_steps=2, bc_episodes=3, bc_success_only=False, filter_groups=False)
    tr = Trainer(cfg, policy=FakePolicy(random_generate(0)), bundle=b)
    m = tr.behaviour_clone()
    assert tr.policy.saved == [f"{tmp_path}/ckpt_bc"]                       # the clone is kept for init_adapter
    assert tr.opt.param_groups[0]["lr"] == pytest.approx(0.1) and tr._lr_base is None   # bc_lr unset: lr
    tr2 = Trainer(dataclasses.replace(cfg, out_dir=str(tmp_path / "c"), bc_lr=0.5), policy=FakePolicy(random_generate(0)),
                  bundle=b)
    tr2.behaviour_clone()
    assert tr2.opt.param_groups[0]["lr"] == pytest.approx(0.5)               # BC ran at bc_lr
    kept, _, _ = tr2.collect()
    for i, t in enumerate(kept[0]):
        t.advantage = 1.0 if i < 2 else -1.0
    assert tr2.update([kept[0]])["train/lr"] == pytest.approx(0.1)          # back to lr for RL
    assert m["bc/steps"] == 2 and 0.0 <= m["bc/expert_success"] <= 1.0 and np.isfinite(m["bc/mean_logp"])
    with pytest.raises(ValueError):                  # an env without a scripted expert (PointMaze has one now)
        Trainer(dataclasses.replace(cfg, out_dir=str(tmp_path / "b")), policy=FakePolicy(random_generate(0)),
                bundle=dataclasses.replace(_bundle(), expert=None)).behaviour_clone()
    tr3 = Trainer(dataclasses.replace(cfg, out_dir=str(tmp_path / "d"), bc_steps=3, bc_dagger_steps=2),
                  policy=FakePolicy(random_generate(0)), bundle=b)
    m3 = tr3.behaviour_clone()                        # 1 expert step, then 2 steps on the clone's own episodes
    assert 0.0 <= m3["bc/expert_success"] <= 1.0 and 0.0 <= m3["bc/dagger_success"] <= 1.0


def test_adv_min_floors_advantages(tmp_path):
    ranks = lambda groups: [[float(i) for i in range(len(g))] for g in groups]      # spread within each group
    tr = _trainer(tmp_path, random_generate(0), reward_fn=ranks, adv_min=0.0, filter_groups=False)
    groups, _, _ = tr.collect()
    advs = [t.advantage for g in groups for t in g]
    assert min(advs) >= 0.0 and max(advs) > 0.0


def test_per_task_success_only_for_multi_task_sets():
    from conftest import make_traj
    from vlmrl.trainer import per_task_success
    a, b = make_traj(success=True), make_traj(success=False)
    a.start_state, b.start_state = {"env_name": "reach-v3"}, {"env_name": "push-v3"}
    c = make_traj(success=True)
    c.start_state = {"env_name": "push-v3"}
    m = per_task_success([a, b, c], "eval/train_tasks")
    assert m == {"eval/train_tasks/task/push-v3/success": 0.5, "eval/train_tasks/task/reach-v3/success": 1.0}
    assert per_task_success([b, c], "eval/x") == {}


def test_bc_rationale_prefixes_expert_replies_with_a_generated_reason(tmp_path):
    import dataclasses
    seen = []

    def gen(convs):
        seen.extend(convs)
        return ['The ball is left of the goal, so move right. {"dx": 9}' for _ in convs]
    b = dataclasses.replace(_bundle(), expert=lambda env, k: np.array([0.3, 0.0]))
    cfg = TrainConfig(out_dir=str(tmp_path), groups_per_step=2, group_size=4, eval_every=0, log_samples_every=0,
                      save_every=0, lr=0.1, bc_steps=1, bc_episodes=2, bc_success_only=False, filter_groups=False,
                      bc_rationale=True, max_frames=1)
    tr = Trainer(cfg, policy=FakePolicy(gen), bundle=b)
    from vlmrl.rollout import expert_rollout
    trajs = expert_rollout(tr.env, b.interface, b.expert, b.train_sampler.sample(2, np.random.default_rng(0)), [0, 1], 3)
    label = trajs[0].messages[2]["content"][0]["text"]
    tr._rationalize(trajs)
    text = trajs[0].messages[2]["content"][0]["text"]
    assert text == "The ball is left of the goal, so move right.\n" + label     # reason, then the expert's own move
    assert trajs[0].turns[0].response == text
    assert "The right move here is" in seen[0][-1]["content"][-1]["text"]     # the hint shows the expert's move
    assert tr.behaviour_clone()["bc/steps"] == 1
