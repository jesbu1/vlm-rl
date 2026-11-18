"""scripts/sweep.py: grid expansion with per-model overrides, and vLLM server shutdown."""
import importlib.util
import os
import subprocess
import sys
import time

import pytest

_spec = importlib.util.spec_from_file_location(
    "sweep", os.path.join(os.path.dirname(__file__), "..", "scripts", "sweep.py"))
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def test_expand_hydra_grid_model_major_and_policy_presets():
    runs = sweep.expand({"group": "g", "project": "p", "vllm_port": 9000,
                         "overrides": ["experiment=pointmaze_q3vl4b_vllm", "steps=3"],
                         "grid": {"reward": ["success", "success+0.5*neg_dist"],
                                  "policy": ["qwen3vl4b_lora", "smolvlm2_2b_lora"]}})
    assert len(runs) == 4
    assert [r["cfg"].model for r in runs] == ["Qwen/Qwen3-VL-4B-Instruct"] * 2 + ["HuggingFaceTB/SmolVLM2-2.2B-Instruct"] * 2
    smol = [r["cfg"] for r in runs if "SmolVLM2" in r["cfg"].model]
    assert all(c.lora_targets.startswith(".*text_model") and c.processor_kwargs == {"do_image_splitting": False} for c in smol)
    assert runs[0]["cfg"].lora_targets.startswith(".*language_model") and runs[0]["cfg"].processor_kwargs == {}
    r0 = next(r for r in runs if r["name"] == "reward=success__policy=qwen3vl4b_lora")
    assert r0["cfg"].out_dir == "runs/g/reward=success__policy=qwen3vl4b_lora" and r0["cfg"].wandb_group == "g"
    assert all(r["cfg"].steps == 3 and r["cfg"].vllm_url == "http://localhost:9000" and r["cfg"].wandb_project == "p"
               for r in runs)
    assert {r["cfg"].reward for r in runs} == {"success", "success+0.5*neg_dist"}


def test_shipped_sweep_composes():
    import yaml
    shipped = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs/sweeps/pointmaze_small.yaml")))
    runs = sweep.expand(shipped)
    assert len(runs) == 8 and runs[-1]["cfg"].model == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert [r["cfg"].model for r in runs[::2]] == ["google/gemma-4-E2B-it", "Qwen/Qwen3-VL-4B-Instruct",
                                                   "openbmb/MiniCPM-V-4.6", "HuggingFaceTB/SmolVLM-256M-Instruct"]
    assert all(r["cfg"].max_frames == 1 and r["cfg"].max_new_tokens == 512 and r["cfg"].reward == "success"
               and r["cfg"].groups_per_step == 16 and r["cfg"].group_size == 16 for r in runs)
    assert [r["cfg"].env_kwargs["maze"] for r in runs[:2]] == ["umaze", "medium"]
    assert all(r["cfg"].wandb_entity == "tri" and r["cfg"].wandb_tags == "pointmaze,pear,lora,sweep" for r in runs)
    assert all(r["cfg"].env_kwargs["prompt_style"] == "pear" and r["cfg"].rollout_backend == "vllm" for r in runs)
    assert "--max-model-len" in shipped["vllm_args"]["HuggingFaceTB/SmolVLM-256M-Instruct"]


def test_shipped_fft_sweep_composes():
    import yaml
    shipped = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs/sweeps/pointmaze_small_fft.yaml")))
    runs = sweep.expand(shipped)
    assert len(runs) == 2 and all(r["cfg"].lora_rank == 0 and r["cfg"].rollout_backend == "hf" for r in runs)
    assert shipped["group"] == yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "configs/sweeps/pointmaze_small.yaml")))["group"]


def _alive(pid: int) -> bool:
    """Running (a killed zombie counts as dead: containers without an init process never reap orphans)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_stop_server_kills_sigterm_ignoring_parent_and_children(tmp_path):
    # parent ignores SIGTERM (like vLLM during startup) and spawns a child (like EngineCore)
    pidfile = tmp_path / "child.pid"
    code = ("import signal, subprocess, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
            f"open({str(pidfile)!r}, 'w').write(str(c.pid)); time.sleep(600)")
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.05)
    child = int(pidfile.read_text())
    t0 = time.time()
    sweep.stop_server(proc, timeout=1)
    assert proc.poll() is not None and time.time() - t0 < 10
    for _ in range(100):  # the child is reparented to init, which reaps it shortly after SIGKILL
        if not _alive(child):
            break
        time.sleep(0.05)
    else:
        pytest.fail("child process survived stop_server")


def test_zeroshot_think_sweeps_compose():
    import yaml
    d = os.path.join(os.path.dirname(__file__), "..", "configs/sweeps")
    runs = sweep.expand(yaml.safe_load(open(f"{d}/zeroshot_think.yaml")))
    assert len(runs) == 32 and all(r["cfg"].steps == 0 and r["cfg"].chat_template_kwargs == {} for r in runs)
    assert {(r["cfg"].env_kwargs["prompt_style"], r["cfg"].env_kwargs["think"]) for r in runs} == \
        {("pear", False), ("pear", True), ("simple", False), ("simple", True)}
    native = sweep.expand(yaml.safe_load(open(f"{d}/zeroshot_native_think.yaml")))
    assert len(native) == 8 and all(r["cfg"].chat_template_kwargs == {"enable_thinking": True}
                                    and r["cfg"].env_kwargs["think"] is False for r in native)


def test_merge_overrides_replaces_keys():
    base = ["experiment=x", "steps=40", "+env_kwargs.maze=umaze", "lr=1e-6"]
    assert sweep.merge_overrides(base, ["steps=2", "env_kwargs.maze=open"]) == \
        ["experiment=x", "lr=1e-6", "steps=2", "env_kwargs.maze=open"]
    assert sweep.merge_overrides(base, []) == base


def test_lanes_assign_disjoint_gpus_and_ports():
    assert sweep.lanes_for(1, {"train_gpu": 3, "vllm_gpu": 5, "vllm_port": 9000}) == \
        [{"train_gpu": 3, "vllm_gpu": 5, "vllm_port": 9000}]
    lanes = sweep.lanes_for(4, {"vllm_port": 8011})
    gpus = [g for l in lanes for g in (l["train_gpu"], l["vllm_gpu"])]
    assert sorted(gpus) == list(range(8)) and [l["vllm_port"] for l in lanes] == [8011, 8012, 8013, 8014]
    ov = sweep.lane_overrides(["a=1", "vllm_url=http://localhost:8011"], lanes[2])
    assert ov == ["a=1", "vllm_url=http://localhost:8013"]


def test_out_and_sync_roots():
    runs = sweep.expand({"group": "g", "overrides": ["experiment=pointmaze_q3vl4b_vllm"], "grid": {"seed": [1]}},
                        out_root="/opt/ml/checkpoints", sync_root="/scratch/sync")
    assert runs[0]["cfg"].out_dir == "/opt/ml/checkpoints/g/seed=1"
    assert runs[0]["cfg"].vllm_sync_dir == "/scratch/sync/g/seed-1"     # no '=' in paths handed to vLLM
    assert sweep.expand({"group": "g", "overrides": [], "grid": {"seed": [1]}})[0]["cfg"].vllm_sync_dir == ""


def test_run_lane_starts_full_ft_server_without_lora(monkeypatch, tmp_path):
    """Full fine-tuning runs get a server with ENABLE_LORA=0 on the lane's GPU/port, trainer on the lane's GPU."""
    import yaml
    shipped = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs/sweeps/sagemaker_gemma4_fft.yaml")))
    runs = sweep.expand(shipped, out_root=str(tmp_path))[:1]
    calls = {}

    class FakeProc:
        pid = 0

        def poll(self):
            return None

    def popen(cmd, env, **kw):
        calls["server"] = (cmd, env)
        return FakeProc()

    def call(cmd, env, **kw):
        calls["train"] = (cmd, env)
        return 0
    monkeypatch.setattr(sweep.subprocess, "Popen", popen)
    monkeypatch.setattr(sweep.subprocess, "call", call)
    monkeypatch.setattr(sweep, "wait_ready", lambda url, proc, **kw: calls.setdefault("url", url))
    monkeypatch.setattr(sweep, "stop_server", lambda proc: None)
    lane = sweep.lanes_for(4, shipped)[3]
    sweep.run_lane(runs, shipped, lane, 3, str(tmp_path))
    cmd, env = calls["server"]
    assert cmd[1:4] == ["google/gemma-4-E2B-it", "8014", "7"] and env["ENABLE_LORA"] == "0"
    tcmd, tenv = calls["train"]
    assert tenv["CUDA_VISIBLE_DEVICES"] == "6" and tcmd[-1] == "vllm_url=http://localhost:8014"
    assert calls["url"] == "http://localhost:8014" and runs[0]["rc"] == 0


def test_shipped_sagemaker_sweep_composes():
    import yaml
    shipped = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs/sweeps/sagemaker_gemma4_fft.yaml")))
    runs = sweep.expand(shipped)
    assert len(runs) == 4 and {r["cfg"].lr for r in runs} == {1e-6, 5e-6}
    c = runs[0]["cfg"]
    assert c.model == "google/gemma-4-E2B-it" and c.lora_rank == 0 and c.fp32_master and c.rollout_backend == "vllm"
    assert "embed_tokens" in c.freeze and c.max_frames == 1 and c.steps == 40


def test_smolvlm_procgen_sweeps_one_per_gpu():
    import yaml
    d = os.path.join(os.path.dirname(__file__), "..", "configs/sweeps")
    files = [yaml.safe_load(open(f"{d}/smolvlm_procgen_gpu{g}.yaml")) for g in (0, 1)]
    assert [(f["train_gpu"], f["vllm_gpu"]) for f in files] == [(0, 0), (1, 1)]
    assert files[0]["vllm_port"] != files[1]["vllm_port"] and files[0]["group"] == files[1]["group"]
    for f in files:
        (r,) = sweep.expand(f)
        c = r["cfg"]
        assert c.lora_rank == 0 and c.rollout_backend == "vllm" and c.reward == "success"
        assert c.env_kwargs["prompt_style"] == "compact" and c.on_invalid == "noop" and c.env_kwargs["maze"] == "gen11x11"
        assert set(c.eval_sets) == {"id_layouts", "ood_layouts", "ood_larger"}
    assert {sweep.expand(f)[0]["cfg"].lr for f in files} == {1e-5, 3e-5}
