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
    assert len(runs) == 9 and runs[-1]["cfg"].model == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    assert all(r["cfg"].wandb_entity == "tri" and r["cfg"].wandb_tags == "pointmaze,medium,lora,sweep" for r in runs)
    assert all(r["cfg"].env_kwargs["maze"] == "medium" and r["cfg"].rollout_backend == "vllm" for r in runs)
    assert "--max-model-len" in shipped["vllm_args"]["HuggingFaceTB/SmolVLM2-2.2B-Instruct"]


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
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("child process survived stop_server")
