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


def test_expand_model_major_with_model_overrides():
    runs = sweep.expand({
        "base": {"model": "a/A", "lora_targets": "lm", "rollout_backend": "vllm", "_comment": "x"},
        "group": "g", "grid": {"reward": ["r1", "r2"], "model": ["a/A", "b/B"]},
        "overrides": {"steps": 3},
        "model_overrides": {"b/B": {"lora_targets": "text", "processor_kwargs": {"do_image_splitting": False}}},
    })
    assert [(r["model"], r["reward"]) for r in runs] == [("a/A", "r1"), ("a/A", "r2"), ("b/B", "r1"), ("b/B", "r2")]
    assert [r["lora_targets"] for r in runs] == ["lm", "lm", "text", "text"]
    assert "processor_kwargs" not in runs[0] and runs[2]["processor_kwargs"] == {"do_image_splitting": False}
    assert runs[2]["out_dir"] == "runs/g/model=B__reward=r1" and runs[2]["steps"] == 3
    assert "_comment" not in runs[0] and runs[0]["vllm_url"].startswith("http://localhost:")


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
