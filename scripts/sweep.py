"""Run a small grid of training runs (e.g. reward fns x models) and log them to one wandb group.

    python scripts/sweep.py configs/sweeps/pointmaze_small.json [--dry_run]

Sweep file:
    {"base": "configs/pointmaze_qwen3vl4b_vllm.json",      # TrainConfig JSON (or inline dict)
     "group": "pointmaze-small-v2",                         # wandb group + output subdir
     "project": "vlm-rl-pointmaze",
     "train_gpu": 0, "vllm_gpu": 1, "vllm_port": 8011,
     "grid": {"model": ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"],
              "reward": ["success", "success+0.5*neg_dist", "pref_oracle_dist"]},
     "overrides": {"steps": 30},
     "model_overrides": {"HuggingFaceTB/SmolVLM2-2.2B-Instruct": {"lora_targets": "...", "processor_kwargs": {...}}},
     "vllm_args": {"HuggingFaceTB/SmolVLM2-2.2B-Instruct": ["--max-model-len", "8192"]}}

`model_overrides` are TrainConfig fields applied after the grid for runs of that model (e.g. LoRA target
names or processor kwargs that differ between architectures); `vllm_args` are extra `vllm serve` flags
for that model's server (later flags override launch_vllm.sh's defaults).

Runs are ordered model-major so each model's vLLM server (rollout_backend == "vllm") is started
once, reused for all of that model's runs, then stopped. Runs whose metrics.jsonl already reached
`steps` are skipped and unfinished runs continue from their last per-step checkpoint
(TrainConfig.resume), so re-running the sweep (e.g. after a crash or reboot) resumes it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slug(v) -> str:
    s = v.split("/")[-1] if isinstance(v, str) else json.dumps(v, sort_keys=True)
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", s).strip("_")[:40]


def expand(sweep: dict) -> list[dict]:
    """Cartesian product of `grid` over `base` + `overrides`; one dict of TrainConfig fields per run."""
    base = sweep["base"]
    if isinstance(base, str):
        with open(os.path.join(ROOT, base) if not os.path.isabs(base) else base) as f:
            base = json.load(f)
    base = {k: v for k, v in base.items() if not k.startswith("_")}
    base.update(sweep.get("overrides", {}))
    keys = list(sweep.get("grid", {}))
    keys.sort(key=lambda k: k != "model")  # model-major ordering
    runs = []
    for combo in itertools.product(*[sweep["grid"][k] for k in keys]):
        cfg = {**base, **dict(zip(keys, combo))}
        cfg.update(sweep.get("model_overrides", {}).get(cfg.get("model"), {}))
        if "env_kwargs" in sweep.get("grid", {}):
            cfg["env_kwargs"] = {**base.get("env_kwargs", {}), **dict(zip(keys, combo))["env_kwargs"]}
        name = "__".join(f"{k}={slug(v)}" for k, v in zip(keys, combo)) or "base"
        cfg["wandb_run_name"] = name
        cfg["wandb_group"] = sweep["group"]
        cfg["wandb_project"] = sweep.get("project", cfg.get("wandb_project"))
        cfg["out_dir"] = f"runs/{sweep['group']}/{name}"
        if cfg.get("rollout_backend") == "vllm":
            cfg["vllm_url"] = f"http://localhost:{sweep.get('vllm_port', 8011)}"
        runs.append(cfg)
    return runs


def finished(cfg) -> bool:
    p = os.path.join(ROOT, cfg["out_dir"], "metrics.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        steps = [json.loads(l)["step"] for l in f if l.strip()]
    return bool(steps) and max(steps) >= cfg.get("steps", 0)


def wait_ready(url, proc, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("vLLM server exited during startup; see its log")
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=5)
            return
        except Exception:
            time.sleep(10)
    raise TimeoutError("vLLM server not ready")


def stop_server(proc, timeout=60):
    """Stop the vLLM server and its children (EngineCore etc.). vLLM can ignore SIGTERM while it is
    still starting up, and SIGKILLing only the parent orphans the GPU-holding engine process, so the
    server runs in its own process group and the whole group is SIGTERMed, then SIGKILLed."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # also reaps children left behind by a clean parent exit
    except ProcessLookupError:
        pass
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    with open(a.sweep) as f:
        sweep = json.load(f)
    runs = expand(sweep)
    print(f"{len(runs)} runs in group {sweep['group']}")
    for r in runs:
        print(("  [done] " if finished(r) else "  ") + r["wandb_run_name"])
    if a.dry_run:
        return
    os.makedirs(os.path.join(ROOT, "runs", sweep["group"]), exist_ok=True)
    server, server_model = None, None
    try:
        for cfg in runs:
            if finished(cfg):
                continue
            if cfg.get("rollout_backend") == "vllm" and server_model != cfg["model"]:
                if server:
                    stop_server(server)
                log = open(os.path.join(ROOT, "runs", sweep["group"], f"vllm_{slug(cfg['model'])}.log"), "w")
                env = {**os.environ, "MAX_LORA_RANK": str(max(16, cfg.get("lora_rank", 64)))}
                server = subprocess.Popen([os.path.join(ROOT, "scripts/launch_vllm.sh"), cfg["model"],
                                           str(sweep.get("vllm_port", 8011)), str(sweep.get("vllm_gpu", 1)),
                                           *sweep.get("vllm_args", {}).get(cfg["model"], [])],
                                          stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                          start_new_session=True)
                server_model = cfg["model"]
                wait_ready(cfg["vllm_url"], server)
            os.makedirs(os.path.join(ROOT, cfg["out_dir"]), exist_ok=True)
            cfg_path = os.path.join(ROOT, cfg["out_dir"], "sweep_config.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"==> {cfg['wandb_run_name']}", flush=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(sweep.get("train_gpu", 0)), "PYTHONWARNINGS": "ignore"}
            with open(os.path.join(ROOT, cfg["out_dir"], "train.log"), "a") as log:  # resumed runs append
                rc = subprocess.call([sys.executable, "scripts/train.py", "--config", cfg_path],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            print(f"    exit {rc}", flush=True)
    finally:
        if server:
            stop_server(server)


if __name__ == "__main__":
    main()
