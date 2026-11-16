"""Run a small grid of training runs (e.g. reward fns x models) into one wandb group.

    python scripts/sweep.py configs/sweeps/pointmaze_small.yaml [--dry_run]

Sweep file (YAML):
    group: pointmaze-small-v2            # wandb group + runs/<group>/<run>/
    project: vlm-rl-pointmaze
    train_gpu: 0
    vllm_gpu: 1
    vllm_port: 8011
    overrides: [experiment=pointmaze_q3vl4b_vllm, steps=30]     # Hydra overrides for every run
    grid: {policy: [qwen3vl2b_lora, qwen3vl4b_lora], reward: [success, "success+0.5*neg_dist"]}
    vllm_args: {HuggingFaceTB/SmolVLM2-2.2B-Instruct: [--max-model-len, "8192"]}   # extra `vllm serve` flags per model

Per-model *training* settings (LoRA target names, processor kwargs) live in the policy presets
(configs/policy/*.yaml), so a grid over `policy` carries them automatically.

Each grid point becomes Hydra overrides for scripts/train.py; the full config is composed up front
(so typos fail before anything launches). Runs are ordered by model so each model's vLLM server
(rollout_backend=vllm) is started once and reused; models keep their grid order. Finished runs are
skipped and unfinished runs continue from their last per-step checkpoint (TrainConfig.resume), so
re-running the sweep (e.g. after a crash or reboot) resumes it.
Plain Hydra multirun (`train.py -m ...`) also works when no per-model vLLM server is needed.
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

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def slug(v) -> str:
    s = str(v).split("/")[-1]
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", s).strip("_")[:40]


def _q(v) -> str:
    """Quote override values containing Hydra-grammar characters."""
    s = str(v)
    return s if re.fullmatch(r"[A-Za-z0-9_./:-]+", s) else "'" + s.replace("'", "\\'") + "'"


def expand(sweep: dict) -> list[dict]:
    """-> [{"name", "overrides", "cfg"(TrainConfig)}], model-major order."""
    from vlmrl.config import compose
    keys = list(sweep.get("grid", {}))
    runs = []
    for combo in itertools.product(*[sweep["grid"][k] for k in keys]):
        name = "__".join(f"{k}={slug(v)}" for k, v in zip(keys, combo)) or "base"
        ov = list(sweep.get("overrides", [])) + [f"{k}={_q(v)}" for k, v in zip(keys, combo)]
        group = sweep["group"]
        ov += [f"wandb_group={_q(group)}", f"wandb_run_name={_q(name)}", f"out_dir={_q(f'runs/{group}/{name}')}"]
        if sweep.get("project"):
            ov.append(f"wandb_project={_q(sweep['project'])}")
        if "vllm_port" in sweep:
            ov.append(f"vllm_url=http://localhost:{sweep['vllm_port']}")
        runs.append({"name": name, "overrides": ov, "cfg": compose(ov)})
    first = {}
    for r in runs:
        first.setdefault(r["cfg"].model, len(first))
    runs.sort(key=lambda r: first[r["cfg"].model])  # group by model, first-appearance order (stable)
    return runs


def finished(run) -> bool:
    p = os.path.join(ROOT, run["cfg"].out_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        steps = [json.loads(line)["step"] for line in f if line.strip()]
    return bool(steps) and max(steps) >= run["cfg"].steps


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
        sweep = yaml.safe_load(f)
    runs = expand(sweep)
    print(f"{len(runs)} runs in group {sweep['group']}")
    for r in runs:
        print(("  [done] " if finished(r) else "  ") + r["name"] + "   " + " ".join(r["overrides"]))
    if a.dry_run:
        return
    os.makedirs(os.path.join(ROOT, "runs", sweep["group"]), exist_ok=True)
    server, server_model = None, None
    try:
        for r in runs:
            cfg = r["cfg"]
            if finished(r):
                continue
            if cfg.rollout_backend == "vllm" and server_model != cfg.model:
                if server:
                    stop_server(server)
                log = open(os.path.join(ROOT, "runs", sweep["group"], f"vllm_{slug(cfg.model)}.log"), "w")
                env = {**os.environ, "MAX_LORA_RANK": str(max(16, cfg.lora_rank))}
                server = subprocess.Popen([os.path.join(ROOT, "scripts/launch_vllm.sh"), cfg.model,
                                           str(sweep.get("vllm_port", 8011)), str(sweep.get("vllm_gpu", 1)),
                                           *map(str, sweep.get("vllm_args", {}).get(cfg.model, []))],
                                          stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                          start_new_session=True)
                server_model = cfg.model
                wait_ready(cfg.vllm_url, server)
            os.makedirs(os.path.join(ROOT, cfg.out_dir), exist_ok=True)
            print(f"==> {r['name']}", flush=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(sweep.get("train_gpu", 0)), "PYTHONWARNINGS": "ignore"}
            with open(os.path.join(ROOT, cfg.out_dir, "train.log"), "a") as log:  # resumed runs append
                rc = subprocess.call([sys.executable, "scripts/train.py", *r["overrides"]],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            print(f"    exit {rc}", flush=True)
    finally:
        if server:
            stop_server(server)


if __name__ == "__main__":
    main()
