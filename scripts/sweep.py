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

Parallel lanes (one big node, e.g. 8 GPUs): `--lanes 4` runs 4 independent copies of the loop above, lane i
training on GPU 2i with its own vLLM server on GPU 2i+1 (port vllm_port+i); runs are dealt round-robin.
`--gpus_per_lane 1` puts a lane's trainer and server on the same GPU (server capped at `vllm_gpu_util_shared`,
default 0.3): for small models on 1-GPU machines.
`--out_root` moves run dirs (default `runs/`; SageMaker uses its checkpoint dir) and `--sync_root` puts the
per-step weight files the trainer hands to vLLM somewhere else (they are large for full fine-tuning and
should not be uploaded with the checkpoints).
`--shard I/N` keeps runs I, I+N, I+2N, ... (after model-major ordering), so N machines (e.g. an AWS Batch array
job) split one sweep; `--only REGEX` keeps runs whose name matches.
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
import threading
import time
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def slug(v) -> str:
    s = str(v).split("/")[-1]
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", s).strip("_")[:40]


def _plain(name: str) -> str:
    """Run name -> a path component with only [A-Za-z0-9._-] (e.g. 'lr=1e-06__maze=umaze' -> 'lr-1e-06__maze-umaze')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def _q(v) -> str:
    """Quote override values containing Hydra-grammar characters."""
    s = str(v)
    return s if re.fullmatch(r"[A-Za-z0-9_./:-]+", s) else "'" + s.replace("'", "\\'") + "'"


def _key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def merge_overrides(base: list[str], extra: list[str]) -> list[str]:
    """`base` with every key that `extra` sets replaced by the `extra` value (Hydra rejects duplicates)."""
    keys = {_key(o) for o in extra}
    return [o for o in base if _key(o) not in keys] + list(extra)


def expand(sweep: dict, out_root: str = "runs", sync_root: str = "") -> list[dict]:
    """-> [{"name", "overrides", "cfg"(TrainConfig)}], model-major order."""
    from vlmrl.config import compose
    keys = list(sweep.get("grid", {}))
    runs = []
    for combo in itertools.product(*[sweep["grid"][k] for k in keys]):
        name = "__".join(f"{k}={slug(v)}" for k, v in zip(keys, combo)) or "base"
        ov = list(sweep.get("overrides", [])) + [f"{k}={_q(v)}" for k, v in zip(keys, combo)]
        group = sweep["group"]
        ov += [f"wandb_group={_q(group)}", f"wandb_run_name={_q(name)}", f"out_dir={_q(f'{out_root}/{group}/{name}')}"]
        if sync_root:
            # plain characters only (no '='): these paths are handed to vLLM's model loader
            ov.append(f"vllm_sync_dir={_q(f'{sync_root}/{slug(group)}/{_plain(name)}')}")
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


def lanes_for(n: int, sweep: dict, gpus_per_lane: int = 2) -> list[dict]:
    """GPU/port assignment per lane: lane i trains on GPU 2i, serves on GPU 2i+1 (1 lane = the sweep file's GPUs).
    ``gpus_per_lane=1``: trainer and vLLM server share GPU i (the server gets ``vllm_gpu_util_shared`` of it)."""
    port = int(sweep.get("vllm_port", 8011))
    if gpus_per_lane == 1:
        return [{"train_gpu": i, "vllm_gpu": i, "vllm_port": port + i, "shared": True} for i in range(max(1, n))]
    if n <= 1:
        return [{"train_gpu": sweep.get("train_gpu", 0), "vllm_gpu": sweep.get("vllm_gpu", 1), "vllm_port": port}]
    return [{"train_gpu": 2 * i, "vllm_gpu": 2 * i + 1, "vllm_port": port + i} for i in range(n)]


def lane_overrides(overrides: list[str], lane: dict) -> list[str]:
    return [o for o in overrides if not o.startswith("vllm_url=")] + [f"vllm_url=http://localhost:{lane['vllm_port']}"]


def select(runs: list[dict], shard: str = "", only: str = "") -> list[dict]:
    """Filter by run-name regex, then keep shard I of N ("I/N")."""
    if only:
        runs = [r for r in runs if re.search(only, r["name"])]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        assert 0 <= i < n, f"bad shard {shard}"
        runs = runs[i::n]
    return runs


def run_lane(runs: list[dict], sweep: dict, lane: dict, idx: int, logdir: str, log_tag: str = ""):
    """Run `runs` one after another on this lane's GPUs, starting one vLLM server per model."""
    server, server_model, tag = None, None, f"[lane {idx}] " if idx >= 0 else ""
    url = f"http://localhost:{lane['vllm_port']}"
    try:
        for r in runs:
            cfg = r["cfg"]
            if finished(r):
                continue
            if cfg.rollout_backend == "vllm" and server_model != cfg.model:
                if server:
                    stop_server(server)
                suffix = (f"_lane{idx}" if idx >= 0 else "") + log_tag
                log = open(os.path.join(logdir, f"vllm_{slug(cfg.model)}{suffix}_{lane['vllm_port']}.log"), "w")
                env = {**os.environ, "MAX_LORA_RANK": str(max(16, cfg.lora_rank)),
                       "ENABLE_LORA": "1" if cfg.lora_rank > 0 else "0"}
                if lane.get("shared"):   # the trainer needs the rest of the GPU (server starts first)
                    env["GPU_UTIL"] = str(sweep.get("vllm_gpu_util_shared", 0.3))
                elif sweep.get("vllm_gpu_util"):
                    env["GPU_UTIL"] = str(sweep["vllm_gpu_util"])
                elif cfg.lora_rank == 0:   # full-weight reloads need headroom for a second copy of the largest tensor
                    env["GPU_UTIL"] = str(sweep.get("vllm_gpu_util_full", 0.7))
                server = subprocess.Popen([os.path.join(ROOT, "scripts/launch_vllm.sh"), cfg.model,
                                           str(lane["vllm_port"]), str(lane["vllm_gpu"]),
                                           *map(str, sweep.get("vllm_args", {}).get(cfg.model, []))],
                                          stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                          start_new_session=True)
                server_model = cfg.model
                wait_ready(url, server)
            os.makedirs(os.path.join(ROOT, cfg.out_dir), exist_ok=True)
            print(f"{tag}==> {r['name']}", flush=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(lane["train_gpu"]), "PYTHONWARNINGS": "ignore"}
            with open(os.path.join(ROOT, cfg.out_dir, "train.log"), "a") as log:  # resumed runs append
                rc = subprocess.call([sys.executable, "scripts/train.py", *lane_overrides(r["overrides"], lane)],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            r["rc"] = rc
            print(f"{tag}    {r['name']}: exit {rc}", flush=True)
    finally:
        if server:
            stop_server(server)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--lanes", type=int, default=1, help="parallel lanes (2 GPUs each)")
    ap.add_argument("--gpus_per_lane", type=int, default=2, choices=(1, 2),
                    help="1: trainer and vLLM server share one GPU per lane")
    ap.add_argument("--out_root", default="runs", help="run dirs go to <out_root>/<group>/<run>")
    ap.add_argument("--sync_root", default="", help="weights handed to vLLM go to <sync_root>/<group>/<run>")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Hydra override for every run (replaces the sweep file's value)")
    ap.add_argument("--group", default="", help="override the sweep file's group (wandb group + run dirs)")
    ap.add_argument("--shard", default="", metavar="I/N", help="keep runs I, I+N, ... (split a sweep over N machines)")
    ap.add_argument("--only", default="", metavar="REGEX", help="keep runs whose name matches")
    a = ap.parse_args()
    with open(a.sweep) as f:
        sweep = yaml.safe_load(f)
    sweep["overrides"] = merge_overrides(sweep.get("overrides", []), a.set)
    if a.group:
        sweep["group"] = a.group
    runs = select(expand(sweep, a.out_root, a.sync_root), a.shard, a.only)
    lanes = lanes_for(a.lanes, sweep, a.gpus_per_lane)
    log_tag = f"_shard{a.shard.split('/')[0]}" if a.shard else ""
    print(f"{len(runs)} runs in group {sweep['group']} on {len(lanes)} lane(s)" + (f", shard {a.shard}" if a.shard else ""))
    for i, r in enumerate(runs):
        lane = f"[lane {i % len(lanes)}] " if len(lanes) > 1 else ""
        print(("  [done] " if finished(r) else "  ") + lane + r["name"] + "   " + " ".join(r["overrides"]))
    if a.dry_run:
        return 0
    logdir = os.path.join(ROOT, a.out_root, sweep["group"])
    os.makedirs(logdir, exist_ok=True)
    if len(lanes) == 1:
        run_lane(runs, sweep, lanes[0], -1, logdir, log_tag)
    else:
        threads = [threading.Thread(target=run_lane, args=(runs[i::len(lanes)], sweep, lane, i, logdir, log_tag))
                   for i, lane in enumerate(lanes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return 0 if all(finished(r) for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
