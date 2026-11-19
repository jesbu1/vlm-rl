"""Inside the training container: run (a shard of) a sweep on this node's GPUs. Started by docker/entrypoint.sh
for both SageMaker training jobs (scripts/sagemaker_submit.py) and AWS Batch container jobs (scripts/batch_submit.py).

Job settings come from SageMaker's hyperparameters file (every value a string; JSON values are decoded) or, when
that file is absent, from environment variables:

    sweep      VLMRL_SWEEP       sweep file in the repo, e.g. configs/sweeps/sagemaker_gemma4_fft.yaml
    lanes      VLMRL_LANES       parallel lanes (default: #GPUs // gpus_per_lane)
    gpus_per_lane  VLMRL_GPUS_PER_LANE  2 (trainer + vLLM GPU) or 1 (shared); default 1 on a 1-GPU machine, else 2
    group      VLMRL_GROUP       optional wandb group / run-dir name override
    overrides  VLMRL_OVERRIDES   optional JSON list of Hydra overrides for every run, e.g. ["steps=2"]
    only       VLMRL_ONLY        optional run-name regex
    shards     VLMRL_SHARDS      split the sweep over N jobs; this job is shard AWS_BATCH_JOB_ARRAY_INDEX (array jobs)
    s3_output  VLMRL_S3_OUTPUT   (Batch) s3://... prefix the run dirs are synced to

Layout:
    SageMaker: /opt/ml/checkpoints/<group>/<run>/ -- SageMaker syncs it to CheckpointConfig.S3Uri and restores it
               on a retry, so a retried / preempted job resumes every run where it stopped.
    Batch:     <scratch>/runs/<group>/<run>/ -- restored from s3_output at start, uploaded every
               VLMRL_SYNC_EVERY seconds (default 300) and at the end (same resume behaviour).
    <scratch>/hf, <scratch>/sync: HF cache and the per-step weights handed to vLLM (never uploaded).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_PATH = "/opt/ml/input/config/hyperparameters.json"
CKPT_DIR = "/opt/ml/checkpoints"
ENV_KEYS = {"sweep": "VLMRL_SWEEP", "lanes": "VLMRL_LANES", "group": "VLMRL_GROUP", "overrides": "VLMRL_OVERRIDES",
            "only": "VLMRL_ONLY", "shards": "VLMRL_SHARDS", "s3_output": "VLMRL_S3_OUTPUT",
            "gpus_per_lane": "VLMRL_GPUS_PER_LANE"}


def _decode(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def read_hyperparameters(path: str = HP_PATH) -> dict:
    """SageMaker writes {name: string}; values that look like JSON are decoded."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: _decode(v) for k, v in json.load(f).items()}


def job_config(hp_path: str = HP_PATH, environ=os.environ) -> dict:
    """SageMaker hyperparameters if present, else the VLMRL_* environment variables; plus the array-job shard."""
    hp = read_hyperparameters(hp_path)
    if not hp:
        hp = {k: _decode(environ[e]) for k, e in ENV_KEYS.items() if environ.get(e)}
    n = int(hp.get("shards") or 1)
    if n > 1:
        hp["shard"] = f"{int(environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))}/{n}"
    return hp


def pick_scratch(candidates=("/opt/ml/scratch", "/tmp", "/dev/shm")) -> str:
    """The candidate with the most free space. SageMaker mounts instance storage at /opt/ml; Batch containers
    often get a small root volume, where the RAM-backed /dev/shm (job definition sharedMemorySize) is bigger."""
    best, best_free = None, -1
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            free = shutil.disk_usage(c).free
        except OSError:
            continue
        if free > best_free:
            best, best_free = c, free
    print(f"[entry] scratch dir {best} ({best_free / 1e9:.0f} GB free)", flush=True)
    return best


def free_port_base(n: int, lo: int = 10000, hi: int = 32000, rng=None) -> int:
    """A random base port with ``n`` consecutive free ports. Batch can place several jobs on one host with host
    networking, so fixed ports (8011, 8012, ...) collide and a lane ends up talking to another job's server.

    The range stays below Linux's ephemeral ports (32768-60999): a running vLLM server opens internal sockets on
    OS-assigned ephemeral ports, and with a base inside that range lane 0's server took lane 1's port (base + 1)
    before lane 1 started ("Address already in use")."""
    import random
    import socket
    rng = rng or random.Random()
    for _ in range(200):
        base = rng.randrange(lo, hi - n)
        try:
            socks = []
            for p in range(base, base + n):
                s = socket.socket()
                s.bind(("127.0.0.1", p))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port range found")


def sweep_command(hp: dict, n_gpus: int, out_root: str, sync_root: str, vllm_port: int | None = None) -> list[str]:
    gpl = int(hp.get("gpus_per_lane") or (1 if n_gpus == 1 else 2))
    lanes = int(hp.get("lanes") or max(1, n_gpus // gpl))
    cmd = [sys.executable, os.path.join(ROOT, "scripts/sweep.py"), hp["sweep"], "--lanes", str(lanes),
           "--gpus_per_lane", str(gpl), "--out_root", out_root, "--sync_root", sync_root]
    if vllm_port:
        cmd += ["--vllm_port", str(vllm_port)]
    if hp.get("group"):
        cmd += ["--group", str(hp["group"])]
    for o in hp.get("overrides") or []:
        cmd += ["--set", str(o)]
    if hp.get("shard"):
        cmd += ["--shard", hp["shard"]]
    if hp.get("only"):
        cmd += ["--only", str(hp["only"])]
    return cmd


def load_sweep_module():
    """scripts/ is not a package, so load scripts/sweep.py by path."""
    import importlib.util
    sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location("vlmrl_sweep", os.path.join(ROOT, "scripts", "sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def models_in(sweep_path: str, hp: dict) -> list[str]:
    """Models of the runs this job will actually train (after overrides, --only and the shard)."""
    import yaml
    sw = load_sweep_module()
    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)
    sweep["overrides"] = sw.merge_overrides(sweep.get("overrides", []), [str(o) for o in hp.get("overrides") or []])
    runs = sw.select(sw.expand(sweep), hp.get("shard", ""), str(hp.get("only") or ""))
    return sorted({r["cfg"].model for r in runs})


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        return 0


def s3_sync_command(src: str, dst: str, include: list[str] | None = None) -> list[str]:
    """``include``: only these paths (globs relative to the synced dir). Machines of one array job share an S3
    prefix, so each must touch only its own runs: syncing everything restored other machines' runs at start and
    later uploaded those stale copies over their newer files."""
    filt = []
    if include is not None:
        filt = ["--exclude", "*"] + [x for p in include for x in ("--include", p)]
    # the trainer rewrites resume/state.pt atomically via state.pt.tmp; never upload the partial file (last
    # filter wins in `aws s3 sync`, so this comes after the includes)
    return ["aws", "s3", "sync", src, dst, "--no-progress", "--only-show-errors", *filt, "--exclude", "*.tmp"]


def own_paths(sweep_path: str, hp: dict) -> list[str]:
    """Sync globs for the runs this job trains: <group>/<run>/* and this shard's server logs."""
    import yaml
    sw = load_sweep_module()
    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)
    sweep["overrides"] = sw.merge_overrides(sweep.get("overrides", []), [str(o) for o in hp.get("overrides") or []])
    if hp.get("group"):
        sweep["group"] = str(hp["group"])
    runs = sw.select(sw.expand(sweep), hp.get("shard", ""), str(hp.get("only") or ""))
    g = sweep["group"]
    logs = f"{g}/vllm_*_shard{hp['shard'].split('/')[0]}_*" if hp.get("shard") else f"{g}/vllm_*"
    return [f"{g}/{r['name']}/*" for r in runs] + [logs]


class S3Syncer:
    """Periodic `aws s3 sync out_root -> s3_output` in a background thread (Batch has no checkpoint sync)."""

    def __init__(self, local: str, remote: str, every: float = 300.0, run=subprocess.call,
                 include: list[str] | None = None):
        self.local, self.remote, self.every, self._run = local, remote.rstrip("/") + "/", every, run
        self.include = include
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def restore(self):
        os.makedirs(self.local, exist_ok=True)
        rc = self._run(s3_sync_command(self.remote, self.local, self.include))
        print(f"[entry] restored run dirs from {self.remote} (exit {rc})", flush=True)

    def push(self):
        rc = self._run(s3_sync_command(self.local, self.remote, self.include))
        if rc:
            print(f"[entry] s3 sync to {self.remote} exited {rc}", flush=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            self.push()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.push()


def main() -> int:
    hp = job_config()
    if "sweep" not in hp:
        print("[entry] no sweep given (SageMaker hyperparameter `sweep` or VLMRL_SWEEP); nothing to do", flush=True)
        return 2
    shown = {k: v for k, v in hp.items()}
    print(f"[entry] vlmrl commit {os.environ.get('VLMRL_GIT_COMMIT', '?')}; job {shown}", flush=True)
    n_gpus = gpu_count()
    print(f"[entry] {n_gpus} GPUs", flush=True)
    scratch = pick_scratch()
    os.environ.setdefault("HF_HOME", os.path.join(scratch, "hf"))
    if not os.environ.get("WANDB_API_KEY"):
        print("[entry] WANDB_API_KEY not set: wandb runs offline (files are still saved with the run dirs)", flush=True)
        os.environ["WANDB_MODE"] = "offline"
    sagemaker = os.path.isdir(CKPT_DIR)
    # several Batch jobs can share a host (and its bind-mounted scratch disk): keep each job's files apart
    job_dir = os.path.join(scratch, "jobs", os.environ.get("AWS_BATCH_JOB_ID", "local").replace(":", "_"))
    out_root = CKPT_DIR if sagemaker else os.path.join(job_dir, "runs")
    syncer = None
    if not sagemaker and hp.get("s3_output"):
        syncer = S3Syncer(out_root, str(hp["s3_output"]), float(os.environ.get("VLMRL_SYNC_EVERY", 300)),
                          include=own_paths(os.path.join(ROOT, hp["sweep"]), hp))
        syncer.restore()
        syncer.start()

    rc = 1
    try:
        # download each model once, before the lanes start (otherwise every trainer + server downloads it)
        from huggingface_hub import snapshot_download
        for m in models_in(os.path.join(ROOT, hp["sweep"]), hp):
            print(f"[entry] downloading {m}", flush=True)
            snapshot_download(m)
        lanes = int(hp.get("lanes") or max(1, n_gpus // int(hp.get("gpus_per_lane") or (1 if n_gpus == 1 else 2))))
        cmd = sweep_command(hp, n_gpus, out_root, os.path.join(job_dir, "sync"),
                            vllm_port=None if sagemaker else free_port_base(lanes))
        print("[entry] " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
    finally:
        if syncer:
            syncer.stop()
    if rc != 0 and sagemaker:
        os.makedirs("/opt/ml/output", exist_ok=True)
        with open("/opt/ml/output/failure", "w") as f:  # shown as the job's FailureReason
            f.write(f"sweep exited {rc}; see the run dirs' train.log and vllm_*.log under the checkpoint S3 URI")
    return rc


if __name__ == "__main__":
    sys.exit(main())
