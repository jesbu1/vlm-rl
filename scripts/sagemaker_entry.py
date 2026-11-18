"""Inside the SageMaker container: run a sweep file on this node's GPUs (launched by docker/entrypoint.sh).

Job hyperparameters (set by scripts/sagemaker_submit.py; SageMaker stores every value as a string):
    sweep      sweep file in the repo, e.g. configs/sweeps/sagemaker_gemma4_fft.yaml
    lanes      parallel lanes (2 GPUs each; default: #GPUs // 2)
    group      optional wandb group / run-dir name override
    overrides  optional JSON list of Hydra overrides applied to every run (e.g. ["steps=2"])

Layout:
    /opt/ml/checkpoints/<group>/<run>/   run dirs (metrics, episodes, resume state, final ckpt). SageMaker
                                         syncs this to CheckpointConfig.S3Uri and restores it on restart, so a
                                         retried/preempted job resumes every run where it stopped.
    <scratch>/hf, <scratch>/sync         HF cache and the per-step weights handed to vLLM (never uploaded)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_PATH = "/opt/ml/input/config/hyperparameters.json"
CKPT_DIR = "/opt/ml/checkpoints"


def read_hyperparameters(path: str = HP_PATH) -> dict:
    """SageMaker writes {name: string}; values that look like JSON are decoded."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        try:
            out[k] = json.loads(v) if isinstance(v, str) else v
        except json.JSONDecodeError:
            out[k] = v
    return out


def pick_scratch(candidates=("/opt/ml/scratch", "/tmp")) -> str:
    """The candidate with the most free space (instance NVMe vs root volume differ per instance type)."""
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


def sweep_command(hp: dict, n_gpus: int, out_root: str, sync_root: str) -> list[str]:
    lanes = int(hp.get("lanes") or max(1, n_gpus // 2))
    cmd = [sys.executable, os.path.join(ROOT, "scripts/sweep.py"), hp["sweep"], "--lanes", str(lanes),
           "--out_root", out_root, "--sync_root", sync_root]
    if hp.get("group"):
        cmd += ["--group", str(hp["group"])]
    for o in hp.get("overrides") or []:
        cmd += ["--set", str(o)]
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
    import yaml
    sw = load_sweep_module()
    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)
    sweep["overrides"] = sw.merge_overrides(sweep.get("overrides", []), [str(o) for o in hp.get("overrides") or []])
    return sorted({r["cfg"].model for r in sw.expand(sweep)})


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        return 0


def main() -> int:
    hp = read_hyperparameters()
    if "sweep" not in hp:
        print("[entry] no `sweep` hyperparameter; nothing to do", flush=True)
        return 2
    print(f"[entry] vlmrl commit {os.environ.get('VLMRL_GIT_COMMIT', '?')}; hyperparameters {hp}", flush=True)
    n_gpus = gpu_count()
    print(f"[entry] {n_gpus} GPUs", flush=True)
    scratch = pick_scratch()
    os.environ.setdefault("HF_HOME", os.path.join(scratch, "hf"))
    if not os.environ.get("WANDB_API_KEY"):
        print("[entry] WANDB_API_KEY not set: wandb runs offline (files are still saved with the run dirs)", flush=True)
        os.environ["WANDB_MODE"] = "offline"
    out_root = CKPT_DIR if os.path.isdir(CKPT_DIR) else os.path.join(scratch, "runs")

    # download each model once, before the lanes start (otherwise every trainer + server downloads it)
    from huggingface_hub import snapshot_download
    for m in models_in(os.path.join(ROOT, hp["sweep"]), hp):
        print(f"[entry] downloading {m}", flush=True)
        snapshot_download(m)

    cmd = sweep_command(hp, n_gpus, out_root, os.path.join(scratch, "sync"))
    print("[entry] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        os.makedirs("/opt/ml/output", exist_ok=True)
        with open("/opt/ml/output/failure", "w") as f:  # shown as the job's FailureReason
            f.write(f"sweep exited {rc}; see the run dirs' train.log and vllm_*.log under the checkpoint S3 URI")
    return rc


if __name__ == "__main__":
    sys.exit(main())
