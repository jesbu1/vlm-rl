"""Submit a sweep to an AWS Batch *container* (ECS) GPU queue, split over parallel machines.

    python scripts/batch_submit.py configs/sweeps/batch_small_fft.yaml --dry_run   # print job definition + job
    python scripts/batch_submit.py configs/sweeps/batch_small_fft.yaml             # one machine per shard
    python scripts/batch_submit.py configs/sweeps/batch_small_fft.yaml --group smoke --set steps=2 --shards 2

The sweep's runs are dealt over `--shards` machines (default: enough for one run per lane), submitted as one Batch
*array job*: child i runs `sweep.py --shard i/N` with `--lanes` lanes (2 GPUs each) on its machine. The image is
the same as for SageMaker (docker/Dockerfile), pushed to this account's ECR. Each submit registers a job
definition revision for the image tag (GPU / vCPU / memory from the config, shared memory for vLLM).

Run dirs are synced to `<s3_output>/<job>/` while the job runs (scripts/job_entry.py), and restored from there
when Batch retries a child, so retries resume. Account settings come from the git-ignored
configs/sagemaker/local_batch.yaml (template: configs/sagemaker/example_batch.yaml); secrets are read from the
submitting shell's environment, like scripts/sagemaker_submit.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "sagemaker", "local_batch.yaml")
REQUIRED = ("region", "queue", "ecr_repository", "job_role_arn", "execution_role_arn", "s3_output", "gpus")

_spec = importlib.util.spec_from_file_location("vlmrl_sagemaker_submit", os.path.join(ROOT, "scripts", "sagemaker_submit.py"))
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def job_definition(cfg: dict, image: str) -> dict:
    """register-job-definition request: the image's entrypoint runs scripts/job_entry.py (`train`)."""
    name = cfg.get("job_definition", f"{cfg['ecr_repository']}-gpu{cfg['gpus']}")
    res = [{"type": "GPU", "value": str(cfg["gpus"])}]
    if cfg.get("vcpus"):
        res.append({"type": "VCPU", "value": str(cfg["vcpus"])})
    if cfg.get("memory_mib"):
        res.append({"type": "MEMORY", "value": str(cfg["memory_mib"])})
    return {
        "jobDefinitionName": name,
        "type": "container",
        "containerProperties": {
            "image": image,
            "command": ["train"],
            "jobRoleArn": cfg["job_role_arn"],
            "executionRoleArn": cfg["execution_role_arn"],
            "resourceRequirements": res,
            # vLLM / torch multiprocessing need a real /dev/shm (Docker's default is 64 MB)
            "linuxParameters": {"sharedMemorySize": int(cfg.get("shm_mib", 65536))},
            "ulimits": [{"name": "nofile", "softLimit": 65536, "hardLimit": 65536}],
        },
        "tags": {k: str(v) for k, v in (cfg.get("tags") or {}).items()},
        "propagateTags": True,
    }


def submit_request(cfg: dict, job_def_arn: str, name: str, sweep: str, shards: int, lanes: int, group: str,
                   overrides: list[str], only: str, max_hours: float, env: dict, gpus_per_lane: int = 2) -> dict:
    """submit-job request. Settings reach scripts/job_entry.py as VLMRL_* environment variables."""
    vlmrl = {"VLMRL_SWEEP": sweep, "VLMRL_LANES": str(lanes), "VLMRL_SHARDS": str(shards),
             "VLMRL_GPUS_PER_LANE": str(gpus_per_lane), "VLMRL_S3_OUTPUT": f"{cfg['s3_output'].rstrip('/')}/{name}/"}
    if group:
        vlmrl["VLMRL_GROUP"] = group
    if overrides:
        vlmrl["VLMRL_OVERRIDES"] = json.dumps(overrides)
    if only:
        vlmrl["VLMRL_ONLY"] = only
    environment = {**{k: str(v) for k, v in (cfg.get("env") or {}).items()}, **vlmrl, **env}
    req = {
        "jobName": name,
        "jobQueue": cfg["queue"],
        "jobDefinition": job_def_arn,
        "containerOverrides": {"environment": [{"name": k, "value": v} for k, v in environment.items()]},
        "timeout": {"attemptDurationSeconds": int(max_hours * 3600)},
        "retryStrategy": {"attempts": int(cfg.get("retry_attempts", 1))},
        "tags": {k: str(v) for k, v in (cfg.get("tags") or {}).items()},
        "propagateTags": True,
    }
    if shards > 1:
        req["arrayProperties"] = {"size": shards}
    return req


def redacted(req: dict, secret_names) -> dict:
    r = json.loads(json.dumps(req))
    for e in r.get("containerOverrides", {}).get("environment", []):
        if e["name"] in secret_names:
            e["value"] = "<redacted>"
    return r


def count_runs(sweep_path: str, overrides: list[str], only: str, group: str) -> int:
    import yaml
    spec = importlib.util.spec_from_file_location("vlmrl_sweep", os.path.join(ROOT, "scripts", "sweep.py"))
    sw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sw)
    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)
    sweep["overrides"] = sw.merge_overrides(sweep.get("overrides", []), overrides)
    if group:
        sweep["group"] = group
    return len(sw.select(sw.expand(sweep), "", only))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep", nargs="?", help="sweep file, relative to the repo root")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="local (git-ignored) account settings")
    ap.add_argument("--tag", default="", help="image tag (default: the current git commit)")
    ap.add_argument("--lanes", type=int, default=None, help="lanes per machine (default: gpus // gpus_per_lane)")
    ap.add_argument("--gpus", type=int, default=None, help="GPUs per machine (overrides the config)")
    ap.add_argument("--vcpus", type=int, default=None, help="vCPUs per machine (overrides the config)")
    ap.add_argument("--memory_mib", type=int, default=None, help="memory per machine (overrides the config)")
    ap.add_argument("--gpus_per_lane", type=int, default=None, choices=(1, 2),
                    help="2: trainer + vLLM GPU per lane; 1: shared GPU (default: 1 if gpus == 1 else 2)")
    ap.add_argument("--shards", type=int, default=None, help="machines (default: ceil(#runs / lanes))")
    ap.add_argument("--group", default="", help="override the sweep's group (wandb group + run dirs)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Hydra override for every run")
    ap.add_argument("--only", default="", metavar="REGEX", help="only runs whose name matches")
    ap.add_argument("--max_hours", type=float, default=None, help="per-machine time limit (default: config max_hours)")
    ap.add_argument("--name", default="", help="job name (default: vlmrl-<group>-<timestamp>)")
    ap.add_argument("--allow_missing_secrets", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--image", action="store_true", help="print the image URI for this commit and exit")
    a = ap.parse_args(argv)
    cfg = sm.load_config(a.config, REQUIRED, "example_batch.yaml")
    for k in ("gpus", "vcpus", "memory_mib"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    account = sm.aws(cfg, "sts", "get-caller-identity")["Account"]
    image = sm.image_uri(cfg, account, a.tag or sm.git_commit())
    if a.image:
        print(image)
        return 0
    if not a.sweep:
        ap.error("sweep file required")
    if os.path.isabs(a.sweep) or not os.path.exists(os.path.join(ROOT, a.sweep)):
        ap.error("sweep must be a path relative to the repo root (it is read inside the image)")
    import yaml
    with open(os.path.join(ROOT, a.sweep)) as f:
        group = a.group or yaml.safe_load(f)["group"]
    gpl = a.gpus_per_lane or (1 if int(cfg["gpus"]) == 1 else 2)
    lanes = a.lanes or max(1, int(cfg["gpus"]) // gpl)
    n_runs = count_runs(os.path.join(ROOT, a.sweep), a.set, a.only, a.group)
    shards = a.shards or max(1, math.ceil(n_runs / lanes))
    names = list(cfg.get("secret_env") or [])
    env = sm.secret_env(names, allow_missing=a.allow_missing_secrets or a.dry_run)
    name = a.name or sm.job_name(group)
    jd = job_definition(cfg, image)
    print(f"{n_runs} runs -> {shards} machine(s) x {lanes} lane(s), {gpl} GPU(s) per lane")
    print(json.dumps(jd, indent=2))
    preview = submit_request(cfg, "<job definition arn>", name, a.sweep, shards, lanes, a.group, a.set, a.only,
                             a.max_hours or float(cfg.get("max_hours", 24)), env, gpl)
    print(json.dumps(redacted(preview, names), indent=2))
    if a.dry_run:
        return 0
    repo, tag = image.rsplit("/", 1)[1].split(":")
    try:
        sm.aws(cfg, "ecr", "describe-images", "--repository-name", repo, "--image-ids", f"imageTag={tag}")
    except RuntimeError as e:
        sys.exit(f"image {image} not found in ECR (build and push it first: docker/build.sh)\n{e}")
    arn = sm.aws(cfg, "batch", "register-job-definition", input_json=jd)["jobDefinitionArn"]
    req = submit_request(cfg, arn, name, a.sweep, shards, lanes, a.group, a.set, a.only,
                         a.max_hours or float(cfg.get("max_hours", 24)), env, gpl)
    r = sm.aws(cfg, "batch", "submit-job", input_json=req)
    prof = f" --profile {cfg['profile']}" if cfg.get("profile") else ""
    print(f"\nsubmitted {name}: jobId {r['jobId']} ({shards} machine(s), job definition {arn.rsplit('/', 1)[1]})\n"
          f"  status: aws batch describe-jobs --jobs {r['jobId']} --region {cfg['region']}{prof}\n"
          f"  output: {cfg['s3_output'].rstrip('/')}/{name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
