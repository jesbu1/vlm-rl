"""Submit a sweep as one SageMaker training job through an AWS Batch service-job queue.

    python scripts/sagemaker_submit.py configs/sweeps/sagemaker_gemma4_fft.yaml            # submit
    python scripts/sagemaker_submit.py configs/sweeps/sagemaker_gemma4_fft.yaml --dry_run  # print the request
    python scripts/sagemaker_submit.py configs/sweeps/sagemaker_gemma4_fft.yaml --set steps=2 --group smoke --max_hours 1
    python scripts/sagemaker_submit.py --image                                             # image URI for docker/build.sh

Account-specific settings (account, queue, role, bucket, cost tags) come from a *local* YAML file that is
git-ignored: copy configs/sagemaker/example.yaml to configs/sagemaker/local.yaml and fill it in.
Secrets (e.g. WANDB_API_KEY) are read from the submitting shell's environment at submit time and passed to
the job as environment variables. They are never printed (--dry_run redacts them) or kept: the request reaches
the AWS CLI as a 0600 file in a private temp dir that is deleted right after the call.

The job runs the image's entrypoint (docker/entrypoint.sh -> scripts/sagemaker_entry.py), which runs
scripts/sweep.py with parallel lanes on the node's GPUs. Run dirs live in /opt/ml/checkpoints, which
SageMaker syncs to <s3_output>/<job>/checkpoints and restores on a retry, so retried jobs resume.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "sagemaker", "local.yaml")
REQUIRED = ("region", "queue", "ecr_repository", "role_arn", "s3_output", "instance_type")


def aws(cfg: dict, *args: str, input_json: dict | None = None) -> dict:
    """Call the AWS CLI (uses its credential cache, e.g. SSO). ``input_json`` (which may hold secrets) is
    passed as a 0600 file in a private temp dir that is deleted as soon as the call returns (never argv)."""
    import tempfile
    cmd = ["aws", *args, "--region", cfg["region"], "--output", "json"]
    if cfg.get("profile"):
        cmd += ["--profile", cfg["profile"]]
    with tempfile.TemporaryDirectory() as d:          # created 0700
        if input_json is not None:
            path = os.path.join(d, "request.json")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(input_json, f)
            cmd += ["--cli-input-json", f"file://{path}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:2])} failed: {r.stderr.strip()}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def load_config(path: str) -> dict:
    import yaml
    if not os.path.exists(path):
        sys.exit(f"{path} not found: copy configs/sagemaker/example.yaml to configs/sagemaker/local.yaml "
                 "and fill in your account's values")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        sys.exit(f"{path}: missing {', '.join(missing)}")
    return cfg


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                          cwd=ROOT, check=True).stdout.strip()


def image_uri(cfg: dict, account: str, tag: str) -> str:
    return f"{account}.dkr.ecr.{cfg['region']}.amazonaws.com/{cfg['ecr_repository']}:{tag}"


def job_name(group: str, now: float | None = None) -> str:
    """SageMaker job names: [a-zA-Z0-9-], <= 63 chars."""
    stamp = time.strftime("%m%d-%H%M%S", time.gmtime(now))
    base = re.sub(r"[^a-zA-Z0-9]+", "-", f"vlmrl-{group}").strip("-")
    return f"{base[:63 - len(stamp) - 1].rstrip('-')}-{stamp}"


def secret_env(names: list[str], environ=os.environ, allow_missing: bool = False) -> dict:
    out, missing = {}, []
    for n in names:
        if environ.get(n):
            out[n] = environ[n]
        else:
            missing.append(n)
    if missing and not allow_missing:
        sys.exit(f"not set in this shell: {', '.join(missing)} (export them, or pass --allow_missing_secrets; "
                 "without WANDB_API_KEY the job logs wandb offline)")
    return out


def build_request(cfg: dict, image: str, name: str, sweep: str, lanes: int | None, group: str,
                  overrides: list[str], max_hours: float, env: dict) -> dict:
    """The CreateTrainingJob request that Batch submits for us (its `serviceRequestPayload`)."""
    out = cfg["s3_output"].rstrip("/")
    hp = {"sweep": sweep}
    if lanes:
        hp["lanes"] = str(lanes)
    if group:
        hp["group"] = group
    if overrides:
        hp["overrides"] = json.dumps(overrides)
    # no "Tags" here: Batch rejects them in the payload; tags go on the Batch job and propagate to the training job
    return {
        "TrainingJobName": name,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "RoleArn": cfg["role_arn"],
        "OutputDataConfig": {"S3OutputPath": f"{out}/output"},
        "ResourceConfig": {"InstanceType": cfg["instance_type"], "InstanceCount": 1,
                           "VolumeSizeInGB": int(cfg.get("volume_gb", 200))},
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_hours * 3600)},
        "CheckpointConfig": {"S3Uri": f"{out}/{name}/checkpoints", "LocalPath": "/opt/ml/checkpoints"},
        "HyperParameters": hp,
        "Environment": {**{k: str(v) for k, v in (cfg.get("env") or {}).items()}, **env},
    }


def redacted(request: dict, secret_names) -> dict:
    r = json.loads(json.dumps(request))
    for k in r.get("Environment", {}):
        if k in secret_names:
            r["Environment"][k] = "<redacted>"
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep", nargs="?", help="sweep file, relative to the repo root")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="local (git-ignored) account settings")
    ap.add_argument("--tag", default="", help="image tag (default: the current git commit)")
    ap.add_argument("--lanes", type=int, default=None, help="parallel lanes (default: #GPUs // 2)")
    ap.add_argument("--group", default="", help="override the sweep's group (wandb group + run dirs)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Hydra override for every run")
    ap.add_argument("--max_hours", type=float, default=None, help="job time limit (default: config max_hours)")
    ap.add_argument("--name", default="", help="job name (default: vlmrl-<group>-<timestamp>)")
    ap.add_argument("--allow_missing_secrets", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--image", action="store_true", help="print the image URI for this commit and exit")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    account = aws(cfg, "sts", "get-caller-identity")["Account"]
    image = image_uri(cfg, account, a.tag or git_commit())
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
    names = list(cfg.get("secret_env") or [])
    env = secret_env(names, allow_missing=a.allow_missing_secrets or a.dry_run)
    name = a.name or job_name(group)
    req = build_request(cfg, image, name, a.sweep, a.lanes, a.group, a.set,
                        a.max_hours or float(cfg.get("max_hours", 8)), env)
    tags = {k: str(v) for k, v in (cfg.get("tags") or {}).items()}
    print(json.dumps(redacted(req, names), indent=2))
    if a.dry_run:
        return 0
    repo, tag = image.rsplit("/", 1)[1].split(":")
    try:
        aws(cfg, "ecr", "describe-images", "--repository-name", repo, "--image-ids", f"imageTag={tag}")
    except RuntimeError as e:
        sys.exit(f"image {image} not found in ECR (build and push it first: docker/build.sh)\n{e}")
    job = {"jobName": name, "jobQueue": cfg["queue"], "serviceJobType": "SAGEMAKER_TRAINING",
           "serviceRequestPayload": json.dumps(req), "shareIdentifier": cfg.get("share_identifier", "default"),
           "schedulingPriority": int(cfg.get("scheduling_priority", 100)),   # required by fair-share queues
           "retryStrategy": {"attempts": int(cfg.get("retry_attempts", 1))},  # retries resume from checkpoints
           "tags": tags}
    r = aws(cfg, "batch", "submit-service-job", input_json=job)
    print(f"\nsubmitted {name}: jobId {r['jobId']}\n"
          f"  status:      aws batch describe-service-job --job-id {r['jobId']} --profile {cfg.get('profile')} "
          f"--region {cfg['region']}\n"
          f"  checkpoints: {req['CheckpointConfig']['S3Uri']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
