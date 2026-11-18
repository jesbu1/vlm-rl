"""SageMaker glue: request building / secrets handling (submit side) and hyperparameter parsing (container side).
No AWS calls."""
import importlib.util
import json
import os
import re
import subprocess

import shutil

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


submit = _load("sagemaker_submit")
entry = _load("sagemaker_entry")

CFG = {"region": "us-west-2", "queue": "q", "ecr_repository": "me-vlm-rl", "role_arn": "arn:aws:iam::1:role/r",
       "s3_output": "s3://b/p/", "instance_type": "ml.p5.48xlarge", "volume_gb": 100,
       "tags": {"owner": "me"}, "env": {"WANDB_BASE_URL": "https://x"}, "secret_env": ["WANDB_API_KEY"]}


def test_job_name_is_valid_sagemaker_name():
    n = submit.job_name("pointmaze-gemma4-fft-v1_with/odd chars and a very long suffix " * 3, now=0)
    assert len(n) <= 63 and re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9])*", n) and n.endswith("0101-000000")


def test_build_request():
    req = submit.build_request(CFG, "img:tag", "job-1", "configs/sweeps/x.yaml", 4, "smoke", ["steps=2"], 1.5,
                               {"WANDB_API_KEY": "secret"})
    assert req["TrainingJobName"] == "job-1" and req["AlgorithmSpecification"]["TrainingImage"] == "img:tag"
    assert req["CheckpointConfig"] == {"S3Uri": "s3://b/p/job-1/checkpoints", "LocalPath": "/opt/ml/checkpoints"}
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 5400
    hp = req["HyperParameters"]
    assert all(isinstance(v, str) for v in hp.values())   # SageMaker requires string values
    assert hp == {"sweep": "configs/sweeps/x.yaml", "lanes": "4", "group": "smoke", "overrides": '["steps=2"]'}
    assert req["Environment"] == {"WANDB_BASE_URL": "https://x", "WANDB_API_KEY": "secret"}
    assert "Tags" not in req   # Batch rejects payload tags; they are set on the Batch job and propagate
    shown = submit.redacted(req, ["WANDB_API_KEY"])
    assert shown["Environment"]["WANDB_API_KEY"] == "<redacted>" and "secret" not in json.dumps(shown)
    assert req["Environment"]["WANDB_API_KEY"] == "secret"  # redaction copies


def test_secret_env_requires_set_variables():
    assert submit.secret_env(["A"], environ={"A": "1"}) == {"A": "1"}
    with pytest.raises(SystemExit):
        submit.secret_env(["A", "B"], environ={"A": "1"})
    assert submit.secret_env(["B"], environ={}, allow_missing=True) == {}


def test_entry_roundtrips_submitted_hyperparameters(tmp_path):
    req = submit.build_request(CFG, "i", "j", "configs/sweeps/sagemaker_gemma4_fft.yaml", 2, "g",
                               ["steps=2", "eval_every=1"], 1, {})
    p = tmp_path / "hp.json"
    p.write_text(json.dumps(req["HyperParameters"]))
    hp = entry.read_hyperparameters(str(p))
    assert hp == {"sweep": "configs/sweeps/sagemaker_gemma4_fft.yaml", "lanes": 2, "group": "g",
                  "overrides": ["steps=2", "eval_every=1"]}
    cmd = entry.sweep_command(hp, 8, "/opt/ml/checkpoints", "/scratch/sync")
    assert cmd[2:] == ["configs/sweeps/sagemaker_gemma4_fft.yaml", "--lanes", "2", "--out_root", "/opt/ml/checkpoints",
                       "--sync_root", "/scratch/sync", "--group", "g", "--set", "steps=2", "--set", "eval_every=1"]
    assert entry.sweep_command({"sweep": "s.yaml"}, 8, "o", "s")[4] == "4"   # default: #GPUs // 2
    assert entry.read_hyperparameters(str(tmp_path / "missing.json")) == {}


def test_entry_models_and_scratch(tmp_path):
    hp = {"sweep": "configs/sweeps/sagemaker_gemma4_fft.yaml", "overrides": ["steps=1"]}
    assert entry.models_in(os.path.join(ROOT, hp["sweep"]), hp) == ["google/gemma-4-E2B-it"]
    (tmp_path / "file").write_text("x")   # a path under a regular file cannot be created -> skipped
    assert entry.pick_scratch((str(tmp_path / "file" / "sub"), str(tmp_path / "a"))) == str(tmp_path / "a")


def test_example_config_has_every_required_key_and_no_account_values():
    import yaml
    path = os.path.join(ROOT, "configs", "sagemaker", "example.yaml")
    cfg = yaml.safe_load(open(path))
    assert all(cfg.get(k) for k in submit.REQUIRED) and cfg["secret_env"] == ["WANDB_API_KEY"]
    assert not re.search(r"\b\d{12}\b", open(path).read())   # no AWS account ids


def test_local_sagemaker_config_is_ignored_by_git_and_docker():
    if not shutil.which("git") or subprocess.run(["git", "-C", ROOT, "rev-parse"], capture_output=True).returncode:
        pytest.skip("not a git checkout")
    r = subprocess.run(["git", "-C", ROOT, "check-ignore", "-q", "configs/sagemaker/local.yaml"])
    assert r.returncode == 0
    assert "configs/sagemaker/local*.yaml" in open(os.path.join(ROOT, ".dockerignore")).read()


def test_no_aws_account_ids_in_tracked_files():
    """The repo is public: account ids belong in configs/sagemaker/local.yaml (git-ignored)."""
    if not shutil.which("git"):
        pytest.skip("git not installed")
    r = subprocess.run(["git", "-C", ROOT, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    pat = re.compile(r"(\d{12}\.dkr\.ecr|arn:aws:iam::\d{12}|arn:aws:batch:[a-z0-9-]+:\d{12})")
    hits = []
    for f in r.stdout.split():
        if f.endswith((".png", ".jpg", ".gif", ".mp4", ".lock")) or not os.path.isfile(os.path.join(ROOT, f)):
            continue
        try:
            text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        hits += [f for _ in pat.finditer(text)]
    assert not hits, f"AWS account-specific identifiers in tracked files: {sorted(set(hits))}"


def test_per_queue_overrides():
    cfg = {**CFG, "queue": "od", "env": {"A": "1"},
           "queues": {"od": {"env": {"R": "1"}}, "spot": {"instance_type": "ml.x", "retry_attempts": 3}}}
    od = submit.for_queue(cfg)
    assert od["queue"] == "od" and od["env"] == {"A": "1", "R": "1"} and "queues" not in od
    sp = submit.for_queue(cfg, "spot")
    assert sp["queue"] == "spot" and sp["env"] == {"A": "1"} and sp["instance_type"] == "ml.x" and sp["retry_attempts"] == 3
    assert submit.for_queue(cfg, "other")["instance_type"] == CFG["instance_type"]
