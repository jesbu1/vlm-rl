"""AWS Batch (ECS) path: sweep sharding, the in-container job config / S3 sync, and the submit requests. No AWS calls."""
import importlib.util
import json
import os
import re

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load("sweep")
entry = _load("job_entry")
batch = _load("batch_submit")

CFG = {"region": "us-east-1", "queue": "gpu-q", "ecr_repository": "me-vlm-rl", "job_role_arn": "arn:job",
       "execution_role_arn": "arn:exec", "s3_output": "s3://b/p/", "gpus": 4, "vcpus": 48, "memory_mib": 262144,
       "tags": {"owner": "me"}, "env": {"WANDB_BASE_URL": "https://x"}, "secret_env": ["WANDB_API_KEY"]}
SWEEP = "configs/sweeps/batch_small_fft.yaml"


def test_select_shards_partition_the_sweep():
    runs = [{"name": f"r{i}"} for i in range(7)]
    parts = [sweep.select(runs, f"{i}/3") for i in range(3)]
    assert sorted(r["name"] for p in parts for r in p) == sorted(r["name"] for r in runs)
    assert [r["name"] for r in parts[1]] == ["r1", "r4"]
    assert [r["name"] for r in sweep.select(runs, "", "r[25]")] == ["r2", "r5"]
    with pytest.raises(AssertionError):
        sweep.select(runs, "3/3")


def test_job_config_from_env_with_array_shard(tmp_path):
    env = {"VLMRL_SWEEP": SWEEP, "VLMRL_LANES": "2", "VLMRL_SHARDS": "4", "VLMRL_OVERRIDES": '["steps=2"]',
           "VLMRL_GROUP": "smoke", "VLMRL_S3_OUTPUT": "s3://b/p/job/", "AWS_BATCH_JOB_ARRAY_INDEX": "3"}
    hp = entry.job_config(str(tmp_path / "none.json"), env)
    assert hp == {"sweep": SWEEP, "lanes": 2, "shards": 4, "overrides": ["steps=2"], "group": "smoke",
                  "s3_output": "s3://b/p/job/", "shard": "3/4"}
    cmd = entry.sweep_command(hp, 4, "/scratch/runs", "/scratch/sync")
    assert cmd[2:] == [SWEEP, "--lanes", "2", "--out_root", "/scratch/runs", "--sync_root", "/scratch/sync",
                       "--group", "smoke", "--set", "steps=2", "--shard", "3/4"]
    # the SageMaker hyperparameters file wins over the environment
    p = tmp_path / "hp.json"
    p.write_text(json.dumps({"sweep": "other.yaml"}))
    assert entry.job_config(str(p), env) == {"sweep": "other.yaml"}
    assert "shard" not in entry.job_config(str(tmp_path / "none.json"), {"VLMRL_SWEEP": SWEEP})


def test_models_downloaded_are_only_this_shards():
    hp = {"sweep": SWEEP, "shards": 4, "shard": "1/4"}
    assert entry.models_in(os.path.join(ROOT, SWEEP), hp) == ["Qwen/Qwen3-VL-2B-Instruct", "openbmb/MiniCPM-V-4.6"]
    assert len(entry.models_in(os.path.join(ROOT, SWEEP), {"sweep": SWEEP})) == 4


def test_s3_syncer_restores_pushes_periodically_and_at_stop(tmp_path):
    calls = []
    s = entry.S3Syncer(str(tmp_path / "runs"), "s3://b/p/job", every=0.05, run=lambda cmd: calls.append(cmd) or 0)
    s.restore()
    assert calls[0][:5] == ["aws", "s3", "sync", "s3://b/p/job/", str(tmp_path / "runs")]
    s.start()
    import time
    time.sleep(0.3)
    s.stop()
    pushes = [c for c in calls[1:] if c[3] == str(tmp_path / "runs")]
    assert len(pushes) >= 2 and all(c[4] == "s3://b/p/job/" and "*.tmp" in c for c in pushes)


def test_job_definition_and_submit_request():
    jd = batch.job_definition(CFG, "acct.dkr.ecr/me-vlm-rl:abc")
    cp = jd["containerProperties"]
    assert jd["jobDefinitionName"] == "me-vlm-rl-gpu4" and cp["command"] == ["train"]
    assert {r["type"]: r["value"] for r in cp["resourceRequirements"]} == {"GPU": "4", "VCPU": "48", "MEMORY": "262144"}
    assert cp["linuxParameters"]["sharedMemorySize"] == 65536 and cp["jobRoleArn"] == "arn:job"
    req = batch.submit_request(CFG, "arn:jd:1", "job-1", SWEEP, 4, 2, "smoke", ["steps=2"], "", 2.0,
                               {"WANDB_API_KEY": "secret"})
    env = {e["name"]: e["value"] for e in req["containerOverrides"]["environment"]}
    assert env["VLMRL_SWEEP"] == SWEEP and env["VLMRL_SHARDS"] == "4" and env["VLMRL_LANES"] == "2"
    assert env["VLMRL_S3_OUTPUT"] == "s3://b/p/job-1/" and json.loads(env["VLMRL_OVERRIDES"]) == ["steps=2"]
    assert env["WANDB_BASE_URL"] == "https://x" and env["WANDB_API_KEY"] == "secret"
    assert req["arrayProperties"] == {"size": 4} and req["timeout"]["attemptDurationSeconds"] == 7200
    assert "arrayProperties" not in batch.submit_request(CFG, "a", "j", SWEEP, 1, 2, "", [], "", 1, {})
    shown = batch.redacted(req, ["WANDB_API_KEY"])
    assert "secret" not in json.dumps(shown) and env["WANDB_API_KEY"] == "secret"


def test_count_runs_and_default_shards():
    assert batch.count_runs(os.path.join(ROOT, SWEEP), [], "", "") == 8
    assert batch.count_runs(os.path.join(ROOT, SWEEP), [], "umaze", "") == 4


def test_example_batch_config_complete_and_generic():
    path = os.path.join(ROOT, "configs", "sagemaker", "example_batch.yaml")
    cfg = yaml.safe_load(open(path))
    assert all(cfg.get(k) for k in batch.REQUIRED)
    assert not re.search(r"\b\d{12}\b", open(path).read())


def test_shipped_small_fft_sweep():
    runs = sweep.expand(yaml.safe_load(open(os.path.join(ROOT, SWEEP))))
    assert len(runs) == 8 and all(r["cfg"].lora_rank == 0 and r["cfg"].fp32_master for r in runs)
    assert {r["cfg"].model for r in runs} == {"Qwen/Qwen3-VL-2B-Instruct", "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                                              "openbmb/MiniCPM-V-4.6", "HuggingFaceTB/SmolVLM-256M-Instruct"}
    assert all(r["cfg"].env_kwargs["prompt_style"] == "simple" and r["cfg"].rollout_backend == "vllm" for r in runs)
