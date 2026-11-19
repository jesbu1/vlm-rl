# vlmrl

**Minimal multi-turn VLM reinforcement learning, PEAR-style, for any Hugging Face VLM, any
environment and any reward.**

📖 **Documentation: <https://www.jessezhang.net/vlm-rl/>** (source in [`docs/`](docs/index.md))

vlmrl is a small (~1.8k-line), readable reproduction of the training recipe from
[PEAR: *Training Reasoning Models in Simulation to Natively Use Real Robots*](https://arxiv.org/abs/2611.18562).
A vision-language model is the policy. Each round it sees the state and camera images, reasons in
text, and writes a JSON action. It is trained with GRPO from task outcomes alone, with no harness and
no demonstrations.

- **Any HF VLM**: presets for Qwen3-VL, Qwen2.5-VL, SmolVLM2 ([models](docs/models/index.md))
- **Any environment**: dynamics, text protocol and start states are separate, so a simulator and a
  learned **world-action model** are interchangeable ([world models](docs/environments/world-models.md))
- **Any reward**: success, shaped, or pairwise **preferences** ([rewards](docs/concepts/rewards.md))
- **PointMaze for VLM-RL** with PEAR's exact prompt protocol and reset-to-state from D4RL/Minari data
  ([PointMaze](docs/environments/pointmaze.md), [prompts](docs/environments/prompts.md))
- **Metaworld** ML1 / ML10 / ML45 with Metaworld's train/test task splits, the same hand + gripper text protocol, and
  scripted experts as an upper bound ([Metaworld](docs/environments/metaworld.md))
- **Faithful PEAR algorithm**: GRPO z-score advantages, DAPO group filter, REINFORCE without clipping,
  score centering ([how PEAR works](docs/pear/index.md), [mapping](docs/pear/mapping.md))
- **Hydra configs, a vLLM rollout server (LoRA or full-weight sync), sweeps, wandb** ([training](docs/training/index.md))
- **Docker image + cloud jobs**: SageMaker via AWS Batch queues, or parallel GPU machines on AWS Batch; multi-run nodes, resumable ([SageMaker](docs/training/sagemaker.md), [AWS Batch](docs/training/batch.md))

## Quickstart

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[pointmaze,offline,lora,wandb,dev]"
make test                                     # fast tests, CPU, ~10 s
make demo                                     # BFS oracle through the text protocol -> outputs/demo/
make smoke                                    # one RL step with a tiny VLM on CPU

# real run: vLLM server on GPU 1, trainer on GPU 0
scripts/launch_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8011 1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py experiment=pointmaze_q3vl4b_vllm wandb_project=vlm-rl-pointmaze
```

On 2×A6000, Qwen3-VL-4B + LoRA reaches PointMaze UMaze eval success **0.625 → 1.0 within 20 RL steps**
([experiment log](docs/training/experiments.md)).

## Docker

One image holds the repo and both Python environments (trainer + vLLM server), pinned to the versions verified
on the dev box (`docker/requirements-*.txt`). `docker/build.sh` builds from `git archive HEAD`, so only committed,
tracked files go in. Uncommitted edits, local configs and keys never do.

```bash
PUSH=0 docker/build.sh vlmrl:dev                       # build only (x86-64 Linux + Docker; no GPU needed)
docker run --rm vlmrl:dev python -m pytest -q          # the test suite inside the image
docker run --rm --gpus all vlmrl:dev python scripts/sweep.py configs/sweeps/sagemaker_gemma4_fft.yaml --lanes 4
```

Inside the image, the trainer env is on `PATH` (`/opt/venv-train`) and the server is `$VLLM_BIN`
(`/opt/venv-vllm`). If the host's NVIDIA driver is older than the image's CUDA 13.2, `docker/entrypoint.sh`
switches to CUDA forward-compatibility libraries automatically.

## SageMaker

A [sweep](docs/training/sweeps.md) runs as **one SageMaker training job on one 8-GPU node**, submitted through
an **AWS Batch service-job queue**. On the node, `scripts/sweep.py --lanes 4` trains 4 runs side by side, each
with one training GPU and one vLLM GPU. Run dirs are synced to S3 continuously, and retried or preempted jobs
resume. Full guide: [docs/training/sagemaker.md](docs/training/sagemaker.md).

```bash
# 1. once: account settings (git-ignored; never commit account ids, roles, buckets or keys) + an ECR repository
cp configs/sagemaker/example.yaml configs/sagemaker/local.yaml        # then fill it in
aws ecr create-repository --repository-name <ecr_repository> --profile <profile> --region <region>

# 2. per commit: build and push the image (tag = git commit). Log in from the machine with AWS credentials
IMAGE=$(python scripts/sagemaker_submit.py --image)
aws ecr get-login-password --profile <profile> --region <region> | docker login --username AWS --password-stdin ${IMAGE%%/*}
docker/build.sh $IMAGE && docker logout ${IMAGE%%/*}

# 3. submit (secrets listed in secret_env are read from your shell at submit time)
export WANDB_API_KEY=...
python scripts/sagemaker_submit.py configs/sweeps/sagemaker_gemma4_fft.yaml --dry_run          # print the request
python scripts/sagemaker_submit.py configs/sweeps/sagemaker_gemma4_fft.yaml \
    --group gemma4-fft-smoke --set steps=2 --max_hours 1                                          # smoke test first
python scripts/sagemaker_submit.py configs/sweeps/sagemaker_gemma4_fft.yaml [--queue <other queue>]

# 4. monitor
aws batch describe-service-job --job-id <jobId> --query '{status:status,reason:statusReason}'
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix AWSBatch<job-name> --follow
aws s3 ls <s3_output>/<job-name>/checkpoints/<group>/
```

**AWS Batch GPU queues** (EC2 instances, no SageMaker): `scripts/batch_submit.py` submits a sweep as an
**array job, one machine per shard, all in parallel**, with run dirs synced to S3
([docs/training/batch.md](docs/training/batch.md)). Same image; settings in git-ignored
`configs/sagemaker/local_batch.yaml` (template `example_batch.yaml`):

```bash
IMAGE=$(python scripts/batch_submit.py --image)       # push the image to that account's ECR as above
python scripts/batch_submit.py configs/sweeps/batch_small_fft.yaml --dry_run
python scripts/batch_submit.py configs/sweeps/batch_small_fft.yaml        # 8 small-model full-FT runs, 4 machines
```

The shipped sweep, `configs/sweeps/sagemaker_gemma4_fft.yaml`, is **Gemma 4 E2B full fine-tuning**. It keeps fp32
master weights and syncs the full weights into vLLM each step, over a maze × learning-rate grid.

## Documentation map

| section | pages |
|---|---|
| [Getting started](docs/getting-started/index.md) | installation · quickstart · your first real run |
| [PEAR (the paper)](docs/pear/index.md) | protocol · RL objective · recipe & systems · results · PEAR ↔ vlmrl |
| [Concepts](docs/concepts/index.md) | architecture · rollouts & episodes · algorithm · rewards & preferences |
| [Environments](docs/environments/index.md) | PointMaze · Metaworld · prompts · world-action models · adding an environment |
| [Models](docs/models/index.md) | Gemma 4 E2B · Qwen3-VL-4B · Qwen3-VL-2B · Qwen2.5-VL-3B · Qwen2.5-VL-7B · MiniCPM-V 4.6 · SmolVLM2-2.2B · SmolVLM-256M · adding a model |
| [Training](docs/training/index.md) | configuration (Hydra) · rollout backends · sweeps · wandb · SageMaker + Docker · AWS Batch · experiment log |
| [Development](docs/development/index.md) | testing · performance · writing docs · roadmap |
| [Reference](docs/reference/index.md) | configuration fields · code layout |

Build the site locally with `make docs-serve` (needs the `docs` extra).

## Citation

```bibtex
@article{tang2026pear,
  title   = {Training Reasoning Models in Simulation to Natively Use Real Robots},
  author  = {Tang, Jiaming and Zhang, Zhuoyang and Guo, Junxian and Zhao, Fangzhou and Zhao, Yilong and
             Li, Zekai and Li, Xingyang and Zhao, Yuyang and Bian, Song and Liu, Zhijian and Han, Song},
  journal = {arXiv preprint arXiv:2611.18562},
  year    = {2026}
}
```
