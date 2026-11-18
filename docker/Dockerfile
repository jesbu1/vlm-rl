# Training image: the repo + two Python environments (trainer, vLLM server), as on the dev box.
#   docker/build.sh <image-uri>     (or: docker build -f docker/Dockerfile -t vlmrl .)
# Used by SageMaker and AWS Batch jobs (docker/entrypoint.sh -> scripts/job_entry.py), and runs anywhere with
#   docker run --gpus all <image> python scripts/sweep.py configs/sweeps/<sweep>.yaml --lanes 4
# The package pins in docker/requirements-*.txt are `uv pip freeze` of the verified environments.
FROM nvidia/cuda:13.2.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1
# gcc + libc headers: Triton compiles small launcher stubs at runtime (vLLM uses torch.compile / Inductor)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgl1 libglib2.0-0 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# vLLM server environment (its own torch/CUDA pins). Managed Python ships the headers vLLM's JIT needs.
COPY docker/requirements-vllm.txt /tmp/requirements-vllm.txt
RUN uv venv --managed-python --python 3.12 /opt/venv-vllm \
    && uv pip install --python /opt/venv-vllm/bin/python -r /tmp/requirements-vllm.txt

# AWS CLI (S3 sync of run dirs for AWS Batch jobs), in its own env so it can't disturb the pinned ones.
RUN uv venv --managed-python --python 3.12 /opt/venv-aws \
    && uv pip install --python /opt/venv-aws/bin/python "awscli>=1.36,<2" \
    && ln -s /opt/venv-aws/bin/aws /usr/local/bin/aws

# Trainer environment.
COPY docker/requirements-train.txt /tmp/requirements-train.txt
RUN uv venv --managed-python --python 3.11 /opt/venv-train \
    && uv pip install --python /opt/venv-train/bin/python -r /tmp/requirements-train.txt

# The repo (tracked files only; see .dockerignore), installed without re-resolving dependencies.
WORKDIR /opt/vlmrl
COPY . /opt/vlmrl
RUN uv pip install --python /opt/venv-train/bin/python --no-deps -e . \
    && chmod +x docker/entrypoint.sh scripts/*.sh

ARG GIT_COMMIT=unknown
ENV VLMRL_GIT_COMMIT=$GIT_COMMIT \
    PATH=/opt/venv-train/bin:$PATH \
    VLLM_BIN=/opt/venv-vllm/bin/vllm \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
LABEL org.opencontainers.image.source="https://github.com/jesbu1/vlm-rl" \
      org.opencontainers.image.revision=$GIT_COMMIT

# SageMaker starts the container as `docker run <image> train`.
ENTRYPOINT ["/opt/vlmrl/docker/entrypoint.sh"]
