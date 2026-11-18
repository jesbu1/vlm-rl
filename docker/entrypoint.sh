#!/usr/bin/env bash
# Container entrypoint. `train` (what SageMaker passes) runs scripts/sagemaker_entry.py; anything else is
# executed as a command, e.g. `docker run <image> python -m pytest -q`.
set -euo pipefail

# CUDA forward compatibility: the image's CUDA (13.2) is newer than some host drivers. If the host driver is
# older than the compat libcuda shipped in the image, put the compat libs first (works on datacenter GPUs).
if [ -f /usr/local/cuda/compat/libcuda.so.1 ] && [ -r /proc/driver/nvidia/version ]; then
  compat=$(readlink /usr/local/cuda/compat/libcuda.so.1 | sed 's/^libcuda\.so\.//')
  # e.g. "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.126.09  Release Build ..."
  driver=$(head -1 /proc/driver/nvidia/version | grep -oE '[0-9]{3,}\.[0-9]+(\.[0-9]+)?' | head -1 || true)
  if [ -n "$driver" ] && [ "$(printf '%s\n%s\n' "$driver" "$compat" | sort -V | head -1)" = "$driver" ] \
      && [ "$driver" != "$compat" ]; then
    echo "[entrypoint] host driver $driver < $compat: using CUDA forward-compat libs"
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
  fi
fi

cd /opt/vlmrl
if [ "${1:-train}" = "train" ]; then
  exec python scripts/sagemaker_entry.py
fi
exec "$@"
