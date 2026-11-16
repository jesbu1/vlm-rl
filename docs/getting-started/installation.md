# Installation

vlmrl uses [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10 (3.11 is tested).

```bash
git clone git@github.com:jesbu1/vlm-rl.git && cd vlm-rl
uv venv -p 3.11 .venv
uv pip install -p .venv -e ".[pointmaze,offline,lora,wandb,dev]"
```

## Extras

| extra | installs | needed for |
|---|---|---|
| `pointmaze` | gymnasium, gymnasium-robotics, mujoco | the PointMaze environment |
| `offline` | minari[hdf5] | D4RL/Minari offline start states |
| `lora` | peft | `lora_rank > 0` (all shipped policy presets except full-FT) |
| `wandb` | wandb | `wandb_project=...` logging, sweep reports |
| `smolvlm` | num2words | [SmolVLM2](../models/smolvlm2-2.2b.md) processors |
| `docs` | mkdocs-material | building this site ([Writing docs](../development/docs.md)) |
| `dev` | pytest | the test suite |

Core dependencies: torch, torchvision (Qwen3-VL processors need it), transformers ≥ 4.57,
hydra-core, accelerate, numpy, pillow, imageio, pyyaml.

## vLLM (optional, for fast rollouts)

vLLM pins its own torch/CUDA stack, so it lives in a **separate venv** and runs as a server. The
trainer talks to it over HTTP (see [Rollout backends](../training/backends.md)).

```bash
uv venv --managed-python -p 3.12 ~/venvs/vllm        # --managed-python: Triton needs Python.h
uv pip install -p ~/venvs/vllm/bin/python vllm
```

!!! warning "Machines without a CUDA toolkit"
    FlashInfer's sampler JIT-compiles with `nvcc`. If `/usr/local/cuda` is missing, the server
    fails at startup with *"Could not find nvcc"*. `scripts/launch_vllm.sh` sets
    `VLLM_USE_FLASHINFER_SAMPLER=0` for that reason.

## Check the install

```bash
make test     # 139 fast tests, CPU only, no downloads, ~10 s
```
