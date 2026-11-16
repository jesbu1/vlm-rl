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
- **Faithful PEAR algorithm**: GRPO z-score advantages, DAPO group filter, REINFORCE without clipping,
  score centering ([how PEAR works](docs/pear/index.md), [mapping](docs/pear/mapping.md))
- **Hydra configs, a vLLM rollout server, sweeps, wandb** ([training](docs/training/index.md))

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

## Documentation map

| section | pages |
|---|---|
| [Getting started](docs/getting-started/index.md) | installation · quickstart · your first real run |
| [PEAR (the paper)](docs/pear/index.md) | protocol · RL objective · recipe & systems · results · PEAR ↔ vlmrl |
| [Concepts](docs/concepts/index.md) | architecture · rollouts & episodes · algorithm · rewards & preferences |
| [Environments](docs/environments/index.md) | PointMaze · prompts · world-action models · adding an environment |
| [Models](docs/models/index.md) | Gemma 4 E2B · Qwen3-VL-4B · Qwen3-VL-2B · Qwen2.5-VL-3B · Qwen2.5-VL-7B · MiniCPM-V 4.6 · SmolVLM2-2.2B · SmolVLM-256M · adding a model |
| [Training](docs/training/index.md) | configuration (Hydra) · rollout backends · sweeps · wandb · experiment log |
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
