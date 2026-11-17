# vlmrl

**Minimal multi-turn VLM reinforcement learning, PEAR-style, for any Hugging Face VLM, any
environment and any reward.**

vlmrl is a small (~1.8k-line), readable reproduction of the training recipe from
[PEAR: *Training Reasoning Models in Simulation to Natively Use Real Robots*](https://arxiv.org/abs/2611.18562)
(MIT HAN Lab). A vision-language model is the policy. Each round it sees the state and camera
images, reasons in text, and writes a JSON action. It is trained with GRPO from task outcomes alone.

<div class="grid cards" markdown>

- **New here?** Start with the [Quickstart](getting-started/quickstart.md), then
  [your first real run](getting-started/first-run.md).
- **Reading the code?** The [Code tour](getting-started/code-tour.md) walks one training step through
  the files and functions, in order.
- **Want to understand PEAR?** Read [the paper, explained](pear/index.md): the
  [protocol](pear/protocol.md), the [objective](pear/objective.md), the [recipe](pear/recipe.md).
- **Bringing your own world model?** See [World-action models](environments/world-models.md) and
  [Rewards & preferences](concepts/rewards.md).
- **Picking a VLM?** Every supported model has [its own page](models/index.md).

</div>

## What it gives you

| | |
|---|---|
| **Any HF VLM** | Anything loadable with `AutoModelForImageTextToText` whose processor has a chat template. Presets for [Qwen3-VL](models/qwen3-vl-4b.md), [Qwen2.5-VL](models/qwen2.5-vl-3b.md) and [SmolVLM2](models/smolvlm2-2.2b.md). |
| **Any environment** | Three swappable pieces (dynamics, text protocol, start states), so a MuJoCo simulator and a learned world-action model are interchangeable. See [Architecture](concepts/architecture.md). |
| **Any reward** | Success, shaped scalars, or pairwise **preferences**, all as one `groups -> scores` function. See [Rewards](concepts/rewards.md). |
| **PointMaze for VLM-RL** | MuJoCo dynamics matching the D4RL/Minari datasets, PEAR's prompt protocol, and exact reset-to-state from offline data. See [PointMaze](environments/pointmaze.md). |
| **Two rollout backends** | In-process HF `generate`, or a vLLM server with prefix caching and LoRA hot-swap (≈9× faster evals). See [Backends](training/backends.md). |
| **Hydra configs, sweeps, wandb** | A type-checked config schema, a reward × model sweep runner, wandb reports. See [Training](training/index.md). |

## How the code is organized

One training step, with the file and function that does each part:

```mermaid
flowchart LR
  subgraph COLLECT["trainer.py: Trainer.collect"]
    direction TB
    A["Sample start states<br/>envs/pointmaze.py"] --> B["Run episodes: VLM replies,<br/>parse action, step env<br/>rollout.py: rollout"]
    B --> C["Score each group<br/>rewards.py"]
    C --> D["GRPO advantages + group filter<br/>algo.py: group_advantages, keep_group"]
  end
  subgraph UPDATE["trainer.py: Trainer.update"]
    direction TB
    E["Tokenize + loss mask<br/>policy.py: HFPolicy.build_batch"] --> F["Log-probs of generated tokens<br/>policy.py: HFPolicy.token_logprobs"]
    F --> G["Loss -A·logπ, backward, AdamW<br/>algo.py: policy_loss_sum"]
  end
  COLLECT --> UPDATE
  UPDATE -->|"new weights (vllm_client.py: sync)"| COLLECT
```

| where is… | file | look for |
|---|---|---|
| the training loop | [`vlmrl/trainer.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/trainer.py) | `Trainer.train`, `collect`, `update` |
| **the GRPO algorithm** (advantages, filter, loss, score centering) | [`vlmrl/algo.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/algo.py) | `group_advantages`, `keep_group`, `policy_loss_sum`, `score_centering_term` |
| multi-turn episodes | [`vlmrl/rollout.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/rollout.py) | `rollout`, `to_samples` |
| the VLM (sampling, log-probs, LoRA) | [`vlmrl/policy.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/policy.py) | `HFPolicy` |
| rewards and preferences | [`vlmrl/rewards.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/rewards.py) | `build_reward`, `preference_reward` |
| the prompt and action parser | [`vlmrl/envs/protocol.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/envs/protocol.py), [`envs/pointmaze.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/envs/pointmaze.py) | `system_prompt`, `parse_action`, `PointMazeTextInterface` |
| the environment | [`vlmrl/envs/pointmaze.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/envs/pointmaze.py) | `PointMazeSimVecEnv` |
| all settings | [`vlmrl/trainer.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/trainer.py) + [`configs/`](https://github.com/jesbu1/vlm-rl/tree/main/configs) | `TrainConfig` ([reference](reference/config.md)) |

Full walkthrough: [Code tour](getting-started/code-tour.md).

## Status

!!! success "Verified"
    On 2×A6000 with Qwen3-VL-4B + LoRA (HF and vLLM backends, wandb logging), PointMaze UMaze eval
    success went **0.625 → 1.0 within 20 RL steps**. See the [experiment log](training/experiments.md).
    165 tests (159 fast CPU + 6 real-VLM) pass locally and on the GPU machine. See [Testing](development/testing.md).

```mermaid
flowchart LR
  O["Observe<br/>state JSON + image"] --> R["Reason<br/>chain of thought"]
  R --> A["Act<br/>JSON change of each quantity"]
  A --> X["Execute<br/>controller until reached and at rest"]
  X -->|"next round, earlier rounds stay in context"| O
```
