# Models

vlmrl's `HFPolicy` works with any checkpoint loadable as `AutoModelForImageTextToText` whose processor
has a chat template. Everything model-specific (image-token expansion, pixel layouts,
`image_grid_thw`, …) stays inside the processor. Each preset in `configs/policy/` sets the model plus
the settings it needs. Pick one with `policy=<name>`.

| model | preset | params | tokens / 336-px frame | status |
|---|---|---|---|---|
| [Qwen3-VL-4B-Instruct](qwen3-vl-4b.md) | `qwen3vl4b_lora` (default) | 4B | 102 | **trained end-to-end**: UMaze 0.625 → 1.0 |
| [Qwen3-VL-2B-Instruct](qwen3-vl-2b.md) | `qwen3vl2b_lora` | 2B | 102 | trains; zero-shot struggles with the PEAR action semantics |
| [Qwen2.5-VL-3B-Instruct](qwen2.5-vl-3b.md) | `qwen25vl3b_lora` | 3B | 146 | processor + template checked; not trained yet |
| [Qwen2.5-VL-7B-Instruct](qwen2.5-vl-7b.md) | `qwen25vl7b_full` | 7B | 146 | full-FT preset for `experiment=pear_recipe`; needs >60 GB |
| [SmolVLM2-2.2B-Instruct](smolvlm2-2.2b.md) | `smolvlm2_2b_lora` | 2.2B | 84 (no tiling) | in the reward × model sweep |
| [SmolVLM-256M-Instruct](smolvlm-256m.md) | `smolvlm256m` | 256M | 67 (no tiling) | tests and CPU smoke runs only |

Token counts are measured with each processor on a 336×336 frame, including the image delimiter
tokens. A PointMaze episode adds one frame per round.

## Things that apply to every model

### Generation config

Checkpoints ship their own sampling defaults, and these differ a lot. PEAR samples at **temperature 1.0
over the full distribution**, so `HFPolicy.generate` always passes `temperature`, `top_p`, `top_k=0` and
`repetition_penalty=1.0`, overriding whatever the checkpoint says:

| checkpoint | shipped `generation_config` |
|---|---|
| Qwen2.5-VL | `temperature=1e-6`, `repetition_penalty=1.05`, which is effectively greedy |
| Qwen3-VL | `temperature=0.7`, `top_p=0.8`, `top_k=20` |
| SmolVLM / SmolVLM2 | none |

Without the override, Qwen2.5-VL would produce identical rollouts within a group, so every group would be
filtered and nothing would train. The vLLM backend sends the same parameters (`top_k=-1`).

### Prefix-stable chat templates

Whole-episode training sequences need the chat template to be **prefix-stable**: rendering
`messages[:k]` must give a token prefix of rendering `messages[:k+1]`. This is checked on the first
training batch, with a clear error if it fails. Templates that rewrite history (e.g. strip earlier
`<think>` blocks) need `max_images=<int>`, which trains each turn as its own exact sample. See
[Algorithm](../concepts/algorithm.md#multi-turn-loss-masks-model-agnostic).

### LoRA targets

`lora_targets` is a PEFT `target_modules` value: `all-linear`, or a regex string. The presets restrict
LoRA to the **language model's** linear layers, because vLLM cannot serve LoRA on the vision tower:

- Qwen family: `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)`
- SmolVLM family: `.*text_model.*\.(…same…)`

`all-linear` (which also adapts the vision tower) works with the HF backend only.

### Image token budget

Every round adds a frame to the context, so tokens per frame drive both memory and speed. Control it
with `processor_kwargs` (e.g. `+processor_kwargs.max_pixels=200704` for Qwen) and
`env_kwargs.image_size`. Idefics3-style processors (SmolVLM) tile images by default: **~1.1–1.4k tokens per
frame**, versus 67–84 with `do_image_splitting: false`, which the presets set.

### Memory

On one 48 GB GPU, use LoRA (all `*_lora` presets) with gradient checkpointing (on by default). Full
fine-tuning a 4B model with AdamW needs about 48 GB before activations. `micro_batch_size` sets the number
of sequences per backward pass. Generation adapts its batch size on CUDA OOM.

## Adding a model

See [Adding a model](adding.md).
