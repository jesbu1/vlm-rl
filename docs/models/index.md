# Models

vlmrl's `HFPolicy` works with any checkpoint loadable as `AutoModelForImageTextToText` whose processor
has a chat template. Everything model-specific (image-token expansion, pixel layouts,
`image_grid_thw`, …) stays inside the processor. Each preset in `configs/policy/` sets the model plus
the settings it needs. Pick one with `policy=<name>`.

| model | preset | params | tokens / 336-px frame | status |
|---|---|---|---|---|
| [Gemma 4 E2B-it](gemma-4-e2b.md) | `gemma4_e2b_lora`, `gemma4_e2b_full` | 5.1B (≈2B effective) | 258 | smoke-tested; in the v4 sweep |
| [Qwen3-VL-4B-Instruct](qwen3-vl-4b.md) | `qwen3vl4b_lora` (default) | 4B | 102 | **trained end-to-end**: UMaze 0.625 → 1.0 |
| [Qwen3.5-4B](qwen3.5-4b.md) | `qwen35_4b_lora` | 4.5B | 64 (patch 16, merge 2 at 224 px) | template + masks checked (thinking on and off); Metaworld ML1 runs |
| [Qwen3-VL-2B-Instruct](qwen3-vl-2b.md) | `qwen3vl2b_lora`, `qwen3vl2b_full` | 2B | 102 | trains; zero-shot struggles with the PEAR action semantics |
| [Qwen2.5-VL-3B-Instruct](qwen2.5-vl-3b.md) | `qwen25vl3b_lora` | 3B | 146 | processor + template checked; not trained yet |
| [Qwen2.5-VL-7B-Instruct](qwen2.5-vl-7b.md) | `qwen25vl7b_full` | 7B | 146 | full-FT preset for `experiment=pear_recipe`; needs >60 GB |
| [MiniCPM-V 4.6](minicpm-v-4.6.md) | `minicpmv46_lora`, `minicpmv46_full` | 1.3B | 70 | smoke-tested; template needs per-turn samples; in the v4 sweep |
| [SmolVLM2-2.2B-Instruct](smolvlm2-2.2b.md) | `smolvlm2_2b_lora`, `smolvlm2_2b_full` | 2.2B | 84 (no tiling) | preset + vLLM flags checked; was in the v1–v3 sweep grids |
| [SmolVLM-256M-Instruct](smolvlm-256m.md) | `smolvlm256m` (full FT), `smolvlm256m_lora` | 256M | 67 (no tiling) | tests, CPU smoke runs; LoRA and full FT in the v4 sweep |

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
| Gemma 4 | `temperature=1.0`, `top_k=64`, `top_p=0.95` |
| MiniCPM-V 4.6 | `temperature=0.7`, `top_k=0`, `top_p=1.0` |
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

- Qwen family, Gemma 4, MiniCPM-V 4.6: `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)`
- SmolVLM family: `.*text_model.*\.(…same…)`

`all-linear` (which also adapts the vision tower) works with the HF backend only.

### Image token budget

Every round adds a frame to the context, so tokens per frame drive both memory and speed. Control it
with `processor_kwargs` (e.g. `+processor_kwargs.max_pixels=200704` for Qwen) and
`env_kwargs.image_size`. Idefics3-style processors (SmolVLM) tile images by default: **~1.1–1.4k tokens per
frame**, versus 67–84 with `do_image_splitting: false`, which the presets set.

### Memory

LoRA (all `*_lora` presets) with gradient checkpointing (on by default) fits every preset on one 48 GB GPU.
`micro_batch_size` sets the number of sequences per backward pass. Generation adapts its batch size on CUDA OOM.

**Full fine-tuning** (`*_full` presets, `lora_rank: 0`) keeps **fp32 master weights** for the trainable
parameters and runs the forward pass under bf16 autocast (`fp32_master: true`, the default). With AdamW that
is about **16 bytes per trainable parameter**: weight 4, gradient 4, two moments 8. Frozen parameters
(`freeze`, a regex) stay bf16 at 2 bytes.

| preset | trainable | ≈ memory before activations | fits |
|---|---|---|---|
| `smolvlm256m` | 0.26B (all) | 4 GB | any GPU |
| `minicpmv46_full` | 1.3B (all) | 21 GB | 24 GB+ |
| `qwen3vl2b_full` | 2.1B (all) | 34 GB | 48 GB (L40S, A6000) |
| `smolvlm2_2b_full` | 2.25B (all) | 36 GB | 48 GB |
| `gemma4_e2b_full` | 2.1B of 5.1B (embeddings + audio tower frozen) | 40 GB | 80 GB (ran out of memory on a 44.4 GB L40S) |
| `gemma4_e2b_full_48gb` | 1.9B of 5.1B (vision tower frozen too) | 36 GB | 48 GB |
| `qwen25vl7b_full` | 8.3B (all) | 130 GB | multi-GPU sharding (not implemented) |

!!! question "Why not keep the weights in bf16 or fp16 and save half the memory?"
    The weights would be stored at 2 bytes, but updates smaller than the format's resolution are lost.
    At PEAR's learning rate (1e-6), an Adam step changes a typical weight (|w| ≈ 0.01–0.05) by about
    1e-6, i.e. 1e-4–1e-5 of its value. bf16 resolves about 4e-3 relative (8 mantissa bits) and fp16 about
    1e-3 (11 bits), so in both formats almost every update rounds back to the old weight and training
    silently does nothing. fp16 also has a narrow range: activations of models pre-trained in bf16
    overflow its 65504 maximum, which needs loss scaling and still risks NaNs. Mixed precision
    (fp32 masters + bf16/fp16 compute) is the standard fix. The memory goes to the fp32 masters and the
    optimizer state, not to the compute dtype, so fp16 compute would not save anything over bf16.
    Ways to cut the 16 bytes that are not implemented here: 8-bit optimizer states (bitsandbytes), bf16
    weights with stochastic rounding or Kahan-compensated updates (e.g. torchao's low-bit AdamW), or
    sharding the optimizer state over GPUs (FSDP / ZeRO).

## Adding a model

See [Adding a model](adding.md).
