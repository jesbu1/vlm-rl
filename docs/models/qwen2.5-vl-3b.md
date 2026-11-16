# Qwen2.5-VL-3B-Instruct

**Preset:** `policy=qwen25vl3b_lora` · **HF:** [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

Same LoRA settings as the Qwen3-VL presets (r=64, α=128, `.*language_model.*` targets, lr 2e-5).

| | |
|---|---|
| tokens per 336-px frame | 146 (28-px patches; Qwen3-VL uses 32) |
| shipped generation config | `temperature=1e-6, repetition_penalty=1.05`: **effectively greedy**, overridden by vlmrl |
| chat template | prefix-stable (checked with the processor on a 3-turn, 3-image chat) |
| image budget | `+processor_kwargs.max_pixels=…` caps tokens for larger frames |

!!! danger "Why the generation override matters most here"
    With the checkpoint's own `temperature=1e-6`, all G rollouts in a group would be identical, every
    group would be filtered, and training would silently do nothing. vlmrl always passes
    `temperature`/`top_p`/`top_k=0`/`repetition_penalty=1.0`.

## Status

The processor and template are checked, and the preset composes (tests). The model hasn't been trained
end-to-end in vlmrl yet. It was planned for the small sweep and then swapped for SmolVLM2.

```bash
python scripts/train.py policy=qwen25vl3b_lora                  # HF backend
scripts/launch_vllm.sh Qwen/Qwen2.5-VL-3B-Instruct 8011 1        # or vLLM
```
