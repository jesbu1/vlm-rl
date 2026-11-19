# Qwen3.5-4B

`policy=qwen35_4b_lora` · [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · `Qwen3_5ForConditionalGeneration`
(transformers ≥ 5, vLLM 0.33 in the image).

* **Hybrid text stack:** 32 layers, 24 Gated DeltaNet (linear attention) + 8 full attention (one in four), plus a
  24-block vision tower. 4.5B parameters, 4.2B of them in the language model.
* **LoRA targets:** the full-attention `q/k/v/o_proj` and all 32 MLPs (`gate/up/down_proj`). The linear-attention
  projections (`in_proj_qkv/z/a/b`, `out_proj`) stay frozen: vLLM serves LoRA on them only through its packed
  `in_proj_qkvz` / `in_proj_ba` modules.
* **No `generation_config.json`.** The stop token comes from the tokenizer (`eos_token = <|im_end|>`), so the loss
  mask still ends at the end-of-turn token (checked on the real template).

## Thinking

Qwen3.5 thinks by default. The preset sets `chat_template_kwargs.enable_thinking: false`; override it per run. The same
kwargs reach vLLM and the trainer's tokenization, so both render the same prompt.

| `enable_thinking` | generation prompt ends with | trained span (per-turn samples) |
|---|---|---|
| `false` | `<think>\n\n</think>\n\n` | the answer + `<|im_end|>` |
| `true` | `<think>\n` | the reasoning + `</think>` + the answer + `<|im_end|>` |

* **Reasoning is per turn.** The chat template renders `<think>…</think>` only for the turn after the last user
  message, so earlier turns' reasoning is dropped from later contexts, both when vLLM generates and when the trainer
  re-renders. Training uses per-turn samples (`max_frames=1`), where the trained reply is always the last message.
  Full-history samples with thinking would not be prefix-stable, and the first batch's check would stop them.
* **Actions are parsed after the last `</think>`** (`rollout.final_answer`). With `enable_thinking: true`, a reply
  that never closes its reasoning (cut off at `max_new_tokens`) is invalid.
* Thinking runs need a larger `max_new_tokens` (512 in `configs/sweeps/metaworld_ml1_q35.yaml`).
