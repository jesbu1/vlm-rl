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
* Thinking runs need a much larger `max_new_tokens`: on Metaworld the reasoning runs ~2,000 characters per turn even when the prompt asks for only the move, and at 512 tokens every reply was cut off before `</think>` (100% invalid). The ML1 runs use 2048.

## Trainer vs vLLM (first ML1 runs)

`train/logp_gap` (trainer mean log-prob − vLLM's, on the same trained replies) on `metaworld-ml1-q35`:

* **Thinking off, reach:** −0.01 at step 1, then it grew with training (−0.13 at step 4, −0.27 at step 8, −0.34
  at step 40), while replies grew from 73 to ~520 characters and invalid replies reached 100% by step 17. A growing
  gap means vLLM was not sampling from the model the trainer updated.
* **Thinking on:** −0.07 to −0.13 already at step 1, before any update: the base models disagree on long sequences.
  The trainer ran the 24 Gated DeltaNet layers on transformers' PyTorch fallback (`flash-linear-attention` was not
  installed); vLLM uses its fla Triton kernels. The image now installs `flash-linear-attention==0.6.0`, which
  transformers uses on GPU.
* At 2048 new tokens, 98% of thinking replies were cut off before `</think>` (~7,500 characters of reasoning per
  turn), so thinking has almost no signal at this budget.

Follow-up runs separate the two suspects: the kernels (fla installed) and LoRA on the gated full attention
(`policy=qwen35_4b_lora_mlp`: MLP-only adapters).
