# SmolVLM-256M-Instruct

**Preset:** `policy=smolvlm256m` · used by `experiment=smoke` and the `--run-slow` tests ·
**HF:** [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

```yaml
model: HuggingFaceTB/SmolVLM-256M-Instruct
processor_kwargs: {do_image_splitting: false}
lora_rank: 0
lr: 1.0e-5
```

Small enough to run on a laptop CPU (~500 MB download). It's for **testing the pipeline**, not for
learning the task.

| | |
|---|---|
| tokens per 336-px frame | 67 without tiling, 1,127 with tiling (the processor default) |
| chat template | prefix-stable; the multi-turn mask tests run on it |
| behavior | does not follow the action format: replies like `[0, 0]` or unterminated tags |

What it's used for:

- `make smoke`: one full RL step on CPU (`on_invalid=noop`, dense reward, so there's something to update).
- `make test-slow`: mask spans decode to exactly the assistant replies, `logits_to_keep` log-probs equal
  a naive full forward, padding invariance, CLI end-to-end, a real weight update.
