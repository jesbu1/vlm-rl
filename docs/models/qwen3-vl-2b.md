# Qwen3-VL-2B-Instruct

**Preset:** `policy=qwen3vl2b_lora` · **HF:** [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

Same settings as [Qwen3-VL-4B](qwen3-vl-4b.md) (LoRA r=64, α=128, language-model targets, lr 2e-5),
and the same processor: 102 tokens per 336-px frame. Shipped generation config:
`temperature=0.7, top_p=0.8, top_k=20` (overridden).

## Observed behavior

!!! warning "Echoes the state instead of writing a change"
    Under the [PEAR protocol](../pear/protocol.md), zero-shot Qwen3-VL-2B usually skips reasoning and
    copies the current state JSON as its action:

    ```text
    USER:      {"ball": {"x": -1.4, "y": 1.4}}  <image>
    ASSISTANT: ```json
               {"ball": {"x": -1.4, "y": 1.4}}
               ```
    ```

    It reads the action as an absolute pose rather than a change. Actions are always valid (0% invalid),
    but with `reward=success` every group fails and gets filtered, so nothing trains. A dense term
    (`reward='success+0.5*neg_dist'`) gives signal. In a 3-step check, rollout success rose from 0 to
    0.19 (16 episodes, so noisy). This matches the paper's note that a small base model needs a
    detailed embodiment description in the prompt.

Options worth comparing for this model: a dense or preference reward, `prompt_style=simple` (whose
action keys `dx`/`dy` make the delta explicit), or a richer ENVIRONMENT section.

## Memory

About half of the 4B model's footprint. Its HF-backend trainer ran on one A6000 next to another
training job.

## Run it

```bash
scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8011 1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py experiment=pointmaze_q3vl4b_vllm policy=qwen3vl2b_lora \
    reward='success+0.5*neg_dist'
```
