# Adding a model

## 1. Write a preset

```yaml
# configs/policy/my_model.yaml
# @package _global_
model: org/My-VLM-Instruct
lora_rank: 64
lora_alpha: 128
lora_targets: '.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
processor_kwargs: {}          # e.g. {do_image_splitting: false} or {max_pixels: 200704}
lr: 2.0e-5
```

Use it with `policy=my_model`. Add it to the parametrized list in
`tests/test_trainer.py::test_policy_presets_compose`.

## 2. Check the four things that differ between models

**Module names for LoRA.** List the language model's linear layers:

```python
import torch
from transformers import AutoConfig, AutoModelForImageTextToText
with torch.device("meta"):   # no weights downloaded or allocated
    m = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained("org/My-VLM-Instruct"))
print(sorted({n.rsplit(".", 2)[0] for n, _ in m.named_modules() if n.endswith("q_proj")})[:3])
```

**Tokens per frame.** Make sure image tiling doesn't blow up the context:

```python
import numpy as np
from PIL import Image
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("org/My-VLM-Instruct")
img = Image.fromarray(np.zeros((336, 336, 3), np.uint8))
n = lambda c: len(p.apply_chat_template([{"role": "user", "content": c}], tokenize=True, return_dict=True)["input_ids"][0])
print(n([{"type": "image", "image": img}, {"type": "text", "text": "hi"}]) - n([{"type": "text", "text": "hi"}]))
```

**Prefix-stable template.** Run one training step (`steps=1`). `build_batch` raises a clear error on the
first batch if rendering the full chat changes earlier turns. If so, set `max_images=<int>`.

**Generation config.** Check `generation_config.json`. vlmrl overrides `temperature`, `top_p`, `top_k`
and `repetition_penalty`, but note anything else unusual (e.g. custom stop tokens).

## 3. For the vLLM backend

- vLLM must support the architecture **and** LoRA for it.
- Pass extra server flags for that model with `vllm_args` in your sweep file (e.g. `--max-model-len`,
  `--mm-processor-kwargs`), or after the GPU argument of `scripts/launch_vllm.sh`.
- Install the processor's extra Python dependencies in the vLLM venv too.

## 4. Document it

Add a page under `docs/models/` (copy [SmolVLM2](smolvlm2-2.2b.md) as a template), a row to the
[Models](index.md) table, and an entry in the `nav:` of `mkdocs.yml`.
