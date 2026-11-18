# SmolVLM2-2.2B-Instruct

**Presets:** `policy=smolvlm2_2b_lora`, `policy=smolvlm2_2b_full` · **HF:** [`HuggingFaceTB/SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)

```yaml
model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
lora_rank: 64
lora_alpha: 128
lora_targets: '.*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
processor_kwargs: {do_image_splitting: false}
lr: 2.0e-5
```

An Idefics3-style architecture, which is a different family from Qwen. That makes it a useful
architecture comparison in the sweep.

## Gotchas

| | |
|---|---|
| LoRA targets | the language model's modules are named **`text_model`**, not `language_model` |
| image tiling | on by default: **1,416 tokens per 336-px frame**. With `do_image_splitting: false`: **84** |
| processor dependency | `num2words` (`uv pip install -e ".[smolvlm]"`), in the **vLLM venv too** |
| context | 8k tokens. Serve with `--max-model-len 8192`; long episodes may need `max_images=<int>` |
| vLLM server | must also disable tiling: `--mm-processor-kwargs '{"do_image_splitting": false}'` |
| shipped generation config | none |

The sweep config passes the server flags automatically:

```yaml
# configs/sweeps/pointmaze_small.yaml
vllm_args:
  HuggingFaceTB/SmolVLM2-2.2B-Instruct: [--max-model-len, "8192", --mm-processor-kwargs, '{"do_image_splitting": false}']
```

## Run it by hand

```bash
scripts/launch_vllm.sh HuggingFaceTB/SmolVLM2-2.2B-Instruct 8011 1 \
    --max-model-len 8192 --mm-processor-kwargs '{"do_image_splitting": false}'
CUDA_VISIBLE_DEVICES=0 python scripts/train.py experiment=pointmaze_q3vl4b_vllm policy=smolvlm2_2b_lora
```

## Status

Was the third model in the `pointmaze-small-v1`–`v3` sweep grids, but those sweeps were stopped before
reaching it, so it has **no training results yet**. It is not in v4. One-step smoke runs on puget confirmed
that the preset and vLLM flags work (see [Experiment log](../training/experiments.md)).

## Full fine-tuning

`policy=smolvlm2_2b_full`: every parameter trained, with fp32 master weights and a bf16 autocast forward pass. See
the memory table in [Models](index.md#memory); it fits one 48 GB GPU. vLLM rollouts use the
[full-weight sync](../training/backends.md#how-weight-sync-works). The saved checkpoint has exactly the original
tensor names (checked on CPU). Shipped sweep: `configs/sweeps/batch_small_fft.yaml` (simple prompt, dense
reward, UMaze + medium), run on L40S machines through [AWS Batch](../training/batch.md).

