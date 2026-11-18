# Configuration (Hydra)

The dataclass `vlmrl.trainer.TrainConfig` is the **schema**. It's registered with Hydra's ConfigStore
as `train_schema`, so every key is type-checked: a typo or a wrong type fails before anything
launches. YAML files in `configs/` only set values. The full field list is in the
[configuration reference](../reference/config.md).

## Layout

```
configs/
  train.yaml        defaults: [train_schema, _self_, policy: qwen3vl4b_lora, task: pointmaze, rollout: hf, experiment: null]
  policy/           which VLM + finetuning mode   (qwen3vl4b_lora, qwen3vl2b_lora, qwen25vl3b_lora,
                                                   qwen25vl7b_full, smolvlm2_2b_lora, smolvlm256m,
                                                   smolvlm256m_lora, gemma4_e2b_lora, minicpmv46_lora)
  task/             env + env_kwargs + reward     (pointmaze, toy_wam)
  rollout/          generation backend            (hf, vllm)
  experiment/       full presets, applied last    (pointmaze_q3vl4b_vllm, pointmaze_q3vl4b_vllm_legacy,
                                                   pointmaze_generalization, pear_recipe, toy_wam_prefs, smoke)
  sweeps/           grids for scripts/sweep.py (not loaded by Hydra)
```

- Every group file starts with `# @package _global_`, so it sets **top-level** fields.
- `experiment/` comes last in the defaults list, so it wins, and it can re-select groups with
  `defaults: [override /policy: …]`. A group chosen on the command line still beats an experiment's choice.
- The groups are named `policy` and `task` because `model` and `env` are already field names.

## Overrides

```bash
python scripts/train.py steps=50 lr=1e-5                                     # set fields
python scripts/train.py policy=qwen3vl2b_lora task=toy_wam rollout=vllm      # pick groups
python scripts/train.py experiment=pear_recipe groups_per_step=16            # preset, then tweak
python scripts/train.py env_kwargs.maze=medium env_kwargs.include_state=false  # nested keys (merged)
python scripts/train.py reward='success+0.5*neg_dist'                        # quote values with + * , etc.
python scripts/train.py +processor_kwargs.max_pixels=200704                  # + adds a new dict key
python scripts/train.py max_context_tokens=null                              # None
python scripts/train.py '~task' env=my.mod:make_bundle +env_kwargs.k=1       # drop a group
```

Useful flags:

```bash
python scripts/train.py --cfg job experiment=pear_recipe     # print the resolved config, don't run
python scripts/train.py --help                               # groups and options
```

## Multirun

For grids that don't need a per-model vLLM server, Hydra's multirun works directly:

```bash
python scripts/train.py -m reward=success,neg_dist policy=qwen3vl2b_lora,qwen3vl4b_lora \
    out_dir='runs/m/${hydra:job.num}'
```

Jobs run sequentially. Give each one its own `out_dir`, as above. For vLLM-backed grids, use
[Sweeps](sweeps.md).

## Where the resolved config lives

Each run writes `out_dir/.hydra/config.yaml` (resolved), `overrides.yaml` and `hydra.yaml`, plus
`out_dir/config.json`. wandb gets the full config, so you can group and filter by any field.

## In Python

```python
from vlmrl.config import compose
from vlmrl.trainer import Trainer

cfg = compose(["experiment=pointmaze_q3vl4b_vllm", "steps=5"])   # -> TrainConfig
Trainer(cfg).train()
# or bypass Hydra entirely: Trainer(TrainConfig(...), policy=..., bundle=..., reward_fn=...)
```
