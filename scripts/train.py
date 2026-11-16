"""Train a HF VLM with multi-turn GRPO on any registered env (Hydra CLI).

    python scripts/train.py                                        # defaults: Qwen3-VL-4B LoRA, PointMaze UMaze, HF rollouts
    python scripts/train.py experiment=pointmaze_q3vl4b_vllm steps=40 wandb_project=vlm-rl-pointmaze
    python scripts/train.py policy=qwen3vl2b_lora env_kwargs.maze=medium reward='success+0.5*neg_dist'
    python scripts/train.py experiment=smoke                       # 1-step CPU smoke test
    python scripts/train.py -m reward=success,neg_dist out_dir='runs/sweep/${hydra:job.num}'   # multirun
    python scripts/train.py --cfg job                              # print the resolved config and exit

Config groups live in configs/ (policy/, task/, rollout/, experiment/); every TrainConfig field is a
type-checked key. The resolved config + overrides are saved to <out_dir>/.hydra/.
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation
sys.path.insert(0, os.getcwd())  # allow env="examples.foo:bar" / reward="mod:fn" from the repo root

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from vlmrl.config import CONFIG_DIR, to_train_config  # noqa: E402  (also registers the schema)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    from vlmrl.trainer import Trainer
    Trainer(to_train_config(cfg)).train()


if __name__ == "__main__":
    main()
