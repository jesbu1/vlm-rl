"""Hydra/OmegaConf glue. ``TrainConfig`` (the dataclass) is the schema; YAML in ``configs/`` sets values."""
from __future__ import annotations

import os

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from vlmrl.trainer import TrainConfig

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

ConfigStore.instance().store(name="train_schema", node=TrainConfig)


def to_train_config(cfg: DictConfig) -> TrainConfig:
    d = OmegaConf.to_container(cfg, resolve=True)
    d.pop("hydra", None)
    return TrainConfig(**d)


def compose(overrides: list[str] | None = None, config_name: str = "train") -> TrainConfig:
    """Build a TrainConfig from the YAML tree + Hydra overrides without running a job (tests, sweeps)."""
    from hydra import compose as _compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return to_train_config(_compose(config_name=config_name, overrides=list(overrides or [])))
