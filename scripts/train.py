"""Train a HF VLM with multi-turn GRPO on any registered env.

    python scripts/train.py --config configs/pointmaze_qwen3b.json --steps 100
    python scripts/train.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
        --env_kwargs '{"maze": "umaze"}' --reward success+0.2*format

Every TrainConfig field is a flag; dict fields take JSON. CLI flags override --config.
"""
import argparse
import dataclasses
import json
import os
import sys
import typing

sys.path.insert(0, os.getcwd())  # allow --env/--reward "examples.foo:bar" from the repo root

from vlmrl.trainer import TrainConfig, Trainer


def parse() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="JSON file with TrainConfig fields")
    hints = typing.get_type_hints(TrainConfig)
    for f in dataclasses.fields(TrainConfig):
        t = hints[f.name]
        if t is bool:
            conv = lambda s: s.lower() in ("1", "true", "yes")
        elif t is dict:
            conv = json.loads
        elif t in (int, float, str):
            conv = t
        else:  # Optional[...]
            inner = [a for a in typing.get_args(t) if a is not type(None)][0]
            conv = (lambda s, inner=inner: None if s.lower() == "none" else inner(s))
        p.add_argument(f"--{f.name}", type=conv, default=argparse.SUPPRESS)
    args = vars(p.parse_args())
    base = {}
    if args.get("config"):
        with open(args.pop("config")) as fh:
            base = json.load(fh)
    args.pop("config", None)
    base.update(args)
    return TrainConfig(**base)


if __name__ == "__main__":
    Trainer(parse()).train()
