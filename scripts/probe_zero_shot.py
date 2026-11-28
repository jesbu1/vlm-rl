"""Zero-shot probe: a base model served by vLLM (no training, no adapter) acting through the text protocol.

Before RL from scratch (PEAR's route: RL only, reasoning before every move) a base model needs *some* success to learn
from, and its reasoning has to end within the token budget. This measures both on any eval sets, without loading the
trainer's copy of the model, so models too large to train on one GPU (Qwen3.5-27B) can be probed on the rollout GPUs
(vLLM tensor parallel). Writes metrics.json (success per set and task, invalid replies, reply length, end reasons) and
a few transcripts per task.

    scripts/launch_vllm.sh Qwen/Qwen3.5-27B 8011 0,1 --tensor-parallel-size 2 &      # ENABLE_LORA=0
    python scripts/probe_zero_shot.py --model Qwen/Qwen3.5-27B --vllm_url http://localhost:8011 \\
        --env_kwargs '{"benchmark": "ML10", "prompt_style": "compact_rel"}' --sets train_tasks=train:50,heldout_tasks=test:50 \\
        --thinking --max_new_tokens 8192 --temperature 0.6 --top_p 0.95 --out runs/probe/qwen35-27b
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import numpy as np

from vlmrl.rollout import RolloutTask, rollout_pooled
from vlmrl.trainer import Trainer, goal_estimate_error, make_env_bundle, per_task_success


def parse_sets(spec: str) -> list[tuple[str, str, int]]:
    """``name=split:episodes,...`` -> [(name, split, episodes)]."""
    out = []
    for part in spec.split(","):
        name, rest = part.split("=")
        split, n = rest.split(":")
        out.append((name.strip(), split.strip(), int(n)))
    return out


def wait_ready(url: str, timeout: float = 3600.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(10)
    raise TimeoutError(f"vLLM server at {url} not ready after {timeout:.0f}s")


def summarize(trajs, name: str, interface) -> dict:
    m = Trainer.eval_metrics(trajs, name)
    m.update(per_task_success(trajs, name))
    m.update(goal_estimate_error(trajs, interface, name))
    replies = [t.response for tr in trajs for t in tr.turns]
    m[f"{name}/reply_chars"] = float(np.mean([len(r) for r in replies])) if replies else 0.0
    m[f"{name}/episodes"] = len(trajs)
    return m


def write_transcripts(trajs, path: str, per_task: int = 2) -> None:
    seen: dict[str, int] = {}
    with open(path, "w") as f:
        for tr in trajs:
            name = (tr.start_state or {}).get("env_name", "?")
            if seen.get(name, 0) >= per_task:
                continue
            seen[name] = seen.get(name, 0) + 1
            f.write(f"\n===== {name} success={tr.success} end={tr.end_reason} turns={len(tr.turns)}\n")
            for i, t in enumerate(tr.turns):
                f.write(f"--- turn {i} (valid={t.action is not None})\n{t.response}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vllm_url", default="http://localhost:8011")
    ap.add_argument("--env", default="metaworld")
    ap.add_argument("--env_kwargs", default="{}", help="JSON, as env_kwargs in the training config")
    ap.add_argument("--sets", default="train_tasks=train:50,heldout_tasks=test:50", help="name=eval_split:episodes,...")
    ap.add_argument("--thinking", action="store_true", help="chat_template_kwargs.enable_thinking=true")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=1, help="earlier turns kept in context (-1 = all)")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--on_invalid", default="noop", choices=("noop", "terminate"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vlmrl.vllm_client import VLLMServerGenerator
    os.makedirs(a.out, exist_ok=True)
    wait_ready(a.vllm_url)
    gen = VLLMServerGenerator(a.vllm_url, a.model, os.path.join(a.out, "unused_adapters"),
                              chat_template_kwargs={"enable_thinking": bool(a.thinking)})
    kw = json.loads(a.env_kwargs)
    tasks, names, itfs = [], [], []
    for name, split, n in parse_sets(a.sets):
        b = make_env_bundle(a.env, {**kw, "eval_split": split})
        starts = b.eval_sampler.sample(n, np.random.default_rng(10_000 + a.seed))
        tasks.append(RolloutTask(b.make_vec_env(), b.interface, starts, list(range(len(starts))), b.max_turns))
        names.append(f"eval/{name}")
        itfs.append(b.interface)
    max_frames = None if a.max_frames < 0 else a.max_frames
    t0 = time.time()
    out = rollout_pooled(lambda convs: gen.generate(convs, a.max_new_tokens, a.temperature, a.top_p),
                         tasks, a.max_images, on_invalid=a.on_invalid, max_frames=max_frames, thinking=a.thinking)
    m = {"model": a.model, "thinking": a.thinking, "max_new_tokens": a.max_new_tokens, "temperature": a.temperature,
         "env_kwargs": kw, "seconds": time.time() - t0}
    for name, trajs, itf in zip(names, out, itfs):
        m.update(summarize(trajs, name, itf))
        write_transcripts(trajs, os.path.join(a.out, f"transcripts_{name.split('/')[-1]}.txt"))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps({k: v for k, v in m.items() if "/task/" not in k}, indent=2))
    for task in tasks:
        task.env.close()


if __name__ == "__main__":
    main()
