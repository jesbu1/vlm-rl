"""Sanity-check the PointMaze VLM env without any model.

Drives the env through the *text* interface with a BFS oracle (or random policy), so the parser,
PD controller, renderer and success logic are all exercised. Saves a GIF and the first prompt.

    python scripts/demo_pointmaze.py --maze medium --policy oracle --episodes 20
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np

from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze, oracle_delta

p = argparse.ArgumentParser()
p.add_argument("--maze", default="umaze")
p.add_argument("--policy", default="oracle", choices=["oracle", "random"])
p.add_argument("--episodes", type=int, default=16)
p.add_argument("--max_turns", type=int, default=20)
p.add_argument("--start_states", default="random")
p.add_argument("--out", default="outputs/demo")
a = p.parse_args()

bundle = make_pointmaze(PointMazeConfig(maze=a.maze, max_turns=a.max_turns, start_states=a.start_states))
env, itf = bundle.make_vec_env(), bundle.interface
rng = np.random.default_rng(0)
starts = bundle.train_sampler.sample(a.episodes, rng)
obs = env.reset(starts)
active, turns, succ = list(range(a.episodes)), np.zeros(a.episodes, int), np.zeros(a.episodes, bool)
frames = [obs[0].image]
for t in range(a.max_turns):
    if not active:
        break
    texts = []
    for k in active:
        o = obs[k]
        d = oracle_delta(env.maze_map, o.state[:2], o.extras["goal"]) if a.policy == "oracle" else rng.uniform(-1, 1, 2)
        texts.append(f'Moving toward the goal. <action>{{"dx": {d[0]:.2f}, "dy": {d[1]:.2f}}}</action>')
    acts = [itf.parse_action(x) for x in texts]
    res = env.step(active, acts)
    nxt = []
    for k, r in zip(active, res):
        obs[k] = r.obs
        turns[k] += 1
        if k == 0:
            frames.append(r.obs.image)
        if r.terminated:
            succ[k] = True
        else:
            nxt.append(k)
    active = nxt

os.makedirs(a.out, exist_ok=True)
imageio.mimsave(f"{a.out}/{a.maze}_{a.policy}.gif", frames, duration=0.4)
with open(f"{a.out}/prompt.txt", "w") as f:
    f.write("SYSTEM:\n" + itf.system_prompt(starts[0]) + "\n\nUSER:\n")
    f.write(itf.observation_message(env.reset(starts[:1])[0], 0, None, False)[1]["text"])
print(f"{a.maze}/{a.policy}: success {succ.mean():.2f}, mean turns {turns.mean():.1f}  -> {a.out}")
