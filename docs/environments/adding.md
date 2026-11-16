# Adding an environment

Write a factory that returns an `EnvBundle`, then point `env=` at it.

```python
# my_pkg/arm_env.py
import numpy as np
from PIL import Image
from vlmrl.envs import protocol
from vlmrl.envs.base import EnvBundle
from vlmrl.types import Obs, StepResult

SCHEMA = {"left_arm": ["x", "y", "z", "roll", "pitch", "yaw"], "left_gripper": ["width"]}
ENVIRONMENT = """Robot and frame:
- The arm's frame has its origin at the robot's base: +x forward, +y left, +z up.
Cameras:
- front: fixed in front of the table, looking at the robot.
Ending:
- The episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet."""

class ArmInterface:
    def system_prompt(self, start_state):
        return protocol.system_prompt(ENVIRONMENT, start_state["instruction"])
    def observation_message(self, obs, turn, last, parse_error):
        text = protocol.format_state(SCHEMA, obs.state)
        return [{"type": "text", "text": text}, {"type": "image", "image": Image.fromarray(obs.image)}]
    def parse_action(self, text):
        return protocol.parse_action(text, SCHEMA)            # None if invalid -> episode ends (PEAR)
    def format_action(self, a):
        return protocol.example_action(SCHEMA, a)
    def noop_action(self):
        return np.zeros(sum(len(q) for q in SCHEMA.values()))

class ArmVecEnv:
    def reset(self, start_states): ...                        # -> list[Obs]
    def step(self, ids, actions): ...                         # -> list[StepResult]; set info["success"]
    def close(self): ...

def make_bundle(max_turns: int = 25, **kw):
    sampler = MyTaskSampler(**kw)                             # .sample(n, rng) -> list[dict]
    return EnvBundle(make_vec_env=lambda: ArmVecEnv(), interface=ArmInterface(),
                     train_sampler=sampler, eval_sampler=sampler, max_turns=max_turns)
```

Make it a named task by adding `configs/task/arm.yaml`:

```yaml
# @package _global_
env: my_pkg.arm_env:make_bundle
env_kwargs: {max_turns: 25}
reward: success
```

and run `python scripts/train.py task=arm`.

!!! warning "Select a task, don't just override `env=`"
    Hydra **merges** dict overrides, so with only `env=...` the default `task=pointmaze`'s
    `env_kwargs` (maze, prompt_style, …) would still be passed to your factory. Selecting `task=arm`
    avoids that. For a one-off without a YAML file, drop the task group and add keys with `+`:
    `python scripts/train.py '~task' env=my_pkg.arm_env:make_bundle +env_kwargs.max_turns=25`.
    You can also register a short name in `vlmrl.trainer.ENV_REGISTRY`.

## Checklist

- `reset` must accept **arbitrary start states** (exact state restore), so that groups share a start state.
- `step(ids, actions)` must only touch the listed slots.
- Report success as `info["success"]`, and terminate on success.
- Keep `observation_message` deterministic. Training re-tokenizes the stored transcript.
- Test it like `tests/test_pointmaze.py`: a scripted policy driven through `format_action` →
  `parse_action` should solve the task.
