# The robot-use protocol

PEAR's prompt (paper Appendix H) has a **fixed protocol** that is the same for every environment,
followed by a task-specific `## ENVIRONMENT` section and the `## TASK` instruction. vlmrl reproduces
the template verbatim in [`vlmrl/envs/protocol.py`](https://github.com/jesbu1/vlm-rl/blob/main/vlmrl/envs/protocol.py).
To see it filled in for PointMaze, go to [Prompts](../environments/prompts.md).

## Sections

### OBSERVATION

> The robot is made of named parts, and each part has named quantities you can set. Every turn
> begins with the current value of every such quantity, grouped by part, followed by the camera
> images: `{"<part>": {"<quantity>": <value>, ...}, ...}`. Lengths are in centimeters and angles in
> radians; any other quantity is a plain number. Nothing beyond these names is given about what
> each quantity corresponds to.

The state lists only quantities the model can **set** (e.g. `left_arm.x … yaw`, `left_gripper.width`),
**not** velocities or object poses. Objects must be found in the images.

### ACTION

> The action changes the listed quantities by these amounts: `{"<part>": {"<quantity>": <change>, ...}, ...}`.
> Write the change of every quantity; 0 leaves a quantity as it is.

So the action is a **delta** with the same structure as the state. On RoboDojo, for example:

```json
{"left_arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
 "right_arm": {"x": 0, "y": 0, "z": -5, "roll": 0, "pitch": 0, "yaw": 0},
 "left_gripper": {"width": 0}, "right_gripper": {"width": -1}}
```

lowers the right gripper by 5 cm and closes it.

### EXECUTION

> The action ends once every target is reached and the robot is at rest. If something keeps a target
> from being reached, the robot stops there and the action ends. Either way, you then receive the
> next observation.

The controller is robot-specific: an operational-space controller on LIBERO's Franka, Cartesian
waypoints with IK on RoboDojo's ARX. In training, the JSON action is executed as a **tool call** to
the simulator, and the call's result is the next observation.

### RESPONSE FORMAT

> End your response with exactly one fenced json block containing the action, and nothing after it.

### ENVIRONMENT

Benchmark-specific: the robot's frame, gripper conventions and how each camera is oriented. For example,
LIBERO's *"agentview: a fixed camera; the robot's base is at the top of its image, +x points toward
the bottom of the image and +y toward its left"*. Every environment also gets an *Ending* note: *"the
episode ends by itself as soon as the task is done: while observations keep coming, the task is not
done yet."* RoboDojo also explains gripper orientation (e.g. *"a roll change of +1.57 points it
straight down"*) and gives a full example action.

!!! note "Prompt detail matters for small models"
    The paper notes that its 27B base model "often lacks knowledge about the embodiment", so harder
    tasks only succeed once the prompt describes the robot and the task in detail. In vlmrl, the 2B
    model shows the same pattern on PointMaze (see [Qwen3-VL-2B](../models/qwen3-vl-2b.md)).

## Episode endings

An episode ends when any of these happens:

| condition | where | in vlmrl |
|---|---|---|
| the simulator reports success | all | `StepResult.terminated` + `info["success"]` |
| the benchmark's step limit | sim | `env_kwargs.max_turns` |
| the context reaches its token budget (256k LIBERO, 160k RoboDojo) | sim | `max_context_tokens` |
| **a response contains no valid action** | sim | `on_invalid=terminate` (default) |
| more than 25 rounds, or >32k reasoning tokens in one round | real robot | `max_turns`, `max_new_tokens` |

On the real robot, an action that IK can't solve is retried, and the retry counts as a round.

## Using the protocol for your own robot

```python
from vlmrl.envs import protocol

SCHEMA = {"left_arm": ["x", "y", "z", "roll", "pitch", "yaw"], "left_gripper": ["width"]}
system = protocol.system_prompt(environment=MY_ENV_SECTION, instruction="Pick up the red block.")
state_line = protocol.format_state(SCHEMA, current_values)      # first text of every user turn
action = protocol.parse_action(reply_text, SCHEMA)               # flat np.ndarray, or None if invalid
```

`parse_action` takes the **last** fenced json block (falling back to `<action>…</action>`), and requires
**every** part and quantity, as the protocol says. A missing or non-finite value makes the action invalid.
