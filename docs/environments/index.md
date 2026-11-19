# Environments

An environment is an `EnvBundle`: dynamics (`VecEnv`), an agent-facing text protocol
(`TextInterface`), train/eval start-state samplers, and `max_turns`. See
[Architecture](../concepts/architecture.md).

| page | |
|---|---|
| [PointMaze](pointmaze.md) | the shipped MuJoCo maze environment: dynamics, controller, rendering, start states, oracle |
| [Metaworld](metaworld.md) | Farama Metaworld ML1 / ML10 / ML45 manipulation: hand + gripper waypoints, train/test task splits, scripted experts |
| [Prompts](prompts.md) | the exact prompts the policy sees, generated from the code |
| [World-action models](world-models.md) | swap the simulator for a learned model; preference rewards from it |
| [Adding an environment](adding.md) | write your own bundle, optionally on PEAR's protocol |

Select an environment with `env=` (a registry name or `module:factory`) and configure it with
`env_kwargs.*`. The `task/` config group bundles both (see [Configuration](../training/configuration.md)).
