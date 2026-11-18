# Code layout

```
vlmrl/
  types.py            Obs, StepResult, Turn, Trajectory (+ end_reason)
  config.py           Hydra glue: TrainConfig registered as the schema; compose() for tests and sweeps
  envs/
    base.py           VecEnv / TextInterface / StartStateSampler protocols, EnvBundle
    protocol.py       PEAR's robot-use protocol (App. H): system prompt, state JSON, fenced-json action parser
    pointmaze.py      MuJoCo PointMaze VecEnv (settle controller), renderer, text interface, samplers, BFS oracle
    world_model.py    WorldActionModel protocol + WorldModelVecEnv adapter
  rollout.py          batched multi-turn rollouts, PEAR termination rules, context windowing, training samples
  rewards.py          group reward fns: success/return/info/format/efficiency, preferences (win-rate, Bradley–Terry)
  algo.py             GRPO advantages, DAPO filter, REINFORCE / PPO-clip token loss, score centering
  policy.py           HFPolicy: generate, multi-turn loss masks, logits_to_keep + chunked fp32 log-probs, LoRA,
                      full FT with fp32 master weights + freeze regex
  vllm_client.py      VLLMServerGenerator: OpenAI-API rollouts + weight sync (LoRA hot-swap / full in-place reload)
  trainer.py          TrainConfig (schema) + Trainer (collect -> filter -> advantages -> update -> eval/log/save)
configs/              Hydra groups: policy/ task/ rollout/ experiment/ + sweeps/ + sagemaker/example.yaml
examples/toy_wam.py   toy kinematic WAM + preference head
scripts/
  train.py            Hydra CLI
  sweep.py            reward x model grids, one vLLM server per model, parallel lanes, resumable
  sagemaker_submit.py submit a sweep as one SageMaker job via an AWS Batch queue (runs where your AWS login is)
  sagemaker_entry.py  in-container runner: hyperparameters -> sweep.py lanes, checkpoints synced to S3
  make_wandb_report.py  wandb Report for a sweep group
  demo_pointmaze.py   oracle / random policy through the text protocol, GIFs + prompt dump
  gen_docs.py         generated docs pages (config reference, prompts)
  launch_vllm.sh      vLLM server launcher
docker/               Dockerfile (trainer + vLLM envs), entrypoint.sh, build.sh, pinned requirements
docs/                 this site (MkDocs Material); mkdocs.yml at the repo root
tests/                pytest suites, one per module, + conftest tiers and fixtures
```

## Main entry points

| you want to… | use |
|---|---|
| train from the CLI | `python scripts/train.py <overrides>` |
| run a sweep on SageMaker | `python scripts/sagemaker_submit.py <sweep>` ([guide](../training/sagemaker.md)) |
| train from Python | `Trainer(compose([...])).train()` or `Trainer(TrainConfig(...), policy=, bundle=, reward_fn=)` |
| roll out without training | `vlmrl.rollout.rollout(generate, env, interface, starts, group_ids, max_turns)` |
| score groups | `vlmrl.rewards.build_reward(spec)(groups)` |
| use PEAR's protocol for a new robot | `vlmrl.envs.protocol.{system_prompt, format_state, parse_action}` |
