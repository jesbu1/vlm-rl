# Concepts

How vlmrl is put together, independent of any particular environment or model.

| page | read it to understand |
|---|---|
| [Architecture](architecture.md) | the five interfaces the trainer sees, and why dynamics are split from the text protocol |
| [Rollouts & episodes](rollouts.md) | the batched multi-turn loop, every way an episode can end, how context is windowed |
| [Algorithm](algorithm.md) | exactly what one training step computes: filter, advantages, loss masks, score centering |
| [Rewards & preferences](rewards.md) | scalar rewards, preference models, win-rate / Bradley–Terry, reward specs |
