# Roadmap & limitations

How vlmrl differs from PEAR's production system, and how each gap would be closed.

| limitation | consequence | path forward |
|---|---|---|
| **Synchronous, single trainer process** (PEAR: fully async veRL, FSDP × SP) | rollouts and updates don't overlap; no multi-GPU training | an async rollout queue with staleness ≤ 2 (score centering is already implemented for exactly this); DDP/FSDP with an all-reduced token count (the loss is already a global-token sum) |
| **Score centering uses μ's top-k** (default 32) | small error from the tail mass outside the top-k | in async mode, take μ's top-k from the rollout engine (vLLM `top_logprobs`) |
| **Full-weight vLLM sync goes through disk** | one checkpoint write + in-place reload per step (seconds for a few-B model) | NCCL weight broadcast into the server (as in FeynRL / veRL) |
| **One node per job** on SageMaker | sweeps scale by lanes on one node, a single run by one trainer GPU | multi-node jobs once the trainer supports DDP/FSDP |
| **No automatic curriculum** | multi-stage training is manual | change `env_kwargs` between runs and resume from `ckpt_step*`; a scheduler keyed on per-task success |
| **Re-tokenization** of responses | rare token-level mismatch with what was sampled | store sampled token ids from vLLM (`return_tokens_as_token_ids`) |
| **No sequence packing** | padding in micro-batches | length-sorted micro-batches today; packing with position ids |
| **Small models misread delta actions** | e.g. [Qwen3-VL-2B](../models/qwen3-vl-2b.md) echoes the state | dense or preference rewards, a richer ENVIRONMENT section, or `prompt_style=simple` |
| **PointMaze state vs WAM** | a WAM may not predict low-dim state | train and evaluate with `include_state=false` so the sim and WAM tasks match |
