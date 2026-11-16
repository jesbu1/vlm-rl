# Testing

```bash
make test        # pytest                        143 fast tests: CPU, no downloads, ~10 s
make test-slow   # pytest --run-slow             + 6 tests with a real SmolVLM-256M
make test-all    # pytest --run-slow --run-gpu   + CUDA-only tests
```

The tiers are defined in `tests/conftest.py`: `@pytest.mark.slow` needs `--run-slow`, and
`@pytest.mark.gpu` needs `--run-gpu` and CUDA. CI (`.github/workflows/tests.yml`) runs the fast tier
on every push and PR, and the slow tier on push.

## The `FakePolicy` trick

`tests/conftest.py` provides `FakePolicy`, a 4-parameter "model" with `HFPolicy`'s interface
(`generate`, `build_batch`, `token_logprobs` including the score-centering `topk`/`gather_ids` paths, and
`save`). Combined with scripted text policies (`oracle_generate`, `random_generate`, `const_generate`,
in both prompt styles), it lets the **entire trainer loop** run in milliseconds. The
real-model behavior (masks, `logits_to_keep`, padding) is covered by the slow tier.

## Coverage by file

| file | covers |
|---|---|
| `test_protocol.py` | PEAR template sections/substitution, state JSON, fenced-json extraction, "every quantity" rule |
| `test_pointmaze.py` | coordinates, renderer, reset-to-state, PD tracking, clipping, walls, force mode, settle (reached / blocked / cap), early success, slot isolation, the oracle solves every maze, both prompt styles and parsers, `format_action` round-trip, samplers |
| `test_rollout.py` | message structure, max-turns, invalid → terminate / noop, empty-step guard, context budget, end reasons, finished episodes stop generating, windowing, rollout context == training sample |
| `test_algo.py` | z-score / center / rloo, filter, REINFORCE gradient, PPO ≡ REINFORCE at ratio 1, clipping, global-token normalization, score centering (0 on-policy; = E_μ∇logπ with full support) |
| `test_rewards.py` | every scalar reward, combine, spec parsing, module-path rewards, win-rate / BT ranking, position-bias cancellation, pair subsampling |
| `test_world_model.py` | WAM adapter bookkeeping and batching, done threshold, carry keys, toy WAM + preferences |
| `test_policy.py` | chunked log-probs (values, grads), adaptive OOM batching; *slow:* generate, mask spans, `logits_to_keep` vs full forward, padding invariance |
| `test_vllm_client.py` | message conversion, request params and order, the LoRA hot-swap protocol (fake HTTP server) |
| `test_trainer.py` | env registry, collect/filter/resample, update direction, PPO path, multi-epoch score-centering wiring, full loop (logs, eval, videos, checkpoints), MP4 encoding (H.264, even-size padding) and the GIF option, sync after update, wandb offline media, Hydra defaults / overrides / type errors, every experiment and policy preset composes; *slow:* CLI end-to-end + a real weight update |
| `test_sweep.py` | Hydra grid expansion (model-major, presets carry per-model settings), the shipped sweep composes, `stop_server` kills a SIGTERM-ignoring server and its children |
| `test_types.py` | trajectory properties |
| `test_docs.py` | generated docs are up to date, every Mermaid block is fenced correctly, every `nav` page exists |

## Writing a test for a new environment

Drive it through the text protocol with a scripted policy, the way `test_oracle_solves_maze` does: build
the reply with `interface.format_action(a)`, parse it back, and check that a known-good policy succeeds.
