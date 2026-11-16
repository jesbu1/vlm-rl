# Results

All numbers are from the paper. Models run at xhigh effort unless noted.

## Headline numbers

| benchmark | base Qwen3.8-27B | PEAR | reference |
|---|---|---|---|
| LIBERO (2,000 episodes) | 1.6% | **74.4%** | GPT-6.1 Sol 76.2% / GPT-6 Astra 80.6% at max effort; 48.1% / 49.3% at medium |
| LIBERO-Pro, unseen task in a training scene | – | 57.4% | π₀.₅ 0.75% |
| LIBERO-Pro, swapped object positions | – | 57.5% | π₀.₅ 20.8% |
| LIBERO-Long, zero-shot (before training on it) | 0.4% | 39.0% | 53.2% after adding it |
| RoboDojo `general_pickup` / `align_blocks` (LIBERO-only model) | 10.7% / 13.0% | 39.9% / 17.0% | 71.2% / 78.8% after RoboDojo RL |
| RoboLab-120 (new benchmark) | 5.4% | 37.3% | Cosmos3-Nano-Policy 36.8%, π₀.₅ 28.0% |
| Real dual-arm YAM, zero-shot sim-to-real | 0/30 | **18/30** | LIBERO-only model 7/30 |
| 9 general benchmarks (AIME, GPQA, LiveCodeBench, MMMU, …) | mean 81.64 | mean 81.49 | every change within its 95% CI |

## LIBERO by suite

| model | Spatial | Object | Goal | Long | avg |
|---|---|---|---|---|---|
| Qwen3.8-27B (base) | 1.2 | 1.8 | 2.8 | 0.4 | 1.6 |
| PEAR LIBERO-only, xhigh | **85.8** | **97.0** | 61.4 | 53.2 | 74.4 |
| PEAR LIBERO-only, low effort | 30.4 | 35.9 | 18.9 | 2.5 | 21.9 |
| π₀.₅ (SFT on LIBERO demos) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |

19% and 33% of Goal and Long episodes end because the context reaches the 256k budget.

## Robustness

PEAR loses about 12 points under perturbation. π₀.₅ loses 11.2 on LIBERO-Plus (observation changes)
but **43.6 on LIBERO-Pro** (task and layout changes). That's because it replays the motion it was shown,
while PEAR reads the instruction and the scene. The largest single LIBERO-Plus sensitivity is the
**robot's initial state** (−36.5 points).

## What failures look like

- Real robot (final model, 12 failed trials): ran out of rounds in 6, blocks not aligned in 3,
  knocked the object over in 2, ball rolled away in 1. Successful trials took 7–24 rounds and 23k–132k tokens of context.
- Full RoboDojo (54 task settings): 5.50% average success, but 68.5% of failures are **context
  exhaustion**, which points to longer-context models or context management.

## Limitations stated by the authors

- **Scale**: only two simulators, a few tasks, and a 27B base, because training stopped for lack of GPUs.
- **Speed**: long reasoning before every move, with no length penalty during RL. Some rounds reach 32k tokens.
- **Small base model**: it lacks embodiment knowledge, so the prompt has to supply it.
- **Single-pose actions**: no action chunks, so results depend on controller quality, and complex
  motions take many rounds.
