PY ?= .venv/bin/python

.PHONY: test test-slow test-all demo smoke
test:          ## fast tests (CPU, no downloads, ~30s)
	$(PY) -m pytest
test-slow:     ## + real tiny VLM (SmolVLM-256M; downloads ~500MB once)
	$(PY) -m pytest --run-slow
test-all:      ## + CUDA tests
	$(PY) -m pytest --run-slow --run-gpu
demo:          ## oracle/random policy through the text interface, writes GIFs to outputs/demo
	$(PY) scripts/demo_pointmaze.py --maze medium --policy oracle
smoke:         ## 1-step training run with the tiny VLM (Hydra experiment=smoke)
	$(PY) scripts/train.py experiment=smoke
