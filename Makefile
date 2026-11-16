PY ?= .venv/bin/python

.PHONY: test test-slow test-all demo smoke docs docs-serve docs-gen
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
docs-gen:      ## regenerate docs pages derived from code (config reference, prompts)
	$(PY) scripts/gen_docs.py
docs:          ## strict build of the docs site into site/
	$(PY) scripts/gen_docs.py --check && $(PY) -m mkdocs build --strict
docs-serve:    ## live-preview the docs at http://127.0.0.1:8000
	$(PY) -m mkdocs serve
