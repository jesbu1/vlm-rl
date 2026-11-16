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
smoke:         ## 1-step training run with the tiny VLM
	$(PY) scripts/train.py --model HuggingFaceTB/SmolVLM-256M-Instruct --processor_kwargs '{"do_image_splitting": false}' \
	  --env_kwargs '{"maze":"umaze","max_turns":2,"image_size":64}' --steps 1 --groups_per_step 1 --group_size 2 \
	  --max_new_tokens 16 --eval_every 1 --eval_episodes 2 --filter_groups false --reward neg_dist --out_dir runs/smoke
