.PHONY: help install test train benchmark clean

help:
	@echo "Distributed LLM Training Commands:"
	@echo "  make install    - Setup venv and install dependencies"
	@echo "  make test       - Run all unit tests with pytest"
	@echo "  make train      - Launch 2-worker distributed training"
	@echo "  make clean      - Clean cache and checkpoint files"

install:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=. ./.venv/bin/pytest tests/ -v

train:
	./run_distributed.sh 2 20

clean:
	rm -rf checkpoints/ .pytest_cache/ __pycache__ dist_llm/**/__pycache__

