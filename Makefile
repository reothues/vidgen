.PHONY: install format lint test train

install:
	python -m venv vidgen-env || true
	./vidgen-env/bin/pip install --upgrade pip
	./vidgen-env/bin/pip install -e .[dev]

format:
	./vidgen-env/bin/ruff check --select I --fix src tests
	./vidgen-env/bin/ruff format src tests

lint:
	./vidgen-env/bin/ruff check src tests
	./vidgen-env/bin/mypy src

test:
	./vidgen-env/bin/pytest

train:
	./vidgen-env/bin/python -m vidgen.cli train --config configs/default.yaml
