.PHONY: install test lint check clean

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

check: lint
	uv run mypy src tests
	uv run pytest

clean:
	rm -rf .venv dist build .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
