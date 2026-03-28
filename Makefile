.PHONY: help sync format lint test clean all

help:
	@echo "Available targets:"
	@echo "  sync       - Sync dependencies with uv"
	@echo "  format     - Format code with ruff"
	@echo "  lint       - Run linters (ruff, mypy)"
	@echo "  test       - Run tests with coverage"
	@echo "  clean      - Remove build artifacts and cache files"
	@echo "  all        - Run sync, format, lint, and test"

sync:
	uv sync --dev

format:
	uv run ruff format src/ tests/ scripts/

lint:
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/ tests/ scripts/

test:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

all: sync format lint test
	@echo "All checks passed!"
