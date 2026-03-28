.PHONY: help install format lint test clean all

help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  format     - Format code with black and isort"
	@echo "  lint       - Run linters (ruff, mypy)"
	@echo "  test       - Run tests with coverage"
	@echo "  clean      - Remove build artifacts and cache files"
	@echo "  all        - Run format, lint, and test"

install:
	pip install -r requirements.txt
	pip install black isort ruff mypy pytest pytest-cov

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint:
	ruff check src/ tests/ scripts/
	mypy src/ tests/ scripts/

test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

all: format lint test
	@echo "All checks passed!"
