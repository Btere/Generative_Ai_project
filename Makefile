.PHONY: help install test test-coverage lint format security clean build deploy-test all

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test: ## Run unit tests
	pytest test_portfolio_manager.py -v

test-coverage: ## Run tests with coverage report
	pytest test_portfolio_manager.py -v --cov=prototype --cov-report=html --cov-report=xml --cov-report=term-missing

test-watch: ## Run tests in watch mode
	pytest-watch test_portfolio_manager.py -v

lint: ## Run all linting checks
	@echo "Running Black..."
	black --check prototype.py test_portfolio_manager.py
	@echo "Running isort..."
	isort --check-only prototype.py test_portfolio_manager.py
	@echo "Running flake8..."
	flake8 prototype.py test_portfolio_manager.py --max-line-length=88 --extend-ignore=E203,W503
	@echo "Running pylint..."
	pylint prototype.py --fail-under=8.0 --disable=C0103,R0903,W0613,C0115

format: ## Auto-format code
	black prototype.py test_portfolio_manager.py
	isort prototype.py test_portfolio_manager.py

security: ## Run security checks
	@echo "Running Bandit security check..."
	bandit -r prototype.py
	@echo "Running Safety dependency check..."
	safety check

type-check: ## Run type checking
	mypy prototype.py --ignore-missing-imports

clean: ## Clean up generated files
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf chroma_db/
	rm -rf test_chroma_db/
	rm -f bandit-report.json
	rm -f safety-report.json

build: ## Build package
	python -m build

install-dev: ## Install development dependencies
	pip install pytest pytest-cov pytest-mock coverage black isort flake8 pylint mypy bandit safety build twine

setup-dev: install install-dev ## Setup development environment

ci-local: clean install-dev lint test-coverage security ## Run full CI pipeline locally

deploy-test: build ## Deploy to test PyPI
	twine upload --repository testpypi dist/*

all: clean setup-dev ci-local ## Run everything