.PHONY: help setup install install-dev test test-cov test-heavy test-gpu test-unstable check format format-check lint clean build publish publish-patch

help:
	@echo "Available commands:"
	@echo "  make setup         - Install with all dev dependencies"
	@echo "  make install       - Install package only"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make test          - Run standard tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make test-heavy    - Run heavy tests only"
	@echo "  make test-gpu      - Run GPU tests only"
	@echo "  make test-unstable - Run unstable tests only"
	@echo "  make check         - Run all checks (format + lint)"
	@echo "  make format        - Format code"
	@echo "  make lint          - Run linter"
	@echo "  make build         - Build package"
	@echo "  make clean         - Clean artifacts"
	@echo "  make publish       - Bump version and publish"

setup:
	python -m pip install --upgrade pip
	pip install -e ".[dev,docs]"
	pre-commit install

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest aydin/ --disable-pytest-warnings --durations=30

test-cov:
	pytest aydin/ --cov=aydin --cov-report=html:reports/coverage --cov-report=xml

test-heavy:
	pytest aydin/ --runheavy --disable-pytest-warnings

test-gpu:
	pytest aydin/ --rungpu --disable-pytest-warnings

test-unstable:
	pytest aydin/ --rununstable --disable-pytest-warnings

check: format-check lint

format:
	isort aydin/
	black aydin/

format-check:
	black --check aydin/
	isort --check-only aydin/

lint:
	flake8 --ignore=E501,E203,E741,W503 aydin/

build: clean
	hatch build

clean:
	hatch clean 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Version format: YYYY.M.D or YYYY.M.D.patch
CURRENT_VERSION := $(shell grep -o '__version__ = "[^"]*"' aydin/__init__.py | cut -d'"' -f2)
TODAY := $(shell date +%Y.%-m.%-d)

publish:
	@echo "Current version: $(CURRENT_VERSION)"
	@if echo "$(CURRENT_VERSION)" | grep -q "^$(TODAY)"; then \
		echo "Error: Already at today's date. Use publish-patch."; \
		exit 1; \
	fi
	@sed -i.bak 's/__version__ = "[^"]*"/__version__ = "$(TODAY)"/' aydin/__init__.py && rm -f aydin/__init__.py.bak
	@sed -i.bak 's/^version = "[^"]*"/version = "$(TODAY)"/' pyproject.toml && rm -f pyproject.toml.bak
	git add aydin/__init__.py pyproject.toml
	git commit -m "chore: bump version to $(TODAY)"
	git tag "v$(TODAY)"
	git push origin master --tags
	@echo "Done! GitHub Actions will publish to PyPI."

publish-patch:
	@echo "Current version: $(CURRENT_VERSION)"
	@if echo "$(CURRENT_VERSION)" | grep -q "^$(TODAY)\."; then \
		PATCH=$$(echo "$(CURRENT_VERSION)" | sed 's/.*\.//'); \
		NEW_PATCH=$$((PATCH + 1)); \
		NEW_VERSION="$(TODAY).$$NEW_PATCH"; \
	elif echo "$(CURRENT_VERSION)" | grep -q "^$(TODAY)$$"; then \
		NEW_VERSION="$(TODAY).1"; \
	else \
		echo "Error: Current version is not today's date. Use publish first."; \
		exit 1; \
	fi; \
	sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"$$NEW_VERSION\"/" aydin/__init__.py && rm -f aydin/__init__.py.bak; \
	sed -i.bak "s/^version = \"[^\"]*\"/version = \"$$NEW_VERSION\"/" pyproject.toml && rm -f pyproject.toml.bak; \
	git add aydin/__init__.py pyproject.toml; \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	git tag "v$$NEW_VERSION"; \
	git push origin master --tags; \
	echo "Done! GitHub Actions will publish to PyPI."
