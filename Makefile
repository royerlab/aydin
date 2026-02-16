.PHONY: help setup install install-dev test test-cov test-cov-check test-heavy test-gpu test-unstable test-gui check format format-check lint clean build publish publish-patch docs docs-build docs-publish

help:
	@echo "Available commands:"
	@echo "  make setup         - Install with all dev dependencies"
	@echo "  make install       - Install package only"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make test          - Run standard tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make test-cov-check - Run tests with coverage threshold check"
	@echo "  make test-heavy    - Run heavy tests only"
	@echo "  make test-gpu      - Run GPU tests only"
	@echo "  make test-unstable - Run unstable tests only"
	@echo "  make test-gui      - Run GUI tests only (requires display or Xvfb)"
	@echo "  make check         - Run all checks (format + lint)"
	@echo "  make format        - Format code (isort + black)"
	@echo "  make format-check  - Check formatting without modifying"
	@echo "  make lint          - Run linter (flake8)"
	@echo "  make build         - Build package"
	@echo "  make clean         - Clean artifacts"
	@echo "  make publish       - Bump version and publish"
	@echo "  make publish-patch - Bump patch version and publish"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build HTML docs (current version only)"
	@echo "  make docs-build    - Build multi-version docs (all tags)"
	@echo "  make docs-publish  - Build multi-version docs and deploy to GitHub Pages"

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
	pytest aydin/ --cov=aydin --cov-report=term-missing --cov-report=html:reports/coverage --cov-report=xml --disable-pytest-warnings --durations=30

test-cov-check:
	pytest aydin/ --cov=aydin --cov-report=term-missing --disable-pytest-warnings -q

test-heavy:
	pytest aydin/ --runheavy --disable-pytest-warnings

test-gpu:
	pytest aydin/ --rungpu --disable-pytest-warnings

test-unstable:
	pytest aydin/ --rununstable --disable-pytest-warnings

test-gui:
	pytest aydin/ --rungui --disable-pytest-warnings --durations=30

check: format-check lint

format:
	isort aydin/
	black aydin/

format-check:
	black --check aydin/
	isort --check-only aydin/

lint:
	flake8 --ignore=E501,E203,E741,W503,E402,F401,E721,E275,E731,E226,F821 aydin/

build: clean
	hatch build

clean:
	hatch clean 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Documentation
docs:
	cd docs && $(MAKE) build

docs-build:
	cd docs && $(MAKE) publish

docs-publish: docs-build
	@command -v ghp-import >/dev/null 2>&1 || { echo "Installing ghp-import..."; pip install ghp-import; }
	ghp-import -n -p -b docs-prod -r upstream -m "docs updated" docs/build/html
	@echo "Docs deployed to https://royerlab.github.io/aydin/"

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
	git push origin main --tags
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
	git push origin main --tags; \
	echo "Done! GitHub Actions will publish to PyPI."
