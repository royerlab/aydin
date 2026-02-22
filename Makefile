.PHONY: help setup install install-dev test test-cov test-cov-check test-heavy test-gpu test-unstable test-gui check format format-check lint validate clean build publish publish-patch docs docs-screenshots docs-build docs-publish docker-build docker-build-cli docker-build-gpu docker-build-studio docker-run-studio docker-test docker-test-all installer installer-icons installer-env installer-clean

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
	@echo "  make validate      - Run pre-publish checks (format + lint + clean tree)"
	@echo "  make build         - Build package"
	@echo "  make clean         - Clean artifacts"
	@echo "  make publish       - Bump version and create release PR"
	@echo "  make publish-patch - Bump patch version and create release PR"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build        - Build all Docker images"
	@echo "  make docker-build-cli    - Build CLI image (CPU)"
	@echo "  make docker-build-gpu    - Build GPU image"
	@echo "  make docker-build-studio - Build Studio GUI image"
	@echo "  make docker-run-studio   - Run Aydin Studio at http://localhost:9876"
	@echo "  make docker-test         - Build + smoke test CLI image"
	@echo "  make docker-test-all     - Build + smoke test CLI + Studio images"
	@echo ""
	@echo "Conda Packaging:"
	@echo "  make installer       - Build native installer for current platform"
	@echo "  make installer-icons - Generate .ico and .icns from source PNG"
	@echo "  make installer-env   - Create conda env with build tools"
	@echo "  make installer-clean - Clean installer build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build HTML docs (regenerates screenshots first)"
	@echo "  make docs-screenshots - Regenerate napari plugin screenshots only"
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
	pytest src/aydin/ --disable-pytest-warnings --durations=30

test-cov:
	pytest src/aydin/ --cov=src/aydin --cov-report=term-missing --cov-report=html:reports/coverage --cov-report=xml --disable-pytest-warnings --durations=30

test-cov-check:
	pytest src/aydin/ --cov=src/aydin --cov-report=term-missing --cov-fail-under=40 --disable-pytest-warnings -q

test-heavy:
	pytest src/aydin/ --runheavy --disable-pytest-warnings

test-gpu:
	pytest src/aydin/ --rungpu --disable-pytest-warnings

test-unstable:
	pytest src/aydin/ --rununstable --disable-pytest-warnings

test-gui:
	pytest src/aydin/ --rungui --disable-pytest-warnings --durations=30

check: format-check lint

format:
	isort src/aydin/
	black src/aydin/

format-check:
	black --check src/aydin/
	isort --check-only src/aydin/

lint:
	flake8 src/aydin/

build: clean
	hatch build

clean:
	hatch clean 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Documentation
docs-screenshots:
	cd docs && $(MAKE) screenshots

docs: ## Build HTML docs (regenerates screenshots first)
	cd docs && $(MAKE) build

docs-build:
	cd docs && $(MAKE) publish

docs-publish: docs-build
	@command -v ghp-import >/dev/null 2>&1 || { echo "Installing ghp-import..."; pip install ghp-import; }
	ghp-import -n -p -b docs-prod -r upstream -m "docs updated" docs/build/html
	@echo "Docs deployed to https://royerlab.github.io/aydin/"

# Pre-publish validation
validate:
	@echo "==> Checking on main branch..."
	@[ "$$(git branch --show-current)" = "main" ] || { echo "Error: Must be on main branch."; exit 1; }
	@echo "==> Checking working tree is clean..."
	@git diff --quiet && git diff --cached --quiet || { echo "Error: Working tree has uncommitted changes."; exit 1; }
	@echo "==> Checking main is up-to-date with origin..."
	@git fetch origin main --quiet 2>/dev/null || true
	@[ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main 2>/dev/null || git rev-parse HEAD)" ] || { echo "Error: Local main differs from origin/main. Run 'git pull' or 'git push'."; exit 1; }
	@echo "==> Checking code formatting..."
	@black --check src/aydin/ --quiet || { echo "Error: Code not formatted. Run 'make format'."; exit 1; }
	@isort --check-only src/aydin/ --quiet || { echo "Error: Imports not sorted. Run 'make format'."; exit 1; }
	@echo "==> Checking lint..."
	@flake8 src/aydin/ || { echo "Error: Lint errors found."; exit 1; }
	@echo "==> All pre-publish checks passed."

# Version format: YYYY.M.D or YYYY.M.D.patch
# Single source of truth: src/aydin/__init__.py (pyproject.toml reads it dynamically)
CURRENT_VERSION := $(shell grep -o '__version__ = "[^"]*"' src/aydin/__init__.py | cut -d'"' -f2)
TODAY := $(shell date +%Y.%-m.%-d)

publish: validate
	@echo "Current version: $(CURRENT_VERSION)"
	@case "$(CURRENT_VERSION)" in $(TODAY)*) echo "Error: Already at today's date. Use publish-patch."; exit 1;; esac
	@command -v gh >/dev/null 2>&1 || { echo "Error: GitHub CLI (gh) is required. Install from https://cli.github.com"; exit 1; }
	@NEW_VERSION="$(TODAY)"; \
	BRANCH="release/v$$NEW_VERSION"; \
	echo "Creating release branch $$BRANCH..."; \
	git rev-parse --verify "$$BRANCH" >/dev/null 2>&1 && { echo "Error: Branch $$BRANCH already exists. Delete it or use publish-patch."; exit 1; }; \
	git checkout -b "$$BRANCH"; \
	sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"$$NEW_VERSION\"/" src/aydin/__init__.py && rm -f src/aydin/__init__.py.bak; \
	git add src/aydin/__init__.py; \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	git push -u origin "$$BRANCH"; \
	gh pr create \
		--title "Release v$$NEW_VERSION" \
		--body "Automated version bump to $$NEW_VERSION. Tag will be created automatically after merge." \
		--base main; \
	git checkout main; \
	echo ""; \
	echo "Done! Release PR created."; \
	echo "Merge it on GitHub, then release.yml will auto-tag and publish to PyPI."

# Docker
# PyQt6 only has wheels for linux/amd64, so we target that platform.
# On Apple Silicon, Docker uses QEMU emulation transparently.
DOCKER_REPO ?= ghcr.io/royerlab
DOCKER_PLATFORM ?= linux/amd64

docker-build: docker-build-cli docker-build-gpu docker-build-studio

docker-build-cli:
	docker build --platform $(DOCKER_PLATFORM) --target aydin -t $(DOCKER_REPO)/aydin:latest -t $(DOCKER_REPO)/aydin:$(CURRENT_VERSION) .

docker-build-gpu:
	docker build --platform $(DOCKER_PLATFORM) --target aydin-gpu -t $(DOCKER_REPO)/aydin:gpu -t $(DOCKER_REPO)/aydin:$(CURRENT_VERSION)-gpu .

docker-build-studio:
	docker build --platform $(DOCKER_PLATFORM) --target aydin-studio -t $(DOCKER_REPO)/aydin-studio:latest -t $(DOCKER_REPO)/aydin-studio:$(CURRENT_VERSION) .

docker-run-studio:
	@echo "Starting Aydin Studio at http://localhost:9876"
	docker run --rm -p 9876:9876 --shm-size=256m -v $$(pwd)/data:/data $(DOCKER_REPO)/aydin-studio:latest

docker-test:
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker is not installed."; exit 1; }
	./docker/test-smoke.sh

docker-test-all:
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker is not installed."; exit 1; }
	./docker/test-smoke.sh --all

# Conda packaging (native installers via conda-constructor)
installer: installer-icons
	python packaging/build_installer.py --output-dir _work

installer-icons:
	python packaging/scripts/generate_icons.py

installer-env:
	conda env create -f packaging/environments/build_installer.yml --force

installer-clean:
	rm -rf _work/

publish-patch: validate
	@echo "Current version: $(CURRENT_VERSION)"
	@command -v gh >/dev/null 2>&1 || { echo "Error: GitHub CLI (gh) is required. Install from https://cli.github.com"; exit 1; }
	@case "$(CURRENT_VERSION)" in \
		$(TODAY).*) \
			PATCH=$$(echo "$(CURRENT_VERSION)" | sed 's/.*\.//'); \
			NEW_PATCH=$$((PATCH + 1)); \
			NEW_VERSION="$(TODAY).$$NEW_PATCH";; \
		$(TODAY)) \
			NEW_VERSION="$(TODAY).1";; \
		*) \
			echo "Error: Current version is not today's date. Use 'make publish' first."; \
			exit 1;; \
	esac; \
	BRANCH="release/v$$NEW_VERSION"; \
	echo "Creating release branch $$BRANCH..."; \
	git rev-parse --verify "$$BRANCH" >/dev/null 2>&1 && { echo "Error: Branch $$BRANCH already exists. Delete it first."; exit 1; }; \
	git checkout -b "$$BRANCH"; \
	sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"$$NEW_VERSION\"/" src/aydin/__init__.py && rm -f src/aydin/__init__.py.bak; \
	git add src/aydin/__init__.py; \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	git push -u origin "$$BRANCH"; \
	gh pr create \
		--title "Release v$$NEW_VERSION" \
		--body "Automated version bump to $$NEW_VERSION. Tag will be created automatically after merge." \
		--base main; \
	git checkout main; \
	echo ""; \
	echo "Done! Release PR created."; \
	echo "Merge it on GitHub, then release.yml will auto-tag and publish to PyPI."
