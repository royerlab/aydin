#!/bin/bash
# =============================================================================
# Docker smoke tests for Aydin
#
# Validates that Docker images build correctly and basic commands work.
# Requires Docker to be installed and running.
#
# Usage:
#   ./docker/test-smoke.sh              # Test CLI image only (fast)
#   ./docker/test-smoke.sh --all        # Test CLI + Studio images
#   ./docker/test-smoke.sh --studio     # Test Studio image only
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
DIM='\033[2m'
RESET='\033[0m'

PASS=0
FAIL=0
CLI_IMAGE="aydin:smoke-test"
STUDIO_IMAGE="aydin-studio:smoke-test"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
TEST_CLI=true
TEST_STUDIO=false

for arg in "$@"; do
    case "$arg" in
        --all)    TEST_STUDIO=true ;;
        --studio) TEST_CLI=false; TEST_STUDIO=true ;;
    esac
done

pass() { echo -e "  ${GREEN}PASS${RESET}  $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${RESET}  $1"; FAIL=$((FAIL + 1)); }

run_test() {
    local name="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        pass "$name"
    else
        fail "$name"
    fi
}

run_test_output() {
    local name="$1"
    local pattern="$2"
    shift 2
    local output
    if output=$("$@" 2>&1) && echo "$output" | grep -qi "$pattern"; then
        pass "$name"
    else
        fail "$name"
        echo -e "    ${DIM}expected pattern: $pattern${RESET}"
    fi
}

# ---- CLI image tests --------------------------------------------------------
if $TEST_CLI; then
    echo ""
    echo "Building CLI image..."
    if ! docker build --platform "$PLATFORM" --target aydin -t "$CLI_IMAGE" . > /dev/null 2>&1; then
        fail "docker build --target aydin"
        echo -e "  ${RED}Build failed — cannot run CLI tests.${RESET}"
    else
        pass "docker build --target aydin"

        echo ""
        echo "CLI smoke tests:"

        run_test_output \
            "aydin --help" \
            "usage" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" --help

        run_test_output \
            "aydin --version" \
            "[0-9]" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" --version

        run_test_output \
            "aydin cite" \
            "zenodo" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" cite

        run_test_output \
            "aydin denoise --help" \
            "denoise" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" denoise --help

        run_test_output \
            "aydin info --help" \
            "info" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" info --help

        run_test_output \
            "aydin psnr --help" \
            "psnr" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" psnr --help

        run_test_output \
            "aydin ssim --help" \
            "ssim" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" ssim --help

        run_test_output \
            "aydin denoise --list-denoisers" \
            "noise2self" \
            docker run --rm --platform "$PLATFORM" "$CLI_IMAGE" denoise --list-denoisers

        run_test \
            "python -c 'import aydin'" \
            docker run --rm --platform "$PLATFORM" --entrypoint python "$CLI_IMAGE" -c "import aydin; print(aydin.__version__)"

        run_test \
            "python -c 'import torch'" \
            docker run --rm --platform "$PLATFORM" --entrypoint python "$CLI_IMAGE" -c "import torch; print(torch.__version__)"

        # Denoise a synthetic image (end-to-end)
        echo ""
        echo "End-to-end denoise test (Butterworth on synthetic noise):"
        run_test \
            "denoise synthetic image (Butterworth)" \
            docker run --rm --platform "$PLATFORM" --entrypoint python "$CLI_IMAGE" -c "
import numpy as np, tempfile, os
from skimage.data import camera
from skimage.io import imsave, imread
d = tempfile.mkdtemp()
img = camera().astype(np.float32)
noisy = img + np.random.normal(0, 25, img.shape).astype(np.float32)
path_in = os.path.join(d, 'noisy.tif')
imsave(path_in, noisy)
os.system(f'aydin denoise {path_in} -d classic-butterworth')
files = [f for f in os.listdir(d) if 'denoised' in f]
assert len(files) == 1, f'Expected 1 denoised file, got {files}'
result = imread(os.path.join(d, files[0]))
assert result.shape == img.shape, f'Shape mismatch: {result.shape} vs {img.shape}'
print('OK')
"
    fi
fi

# ---- Studio image tests -----------------------------------------------------
if $TEST_STUDIO; then
    echo ""
    echo "Building Studio image..."
    if ! docker build --platform "$PLATFORM" --target aydin-studio -t "$STUDIO_IMAGE" . > /dev/null 2>&1; then
        fail "docker build --target aydin-studio"
        echo -e "  ${RED}Build failed — cannot run Studio tests.${RESET}"
    else
        pass "docker build --target aydin-studio"

        echo ""
        echo "Studio smoke tests:"

        run_test \
            "xpra is installed" \
            docker run --rm --platform "$PLATFORM" --entrypoint xpra "$STUDIO_IMAGE" --version

        run_test \
            "Xvfb is installed" \
            docker run --rm --platform "$PLATFORM" --entrypoint which "$STUDIO_IMAGE" Xvfb

        run_test \
            "/usr/share/xpra/www/ exists (HTML5 client)" \
            docker run --rm --platform "$PLATFORM" --entrypoint test "$STUDIO_IMAGE" -d /usr/share/xpra/www

        run_test \
            "startup script exists and is executable" \
            docker run --rm --platform "$PLATFORM" --entrypoint test "$STUDIO_IMAGE" -x /usr/local/bin/xpra-start.sh

        # Start studio, wait for Xpra to listen, then stop
        echo ""
        echo "Studio startup test (start, verify port, stop):"
        CONTAINER_ID=$(docker run -d --rm --platform "$PLATFORM" -p 19876:9876 --shm-size=256m "$STUDIO_IMAGE" 2>/dev/null)
        if [ -z "$CONTAINER_ID" ]; then
            fail "start studio container"
        else
            # Wait for xpra to start listening (up to 30 seconds)
            STARTED=false
            for i in $(seq 1 30); do
                if curl -sf http://localhost:19876/ > /dev/null 2>&1; then
                    STARTED=true
                    break
                fi
                sleep 1
            done

            if $STARTED; then
                pass "xpra HTML5 client responds on port 19876"
            else
                fail "xpra HTML5 client responds on port 19876 (timed out after 30s)"
            fi

            docker stop "$CONTAINER_ID" > /dev/null 2>&1 || true
        fi
    fi
fi

# ---- Summary ----------------------------------------------------------------
echo ""
echo "========================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================"
echo ""

[ "$FAIL" -eq 0 ] || exit 1
