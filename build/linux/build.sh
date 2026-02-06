#!/usr/bin/env bash
# Aydin PyInstaller build script for Linux
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Aydin Linux Build ==="
echo "Working directory: $(pwd)"

# Get version dynamically
VERSION=$(python3 -c "import aydin; print(aydin.__version__)")
echo "Building Aydin v${VERSION}..."

echo ""
echo "Cleaning old build artifacts..."
rm -rf build dist

echo ""
echo "Checking problematic packages..."
pip uninstall -y enum34 2>/dev/null || true
pip uninstall -y imagecodecs 2>/dev/null || true

echo ""
echo "Building with PyInstaller..."
pyinstaller -y --clean aydin.spec

echo ""
echo "Copying launcher script..."
cp run_aydin.sh dist/

echo ""
echo "Fixing numba jitclass _box binary..."
# Find and copy the _box binary to the correct location
NUMBA_PATH=$(python3 -c "import numba; import os; print(os.path.dirname(numba.__file__))")
BOX_SRC="${NUMBA_PATH}/experimental/jitclass/_box"*.so
BOX_DST="dist/aydin/_internal/numba/experimental/jitclass/"
mkdir -p "$BOX_DST"
cp $BOX_SRC "$BOX_DST" 2>/dev/null && echo "Copied _box binary" || echo "Warning: _box binary not found"

echo ""
echo "=== Build complete! ==="
echo "Output directory: $(pwd)/dist/aydin"
echo ""
echo "To run Aydin:"
echo "  cd dist && ./run_aydin.sh"
echo "  # or: ./dist/aydin/aydin"
