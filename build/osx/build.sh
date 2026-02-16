#!/usr/bin/env bash
# Aydin PyInstaller build script for macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Aydin macOS Build ==="
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
echo "=== Build complete! ==="
echo "Output directory: $(pwd)/dist"
echo ""
echo "App bundle: dist/Aydin.app"
echo "Directory build: dist/aydin/"
echo ""
echo "To run Aydin:"
echo "  open dist/Aydin.app"
echo "  # or: ./dist/aydin/aydin"
