#!/bin/bash
# Post-installation script for Aydin Studio (Linux / macOS)
#
# This runs inside the installed conda environment after constructor
# finishes extracting and linking all conda packages.
#
# Use this to pip-install any packages not yet available on conda-forge.
# Once all dependencies are on conda-forge, this script becomes a no-op.

set -euo pipefail

echo "Aydin Studio post-install: finalizing installation..."

# --- pip fallback for packages not yet on conda-forge ---
# Uncomment lines below if needed during the bootstrap period:
# "${PREFIX}/bin/pip" install --no-deps --quiet arbol
# "${PREFIX}/bin/pip" install --no-deps --quiet czifile

echo ""
echo "================================================================"
echo "  Aydin Studio installation complete!"
echo ""
echo "  Launch from your Applications menu, or run:"
echo "    ${PREFIX}/bin/aydin"
echo ""
echo "  For GPU support (NVIDIA), activate the environment and run:"
echo "    conda install pytorch-cuda=12.6 -c conda-forge"
echo "================================================================"
echo ""
