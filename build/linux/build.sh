#!/usr/bin/env bash

echo "removing old files..."
rm -rf build
rm -rf dist

# Check if error introducing packages are still there
pip uninstall enum34
pip uninstall imagecodecs


echo "building app..."
#onefile
pyinstaller -w -F -y --clean aydin.spec

mkdir -p dist/aydin/numba/experimental/jitclass
cp /PATH/TO/numba/experimental/jitclass/_box.cpython-39-x86_64-linux-gnu.so dist/aydin/numba/experimental/jitclass/.
cp run_aydin.sh dist/.
