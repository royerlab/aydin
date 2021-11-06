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

cp run_aydin.sh dist/.
