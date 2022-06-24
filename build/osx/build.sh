#!/usr/bin/env bash
echo "removing old files..."
rm -rf build
rm -rf dist

# Check if error introducing packages are still there
pip uninstall enum34
pip uninstall imagecodecs

echo "building app..."
#onefolder
pyinstaller -y --clean aydin.spec # -D -y --clean


mkdir -p dist/aydin_0.1.5rc12.app/Contents/MacOS
mkdir -p dist/aydin_0.1.5rc12.app/Contents/Resources

cp /PATH/TO/sklearn/cluster/*.so dist/aydin/sklearn/cluster/.
cp Info.plist dist/aydin_0.1.5rc12.app/Contents/.
cp icon-windowed.icns dist/aydin_0.1.5rc12.app/Contents/Resources/.
cp -rf -p dist/aydin/* dist/aydin_0.1.5rc12.app/Contents/MacOS/.
cp aydin_0.1.5rc3 dist/aydin_0.1.5rc12.app/Contents/MacOS/.
