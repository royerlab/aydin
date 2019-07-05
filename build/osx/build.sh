echo "removing old files..."
rm -rf build
rm -rf dist


echo "building app..."
#onefile
pyinstaller -w -F -y --clean pitl.spec


# echo "creating the dmg..."
# hdiutil create dist/spimagine_v${version}.dmg -srcfolder dist/spimagine.app/

