version=`python3 -c "import os, sys;tmp = sys.stdout;sys.stdout = open(os.devnull,'w');sys.stderr= open(os.devnull,'w');import aydin;sys.stdout = tmp;print(aydin.__version__)"`

del -rf build
del -rf dist

pip uninstall imagecodecs
pip uninstall enum34

pyinstaller.exe -w -D -y --clean aydin.spec

copy run_aydin.bat dist\run_aydin.bat
mkdir -p dist\aydin\numba\experimental\jitclass
copy \PATH\TO\numba\experimental\jitclass\_box.cp39-win_amd64.pyd dist\aydin\numba\experimental\jitclass\_box.cp39-win_amd64.pyd