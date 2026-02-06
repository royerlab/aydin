@echo off
REM Aydin PyInstaller build script for Windows
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo === Aydin Windows Build ===
echo Working directory: %CD%

REM Get version dynamically
for /f "tokens=*" %%i in ('python -c "import aydin; print(aydin.__version__)"') do set VERSION=%%i
echo Building Aydin v%VERSION%...

echo.
echo Cleaning old build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Checking problematic packages...
pip uninstall -y enum34 2>nul
pip uninstall -y imagecodecs 2>nul

echo.
echo Building with PyInstaller...
pyinstaller.exe -y --clean aydin.spec

echo.
echo Copying launcher script...
copy run_aydin.bat dist\

echo.
echo === Build complete! ===
echo Output directory: %CD%\dist\aydin
echo.
echo To run Aydin:
echo   cd dist ^&^& run_aydin.bat
echo   or: dist\aydin\aydin.exe

endlocal
