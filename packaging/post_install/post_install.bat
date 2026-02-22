@echo off
REM Post-installation script for Aydin Studio (Windows)
REM
REM This runs inside the installed conda environment after constructor
REM finishes extracting and linking all conda packages.
REM
REM Use this to pip-install any packages not yet available on conda-forge.
REM Once all dependencies are on conda-forge, this script becomes a no-op.

echo Aydin Studio post-install: finalizing installation...

REM --- pip fallback for packages not yet on conda-forge ---
REM Uncomment lines below if needed during the bootstrap period:
REM "%PREFIX%\Scripts\pip.exe" install --no-deps --quiet arbol
REM "%PREFIX%\Scripts\pip.exe" install --no-deps --quiet czifile

echo.
echo ================================================================
echo   Aydin Studio installation complete!
echo.
echo   Launch from your Start Menu or Desktop shortcut.
echo.
echo   For GPU support (NVIDIA), open Anaconda Prompt and run:
echo     conda install pytorch-cuda -c conda-forge
echo ================================================================
echo.
