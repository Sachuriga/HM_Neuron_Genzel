@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  HM_Tracker_2025 - Environment Setup
echo ============================================================
echo.

:: ---- Step 1: create conda environment ----
echo [1/3] Creating conda environment (Python 3.10)...
call conda env create -f reproduce.yml
if errorlevel 1 (
    echo.
    echo [ERROR] conda env create failed.
    echo   - If the environment already exists, remove it first:
    echo       conda env remove -n HM_neuron
    echo   - If conda is not on PATH, open an Anaconda Prompt and re-run this script.
    pause
    exit /b 1
)
echo.

:: ---- Step 2: activate ----
echo [2/3] Activating HM_neuron...
call conda activate HM_neuron
if errorlevel 1 (
    echo.
    echo [ERROR] Could not activate HM_neuron.
    echo   Try running:  conda activate HM_neuron
    echo   then:         pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

:: ---- Step 3: pip install ----
echo [3/3] Installing Python packages (this may take 10-20 minutes)...
echo       PyTorch (CUDA 12.8) will be fetched from download.pytorch.org
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed.
    echo   - Check your internet connection.
    echo   - If a specific package is failing, install it manually:
    echo       pip install ^<package^>
    echo   - For torch specifically:
    echo       pip install torch==2.10.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Installation complete.
echo  Activate the environment with:  conda activate HM_neuron
echo ============================================================
pause
