@echo off
setlocal EnableDelayedExpansion

:: ========================================================
::               CONFIGURATION (SHARED)
:: ========================================================
cd /d "%~dp0"

:: Thresholds
set MAX_CPU=90
set MAX_GPU=90
set WAIT_SECONDS=10

:: Paths
set "ONNX_WEIGHTS_PATH=C:\Users\gl_pc\Desktop\track\yolov3_training_best.onnx"
set "TRODES_EXPORT_CMD=C:\Users\gl_pc\Desktop\track\trodes\trodesexport.exe"
set FREQ=30000

:: ========================================================
::           MODE CHECK: MASTER OR WORKER?
:: ========================================================
if "%~1"==":WORKER" goto :WORKER_ROUTINE

echo ========================================================
echo          SMART PARALLEL DEBUG MODE (Resource Aware)
echo ========================================================
:: FIXED LINE BELOW: Added ^ before |
echo [CONFIG] Max CPU: %MAX_CPU%%% ^| Max GPU: %MAX_GPU%%%
echo.

:: 3. Handle Input (Master Mode)
if "%~1"=="" (
    echo Usage: runner_windows.bat "path_to_data_folder"
    exit /b 1
)

pushd "%~1"
set "ROOT_DIR=%CD%"
popd
echo [DEBUG] Target Root Directory: [%ROOT_DIR%]

:: 4. Scan Loop (Master Mode)
set count=0

for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    set "NUM=!DIR_NAME:ip=!"
    set "OP_PATH=%ROOT_DIR%\op!NUM!"

    if exist "!OP_PATH!\" (
        
        echo.
        echo [QUEUE] Preparing to launch: !DIR_NAME!
        
        :: >>> RESOURCE CHECK BEFORE LAUNCH <<<
        call :WAIT_FOR_RESOURCES
        
        set /a count+=1
        echo [LAUNCH] System clear. Spawning worker for !DIR_NAME!
        start "Job-!DIR_NAME!" cmd /k call "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!"
        
        :: Small safety buffer to let the new process register its load
        timeout /t 3 /nobreak >nul
        
    ) else (
        echo [SKIP] !OP_PATH! does not exist.
    )
)

echo.
echo ========================================================
echo [MASTER] Launched !count! jobs.
echo ========================================================
pause
exit /b

:: ========================================================
::            RESOURCE MONITOR SUBROUTINE
:: ========================================================
:WAIT_FOR_RESOURCES
:CHECK_AGAIN
    :: 1. GET CPU LOAD
    set CPU_LOAD=0
    for /f "skip=1" %%P in ('wmic cpu get loadpercentage') do (
        if "%%P" neq "" set CPU_LOAD=%%P
        goto :break_cpu
    )
    :break_cpu

    :: 2. GET GPU LOAD (Requires nvidia-smi)
    set GPU_LOAD=0
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits > gpu_temp.txt 2>nul
    if exist gpu_temp.txt (
        set /p GPU_LOAD=<gpu_temp.txt
        del gpu_temp.txt
    )
    
    :: Handle case where GPU might return multiple lines or empty
    if "%GPU_LOAD%"=="" set GPU_LOAD=0

    :: 3. COMPARE
    if !CPU_LOAD! GTR %MAX_CPU% (
        echo    [WAIT] High CPU Load: !CPU_LOAD!%%. Pausing %WAIT_SECONDS%s...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )

    if !GPU_LOAD! GTR %MAX_GPU% (
        echo    [WAIT] High GPU Load: !GPU_LOAD!%%. Pausing %WAIT_SECONDS%s...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )

    echo    [CHECK] CPU: !CPU_LOAD!%% ^| GPU: !GPU_LOAD!%% - OK.
exit /b

:: ========================================================
::               THE WORKER SUBROUTINE
:: ========================================================
:WORKER_ROUTINE
set "IP=%~2"
set "OP=%~3"
color 0A 

echo.
echo ^> ^> ^> WORKER STARTED ^< ^< ^<
echo [INFO] IP: %IP%
echo [INFO] OP: %OP%

REM 1. TRODES CHECK
echo.
echo [STEP 1] Running Trodes...
if exist "%TRODES_EXPORT_CMD%" (
    for %%F in ("%IP%\*.rec") do (
        "%TRODES_EXPORT_CMD%" -dio -rec "%%F"
    )
)

REM 2. SYNC CHECK
echo.
echo [STEP 2] Running Sync Script...
if exist ".\src\tracker\Video_LED_Sync_using_ICA.py" (
    python -u ./src/tracker/Video_LED_Sync_using_ICA.py -i "%IP%" -o "%OP%" -f %FREQ%
)

REM 3. STITCH CHECK
echo.
echo [STEP 3] Running Stitching...
if exist ".\src\tracker\join_views.py" (
    python -u ./src/tracker/join_views.py "%IP%"
)

REM 4. TRACKER CHECK
echo.
echo [STEP 4] Running Tracker...
if exist "%IP%\stitched.mp4" (
    if exist ".\src\tracker\TrackerYolov11.py" (
        python -u ./src/tracker/TrackerYolov11.py --input_folder "%IP%" --output_folder "%OP%" --onnx_weight "%ONNX_WEIGHTS_PATH%"
    )
)

REM 5. PLOT CHECK
echo.
echo [STEP 5] Running Plotting...
if exist ".\src\tracker\plot_trials.py" (
    python -u ./src/tracker/plot_trials.py -o "%OP%"
)

echo.
echo [COMPLETE] Worker finished.
timeout /t 5
exit