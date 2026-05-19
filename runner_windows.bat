@echo off
setlocal EnableDelayedExpansion

:: ========================================================
::                CONFIGURATION (SHARED)
:: ========================================================
cd /d "%~dp0"

:: Thresholds
set MAX_CPU=90
set MAX_GPU=90
set MAX_MEM=65
set WAIT_SECONDS=30

:: Paths - loaded from Desktop config file
set "CONFIG_FILE=%USERPROFILE%\Desktop\hm_tracker_paths.txt"
if not exist "%CONFIG_FILE%" (
    echo [ERROR] Config file not found: %CONFIG_FILE%
    echo.
    echo Please create hm_tracker_paths.txt on your Desktop with these lines:
    echo   FFMPEG_CMD=C:\path\to\ffmpeg.exe
    echo   ONNX_WEIGHTS_PATH=C:\path\to\weights.pt
    echo   TRODES_EXPORT_CMD=C:\path\to\trodesexport.exe
    echo   TRODES_EXPORT_LFP=C:\path\to\exportLFP.exe
    echo   LFP_CHANNELS=1 2 3 4 5 6 7 8
    echo.
    echo See hm_tracker_paths.example.txt in the repo for a template.
    pause
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%A in ("%CONFIG_FILE%") do (
    if not "%%A"=="" if not "!A:~0,1!"=="#" set "%%A=%%B"
)
set FREQ=30000

:: ========================================================
::            MODE CHECK: MASTER OR WORKER?
:: ========================================================
:: If a 4th argument exists, it's the user's selection passed from Master
if "%~1"==":WORKER" (
    set "STEPS_TO_RUN=%~4"
    set "PENDING_FILE=%~5"
    goto :WORKER_ROUTINE
)

echo ========================================================
echo           SMART PARALLEL MODE (Multi-Step)
echo ========================================================
echo [CONFIG] Max CPU: %MAX_CPU%%% ^| Max GPU: %MAX_GPU%%%
echo.

:: 3. Handle Input (Master Mode)
if "%~1"=="" (
    echo Usage: runner_windows.bat "path_to_data_folder"
    exit /b 1
)

:: --- NEW: STEP SELECTION MENU ---
echo Select steps to run (e.g., 123 for steps 1, 2, and 3):
echo [1] Trodes Export (DIO/Raw)
echo [e] Trodes Export LFP (per channel)
echo [2] Sync Script
echo [3] Stitching
echo [4] Tracker
echo [5] Plotting
echo [6] Compression
echo [7] Sorting
echo [8] LFP
echo [d] deeplabcut
echo [9] Cleaning
echo [n] Node Analysis
echo.
set /p "MY_SELECTION=Enter steps: "

:: --- Steps 7 and 9 run after all parallel steps, sequentially ---
:: Strip "7" and "9" from the selection passed to parallel workers
set "PARALLEL_STEPS=%MY_SELECTION%"
set "HAS_SORT=0"
set "HAS_CLEAN=0"
echo %MY_SELECTION% | findstr "7" >nul
if %errorlevel% equ 0 (
    set "HAS_SORT=1"
    call set "PARALLEL_STEPS=%%PARALLEL_STEPS:7=%%"
)
echo %MY_SELECTION% | findstr "9" >nul
if %errorlevel% equ 0 (
    set "HAS_CLEAN=1"
    call set "PARALLEL_STEPS=%%PARALLEL_STEPS:9=%%"
)
REM Trim spaces so empty-check works. Guard against undefined PARALLEL_STEPS:
REM set "X=" deletes the variable, and substring substitution on an undefined
REM variable can return literal text instead of empty, falsely passing the
REM emptiness check that gates worker spawning.
set "PARALLEL_STEPS_TRIM="
if defined PARALLEL_STEPS set "PARALLEL_STEPS_TRIM=!PARALLEL_STEPS: =!"

pushd "%~1"
set "ROOT_DIR=%CD%"
popd
echo [DEBUG] Target Root Directory: [%ROOT_DIR%]

:: 4. Scan Loop (Master Mode)
set count=0
set sort_count=0

for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    set "NUM=!DIR_NAME:ip=!"
    set "OP_PATH=%ROOT_DIR%\op!NUM!"

    :: Collect unconditionally — sorting.py creates op* if it doesn't exist yet
    set /a sort_count+=1
    set "SORT_IP_!sort_count!=!IP_PATH!"
    set "SORT_OP_!sort_count!=!OP_PATH!"
    set "SORT_DIR_!sort_count!=!DIR_NAME!"

    :: Only launch a parallel worker if there are non-sort steps to run
    if not "!PARALLEL_STEPS_TRIM!"=="" (
        echo.
        echo [QUEUE] Preparing: !DIR_NAME!

        call :WAIT_FOR_RESOURCES

        set /a count+=1
        :: Create a pending sentinel file; worker deletes it when done
        set "PENDING_FILE=%TEMP%\hm_worker_!DIR_NAME!.pending"
        echo . > "!PENDING_FILE!"

        start "Job-!DIR_NAME!" cmd /k call "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!" "!PARALLEL_STEPS!" "!PENDING_FILE!"

        echo [MASTER] Job launched. Waiting 20s for stability...
        timeout /t 20 /nobreak >nul
    )
)

:: Wait for all parallel workers to finish before running sorting
if !count! gtr 0 (
    echo.
    echo ========================================================
    echo [MASTER] Launched !count! parallel job(s). Waiting for all to finish...
    echo ========================================================
    call :WAIT_ALL_WORKERS
    echo [MASTER] All parallel workers have completed.
)

:: Run sorting sequentially — one folder at a time
if !HAS_SORT!==1 (
    echo.
    echo ========================================================
    echo [MASTER] Running SORTING sequentially ^(1 folder at a time^)...
    echo ========================================================
    for /l %%i in (1,1,!sort_count!) do (
        set "CUR_IP=!SORT_IP_%%i!"
        set "CUR_OP=!SORT_OP_%%i!"
        echo.
        echo [SORT %%i/!sort_count!] Processing: !CUR_IP!
        if exist ".\src\sorter\sorting.py" (
            python -u ./src/sorter/sorting.py --input_folder "!CUR_IP!" --output_folder "!CUR_OP!"
        )
    )
    echo.
    echo [MASTER] Sorting complete for all !sort_count! folder(s).
)

:: Run cleaning sequentially after sorting — one folder at a time
if !HAS_CLEAN!==1 (
    echo.
    echo ========================================================
    echo [MASTER] Running CLEANING sequentially ^(after sorting^)...
    echo ========================================================
    for /l %%i in (1,1,!sort_count!) do (
        set "CUR_IP=!SORT_IP_%%i!"
        echo.
        echo [CLEAN %%i/!sort_count!] Cleaning: !CUR_IP!
        for /d %%D in ("!CUR_IP!\*.DIO" "!CUR_IP!\*.raw" "!CUR_IP!\*timestampoffset*") do (
            if exist "%%D" (
                echo     Deleting: %%~nxD
                rmdir /s /q "%%D"
            )
        )
    )
    echo.
    echo [MASTER] Cleaning complete for all !sort_count! folder(s).
)

echo.
echo ========================================================
echo [MASTER] Done. Parallel jobs: !count! ^| Sorting: !HAS_SORT!
echo ========================================================
pause
exit /b

:: ========================================================
::             RESOURCE MONITOR SUBROUTINE
:: ========================================================
:WAIT_FOR_RESOURCES
:CHECK_AGAIN
    :: --- Check CPU (Using PowerShell instead of wmic) ---
    set CPU_LOAD=0
    for /f "delims=" %%P in ('powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average)" 2^>nul') do (
        if "%%P" neq "" set CPU_LOAD=%%P
    )

    :: --- Check GPU ---
    set GPU_LOAD=0
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits > gpu_temp.txt 2>nul
    if exist gpu_temp.txt (
        set /p GPU_LOAD=<gpu_temp.txt
        del gpu_temp.txt
    )
    if "%GPU_LOAD%"=="" set GPU_LOAD=0

    :: --- Check Memory (Using PowerShell to bypass 32-bit batch math limits) ---
    set MEM_USAGE=0
    for /f "delims=" %%M in ('powershell -NoProfile -Command "$os = Get-CimInstance Win32_OperatingSystem; [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100)" 2^>nul') do (
        if "%%M" neq "" set MEM_USAGE=%%M
    )

    :: --- Validation Logic ---
    if !CPU_LOAD! GTR %MAX_CPU% (
        echo     [WAIT] High CPU: !CPU_LOAD!%%. Pausing %WAIT_SECONDS%s...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )
    if !GPU_LOAD! GTR %MAX_GPU% (
        echo     [WAIT] High GPU: !GPU_LOAD!%%. Pausing %WAIT_SECONDS%s...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )
    if !MEM_USAGE! GTR %MAX_MEM% (
        echo     [WAIT] High MEM: !MEM_USAGE!%%. Pausing %WAIT_SECONDS%s...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )

    echo     [CHECK] CPU: !CPU_LOAD!%% ^| GPU: !GPU_LOAD!%% ^| MEM: !MEM_USAGE!%% - OK.
exit /b

:: ========================================================
::            WAIT FOR ALL WORKERS SUBROUTINE
:: ========================================================
:WAIT_ALL_WORKERS
    set "ALL_DONE=1"
    for /l %%i in (1,1,!sort_count!) do (
        set "_CHK_DIR=!SORT_DIR_%%i!"
        if exist "%TEMP%\hm_worker_!_CHK_DIR!.pending" set "ALL_DONE=0"
    )
    if !ALL_DONE!==0 (
        timeout /t 15 /nobreak >nul
        goto :WAIT_ALL_WORKERS
    )
exit /b

:: ========================================================
::                THE WORKER SUBROUTINE
:: ========================================================
:WORKER_ROUTINE
set "IP=%~2"
set "OP=%~3"
color 0A

echo.
echo [INFO] Running steps [%STEPS_TO_RUN%] for !IP!

REM Guard against empty STEPS_TO_RUN. With an empty variable, echo emits
REM the localized "ECHO is on/off." status line, which can contain step
REM letters (notably n) and falsely trigger steps via findstr below.
if not defined STEPS_TO_RUN (
    echo [WORKER] No steps to run, exiting.
    if not "!PENDING_FILE!"=="" if exist "!PENDING_FILE!" del /q "!PENDING_FILE!"
    exit /b
)

:: --- STEP 1 (DIO/Raw only) ---
echo %STEPS_TO_RUN% | findstr "1" >nul
if %errorlevel% equ 0 (
    echo [STEP 1] Running Trodes DIO/Raw Export...
    if exist "%TRODES_EXPORT_CMD%" (
        for %%F in ("%IP%\*.rec") do ("%TRODES_EXPORT_CMD%" -dio -raw -rec "%%F")
    ) else (
        echo [WARNING] trodesexport not found at: %TRODES_EXPORT_CMD%
    )
)

:: --- STEP e (LFP export) ---
echo %STEPS_TO_RUN% | findstr "e" >nul
if %errorlevel% equ 0 (
    echo [STEP e] Running Trodes LFP Export ^(1000Hz, LP 500Hz^)...
    if exist "%TRODES_EXPORT_LFP%" (
        for %%F in ("%IP%\*.rec") do (
            echo     Exporting LFP from %%~nxF
            "%TRODES_EXPORT_LFP%" -rec "%%F" -outputrate 1000 -lfplowpass 500
        )
    ) else (
        echo [WARNING] exportLFP not found at: %TRODES_EXPORT_LFP%
    )
)

:: --- STEP 2 ---
echo %STEPS_TO_RUN% | findstr "2" >nul
if %errorlevel% equ 0 (
    echo [STEP 2] Running Sync Script...
    if exist ".\src\Video_LED_Sync_using_ICA.py" (
        python -u ./src/Video_LED_Sync_using_ICA.py -i "%IP%" -o "%OP%" -f %FREQ%
    )
)

:: --- STEP 3 ---
echo %STEPS_TO_RUN% | findstr "3" >nul
if %errorlevel% equ 0 (
    echo [STEP 3] Running Stitching...
    if exist ".\src\join_views.py" (
        python -u ./src/join_views.py "%IP%"
    )
)

:: --- STEP 4 ---
echo %STEPS_TO_RUN% | findstr "4" >nul
if %errorlevel% equ 0 (
    echo [STEP 4] Running Tracker...
    if exist "%IP%\stitched.mp4" (
        python -u ./src/TrackerYolov11.py --input_folder "%IP%" --output_folder "%OP%" --onnx_weight "%ONNX_WEIGHTS_PATH%"
    )
)

:: --- STEP 5 ---
echo %STEPS_TO_RUN% | findstr "5" >nul
if %errorlevel% equ 0 (
    echo [STEP 5] Running Plotting...
    if exist ".\src\plot_trials.py" (
        python -u ./src/plot_trials.py --input_folder "%IP%" --output_folder "%OP%"
    )
)

:: --- STEP 6 ---
echo %STEPS_TO_RUN% | findstr "6" >nul
if %errorlevel% equ 0 (
    echo [STEP 6] Running Compression - GPU Accelerated...
    set "VIDEO_FILE="
    set "TEMP_FILE=!OP!\__temp_compressed.mp4"
    
    :: Clean up any leftover temp files from previous crashed runs
    if exist "!TEMP_FILE!" del /q "!TEMP_FILE!"
    
    :: Safely grab the first .mp4 found that is NOT the temp file
    for %%f in ("!OP!\*.mp4") do (
        if "!VIDEO_FILE!"=="" (
            if /I not "%%~nxf"=="__temp_compressed.mp4" set "VIDEO_FILE=%%~f"
        )
    )
    
    if not "!VIDEO_FILE!"=="" (
        :: Added -hide_banner -loglevel warning -stats to show the live progress line
        "%FFMPEG_CMD%" -nostdin -y -hide_banner -loglevel warning -stats -i "!VIDEO_FILE!" -c:v h264_nvenc -preset p6 -cq 28 -c:a copy "!TEMP_FILE!" && (
            echo.
            move /Y "!TEMP_FILE!" "!VIDEO_FILE!" >nul
            echo [SUCCESS] Video compressed using GPU: !VIDEO_FILE!
        ) || (
            echo.
            echo [ERROR] FFmpeg compression failed for !VIDEO_FILE!
            echo         Check for NVENC concurrent session limits or GPU memory issues.
            if exist "!TEMP_FILE!" del /q "!TEMP_FILE!"
        )
    ) else (
        echo [WARNING] No valid .mp4 file found in "!OP!" to compress.
    )
)

:: --- STEP 8 ---
echo %STEPS_TO_RUN% | findstr "8" >nul
if %errorlevel% equ 0 (
    echo [STEP 8] Running LFP Extraction...
    if exist ".\src\sorter\export_lfp.py" (
        python -u ./src/sorter/export_lfp.py --input_folder "%IP%" --output_folder "%OP%"
    )

)

:: --- STEP d ---
echo %STEPS_TO_RUN% | findstr "d" >nul
if %errorlevel% equ 0 (
    echo [STEP d] Exporting video for DeepLabCut...
    if exist ".\src\dlc\tracking_eyes.py" (
        python -u ./src/dlc/tracking_eyes.py --input_folder "%IP%" --output_folder "%OP%"
    )

)

:: --- STEP 9 ---
echo %STEPS_TO_RUN% | findstr "9" >nul
if %errorlevel% equ 0 (
    echo [STEP 9] Cleaning up .DIO, .raw, and timestampoffset folders...
    for /d %%D in ("%IP%\*.DIO" "%IP%\*.raw" "%IP%\*timestampoffset*") do (
        if exist "%%D" (
            echo Deleting folder: %%~nxD
            rmdir /s /q "%%D"
        )
    )
)

:: --- STEP n ---
echo %STEPS_TO_RUN% | findstr "n" >nul
if %errorlevel% equ 0 (
    echo [STEP n] Running Node Analysis...
    if exist ".\src\node_analysis\hex_maze_analysis.py" (
        python -u ./src/node_analysis/hex_maze_analysis.py --input_folder "%IP%" --output_folder "%OP%"
    )
)

:: Signal master that this worker is done
if not "!PENDING_FILE!"=="" (
    if exist "!PENDING_FILE!" del /q "!PENDING_FILE!"
)

echo.
echo [COMPLETE] Worker finished.
REM timeout /t 15
REM exit