@echo off
REM ========================================================
REM   HM Tracker — Windows launcher.
REM
REM   Thin wrapper: all pipeline logic lives in runner.py (the single
REM   cross-platform source of truth, shared with runner_unix.sh).
REM   Edit the menu / steps THERE, not here.
REM
REM   Usage:  scripts\runner_windows.bat "path_to_data_folder"
REM ========================================================
cd /d "%~dp0.."
python runner.py %*
pause
