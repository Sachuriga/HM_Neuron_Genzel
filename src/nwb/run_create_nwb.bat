@echo off
cd /d "%~dp0"
echo Current directory is: %cd%

:: ===== EDIT THESE VARIABLES =====
set "rat_number=1"

:: Folder containing session folders
set "input_folder=nwb_data"
:: Folder where nwb files should be stored
set "output_folder=nwb_data"
:: Specific session folder in input_folder that should be run (OPTIONAL)
set "session_folder=20200914_Rat1"

:: ===== ADD OWN PYTHON INSTALLATION =====
call "C:\Users\Jacob\anaconda3\condabin\conda.bat" activate base

:: use --noroot to specify whole filepath 
:: use --usecd (and omit --noroot) to use current directory as root and specify only folders with ip and op
:: == Use this if you want to run all folders in input_folder (for just multiple use sess_i and sess_f) ==
::python create_nwb.py --rat_nr %rat_number% --noroot --ip "%input_folder%" --op "%output_folder%"


:: === Use this if you want to run a specific folder ===
python create_nwb.py --rat_nr %rat_number% --usecd --session_folder "%session_folder%" --ip "%input_folder%" --op "%output_folder%"

pause