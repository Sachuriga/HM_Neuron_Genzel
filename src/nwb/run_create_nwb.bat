@echo off
cd /d "%~dp0"
echo Current directory is: %cd%

:: ===== EDIT THESE VARIABLES =====
set "rat_number=1"

set "input_folder=S:\data\Rat%rat_number%"
set "output_folder=S:\data\nwb_rat%rat_number%"

::set "input_folder=nwb_data"
::set "output_folder=nwb_data"

:: ===== ADD OWN PYTHON INSTALLATION =====
call "C:\Users\Jacob\anaconda3\condabin\conda.bat" activate base

:: use --noroot to specify whole filepath 
:: use --usecd (and omit --noroot) to use current directory as root and specify only folders with ip and op
python create_nwb.py --rat_nr %rat_number% --noroot --ip "%input_folder%" --op "%output_folder%"

pause