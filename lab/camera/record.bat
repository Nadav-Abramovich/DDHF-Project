@echo off
REM Record a new event + RGB session into lab\lab<N> (the folder above camera\).
REM The folder number is picked automatically, so nothing is ever overwritten.
REM
REM   record.bat                 5 seconds (default)
REM   record.bat -d 10           10 seconds
REM   record.bat -d 10 --no-rgb  events only
REM
REM Any main.py flag can be passed through.

setlocal
REM Use lab\.venv (see ..\README.md); otherwise "python" on PATH.
set "VENV="
if exist "%~dp0..\.venv\Scripts\activate.bat" set "VENV=%~dp0..\.venv"
if defined VENV (
    call "%VENV%\Scripts\activate.bat"
    if errorlevel 1 (
        echo failed to activate the venv
        exit /b 1
    )
)

cd /d "%~dp0"
python main.py %*
set RC=%ERRORLEVEL%

if %RC% neq 0 (
    echo.
    echo recording FAILED with exit code %RC%
)
endlocal & exit /b %RC%
