@echo off
REM Open debug_viewer_pooled.py on a recorded session, using the parameters from
REM working_point_bayes_pixelize_track_ss_r80_validation.json.
REM
REM   view.bat                     the most recently recorded lab* session
REM   view.bat lab9                that session under lab
REM   view.bat C:\path\to\session  an explicit folder
REM   view.bat lab9 --frame 100    extra flags are passed to the viewer
REM
REM NOTE: only 9 of the working point's 19 parameters exist in this viewer. The
REM SNR gates (min_snr_sum / min_snr_avg), persistence and the multi-track
REM coast/confirm/acquire/predict settings have no equivalent here, so this is an
REM approximation - for an exact replay pass --wp to the viewer.

setlocal
for %%I in ("%~dp0..") do set "DATASET=%%~fI"
for %%I in ("%~dp0..\code\common") do set "CODEDIR=%%~fI"

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

set "SESSION=%~1"
if not "%SESSION%"=="" shift

REM Collect any remaining arguments to forward to the viewer.
set "EXTRA="
:collect
if "%~1"=="" goto resolve
set "EXTRA=%EXTRA% %1"
shift
goto collect

:resolve
REM No argument: use the most recently modified lab* folder.
if "%SESSION%"=="" (
    for /f "delims=" %%d in ('dir /b /ad /o-d "%DATASET%\lab*" 2^>nul') do (
        set "SESSION=%DATASET%\%%d"
        goto check
    )
    echo no lab* sessions found in %DATASET%
    exit /b 1
)

REM A bare name like "lab9" is resolved against the dataset root.
if not exist "%SESSION%\Event" set "SESSION=%DATASET%\%SESSION%"

:check
if not exist "%SESSION%\Event\events.raw" (
    echo no Event\events.raw in "%SESSION%"
    exit /b 1
)

echo viewing %SESSION%
cd /d "%CODEDIR%"
python debug_viewer_pooled.py --seq "%SESSION%" ^
    --mode pixelize ^
    --tau-sf 0.95 ^
    --tau-comb 3.2163 ^
    --min-events 15 ^
    --min-box-area 1 ^
    --alpha 3.5867 ^
    --min-rotor-pixels 4 ^
    --track ^
    --sigma 20.0 %EXTRA%
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
