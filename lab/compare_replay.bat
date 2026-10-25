@echo off
REM compare_replay.bat [rec_folder] [args...] - play a dual_record.py recording as
REM RGB / events / overlay, and pick the RGB-to-event registration pairs that
REM export_dataset.bat uses. Without a folder it lists the recordings in
REM runs\ and asks which one. Extra arguments go to code\compare_replay.py.
setlocal EnableDelayedExpansion
REM Python: lab\.venv (see README.md), else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"

set "RUN=%~1"
if not "%RUN%"=="" goto have_run
if not exist "%~dp0runs" (
    echo [compare_replay] no recordings: %~dp0runs does not exist.
    echo                  record one with camera\dual_record.py first.
    exit /b 1
)
set /a N=0
for /f "delims=" %%d in ('dir /b /ad /o-d "%~dp0runs\rec_*" 2^>nul') do (
    set /a N+=1
    set "RUN_!N!=%~dp0runs\%%d"
    echo   !N!  %%d
)
if %N%==0 (
    echo [compare_replay] no rec_* recordings in %~dp0runs
    exit /b 1
)
set "PICK=1"
set /p "PICK=Recording number [1 = newest]: "
if not defined RUN_%PICK% (
    echo [compare_replay] no recording number %PICK%
    exit /b 1
)
set "RUN=!RUN_%PICK%!"
"%PY%" "%~dp0code\compare_replay.py" "!RUN!"
exit /b

:have_run
REM a bare rec_* name is looked up in runs\
if not exist "%RUN%" if exist "%~dp0runs\%RUN%" set "RUN=%~dp0runs\%RUN%"
shift
set "EXTRA="
:collect
if "%~1"=="" goto run
set "EXTRA=!EXTRA! %1"
shift
goto collect
:run
"%PY%" "%~dp0code\compare_replay.py" "%RUN%" !EXTRA!
endlocal
