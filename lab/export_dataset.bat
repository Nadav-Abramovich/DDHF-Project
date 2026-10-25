@echo off
REM export_dataset.bat [run] [args...] - turn a dual_record.py recording into a
REM FRED-style dataset in runs\dataset\, using the lens calibrations and the
REM registration pairs picked in compare_replay.bat. Without a run it lists the
REM recordings and asks which one. Arguments go to code\export_dataset.py.
setlocal
REM Python: lab\.venv (see README.md), else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"
"%PY%" "%~dp0code\export_dataset.py" %*
endlocal
