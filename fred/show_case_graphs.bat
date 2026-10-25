@echo off
REM show_case_graphs.bat [args...] - per-case metric graphs (uses the newest
REM working_point_eval_*.csv unless --eval-csv is given).
REM
REM Launcher for the self-contained copy in script\. It picks
REM the first interpreter that exists, so no path has to be edited by hand:
REM   1. a venv inside script  (script\.venv\ or script\.venv312\)
REM   2. the shared venv: drone_project_final\common\.venv (see how_to.txt),
REM      else common\.venv312
REM   3. the PYTHON env var, if you set one
REM   4. whatever "python" is on PATH
setlocal

set "PY="
if exist "%~dp0script\.venv\Scripts\python.exe"    set "PY=%~dp0script\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0script\.venv312\Scripts\python.exe" set "PY=%~dp0script\.venv312\Scripts\python.exe"
if not defined PY if exist "%~dp0..\common\.venv\Scripts\python.exe" set "PY=%~dp0..\common\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\common\.venv312\Scripts\python.exe" set "PY=%~dp0..\common\.venv312\Scripts\python.exe"
if not defined PY if defined PYTHON if exist "%PYTHON%" set "PY=%PYTHON%"
if not defined PY set "PY=python"

"%PY%" "%~dp0script\show_case_graphs.py" %*
endlocal
