@echo off
REM show_lab_case_graphs.bat [args...] - per-case graphs of the lab working points
REM (lab\data\working_points). Click a run name to open it in the lab viewer or
REM frame_explorer; a missing feature cache is built first (lab\frame_data).
REM All arguments are passed through to code\show_lab_case_graphs.py.
setlocal
REM Python: lab\.venv (see README.md), else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"
"%PY%" "%~dp0code\show_lab_case_graphs.py" %*
endlocal
