@echo off
REM gt_editor.bat [run] [args...] - draw / edit the ground-truth drone boxes on a
REM run exported by export_dataset.bat. Without a run it lists the exported runs
REM and asks which one. Arguments go to code\gt_editor.py.
setlocal
REM Python: lab\.venv (see README.md), else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"
"%PY%" "%~dp0code\gt_editor.py" %*
endlocal
