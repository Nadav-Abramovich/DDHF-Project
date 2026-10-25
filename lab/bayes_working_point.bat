@echo off
REM bayes_working_point.bat [args...] - Bayesian search for a lab working point.
REM   e.g.  bayes_working_point.bat                    asks for the detector mode
REM         bayes_working_point.bat --mode pixelize --track
REM         bayes_working_point.bat --mode pixelize --track --workers 8
REM         bayes_working_point.bat --mode baseline --track --stride 3 --iterations 40
REM It searches over the exported lab runs cached for that mode in frame_data
REM (build them with cache_frames.bat). Without --mode it asks which mode.
REM Resumable: rerun the same command after a stop and it continues.
REM View the winner's per-case scores with show_lab_case_graphs.bat.
REM All arguments are passed through to code\lab_bayes_working_point.py.
setlocal
set "MODEARG="
echo( %* | findstr /i /c:"--mode" >nul
if not errorlevel 1 goto run

set "DATA=%~dp0frame_data"
for %%M in (pixelize baseline pool components) do call :count %%M
echo Which detector mode should the working point be searched for?
echo   1  pixelize    3x3 blocks             (%N_pixelize% cached runs)
echo   2  baseline    each pixel on its own  (%N_baseline% cached runs)
echo   3  pool        3x3 neighbourhood      (%N_pool% cached runs)
echo   4  components  connected event blobs  (%N_components% cached runs)
choice /c 1234 /n /m "Mode [1-4]: "
if errorlevel 4 (set "MODEARG=--mode components" & goto run)
if errorlevel 3 (set "MODEARG=--mode pool" & goto run)
if errorlevel 2 (set "MODEARG=--mode baseline" & goto run)
set "MODEARG=--mode pixelize"

:run
REM Python: lab\.venv (see README.md), else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"
"%PY%" "%~dp0code\lab_bayes_working_point.py" %MODEARG% %*
endlocal
exit /b

:count
REM N_<mode> = number of cached runs for <mode> (one meta_<run>_<mode>.json each)
set "N_%1=0"
for %%F in ("%DATA%\meta_*_%1.json") do set /a N_%1+=1
exit /b
