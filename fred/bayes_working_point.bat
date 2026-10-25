@echo off
REM bayes_working_point.bat [args...] - Bayesian search for a global working point.
REM   e.g.  bayes_working_point.bat                    asks for the detector mode
REM         bayes_working_point.bat --mode pixelize --track
REM         bayes_working_point.bat --mode pixelize --track --workers 8
REM         bayes_working_point.bat --mode baseline --track --stride 3 --iterations 40
REM It searches over the cases cached for that mode in utility\frame_data
REM (build them with cache_frames.bat). Without --mode it asks which mode.
REM Resumable: rerun the same command after a stop and it continues.
REM View the winner's per-case scores with show_case_graphs.bat.
REM All arguments are passed through to bayes_working_point.py.
setlocal
set "MODEARG="
echo( %* | findstr /i /c:"--mode" >nul
if not errorlevel 1 goto run

set "DATA=%~dp0utility\frame_data"
for %%M in (pixelize baseline pool components) do call :count %%M
echo Which detector mode should the working point be searched for?
echo   1  pixelize    3x3 blocks             (%N_pixelize% cached cases)
echo   2  baseline    each pixel on its own  (%N_baseline% cached cases)
echo   3  pool        3x3 neighbourhood      (%N_pool% cached cases)
echo   4  components  connected event blobs  (%N_components% cached cases)
choice /c 1234 /n /m "Mode [1-4]: "
if errorlevel 4 (set "MODEARG=--mode components" & goto run)
if errorlevel 3 (set "MODEARG=--mode pool" & goto run)
if errorlevel 2 (set "MODEARG=--mode baseline" & goto run)
set "MODEARG=--mode pixelize"

:run
REM Python: common\.venv (see how_to.txt), else common\.venv312, else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0..\common\.venv\Scripts\python.exe" set "PY=%~dp0..\common\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\common\.venv312\Scripts\python.exe" set "PY=%~dp0..\common\.venv312\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"
"%PY%" "%~dp0..\common\utility\bayes_working_point.py" %MODEARG% %*
endlocal
exit /b

:count
REM N_<mode> = number of cached cases for <mode> (one meta_<seq>_<mode>.json each)
set "N_%1=0"
for %%F in ("%DATA%\meta_*_%1.json") do set /a N_%1+=1
exit /b
