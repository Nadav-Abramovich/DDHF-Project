@echo off
REM ============================================================================
REM  cache_frames.bat [mode]
REM  Build the frame_data feature cache for every FRED case (resumable - done
REM  cases skip). Missing cases are downloaded from HuggingFace and deleted again
REM  once cached (~1 GB each at a time); cases already on disk are kept.
REM    mode: pixelize | baseline | pool | components
REM  Without a mode (or with an unknown one) it asks which mode to cache.
REM  Examples:
REM    cache_frames.bat                 ask for the mode
REM    cache_frames.bat pixelize        cache for the bayes_pixelize_* working points
REM  Crashes (out of disk/memory, reboot) are retried up to 10 times, resuming
REM  where it stopped.
REM ============================================================================
setlocal
set "MODE=%~1"
if /i "%MODE%"=="pixelize"   (set "MODE=pixelize" & goto mode_ok)
if /i "%MODE%"=="baseline"   (set "MODE=baseline" & goto mode_ok)
if /i "%MODE%"=="pool"       (set "MODE=pool" & goto mode_ok)
if /i "%MODE%"=="components" (set "MODE=components" & goto mode_ok)
if not "%MODE%"=="" echo [cache_frames] unknown mode "%MODE%".

echo Which detector mode should be cached?
echo   1  pixelize    3x3 blocks             (bayes_pixelize_* working points)
echo   2  baseline    each pixel on its own  (baseline / bayes_baseline_* working points)
echo   3  pool        3x3 neighbourhood per pixel
echo   4  components  connected event blobs
choice /c 1234 /n /m "Mode [1-4]: "
if errorlevel 4 (set "MODE=components" & goto mode_ok)
if errorlevel 3 (set "MODE=pool" & goto mode_ok)
if errorlevel 2 (set "MODE=baseline" & goto mode_ok)
set "MODE=pixelize"

:mode_ok
REM Python: common\.venv (see how_to.txt), else common\.venv312, else %PYTHON%, else python on PATH.
set "PY="
if exist "%~dp0..\common\.venv\Scripts\python.exe" set "PY=%~dp0..\common\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\common\.venv312\Scripts\python.exe" set "PY=%~dp0..\common\.venv312\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"
set "SCRIPT=%~dp0..\common\utility\frame_features.py"
set /a TRIES=0

:loop
echo [cache_frames] running mode %MODE% (attempt %TRIES%) ...
"%PY%" "%SCRIPT%" --mode %MODE% --workers 8
if %ERRORLEVEL%==0 goto done
set /a TRIES+=1
if %TRIES% GEQ 10 goto fail
echo [cache_frames] exited with an error - retrying in 30 s (%TRIES%/10) ...
timeout /t 30 /nobreak >nul
goto loop

:done
echo [cache_frames] mode %MODE% finished.
exit /b 0

:fail
echo [cache_frames] gave up after 10 retries.
exit /b 1
