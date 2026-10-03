@echo off
rem ===========================================================================
rem  Launch the 3D train sandbox (track editor).
rem
rem  Usage:
rem      run.bat                                start with an empty field
rem      run.bat --list-scenes                  list ready-made scenes
rem      run.bat --scene valley                 start in a scene (train on the track)
rem      run.bat --scene gorge                  ... or gorge / town / lake
rem      run.bat --open saves\yard.json         open a station yard, keep building
rem      run.bat --open saves\figure8.json      open the figure-8 (already a closed loop)
rem      run.bat --help                         list every option
rem
rem  This script only saves you from running `conda activate` first; it does
rem  exactly the same thing as:  python app/main.py ...
rem
rem  NOTE: keep this file ASCII-only. cmd.exe parses .bat files using the OEM
rem  code page (GBK on a Chinese Windows), so UTF-8 Chinese comments get decoded
rem  into bytes that look like shell metacharacters ("|", "&") and cmd then tries
rem  to execute the fragments as commands.
rem ===========================================================================
setlocal

set "PY=C:\Users\moren\anaconda3\envs\train3d\python.exe"
if not exist "%PY%" set "PY=python"

rem Do NOT force PYTHONIOENCODING here: the console is cp936 and Python's
rem default already matches it, so Chinese output renders correctly.

"%PY%" "%~dp0app\main.py" %*
set "CODE=%ERRORLEVEL%"

rem Keep the window open on failure so the traceback stays readable.
if not "%CODE%"=="0" (
    echo.
    echo [launch failed] exit code %CODE%
    pause
)
exit /b %CODE%
