@echo off
chcp 65001 >nul
cd /d "%~dp0"

set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8

if exist "dist\VideoDubber\VideoDubber.exe" (
    start "" "dist\VideoDubber\VideoDubber.exe"
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
    exit /b 0
)

echo [LỖI] Không tìm thấy file thực thi!
pause
