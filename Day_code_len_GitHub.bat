@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================================
echo   ĐẨY MÃ NGUỒN VIDEO DUBBER LÊN GITHUB
echo ==============================================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [!] Máy tính chưa cài đặt Git CLI.
    echo Đang tự động cài đặt Git thông qua Windows Winget...
    winget install --id Git.Git -e --source winget
    echo.
    echo Vui lòng tắt cửa sổ này và mở lại sau khi Git cài đặt xong!
    pause
    exit /b 1
)

if not exist ".git" (
    git init
    git branch -M main
)

set /p GH_URL="Nhập đường dẫn GitHub Repo của bạn (VD: https://github.com/username/video-dubber.git): "
if "%GH_URL%"=="" (
    echo [LỖI] Bạn chưa nhập đường dẫn GitHub Repo!
    pause
    exit /b 1
)

git remote remove origin >nul 2>&1
git remote add origin "%GH_URL%"

git add .
git commit -m "chore: release Video Dubber v2.0.0 with anti-slur TTS, smart OCR inpaint, multi-voice & auto-updater"
git push -u origin main

echo.
echo ==============================================================
echo   HOÀN TẤT ĐẨY LÊN GITHUB!
echo ==============================================================
pause
