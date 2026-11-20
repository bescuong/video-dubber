@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ====================================================
echo   Đóng gói Video Dubber thành file EXE và Installer
echo ====================================================
echo.

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo [LỖI] Không tìm thấy PyInstaller trong .venv!
    pause
    exit /b 1
)

echo [1/2] Đang đóng gói ứng dụng với PyInstaller...
.venv\Scripts\pyinstaller.exe --noconfirm VideoDubber.spec
if errorlevel 1 (
    echo [LỖI] Quá trình đóng gói PyInstaller thất bại!
    pause
    exit /b 1
)

echo.
echo [2/2] Đang tạo file cài đặt Windows Installer (Setup.exe)...
set "ISCC_PATH="
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
)

if defined ISCC_PATH (
    "%ISCC_PATH%" installer.iss
    if errorlevel 1 (
        echo [CẢNH BÁO] Không thể tạo file Setup.exe!
    ) else (
        echo.
        echo ====================================================
        echo   HOÀN TẤT!
        echo   1. Thư mục chạy trực tiếp: dist\VideoDubber\VideoDubber.exe
        echo   2. File cài đặt Setup:     Output\VideoDubber_v2.5.1_Setup.exe
        echo ====================================================
    )
) else (
    echo [THÔNG BÁO] Inno Setup chưa được cài đặt.
    echo Bạn có thể chạy trực tiếp ứng dụng tại: dist\VideoDubber\VideoDubber.exe
)

pause
