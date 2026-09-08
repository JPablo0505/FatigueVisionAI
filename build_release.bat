@echo off
REM Quick build script for Windows — double-click to package a release
REM Requires: Python 3.12+, Node.js 24+, pnpm 11+
REM Optional: Inno Setup 6+ (for installer generation)

echo ==========================================
echo   FatigueVision AI Release Builder
echo ==========================================
echo.

REM Check prerequisites
where python >nul 2>&1 || (echo [ERROR] Python not found in PATH & pause & exit /b 1)
where node >nul 2>&1 || (echo [ERROR] Node.js not found in PATH & pause & exit /b 1)
where pnpm >nul 2>&1 || (echo [ERROR] pnpm not found in PATH & pause & exit /b 1)

REM Install backend PyInstaller dependency
echo [0/4] Installing PyInstaller...
pip install pyinstaller --quiet 2>nul

REM Run the packaging pipeline
echo.
python scripts\package_release.py %*

echo.
echo ==========================================
echo   Done! Check dist\ for artifacts.
echo ==========================================
pause
