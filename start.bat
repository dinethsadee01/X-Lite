@echo off
REM X-Lite Startup Script for Windows
REM This script starts both the backend API and frontend development server
setlocal

echo ========================================
echo X-Lite - Chest X-Ray Classification
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Error: Python virtual environment not found at .venv\Scripts\python.exe
    echo Please create it first with: python -m venv .venv
    exit /b 1
)

if not exist "frontend\package.json" (
    echo Error: frontend\package.json not found.
    echo Run this script from the project root directory.
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo Error: npm is not installed or not on PATH.
    echo Please install Node.js LTS and retry.
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        echo Error: npm install failed.
        exit /b 1
    )
    popd
)

echo.
echo ========================================
echo Starting Backend API...
echo ========================================
REM Start backend in a new window
start "X-Lite Backend" cmd /k "cd /d %cd% && .venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3

echo.
echo ========================================
echo Starting Frontend...
echo ========================================
REM Start frontend in a new window
start "X-Lite Frontend" cmd /k "cd /d %cd%\frontend && npm start"

echo.
echo ========================================
echo Startup in progress...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/api/docs
echo ========================================
echo.
pause
