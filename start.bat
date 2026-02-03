@echo off
REM X-Lite Startup Script for Windows
REM This script starts both the backend API and frontend development server

echo ========================================
echo X-Lite - Chest X-Ray Classification
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating Python virtual environment...
    call .venv\Scripts\activate.bat
    if errorlevel 1 (
        echo Error: Virtual environment not found. Please create it first.
        exit /b 1
    )
)

echo.
echo ========================================
echo Starting Backend API...
echo ========================================
REM Start backend in a new window
start "X-Lite Backend" cmd /k "cd /d %cd% && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

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
