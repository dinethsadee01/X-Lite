#!/bin/bash
# X-Lite Startup Script for macOS/Linux
# This script starts both the backend API and frontend development server

set -u

echo "========================================"
echo "X-Lite - Chest X-Ray Classification"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Error: .venv/bin/python not found."
    echo "Please create and initialize the virtual environment first."
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo "Error: frontend/package.json not found."
    echo "Run this script from the project root directory."
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm is not installed or not on PATH."
    echo "Please install Node.js LTS and retry."
    exit 1
fi

# Activate virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate

if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd frontend && npm install) || {
        echo "Error: npm install failed."
        exit 1
    }
fi

echo ""
echo "========================================"
echo "Starting Backend API..."
echo "========================================"
# Start backend in background
.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

echo ""
echo "========================================"
echo "Starting Frontend..."
echo "========================================"
# Start frontend
(cd frontend && npm start) &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "Startup in progress..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/api/docs"
echo "========================================"
echo ""

cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
}

trap cleanup INT TERM

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
