#!/bin/bash

# AI MATE - React Frontend Startup Script
# This script starts Ollama, backend, and frontend servers in the correct order

echo "🚀 Starting AI MATE React Application..."

# Function to check if port is in use
check_port() {
    if lsof -i :$1 >/dev/null 2>&1; then
        echo "⚠️  Port $1 is already in use. Please stop the service or use a different port."
        return 1
    fi
    return 0
}

# Function to check if Ollama is installed
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        echo "⚠️  Ollama is not installed"
        echo "ℹ️  Run ./setup_ollama.sh to install Ollama first"
        return 1
    fi
    return 0
}

# Check if required directories exist
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found"
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f "backend/venv/bin/activate" ]; then
    echo "❌ Python virtual environment not found"
    echo "ℹ️  Run: cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if ports are available
if ! check_port 11434; then
    echo "❌ Ollama port 11434 is busy"
    exit 1
fi

if ! check_port 8000; then
    echo "❌ Backend port 8000 is busy"
    exit 1
fi

if ! check_port 3000; then
    echo "⚠️  Frontend port 3000 is busy. React will try to use port 3001."
fi

# Start Ollama API server first
echo "🤖 Starting Ollama API server on port 11434..."
if check_ollama; then
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 5
    
    if ps -p $OLLAMA_PID > /dev/null 2>&1; then
        echo "✅ Ollama API server started successfully (PID: $OLLAMA_PID)"
    else
        echo "⚠️  Ollama may have failed to start, but continuing..."
        OLLAMA_PID=""
    fi
else
    echo "⚠️  Continuing without Ollama..."
    OLLAMA_PID=""
fi

# Start backend server second
echo "🔧 Starting FastAPI backend server on port 8000..."
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ps -p $BACKEND_PID > /dev/null; then
    echo "✅ Backend server started successfully (PID: $BACKEND_PID)"
else
    echo "❌ Failed to start backend server"
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null
    fi
    exit 1
fi

# Start frontend server last
echo "🎨 Starting React frontend server on port 3000..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "⚠️  Node modules not found, installing..."
    npm install
fi

npm start > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 5

# Check if frontend started successfully
if ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ Frontend server started successfully (PID: $FRONTEND_PID)"
else
    echo "❌ Failed to start frontend server"
    kill $BACKEND_PID 2>/dev/null
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null
    fi
    exit 1
fi

echo ""
echo "🎉 AI MATE is now running!"
echo ""
if [ -n "$OLLAMA_PID" ]; then
    echo "🤖 Ollama:   http://localhost:11434"
fi
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:8000"
echo "📋 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Use './restart.sh --status' to check service status"
echo "💡 Use './restart.sh --help' for individual service control"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null
    fi
    
    # Additional cleanup for any remaining processes
    pkill -f "react-scripts start" 2>/dev/null || true
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    
    echo "✅ All servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait 