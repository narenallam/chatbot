#!/bin/bash

echo "🔄 Resetting AI Chatbot System..."
echo "=================================="

# Function to stop all services
stop_all_services() {
    echo "📛 Stopping all running services..."
    
    # Stop React frontend
    echo "   • Stopping React frontend..."
    pkill -f "react-scripts start" 2>/dev/null || true
    pkill -f "npm start" 2>/dev/null || true
    
    # Stop FastAPI backend
    echo "   • Stopping FastAPI backend..."
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "python -m uvicorn" 2>/dev/null || true
    
    # Stop Ollama API server
    echo "   • Stopping Ollama API server..."
    pkill -f "ollama serve" 2>/dev/null || true
    pkill -f "ollama" 2>/dev/null || true
    
    # Kill any remaining Node.js processes on port 3000
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    
    # Kill any remaining Python processes on port 8000
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    
    # Kill any remaining Ollama processes on port 11434
    lsof -ti:11434 | xargs kill -9 2>/dev/null || true
    
    sleep 3
    echo "✅ All services stopped"
}

# Function to clean all data
clean_all_data() {
    echo "🗑️  Cleaning all system data..."
    
    # Navigate to backend directory
    cd backend 2>/dev/null || { echo "❌ Backend directory not found!"; exit 1; }

    echo "   • Clearing ChromaDB (Vector Database)..."
    if [ -d "embeddings" ]; then
        rm -rf embeddings/*
        echo "     ✅ ChromaDB cleared"
    else
        echo "     ℹ️  ChromaDB directory not found"
    fi

    echo "   • Clearing SQLite Database..."
    if [ -f "data/chatbot.db" ]; then
        rm -f data/chatbot.db
        echo "     ✅ SQLite database cleared"
    else
        echo "     ℹ️  SQLite database not found"
    fi

    echo "   • Clearing uploaded documents..."
    if [ -d "data" ]; then
        # Keep the data directory but remove uploaded files
        find data -type f ! -name "chatbot.db" -delete 2>/dev/null || true
        echo "     ✅ Uploaded documents cleared"
    else
        echo "     ℹ️  Data directory not found"
    fi

    echo "   • Clearing Python cache files..."
    # Remove any .pyc files
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "     ✅ Python cache cleared"

    # Navigate back to root
    cd ..

    echo "   • Clearing frontend cache..."
    if [ -d "frontend/node_modules/.cache" ]; then
        rm -rf frontend/node_modules/.cache
        echo "     ✅ Frontend cache cleared"
    fi
    
    echo "   • Frontend localStorage will be cleared on next app start..."
    echo "     ℹ️  localStorage data requires browser refresh to clear"
    
    echo "✅ All data cleaned"
}

# Function to start services in correct order
start_all_services() {
    echo "🚀 Starting services in correct order..."
    
    # Start Ollama API server first
    echo "   1️⃣ Starting Ollama API server..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 5
    
    if ps -p $OLLAMA_PID > /dev/null 2>&1; then
        echo "     ✅ Ollama API server started (PID: $OLLAMA_PID)"
    else
        echo "     ⚠️  Ollama may not be installed or failed to start"
        echo "     ℹ️  Run ./setup_ollama.sh to install Ollama first"
    fi
    
    # Start FastAPI backend second
    echo "   2️⃣ Starting FastAPI backend..."
    cd backend
    source venv/bin/activate
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
    BACKEND_PID=$!
    cd ..
    sleep 3
    
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "     ✅ Backend server started (PID: $BACKEND_PID)"
    else
        echo "     ❌ Failed to start backend server"
        return 1
    fi
    
    # Start React frontend last
    echo "   3️⃣ Starting React frontend..."
    cd frontend
    npm start > /dev/null 2>&1 &
    FRONTEND_PID=$!
    cd ..
    sleep 5
    
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "     ✅ Frontend server started (PID: $FRONTEND_PID)"
    else
        echo "     ❌ Failed to start frontend server"
        return 1
    fi
    
    echo "✅ All services started successfully"
    return 0
}

# Main execution
stop_all_services
clean_all_data

echo ""
echo "✨ System Reset Complete!"
echo "========================"
echo "📊 What was cleaned:"
echo "   • All running services stopped"
echo "   • ChromaDB vector database"
echo "   • SQLite chat database" 
echo "   • All uploaded documents"
echo "   • Python cache files"
echo "   • Frontend cache"
echo "   • Frontend localStorage (on next browser refresh)"
echo ""

# Ask user if they want to restart services
read -p "🚀 Would you like to restart all services now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    start_all_services
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 AI MATE is now running!"
        echo ""
        echo "📱 Frontend: http://localhost:3000"
        echo "🔧 Backend:  http://localhost:8000"
        echo "📋 API Docs: http://localhost:8000/docs"
        echo "🤖 Ollama:   http://localhost:11434"
        echo ""
        echo "Use ./restart.sh to control individual services"
        echo "Press Ctrl+C to stop all services"
        
        # Wait for user to stop
        trap 'echo ""; stop_all_services; exit 0' SIGINT SIGTERM
        wait
    fi
else
    echo ""
    echo "🔄 To start services manually, use:"
    echo "   ./restart.sh --ollama --backend --frontend"
    echo "   or"
    echo "   ./start_react_app.sh"
fi 