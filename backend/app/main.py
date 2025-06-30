"""
Personal Assistant AI Chatbot - Main FastAPI Application
"""

# Fix HuggingFace tokenizers parallelism warning BEFORE any imports that might use tokenizers
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import json
from typing import Dict, Set
from datetime import datetime
from dotenv import load_dotenv

from app.api.routes import chat, documents, upload, generate, admin
from app.core.config import settings
from app.services.database_service import DatabaseService

# Load environment variables
load_dotenv()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        message_str = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.add(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Cache control middleware
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add no-cache headers for all API endpoints and static files
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Accel-Expires"] = "0"

        return response


# Global connection manager instance
manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(
    title="Personal Assistant AI Chatbot",
    description="AI chatbot with document ingestion and RAG capabilities",
    version="1.0.0",
)

# Add no-cache middleware first
app.add_middleware(NoCacheMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket endpoint for real-time logs
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for streaming processing logs"""
    await manager.connect(websocket)
    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps(
                {
                    "type": "connection",
                    "message": "Connected to processing logs",
                    "timestamp": datetime.now().isoformat(),
                    "level": "info",
                }
            ),
            websocket,
        )

        # Keep connection alive
        while True:
            # Wait for any message from client (can be heartbeat)
            try:
                data = await websocket.receive_text()
                # Echo heartbeat or handle commands
                if data == "ping":
                    await manager.send_personal_message(
                        json.dumps(
                            {"type": "pong", "timestamp": datetime.now().isoformat()}
                        ),
                        websocket,
                    )
            except WebSocketDisconnect:
                break
            except:
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Function to broadcast logs to all connected clients
async def broadcast_log(level: str, message: str, details: dict = None):
    """Broadcast log message to all connected WebSocket clients"""
    log_message = {
        "type": "log",
        "level": level,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
    }
    await manager.broadcast(log_message)


# Include API routes with proper organization
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(generate.router, prefix="/api/generate", tags=["generate"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# Initialize WebSocket logging for upload module and document services
from app.api.routes import upload
from app.services import document_service

upload.broadcast_log_func = broadcast_log
document_service.broadcast_log_func = broadcast_log

# Mount static files
if os.path.exists("./data"):
    app.mount("/static", StaticFiles(directory="./data"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize database (constructor already initializes it)
    db_service = DatabaseService()

    # Create necessary directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./embeddings", exist_ok=True)

    # Broadcast startup message
    await broadcast_log("info", "🚀 Backend server started and ready")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Personal Assistant AI Chatbot API",
        "version": "1.0.0",
        "status": "active",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# Make manager available globally for other modules
app.state.ws_manager = manager
app.state.broadcast_log = broadcast_log
