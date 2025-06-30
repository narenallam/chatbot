"""
Admin API routes for system management
"""

from fastapi import APIRouter, HTTPException
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reset-database")
async def reset_database():
    """
    Reset all databases and clear uploaded documents

    Returns:
        Success message with cleared items
    """
    try:
        cleared_items = []

        # Clear ChromaDB vector database
        data_path = Path("data")
        if data_path.exists():
            shutil.rmtree(data_path)
            data_path.mkdir(exist_ok=True)
            cleared_items.append("ChromaDB vector database")

        # Clear SQLite chat database
        chat_db_path = Path("chat_history.db")
        if chat_db_path.exists():
            chat_db_path.unlink()
            cleared_items.append("SQLite chat database")

        # Clear uploaded documents
        uploads_path = Path("uploads")
        if uploads_path.exists():
            shutil.rmtree(uploads_path)
            uploads_path.mkdir(exist_ok=True)
            cleared_items.append("Uploaded documents")

        # Clear embeddings
        embeddings_path = Path("embeddings")
        if embeddings_path.exists():
            shutil.rmtree(embeddings_path)
            embeddings_path.mkdir(exist_ok=True)
            cleared_items.append("Embeddings")

        # Clear Python cache files
        for root, dirs, files in os.walk("."):
            # Remove __pycache__ directories
            if "__pycache__" in dirs:
                cache_path = Path(root) / "__pycache__"
                shutil.rmtree(cache_path)

            # Remove .pyc files
            for file in files:
                if file.endswith(".pyc"):
                    pyc_path = Path(root) / file
                    pyc_path.unlink()

        cleared_items.append("Python cache files")

        logger.info(f"Database reset completed. Cleared: {', '.join(cleared_items)}")

        return {
            "message": "Database reset completed successfully",
            "cleared_items": cleared_items,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset database: {str(e)}"
        )


@router.get("/status")
async def get_system_status():
    """
    Get detailed system status for admin purposes

    Returns:
        System status information
    """
    try:
        status = {
            "databases": {
                "chromadb": Path("data").exists(),
                "chat_history": Path("chat_history.db").exists(),
            },
            "directories": {
                "uploads": Path("uploads").exists(),
                "embeddings": Path("embeddings").exists(),
            },
            "document_count": 0,
            "upload_count": 0,
        }

        # Count documents in uploads directory
        uploads_path = Path("uploads")
        if uploads_path.exists():
            status["upload_count"] = len(list(uploads_path.glob("*")))

        return status

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )
