"""
Database service for the Personal Assistant AI Chatbot
Handles conversation history and session management using SQLite
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for managing conversations and sessions"""

    def __init__(self, db_path: str = "data/chatbot.db"):
        """
        Initialize database service

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create conversations table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_message TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        sources TEXT,
                        timestamp DATETIME NOT NULL,
                        metadata TEXT
                    )
                """
                )

                # Create sessions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at DATETIME NOT NULL,
                        last_activity DATETIME NOT NULL,
                        metadata TEXT
                    )
                """
                )

                # Create documents table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        content TEXT NOT NULL,
                        doc_type TEXT NOT NULL,
                        upload_date DATETIME NOT NULL,
                        metadata TEXT
                    )
                """
                )

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def create_session(self) -> str:
        """
        Create a new chat session

        Returns:
            Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sessions (id, created_at, last_activity, metadata)
                    VALUES (?, ?, ?, ?)
                """,
                    (session_id, timestamp, timestamp, json.dumps({})),
                )
                conn.commit()

            return session_id

        except Exception as e:
            logger.error(f"Create session error: {e}")
            return str(uuid.uuid4())  # Fallback to in-memory session

    def save_conversation(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        sources: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Save a conversation turn

        Args:
            session_id: Session ID
            user_message: User's message
            ai_response: AI's response
            sources: Source documents used
            metadata: Additional metadata

        Returns:
            Conversation ID
        """
        try:
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            sources_json = json.dumps(sources or [])
            metadata_json = json.dumps(metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Save conversation
                cursor.execute(
                    """
                    INSERT INTO conversations 
                    (id, session_id, user_message, ai_response, sources, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        conversation_id,
                        session_id,
                        user_message,
                        ai_response,
                        sources_json,
                        timestamp,
                        metadata_json,
                    ),
                )

                # Update session last activity
                cursor.execute(
                    """
                    UPDATE sessions SET last_activity = ? WHERE id = ?
                """,
                    (timestamp, session_id),
                )

                conn.commit()

            return conversation_id

        except Exception as e:
            logger.error(f"Save conversation error: {e}")
            return str(uuid.uuid4())  # Return a fallback ID

    def get_conversation_history(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session

        Args:
            session_id: Session ID
            limit: Maximum number of conversations to return

        Returns:
            List of conversation dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (session_id, limit),
                )

                rows = cursor.fetchall()

                conversations = []
                for row in rows:
                    conversation = {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "user_message": row["user_message"],
                        "ai_response": row["ai_response"],
                        "sources": json.loads(row["sources"]) if row["sources"] else [],
                        "timestamp": row["timestamp"],
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                    }
                    conversations.append(conversation)

                return list(reversed(conversations))  # Return in chronological order

        except Exception as e:
            logger.error(f"Get conversation history error: {e}")
            return []

    def save_document(
        self,
        filename: str,
        content: str,
        doc_type: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Save a document to the database

        Args:
            filename: Document filename
            content: Document content
            doc_type: Document type (pdf, txt, etc.)
            metadata: Additional metadata

        Returns:
            Document ID
        """
        try:
            document_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO documents (id, filename, content, doc_type, upload_date, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        document_id,
                        filename,
                        content,
                        doc_type,
                        timestamp,
                        metadata_json,
                    ),
                )
                conn.commit()

            return document_id

        except Exception as e:
            logger.error(f"Save document error: {e}")
            raise

    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents

        Returns:
            List of document dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
                rows = cursor.fetchall()

                documents = []
                for row in rows:
                    document = {
                        "id": row["id"],
                        "filename": row["filename"],
                        "doc_type": row["doc_type"],
                        "upload_date": row["upload_date"],
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                    }
                    documents.append(document)

                return documents

        except Exception as e:
            logger.error(f"Get documents error: {e}")
            return []


# Global database service instance
database_service = DatabaseService()
