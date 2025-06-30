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
                        file_hash TEXT,
                        metadata TEXT
                    )
                """
                )

                # Create files table for tracking uploaded files
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS files (
                        id TEXT PRIMARY KEY,
                        full_filename TEXT NOT NULL,
                        file_hash TEXT NOT NULL UNIQUE,
                        uploaded_datetime DATETIME NOT NULL,
                        new_filename TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        file_data_hash TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        status TEXT DEFAULT 'uploaded',
                        metadata TEXT
                    )
                """
                )

                # Add file_hash column if it doesn't exist (for existing databases)
                try:
                    cursor.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

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
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, user_message, ai_response, sources, timestamp, metadata
                    FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (session_id, limit),
                )

                conversations = []
                for row in cursor.fetchall():
                    conversations.append(
                        {
                            "id": row[0],
                            "user_message": row[1],
                            "ai_response": row[2],
                            "sources": json.loads(row[3]) if row[3] else [],
                            "timestamp": row[4],
                            "metadata": json.loads(row[5]) if row[5] else {},
                        }
                    )

                return conversations

        except Exception as e:
            logger.error(f"Get conversation history error: {e}")
            return []

    def get_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent sessions

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, created_at, last_activity, metadata
                    FROM sessions 
                    ORDER BY last_activity DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                sessions = []
                for row in cursor.fetchall():
                    sessions.append(
                        {
                            "id": row[0],
                            "created_at": row[1],
                            "last_activity": row[2],
                            "metadata": json.loads(row[3]) if row[3] else {},
                        }
                    )

                return sessions

        except Exception as e:
            logger.error(f"Get sessions error: {e}")
            return []

    def save_document(
        self,
        filename: str,
        content: str,
        doc_type: str,
        file_hash: str = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Save document content to database

        Args:
            filename: Original filename
            content: Document content
            doc_type: Document type
            file_hash: File hash for deduplication
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
                    INSERT INTO documents (id, filename, content, doc_type, upload_date, file_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        document_id,
                        filename,
                        content,
                        doc_type,
                        timestamp,
                        file_hash,
                        metadata_json,
                    ),
                )
                conn.commit()

            return document_id

        except Exception as e:
            logger.error(f"Save document error: {e}")
            return str(uuid.uuid4())  # Return a fallback ID

    def save_file_info(
        self,
        full_filename: str,
        file_hash: str,
        new_filename: str,
        file_size: int,
        file_data_hash: str,
        content_type: str,
        metadata: Dict[str, Any] = None,
        file_id: str = None,
    ) -> str:
        """
        Save file information to database

        Args:
            full_filename: Original filename
            file_hash: File hash for deduplication
            new_filename: Hashed filename
            file_size: File size in bytes
            file_data_hash: Hash of file data + size
            content_type: MIME content type
            metadata: Additional metadata

        Returns:
            File ID
        """
        try:
            if file_id is None:
                file_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO files (id, full_filename, file_hash, uploaded_datetime, new_filename, file_size, file_data_hash, content_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        file_id,
                        full_filename,
                        file_hash,
                        timestamp,
                        new_filename,
                        file_size,
                        file_data_hash,
                        content_type,
                        metadata_json,
                    ),
                )
                conn.commit()

            return file_id

        except Exception as e:
            logger.error(f"Save file info error: {e}")
            return str(uuid.uuid4())  # Return a fallback ID

    def check_file_exists(self, file_data_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if file with this data hash already exists

        Args:
            file_data_hash: Hash of file data + size

        Returns:
            File info if exists, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, full_filename, file_hash, uploaded_datetime, new_filename, file_size, content_type, metadata
                    FROM files 
                    WHERE file_data_hash = ?
                """,
                    (file_data_hash,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "full_filename": row[1],
                        "file_hash": row[2],
                        "uploaded_datetime": row[3],
                        "new_filename": row[4],
                        "file_size": row[5],
                        "content_type": row[6],
                        "metadata": json.loads(row[7]) if row[7] else {},
                    }
                return None

        except Exception as e:
            logger.error(f"Check file exists error: {e}")
            return None

    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents

        Returns:
            List of document dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, filename, doc_type, upload_date, file_hash, metadata
                    FROM documents 
                    ORDER BY upload_date DESC
                """
                )

                documents = []
                for row in cursor.fetchall():
                    documents.append(
                        {
                            "id": row[0],
                            "filename": row[1],
                            "doc_type": row[2],
                            "upload_date": row[3],
                            "file_hash": row[4],
                            "metadata": json.loads(row[5]) if row[5] else {},
                        }
                    )

                return documents

        except Exception as e:
            logger.error(f"Get documents error: {e}")
            return []

    def get_files(self) -> List[Dict[str, Any]]:
        """
        Get all uploaded files

        Returns:
            List of file dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, full_filename, file_hash, uploaded_datetime, new_filename, file_size, content_type, status, metadata
                    FROM files 
                    ORDER BY uploaded_datetime DESC
                """
                )

                files = []
                for row in cursor.fetchall():
                    files.append(
                        {
                            "id": row[0],
                            "full_filename": row[1],
                            "file_hash": row[2],
                            "uploaded_datetime": row[3],
                            "new_filename": row[4],
                            "file_size": row[5],
                            "content_type": row[6],
                            "status": row[7],
                            "metadata": json.loads(row[8]) if row[8] else {},
                        }
                    )

                return files

        except Exception as e:
            logger.error(f"Get files error: {e}")
            return []

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file record

        Args:
            file_id: File ID to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Delete file error: {e}")
            return False


# Create a global instance
database_service = DatabaseService()
