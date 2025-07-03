"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document types"""

    PDF = "pdf"
    TEXT = "text"
    DOCX = "docx"
    URL = "url"
    CSV = "csv"
    MARKDOWN = "markdown"


class ChatRole(str, Enum):
    """Chat message roles"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentType(str, Enum):
    """Content generation types"""

    EMAIL = "email"
    LETTER = "letter"
    SOCIAL_POST = "social_post"
    BLOG_POST = "blog_post"
    SUMMARY = "summary"
    REPORT = "report"


# Document Schemas
class DocumentBase(BaseModel):
    """Base document schema"""

    filename: str
    title: Optional[str] = None
    content_type: DocumentType
    description: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document"""

    content: Optional[str] = None
    url: Optional[str] = None


class DocumentResponse(BaseModel):
    """Schema for document response"""

    id: str
    status: DocumentStatus
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    chunks_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for listing documents"""

    documents: List[DocumentResponse]
    total: int
    page: int = 1
    size: int = 10


# Chat Schemas
class ChatMessage(BaseModel):
    """Chat message schema"""

    role: ChatRole
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request schema"""

    message: str
    conversation_id: Optional[str] = None
    use_context: bool = True
    include_web_search: bool = True
    selected_search_engine: Optional[str] = Field(default="duckduckgo", description="Search engine to use: duckduckgo, brave, bing")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """Chat response schema"""

    message: str
    conversation_id: str
    sources: List[Dict[str, Any]] = []
    tokens_used: Optional[int] = None


class ConversationHistory(BaseModel):
    """Conversation history schema"""

    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


# Upload Schemas
class UploadResponse(BaseModel):
    """Upload response schema"""

    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class URLUploadRequest(BaseModel):
    """URL upload request schema"""

    url: str
    title: Optional[str] = None
    description: Optional[str] = None


# Content Generation Schemas
class ContentGenerationRequest(BaseModel):
    """Content generation request schema"""

    content_type: ContentType
    prompt: str
    context_query: Optional[str] = None
    tone: Optional[str] = Field(default="professional", description="Content tone")
    length: Optional[str] = Field(default="medium", description="Content length")
    additional_instructions: Optional[str] = None


class ContentGenerationResponse(BaseModel):
    """Content generation response schema"""

    content: str
    content_type: ContentType
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


# Search Schemas
class SearchRequest(BaseModel):
    """Search request schema"""

    query: str
    document_types: Optional[List[DocumentType]] = None
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    """Search result schema"""

    document_id: str
    filename: str
    chunk_text: str
    similarity_score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Search response schema"""

    results: List[SearchResult]
    total_results: int
    query: str


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema"""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# Health Check Schema
class HealthResponse(BaseModel):
    """Health check response schema"""

    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str] = {}
