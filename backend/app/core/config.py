"""
Configuration settings for the Personal Assistant AI Chatbot
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    api_title: str = "Personal Assistant AI Chatbot"
    api_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # LLM Settings
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")  # "ollama" or "openai"
    
    # OpenAI Settings (fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")

    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3:8b-instruct-q8_0", env="OLLAMA_MODEL")

    # ChromaDB Settings
    chroma_db_path: str = Field(default="./embeddings", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(
        default="documents", env="CHROMA_COLLECTION_NAME"
    )

    # File Storage Settings
    data_storage_path: str = Field(default="./data", env="DATA_STORAGE_PATH")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: list = Field(
        default=[".pdf", ".txt", ".docx", ".doc", ".md", ".csv"],
        env="ALLOWED_FILE_TYPES",
    )

    # Database Settings
    database_url: str = Field(default="sqlite:///./chatbot.db", env="DATABASE_URL")

    # RAG Settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")

    # Embedding Settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # Chat Settings
    max_chat_history: int = Field(default=10, env="MAX_CHAT_HISTORY")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
