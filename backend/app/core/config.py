"""
Configuration settings for the Personal Assistant AI Chatbot
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List
import os
import json


class Settings(BaseSettings):
    """Application settings"""

    # Application Settings
    app_name: str = Field(default="Personal Assistant AI Chatbot", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=True, env="RELOAD")
    
    # CORS Settings
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")

    # LLM Settings
    llm_provider: str = Field(
        default="ollama", env="LLM_PROVIDER"
    )  # "ollama" or "openai"

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")

    # Gemini Settings
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")

    # Web Search Settings
    web_search_enabled: bool = Field(default=True, env="WEB_SEARCH_ENABLED")
    serpapi_api_key: Optional[str] = Field(default=None, env="SERPAPI_API_KEY")
    brave_api_key: Optional[str] = Field(default=None, env="BRAVE_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_search_engine_id: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_ENGINE_ID")
    bing_api_key: Optional[str] = Field(default=None, env="BING_API_KEY")

    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")

    # ChromaDB Settings
    chroma_db_path: str = Field(default="./embeddings", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    chroma_openai_db_path: str = Field(default="./embeddings_openai", env="CHROMA_OPENAI_DB_PATH")
    chroma_openai_collection_name: str = Field(default="documents_openai", env="CHROMA_OPENAI_COLLECTION_NAME")

    # File Storage Settings
    data_storage_path: str = Field(default="./data", env="DATA_STORAGE_PATH")
    original_files_path: str = Field(default="./data/original_files", env="ORIGINAL_FILES_PATH")
    hashed_files_path: str = Field(default="./data/hashed_files", env="HASHED_FILES_PATH")
    converted_files_path: str = Field(default="./data/converted", env="CONVERTED_FILES_PATH")
    temp_files_path: str = Field(default="./data/temp", env="TEMP_FILES_PATH")
    metadata_path: str = Field(default="./data/metadata", env="METADATA_PATH")
    logs_path: str = Field(default="./data/logs", env="LOGS_PATH")
    max_file_size_mb: int = Field(default=500, env="MAX_FILE_SIZE_MB")
    allowed_file_types: list = Field(
        default=[
            ".pdf",
            ".txt",
            ".docx",
            ".doc",
            ".md",
            ".csv",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".html",
            ".htm",
            ".png",
            ".jpg",
            ".jpeg",
            ".heic",
            ".bmp",
            ".gif",
            ".tiff",
        ],
        env="ALLOWED_FILE_TYPES",
    )

    # Database Settings
    database_url: str = Field(default="sqlite:///./data/chatbot.db", env="DATABASE_URL")

    # RAG Settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    min_relevance_score: float = Field(default=0.5, env="MIN_RELEVANCE_SCORE")
    search_type: str = Field(default="similarity", env="SEARCH_TYPE")
    hybrid_search_alpha: float = Field(default=0.5, env="HYBRID_SEARCH_ALPHA")

    # Embedding Settings
    embedding_provider: str = Field(default="sentence_transformers", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # Chat Settings
    max_chat_history: int = Field(default=10, env="MAX_CHAT_HISTORY")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    conversation_memory_type: str = Field(default="buffer_window", env="CONVERSATION_MEMORY_TYPE")

    # Performance Settings
    enable_parallel_processing: bool = Field(default=True, env="ENABLE_PARALLEL_PROCESSING")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    parallel_threshold_mb: int = Field(default=10, env="PARALLEL_THRESHOLD_MB")
    batch_size: int = Field(default=100, env="BATCH_SIZE")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    multiprocessing_enabled: bool = Field(default=True, env="MULTIPROCESSING_ENABLED")

    # OCR Settings
    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    ocr_languages: str = Field(default="eng", env="OCR_LANGUAGES")
    ocr_dpi: int = Field(default=300, env="OCR_DPI")

    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    log_max_size_mb: int = Field(default=100, env="LOG_MAX_SIZE_MB")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    enable_detailed_logging: bool = Field(default=True, env="ENABLE_DETAILED_LOGGING")

    # Vector Search Settings
    vector_search_k: int = Field(default=10, env="VECTOR_SEARCH_K")
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")
    use_reranking: bool = Field(default=True, env="USE_RERANKING")
    
    # Security Settings
    secret_key: str = Field(default="your_secret_key_here_generate_a_secure_one", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=3600, env="RATE_LIMIT_PERIOD")
    
    # HuggingFace Settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    tokenizers_parallelism: str = Field(default="false", env="TOKENIZERS_PARALLELISM")
    
    # Cache Settings
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Feature Flags
    enable_web_search: bool = Field(default=True, env="ENABLE_WEB_SEARCH")
    enable_document_upload: bool = Field(default=True, env="ENABLE_DOCUMENT_UPLOAD")
    enable_chat_history: bool = Field(default=True, env="ENABLE_CHAT_HISTORY")
    enable_analytics: bool = Field(default=False, env="ENABLE_ANALYTICS")

    @validator('cors_origins', 'cors_allow_methods', 'cors_allow_headers', pre=True)
    def parse_json_list(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v.split(',')
        return v
    
    @validator('allowed_file_types', pre=True)
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return v.split(',')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
