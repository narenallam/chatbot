"""
Technology-Agnostic Interfaces for AI Components
Provides abstraction layers for LLMs, Vector DBs, and Embedding Models
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

@dataclass
class Document:
    """Universal document representation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Universal search result representation"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    search_type: str
    matched_tokens: Optional[List[str]] = None
    coordinates: Optional[Tuple[int, int]] = None

@dataclass
class QueryAnalysis:
    """Universal query analysis result"""
    query: str
    intent: str  # 'table_specific', 'numerical', 'temporal', 'general'
    confidence: float
    extracted_entities: List[str]
    filters: Dict[str, Any]
    suggested_strategy: str

class SearchStrategy(Enum):
    """Search strategy enumeration"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"
    PRECISE = "precise"
    AUTO = "auto"

class ModelType(Enum):
    """Model type enumeration"""
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"

# Abstract Base Classes for Technology Independence

class EmbeddingModelInterface(ABC):
    """Abstract interface for embedding models"""
    
    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    async def encode_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector database"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the database"""
        pass
    
    @abstractmethod
    async def update_document(self, document: Document) -> bool:
        """Update a document in the database"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass
    
    @abstractmethod
    async def create_index(self, index_config: Dict[str, Any]) -> bool:
        """Create or rebuild index"""
        pass

class PreciseRetrievalInterface(ABC):
    """Abstract interface for precise retrieval systems (like ColBERT)"""
    
    @abstractmethod
    async def index_documents(self, documents: List[Document]) -> bool:
        """Index documents for precise retrieval"""
        pass
    
    @abstractmethod
    async def search_precise(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform precise token-level search"""
        pass
    
    @abstractmethod
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for intent and structure"""
        pass
    
    @abstractmethod
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        pass

class LLMInterface(ABC):
    """Abstract interface for Large Language Models"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat-based text generation"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        pass

class SearchServiceInterface(ABC):
    """Abstract interface for search services"""
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        strategy: SearchStrategy = SearchStrategy.AUTO,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform search with specified strategy"""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to all relevant indexes"""
        pass
    
    @abstractmethod
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine optimal search strategy"""
        pass
    
    @abstractmethod
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        pass

class DocumentProcessorInterface(ABC):
    """Abstract interface for document processing"""
    
    @abstractmethod
    async def extract_content(self, file_content: bytes, filename: str) -> str:
        """Extract text content from file"""
        pass
    
    @abstractmethod
    async def extract_tables(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract structured table data"""
        pass
    
    @abstractmethod
    async def chunk_content(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk content into documents"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        pass

# Factory Pattern for Technology Independence

class ServiceFactory:
    """Factory for creating technology-specific implementations"""
    
    _embedding_models = {}
    _vector_databases = {}
    _precise_retrievers = {}
    _llm_models = {}
    _document_processors = {}
    
    @classmethod
    def register_embedding_model(cls, name: str, implementation_class):
        """Register an embedding model implementation"""
        cls._embedding_models[name] = implementation_class
    
    @classmethod
    def register_vector_database(cls, name: str, implementation_class):
        """Register a vector database implementation"""
        cls._vector_databases[name] = implementation_class
    
    @classmethod
    def register_precise_retriever(cls, name: str, implementation_class):
        """Register a precise retriever implementation"""
        cls._precise_retrievers[name] = implementation_class
    
    @classmethod
    def register_llm_model(cls, name: str, implementation_class):
        """Register an LLM implementation"""
        cls._llm_models[name] = implementation_class
    
    @classmethod
    def register_document_processor(cls, name: str, implementation_class):
        """Register a document processor implementation"""
        cls._document_processors[name] = implementation_class
    
    @classmethod
    def create_embedding_model(cls, name: str, config: Dict[str, Any] = None) -> EmbeddingModelInterface:
        """Create an embedding model instance"""
        if name not in cls._embedding_models:
            raise ValueError(f"Unknown embedding model: {name}")
        return cls._embedding_models[name](config or {})
    
    @classmethod
    def create_vector_database(cls, name: str, config: Dict[str, Any] = None) -> VectorDatabaseInterface:
        """Create a vector database instance"""
        if name not in cls._vector_databases:
            raise ValueError(f"Unknown vector database: {name}")
        return cls._vector_databases[name](config or {})
    
    @classmethod
    def create_precise_retriever(cls, name: str, config: Dict[str, Any] = None) -> PreciseRetrievalInterface:
        """Create a precise retriever instance"""
        if name not in cls._precise_retrievers:
            raise ValueError(f"Unknown precise retriever: {name}")
        return cls._precise_retrievers[name](config or {})
    
    @classmethod
    def create_llm_model(cls, name: str, config: Dict[str, Any] = None) -> LLMInterface:
        """Create an LLM instance"""
        if name not in cls._llm_models:
            raise ValueError(f"Unknown LLM model: {name}")
        return cls._llm_models[name](config or {})
    
    @classmethod
    def create_document_processor(cls, name: str, config: Dict[str, Any] = None) -> DocumentProcessorInterface:
        """Create a document processor instance"""
        if name not in cls._document_processors:
            raise ValueError(f"Unknown document processor: {name}")
        return cls._document_processors[name](config or {})
    
    @classmethod
    def list_available_implementations(cls) -> Dict[str, List[str]]:
        """List all available implementations"""
        return {
            "embedding_models": list(cls._embedding_models.keys()),
            "vector_databases": list(cls._vector_databases.keys()),
            "precise_retrievers": list(cls._precise_retrievers.keys()),
            "llm_models": list(cls._llm_models.keys()),
            "document_processors": list(cls._document_processors.keys())
        }

# Configuration Management for Technology Independence

@dataclass
class AIConfig:
    """Configuration for AI components"""
    embedding_model: str
    embedding_config: Dict[str, Any]
    vector_database: str
    vector_config: Dict[str, Any]
    precise_retriever: Optional[str]
    precise_config: Optional[Dict[str, Any]]
    llm_model: str
    llm_config: Dict[str, Any]
    search_config: Dict[str, Any]
    fallback_configs: Optional[Dict[str, Any]] = None

class ConfigManager:
    """Manages configuration for technology-agnostic AI components"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.current_config: Optional[AIConfig] = None
        
    async def load_config(self, config: Union[Dict[str, Any], AIConfig]) -> AIConfig:
        """Load configuration from dict or AIConfig object"""
        if isinstance(config, dict):
            self.current_config = AIConfig(**config)
        else:
            self.current_config = config
        return self.current_config
    
    async def save_config(self, config: AIConfig, file_path: Optional[str] = None):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        save_path = file_path or self.config_file
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
    
    def get_current_config(self) -> Optional[AIConfig]:
        """Get current configuration"""
        return self.current_config
    
    async def validate_config(self, config: AIConfig) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # Check if all required implementations are registered
        available = ServiceFactory.list_available_implementations()
        
        if config.embedding_model not in available["embedding_models"]:
            errors.append(f"Embedding model '{config.embedding_model}' not registered")
        
        if config.vector_database not in available["vector_databases"]:
            errors.append(f"Vector database '{config.vector_database}' not registered")
        
        if config.llm_model not in available["llm_models"]:
            errors.append(f"LLM model '{config.llm_model}' not registered")
        
        
        if config.precise_retriever and config.precise_retriever not in available["precise_retrievers"]:
            errors.append(f"Precise retriever '{config.precise_retriever}' not registered")
        
        return len(errors) == 0, errors

# Global instances
service_factory = ServiceFactory()
config_manager = ConfigManager()