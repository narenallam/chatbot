"""
Service Registry - Registers all AI component implementations
"""

import logging

logger = logging.getLogger(__name__)

def register_all_implementations():
    """Register all available AI component implementations"""
    try:
        # Import and register embedding models
        from app.implementations.embedding_models import (
            SentenceTransformersEmbedding,
            OpenAIEmbedding,
            HuggingFaceEmbedding
        )
        from app.core.interfaces import ServiceFactory
        
        # Register embedding models
        ServiceFactory.register_embedding_model("sentence_transformers", SentenceTransformersEmbedding)
        ServiceFactory.register_embedding_model("openai", OpenAIEmbedding)
        ServiceFactory.register_embedding_model("huggingface", HuggingFaceEmbedding)
        
        # Import and register vector databases
        from app.implementations.vector_databases import (
            ChromaDBImplementation,
            FAISSImplementation
        )
        
        ServiceFactory.register_vector_database("chromadb", ChromaDBImplementation)
        ServiceFactory.register_vector_database("faiss", FAISSImplementation)
        
        # Import and register precise retrievers
        from app.implementations.precise_retrievers import (
            ColBERTImplementation,
            BM25Implementation
        )
        
        ServiceFactory.register_precise_retriever("colbert", ColBERTImplementation)
        ServiceFactory.register_precise_retriever("bm25", BM25Implementation)
        
        # Import and register LLM models
        from app.implementations.llm_models import (
            OllamaLLM,
            OpenAILLM,
            HuggingFaceLLM
        )
        
        ServiceFactory.register_llm_model("ollama", OllamaLLM)
        ServiceFactory.register_llm_model("openai", OpenAILLM)
        ServiceFactory.register_llm_model("huggingface", HuggingFaceLLM)
        
        logger.info("✅ All AI component implementations registered successfully")
        
        # Log available implementations
        available = ServiceFactory.list_available_implementations()
        logger.info(f"Available implementations: {available}")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import implementation: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to register implementations: {e}")
        raise