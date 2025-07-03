"""
Technology-Specific Embedding Model Implementations
"""

import logging
from typing import List, Dict, Any
import asyncio

from app.core.interfaces import EmbeddingModelInterface, ServiceFactory

logger = logging.getLogger(__name__)

class SentenceTransformersEmbedding(EmbeddingModelInterface):
    """Sentence Transformers implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.device = config.get('device', 'cpu')
        self.batch_size = config.get('batch_size', 32)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Initialized SentenceTransformers model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformers model: {e}")
            raise
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.encode, batch
                )
                all_embeddings.extend(embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def encode_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")
            
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.model.encode, [text]
            )
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 768  # Default for most models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "sentence_transformers",
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.get_dimension(),
            "batch_size": self.batch_size
        }

class OpenAIEmbedding(EmbeddingModelInterface):
    """OpenAI embedding implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name', 'text-embedding-ada-002')
        self.batch_size = config.get('batch_size', 100)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI embedding client: {self.model_name}")
        except ImportError:
            logger.error("openai not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    async def encode_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            if not self.client:
                raise RuntimeError("Client not initialized")
            
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating single OpenAI embedding: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        # OpenAI embedding dimensions
        dimension_map = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        return dimension_map.get(self.model_name, 1536)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "openai",
            "model_name": self.model_name,
            "dimension": self.get_dimension(),
            "batch_size": self.batch_size,
            "provider": "OpenAI"
        }

class HuggingFaceEmbedding(EmbeddingModelInterface):
    """HuggingFace Transformers embedding implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = config.get('device', 'cpu')
        self.batch_size = config.get('batch_size', 16)
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info(f"Initialized HuggingFace model: {self.model_name}")
            
        except ImportError:
            logger.error("transformers not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            import torch
            
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                if self.device == 'cuda':
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating HuggingFace embeddings: {e}")
            raise
    
    async def encode_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        result = await self.encode([text])
        return result[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.config.hidden_size
        return 768  # Default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "huggingface",
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.get_dimension(),
            "batch_size": self.batch_size,
            "provider": "HuggingFace"
        }

# Register implementations
ServiceFactory.register_embedding_model("sentence_transformers", SentenceTransformersEmbedding)
ServiceFactory.register_embedding_model("openai", OpenAIEmbedding)
ServiceFactory.register_embedding_model("huggingface", HuggingFaceEmbedding)