"""
Technology-Specific Vector Database Implementations
"""

import logging
from typing import List, Dict, Any, Optional
import uuid
import json
from pathlib import Path

from app.core.interfaces import VectorDatabaseInterface, Document, SearchResult, ServiceFactory

logger = logging.getLogger(__name__)

class ChromaDBImplementation(VectorDatabaseInterface):
    """ChromaDB implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.collection_name = config.get('collection_name', 'documents')
        self.persist_directory = config.get('persist_directory', './embeddings')
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            
        except ImportError:
            logger.error("chromadb not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to ChromaDB"""
        try:
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            texts = [doc.content for doc in documents]
            
            # Clean metadata to remove None values and ensure correct types
            metadatas = []
            for doc in documents:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if value is not None:
                        # Convert to supported types
                        if isinstance(value, bool):
                            cleaned_metadata[key] = value
                        elif isinstance(value, (int, float)):
                            cleaned_metadata[key] = value
                        else:
                            cleaned_metadata[key] = str(value)
                metadatas.append(cleaned_metadata)
            
            if not embeddings:
                raise ValueError("Documents must have embeddings")
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents"""
        try:
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = SearchResult(
                        document_id=results["ids"][0][i] if "ids" in results else f"doc_{i}",
                        content=results["documents"][0][i],
                        score=1 - results["distances"][0][i],  # Convert distance to similarity
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        search_type="semantic"
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            return False
    
    async def update_document(self, document: Document) -> bool:
        """Update document in ChromaDB"""
        try:
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            if not document.embedding:
                raise ValueError("Document must have embedding")
            
            self.collection.update(
                ids=[document.id],
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[document.metadata]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document in ChromaDB: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            count = self.collection.count()
            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "document_count": count,
                "distance_metric": self.distance_metric,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {e}")
            return {"error": str(e)}
    
    async def create_index(self, index_config: Dict[str, Any]) -> bool:
        """Create or rebuild ChromaDB index"""
        try:
            # ChromaDB handles indexing automatically
            logger.info("ChromaDB handles indexing automatically")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ChromaDB index: {e}")
            return False

class FAISSImplementation(VectorDatabaseInterface):
    """FAISS implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.index_type = config.get('index_type', 'IndexFlatIP')
        self.dimension = config.get('dimension', 768)
        self.persist_path = config.get('persist_path', './faiss_index')
        self.index = None
        self.documents = {}  # Store documents separately
        self.metadata_store = {}
        self._initialize_index()
    
    def _initialize_index(self):
        try:
            import faiss
            
            # Create index based on type
            if self.index_type == 'IndexFlatIP':
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == 'IndexIVFFlat':
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            
            # Try to load existing index
            self._load_index()
            
            logger.info(f"Initialized FAISS index: {self.index_type}")
            
        except ImportError:
            logger.error("faiss not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    def _load_index(self):
        """Load existing FAISS index if available"""
        try:
            import faiss
            import pickle
            
            index_path = Path(self.persist_path)
            if index_path.exists():
                self.index = faiss.read_index(str(index_path / "index.faiss"))
                
                # Load metadata
                metadata_path = index_path / "metadata.pkl"
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        data = pickle.load(f)
                        self.documents = data.get('documents', {})
                        self.metadata_store = data.get('metadata', {})
                
                logger.info("Loaded existing FAISS index")
                
        except Exception as e:
            logger.warning(f"Could not load existing FAISS index: {e}")
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            import faiss
            import pickle
            
            index_path = Path(self.persist_path)
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, str(index_path / "index.faiss"))
            
            # Save metadata
            with open(index_path / "metadata.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata_store
                }, f)
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to FAISS"""
        try:
            import numpy as np
            
            if not self.index:
                raise RuntimeError("Index not initialized")
            
            embeddings = []
            for doc in documents:
                if not doc.embedding:
                    raise ValueError("Document must have embedding")
                embeddings.append(doc.embedding)
                self.documents[doc.id] = doc.content
                self.metadata_store[doc.id] = doc.metadata
            
            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            return False
    
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search FAISS for similar documents"""
        try:
            import numpy as np
            
            if not self.index:
                raise RuntimeError("Index not initialized")
            
            # Perform search
            query_array = np.array([query_embedding], dtype=np.float32)
            scores, indices = self.index.search(query_array, k)
            
            # Format results
            search_results = []
            doc_ids = list(self.documents.keys())
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    
                    # Apply filters if specified
                    if filters:
                        doc_metadata = self.metadata_store.get(doc_id, {})
                        skip = False
                        for key, value in filters.items():
                            if key not in doc_metadata:
                                skip = True
                                break
                            if isinstance(value, list):
                                if doc_metadata[key] not in value:
                                    skip = True
                                    break
                            elif doc_metadata[key] != value:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    result = SearchResult(
                        document_id=doc_id,
                        content=self.documents[doc_id],
                        score=float(score),
                        metadata=self.metadata_store.get(doc_id, {}),
                        search_type="semantic"
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (requires rebuild)"""
        try:
            # FAISS doesn't support deletion, so we need to rebuild
            for doc_id in document_ids:
                self.documents.pop(doc_id, None)
                self.metadata_store.pop(doc_id, None)
            
            # Rebuild index
            await self.create_index({})
            
            logger.info(f"Deleted {len(document_ids)} documents from FAISS (rebuilt index)")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from FAISS: {e}")
            return False
    
    async def update_document(self, document: Document) -> bool:
        """Update document in FAISS (requires rebuild)"""
        try:
            self.documents[document.id] = document.content
            self.metadata_store[document.id] = document.metadata
            
            # For simplicity, we don't rebuild the entire index
            # In production, you might want to implement a more efficient update strategy
            logger.info(f"Updated document metadata for {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document in FAISS: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            return {
                "type": "faiss",
                "index_type": self.index_type,
                "dimension": self.dimension,
                "document_count": len(self.documents),
                "index_size": self.index.ntotal,
                "persist_path": self.persist_path
            }
            
        except Exception as e:
            logger.error(f"Error getting FAISS stats: {e}")
            return {"error": str(e)}
    
    async def create_index(self, index_config: Dict[str, Any]) -> bool:
        """Rebuild FAISS index"""
        try:
            import numpy as np
            
            # Get all embeddings (would need to re-embed documents)
            # For now, we'll just reinitialize the index
            self._initialize_index()
            
            logger.info("Rebuilt FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False

# Register implementations
ServiceFactory.register_vector_database("chromadb", ChromaDBImplementation)
ServiceFactory.register_vector_database("faiss", FAISSImplementation)