"""
Vector Database Service using ChromaDB for embeddings and similarity search
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from langchain.schema import Document

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorService:
    """Vector database service for storing and querying document embeddings"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create embeddings directory if it doesn't exist
            Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "Personal Assistant AI Chatbot documents"},
            )

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.embedding_model)

            logger.info(
                f"Vector service initialized with collection: {settings.chroma_collection_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {e}")
            raise

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add document chunks to the vector database

        Args:
            documents: List of Document objects with page_content and metadata

        Returns:
            List of chunk IDs
        """
        try:
            if not documents:
                return []

            # Extract texts and metadatas from Document objects
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Generate unique IDs for each chunk
            chunk_ids = []
            for i, metadata in enumerate(metadatas):
                chunk_id = metadata.get("chunk_id", f"chunk_{uuid.uuid4()}")
                chunk_ids.append(chunk_id)

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=chunk_ids,
            )

            logger.info(f"Added {len(documents)} chunks to vector database")
            return chunk_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    async def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of search results with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1
                        - results["distances"][0][i],  # Convert distance to similarity
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    }
                    formatted_results.append(result)

            logger.info(
                f"Found {len(formatted_results)} similar chunks for query: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise

    async def delete_documents(self, chunk_ids: List[str]) -> bool:
        """
        Delete specific chunks by their IDs

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            True if successful
        """
        try:
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks from vector database")

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    def get_document_chunks(self, file_hash: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific file

        Args:
            file_hash: File hash

        Returns:
            List of document chunks
        """
        try:
            results = self.collection.get(
                where={"file_hash": file_hash}, include=["documents", "metadatas"]
            )

            chunks = []
            if results["documents"]:
                for i, text in enumerate(results["documents"]):
                    chunk = {
                        "text": text,
                        "metadata": (
                            results["metadatas"][i] if results["metadatas"] else {}
                        ),
                        "file_hash": file_hash,
                    }
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            raise

    def delete_document(self, file_hash: str) -> bool:
        """
        Delete all chunks for a file

        Args:
            file_hash: File hash to delete

        Returns:
            True if successful
        """
        try:
            # Get all chunk IDs for the file
            results = self.collection.get(
                where={"file_hash": file_hash}, include=["documents"]
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for file {file_hash}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics

        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.collection.count()

            # Get unique file count
            all_results = self.collection.get(include=["metadatas"])
            unique_files = set()
            if all_results["metadatas"]:
                for metadata in all_results["metadatas"]:
                    file_hash = metadata.get("file_hash")
                    if file_hash:
                        unique_files.add(file_hash)

            return {
                "total_chunks": count,
                "unique_files": len(unique_files),
                "collection_name": settings.chroma_collection_name,
                "embedding_model": settings.embedding_model,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_chunks": 0,
                "unique_files": 0,
                "collection_name": settings.chroma_collection_name,
                "embedding_model": settings.embedding_model,
            }

    def reset_collection(self) -> bool:
        """
        Reset the entire collection (delete all data)

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(settings.chroma_collection_name)
            self.collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "Personal Assistant AI Chatbot documents"},
            )
            logger.info("Collection reset successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

    def hybrid_search(
        self, query: str, n_results: int = 5, file_hashes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic similarity with keyword matching

        Args:
            query: Search query
            n_results: Number of results to return
            file_hashes: Optional list of file hashes to filter by

        Returns:
            List of search results with enhanced relevance scoring
        """
        try:
            # Step 1: Perform semantic search
            semantic_results = self.search_similar(query, n_results * 2, file_hashes)

            if not semantic_results:
                return []

            # Step 2: Add keyword scoring
            query_words = query.lower().split()

            for result in semantic_results:
                text = result.get("text", "").lower()

                # Calculate keyword match score
                keyword_matches = sum(1 for word in query_words if word in text)
                keyword_score = keyword_matches / len(query_words) if query_words else 0

                # Combine semantic and keyword scores
                semantic_score = result.get("similarity_score", 0)
                # Ensure semantic score is positive (convert distance to similarity if needed)
                if semantic_score < 0:
                    semantic_score = 1 / (1 + abs(semantic_score))

                hybrid_score = (semantic_score * 0.7) + (keyword_score * 0.3)

                result["hybrid_score"] = hybrid_score
                result["keyword_score"] = keyword_score

            # Step 3: Sort by hybrid score and return top results
            sorted_results = sorted(
                semantic_results, key=lambda x: x.get("hybrid_score", 0), reverse=True
            )
            return sorted_results[:n_results]

        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            # Fallback to regular semantic search
            return self.search_similar(query, n_results, file_hashes)

    def search_similar(
        self, query: str, n_results: int = 5, file_hashes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            file_hashes: Optional list of file hashes to filter by

        Returns:
            List of search results with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Prepare where clause for filtering
            where_clause = None
            if file_hashes:
                where_clause = {"file_hash": {"$in": file_hashes}}

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1
                        - results["distances"][0][i],  # Convert distance to similarity
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    }
                    formatted_results.append(result)

            logger.info(
                f"Found {len(formatted_results)} similar chunks for query: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise


# Global vector service instance
vector_service = VectorService()
