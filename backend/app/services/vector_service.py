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

    def add_documents(
        self, texts: List[str], metadatas: List[Dict[str, Any]], document_id: str
    ) -> List[str]:
        """
        Add document chunks to the vector database

        Args:
            texts: List of text chunks
            metadatas: List of metadata for each chunk
            document_id: ID of the source document

        Returns:
            List of chunk IDs
        """
        try:
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(texts))]

            # Add document ID to metadata
            for metadata in metadatas:
                metadata["document_id"] = document_id

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=chunk_ids,
            )

            logger.info(f"Added {len(texts)} chunks for document {document_id}")
            return chunk_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search_similar(
        self, query: str, n_results: int = 5, document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of search results with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Prepare where clause for filtering
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}

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
                        "document_id": results["metadatas"][0][i].get("document_id"),
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

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document

        Args:
            document_id: Document ID

        Returns:
            List of document chunks
        """
        try:
            results = self.collection.get(
                where={"document_id": document_id}, include=["documents", "metadatas"]
            )

            chunks = []
            if results["documents"]:
                for i, text in enumerate(results["documents"]):
                    chunk = {
                        "text": text,
                        "metadata": (
                            results["metadatas"][i] if results["metadatas"] else {}
                        ),
                        "document_id": document_id,
                    }
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"document_id": document_id}, include=["documents"]
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {document_id}"
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

            # Get unique document count
            all_results = self.collection.get(include=["metadatas"])
            unique_docs = set()
            if all_results["metadatas"]:
                for metadata in all_results["metadatas"]:
                    doc_id = metadata.get("document_id")
                    if doc_id:
                        unique_docs.add(doc_id)

            return {
                "total_chunks": count,
                "unique_documents": len(unique_docs),
                "collection_name": settings.chroma_collection_name,
                "embedding_model": settings.embedding_model,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_chunks": 0,
                "unique_documents": 0,
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


# Global vector service instance
vector_service = VectorService()
