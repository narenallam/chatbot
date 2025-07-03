"""
ColBERT Service for Precise Table Retrieval (Phase 2)
Enhanced table query processing with token-level precision
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime

# Core imports
import torch
import numpy as np
from dataclasses import dataclass

# ColBERT specific imports
try:
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries, Collection
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False
    # Fallback classes for development
    class Indexer: pass
    class Searcher: pass
    class RunConfig: pass
    class ColBERTConfig: pass

# Additional utilities
try:
    import rapidfuzz
    import dateparser
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class TableQuery:
    """Structured representation of a table-specific query"""
    query: str
    query_type: str  # 'numerical_range', 'exact_match', 'fuzzy_match', 'temporal'
    target_columns: List[str]
    filters: Dict[str, Any]
    confidence: float

@dataclass
class ColBERTResult:
    """ColBERT search result with enhanced metadata"""
    passage_id: str
    text: str
    score: float
    table_metadata: Dict[str, Any]
    matched_tokens: List[str]
    cell_coordinates: Optional[Tuple[int, int]]  # (row, column) if available

class ColBERTService:
    """
    ColBERT-based service for precise table cell retrieval and token-level search
    """
    
    def __init__(self):
        self.config = None
        self.indexer = None
        self.searcher = None
        self.index_path = Path("colbert_indexes")
        self.current_index_name = "table_precise_v1"
        self.is_initialized = False
        self.table_metadata_store = {}
        
        # Initialize if ColBERT is available
        if COLBERT_AVAILABLE:
            self._initialize_colbert()
        else:
            logger.warning("ColBERT not available. Install with: pip install colbert-ai")
    
    def _initialize_colbert(self):
        """Initialize ColBERT configuration and components"""
        try:
            # Create index directory
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Configure ColBERT for table-specific processing
            self.config = ColBERTConfig(
                nbits=2,  # 2-bit quantization for efficiency
                doc_maxlen=512,  # Longer docs for table content
                query_maxlen=64,  # Reasonable query length
                similarity="cosine",  # Cosine similarity
                index_path=str(self.index_path),
                experiment="table_precise_search"
            )
            
            # Check if index exists
            index_full_path = self.index_path / self.current_index_name
            if index_full_path.exists():
                self._load_existing_index()
            
            self.is_initialized = True
            logger.info("ColBERT service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ColBERT: {e}")
            self.is_initialized = False
    
    def _load_existing_index(self):
        """Load existing ColBERT index if available"""
        try:
            with Run().context(RunConfig(experiment="table_precise_search")):
                self.searcher = Searcher(index=self.current_index_name, config=self.config)
            
            # Load metadata store
            metadata_path = self.index_path / f"{self.current_index_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.table_metadata_store = json.load(f)
            
            logger.info(f"Loaded existing ColBERT index: {self.current_index_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            self.searcher = None
    
    async def index_table_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index table documents with ColBERT for precise retrieval
        
        Args:
            documents: List of document chunks with table metadata
            
        Returns:
            Success status
        """
        if not COLBERT_AVAILABLE or not self.is_initialized:
            logger.warning("ColBERT not available for indexing")
            return False
        
        try:
            # Filter documents to only table-containing chunks
            table_documents = [
                doc for doc in documents 
                if doc.get('metadata', {}).get('contains_table', False)
            ]
            
            if not table_documents:
                logger.info("No table documents to index")
                return True
            
            logger.info(f"Indexing {len(table_documents)} table documents with ColBERT")
            
            # Prepare passages for indexing
            passages = []
            metadata_store = {}
            
            for i, doc in enumerate(table_documents):
                passage_id = str(i)
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                # Enhance text with table structure markers for better token matching
                enhanced_text = self._enhance_text_for_indexing(text, metadata)
                passages.append(enhanced_text)
                
                # Store metadata separately
                metadata_store[passage_id] = {
                    'original_text': text,
                    'metadata': metadata,
                    'document_id': doc.get('id'),
                    'source': metadata.get('source'),
                    'table_type': metadata.get('table_type'),
                    'contains_table': metadata.get('contains_table'),
                    'row_count': metadata.get('row_count', 0),
                    'numeric_data': metadata.get('numeric_data', False)
                }
            
            # Create Collection for ColBERT
            collection = Collection(data=passages)
            
            # Index the collection
            with Run().context(RunConfig(experiment="table_precise_search")):
                self.indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=self.config)
                self.indexer.index(name=self.current_index_name, collection=collection)
            
            # Save metadata store
            self.table_metadata_store.update(metadata_store)
            metadata_path = self.index_path / f"{self.current_index_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.table_metadata_store, f, indent=2)
            
            # Load the searcher
            self._load_existing_index()
            
            logger.info(f"Successfully indexed {len(table_documents)} table documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index table documents: {e}")
            return False
    
    def _enhance_text_for_indexing(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Enhance text with table structure information for better token matching
        
        Args:
            text: Original text content
            metadata: Document metadata
            
        Returns:
            Enhanced text for indexing
        """
        try:
            enhanced_parts = [text]
            
            # Add table type information
            if metadata.get('table_type'):
                enhanced_parts.append(f"TABLE_TYPE:{metadata['table_type']}")
            
            # Add numeric indicators for better numerical queries
            if metadata.get('numeric_data'):
                enhanced_parts.append("CONTAINS_NUMBERS")
            
            # Add row count information
            if metadata.get('row_count', 0) > 0:
                enhanced_parts.append(f"ROWS:{metadata['row_count']}")
            
            # Add source information
            if metadata.get('source'):
                enhanced_parts.append(f"SOURCE:{metadata['source']}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing text for indexing: {e}")
            return text
    
    async def search_tables(self, query: str, k: int = 10) -> List[ColBERTResult]:
        """
        Search table content with ColBERT precision
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of ColBERT search results
        """
        if not COLBERT_AVAILABLE or not self.searcher:
            logger.warning("ColBERT searcher not available")
            return []
        
        try:
            # Analyze query for table-specific enhancements
            enhanced_query = await self._enhance_query_for_search(query)
            
            # Perform ColBERT search
            results = self.searcher.search(enhanced_query, k=k)
            
            # Process and format results
            formatted_results = []
            for passage_id, rank, score in zip(results[0], results[1], results[2]):
                passage_id_str = str(passage_id)
                
                if passage_id_str in self.table_metadata_store:
                    metadata = self.table_metadata_store[passage_id_str]
                    
                    # Extract matched tokens (simplified approach)
                    matched_tokens = self._extract_matched_tokens(
                        enhanced_query, metadata['original_text']
                    )
                    
                    result = ColBERTResult(
                        passage_id=passage_id_str,
                        text=metadata['original_text'],
                        score=float(score),
                        table_metadata=metadata['metadata'],
                        matched_tokens=matched_tokens,
                        cell_coordinates=None  # Could be enhanced with table parsing
                    )
                    
                    formatted_results.append(result)
            
            logger.info(f"ColBERT search returned {len(formatted_results)} results for: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in ColBERT search: {e}")
            return []
    
    async def _enhance_query_for_search(self, query: str) -> str:
        """
        Enhance query for better table matching
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query
        """
        try:
            enhanced_parts = [query]
            
            # Detect numerical queries
            if any(char.isdigit() for char in query):
                enhanced_parts.append("CONTAINS_NUMBERS")
            
            # Detect common table keywords
            table_keywords = ['total', 'sum', 'revenue', 'sales', 'price', 'quantity', 'year', 'quarter']
            found_keywords = [kw for kw in table_keywords if kw.lower() in query.lower()]
            if found_keywords:
                enhanced_parts.extend(found_keywords)
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def _extract_matched_tokens(self, query: str, text: str) -> List[str]:
        """
        Extract tokens that likely matched between query and text
        
        Args:
            query: Search query
            text: Retrieved text
            
        Returns:
            List of matched tokens
        """
        try:
            if not ENHANCED_UTILS_AVAILABLE:
                # Basic token matching without rapidfuzz
                query_tokens = query.lower().split()
                text_tokens = text.lower().split()
                return [token for token in query_tokens if token in text_tokens]
            
            # Enhanced fuzzy matching
            query_tokens = query.lower().split()
            text_tokens = text.lower().split()
            matched_tokens = []
            
            for q_token in query_tokens:
                best_match = rapidfuzz.process.extractOne(
                    q_token, text_tokens, score_cutoff=80
                )
                if best_match:
                    matched_tokens.append(best_match[0])
            
            return matched_tokens
            
        except Exception as e:
            logger.error(f"Error extracting matched tokens: {e}")
            return []
    
    async def analyze_table_query(self, query: str) -> TableQuery:
        """
        Analyze query to determine if it's table-specific and extract structure
        
        Args:
            query: User query
            
        Returns:
            Structured table query analysis
        """
        try:
            # Initialize with defaults
            query_type = "general"
            target_columns = []
            filters = {}
            confidence = 0.0
            
            query_lower = query.lower()
            
            # Detect numerical range queries
            numerical_patterns = [
                r'>\s*\$?\d+', r'<\s*\$?\d+', r'between\s+\$?\d+.*\$?\d+',
                r'greater than', r'less than', r'more than', r'higher than'
            ]
            
            import re
            for pattern in numerical_patterns:
                if re.search(pattern, query_lower):
                    query_type = "numerical_range"
                    confidence += 0.3
                    break
            
            # Detect exact match queries
            exact_patterns = ['what is', 'show me', 'find', 'get']
            if any(pattern in query_lower for pattern in exact_patterns):
                if query_type == "general":
                    query_type = "exact_match"
                confidence += 0.2
            
            # Detect temporal queries
            temporal_keywords = ['year', 'quarter', 'month', 'q1', 'q2', 'q3', 'q4', '2023', '2024']
            if any(keyword in query_lower for keyword in temporal_keywords):
                query_type = "temporal"
                confidence += 0.3
            
            # Detect table-specific keywords
            table_keywords = [
                'revenue', 'sales', 'profit', 'total', 'sum', 'average',
                'price', 'quantity', 'amount', 'cost', 'table', 'row', 'column'
            ]
            
            found_table_keywords = [kw for kw in table_keywords if kw in query_lower]
            if found_table_keywords:
                confidence += 0.4
                target_columns = found_table_keywords
            
            # Final confidence adjustment
            confidence = min(confidence, 1.0)
            
            return TableQuery(
                query=query,
                query_type=query_type,
                target_columns=target_columns,
                filters=filters,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing table query: {e}")
            return TableQuery(
                query=query,
                query_type="general",
                target_columns=[],
                filters={},
                confidence=0.0
            )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ColBERT index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = {
                "is_available": COLBERT_AVAILABLE,
                "is_initialized": self.is_initialized,
                "has_searcher": self.searcher is not None,
                "index_name": self.current_index_name,
                "indexed_documents": len(self.table_metadata_store),
                "index_path": str(self.index_path)
            }
            
            if self.index_path.exists():
                index_size = sum(f.stat().st_size for f in self.index_path.rglob('*') if f.is_file())
                stats["index_size_mb"] = round(index_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    async def rebuild_index(self, force: bool = False) -> bool:
        """
        Rebuild the ColBERT index from scratch
        
        Args:
            force: Force rebuild even if index exists
            
        Returns:
            Success status
        """
        try:
            if not force and self.searcher is not None:
                logger.info("Index already exists. Use force=True to rebuild.")
                return True
            
            logger.info("Rebuilding ColBERT index...")
            
            # Clear existing index
            if self.index_path.exists():
                import shutil
                shutil.rmtree(self.index_path)
                self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Reset components
            self.searcher = None
            self.indexer = None
            self.table_metadata_store = {}
            
            # Re-initialize
            self._initialize_colbert()
            
            logger.info("ColBERT index rebuild initiated. Add documents to complete indexing.")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False

# Global ColBERT service instance
colbert_service = ColBERTService()