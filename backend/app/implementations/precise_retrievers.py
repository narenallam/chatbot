"""
Technology-Specific Precise Retrieval Implementations
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import json
import re

from app.core.interfaces import PreciseRetrievalInterface, Document, SearchResult, QueryAnalysis, ServiceFactory

logger = logging.getLogger(__name__)

class ColBERTImplementation(PreciseRetrievalInterface):
    """ColBERT precise retrieval implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'colbert-ir/colbertv2.0')
        self.index_name = config.get('index_name', 'table_precise_v1')
        self.index_path = Path(config.get('index_path', './colbert_indexes'))
        self.max_doc_length = config.get('max_doc_length', 512)
        self.max_query_length = config.get('max_query_length', 64)
        
        self.indexer = None
        self.searcher = None
        self.is_initialized = False
        self.metadata_store = {}
        
        self._initialize_colbert()
    
    def _initialize_colbert(self):
        """Initialize ColBERT components"""
        try:
            from colbert import Indexer, Searcher
            from colbert.infra import Run, RunConfig, ColBERTConfig
            
            # Create index directory
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Configure ColBERT
            self.config = ColBERTConfig(
                nbits=2,
                doc_maxlen=self.max_doc_length,
                query_maxlen=self.max_query_length,
                similarity="cosine",
                index_path=str(self.index_path),
                experiment="table_precise_search"
            )
            
            # Check if index exists
            index_full_path = self.index_path / self.index_name
            if index_full_path.exists():
                self._load_existing_index()
            
            self.is_initialized = True
            logger.info("ColBERT implementation initialized")
            
        except ImportError:
            logger.warning("ColBERT not available. Install with: pip install colbert-ai")
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize ColBERT: {e}")
            self.is_initialized = False
    
    def _load_existing_index(self):
        """Load existing ColBERT index"""
        try:
            from colbert import Searcher
            from colbert.infra import Run, RunConfig
            
            with Run().context(RunConfig(experiment="table_precise_search")):
                self.searcher = Searcher(index=self.index_name, config=self.config)
            
            # Load metadata
            metadata_path = self.index_path / f"{self.index_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)
            
            logger.info(f"Loaded existing ColBERT index: {self.index_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load existing ColBERT index: {e}")
            self.searcher = None
    
    async def index_documents(self, documents: List[Document]) -> bool:
        """Index documents with ColBERT"""
        if not self.is_initialized:
            logger.warning("ColBERT not initialized")
            return False
        
        try:
            from colbert import Indexer
            from colbert.infra import Run, RunConfig
            from colbert.data import Collection
            
            # Filter for table documents
            table_documents = [
                doc for doc in documents 
                if doc.metadata.get('contains_table', False)
            ]
            
            if not table_documents:
                logger.info("No table documents to index with ColBERT")
                return True
            
            logger.info(f"Indexing {len(table_documents)} table documents with ColBERT")
            
            # Prepare passages
            passages = []
            metadata_store = {}
            
            for i, doc in enumerate(table_documents):
                passage_id = str(i)
                enhanced_text = self._enhance_text_for_indexing(doc.content, doc.metadata)
                passages.append(enhanced_text)
                
                metadata_store[passage_id] = {
                    'document_id': doc.id,
                    'original_text': doc.content,
                    'metadata': doc.metadata
                }
            
            # Create collection and index
            collection = Collection(data=passages)
            
            with Run().context(RunConfig(experiment="table_precise_search")):
                self.indexer = Indexer(checkpoint=self.model_name, config=self.config)
                self.indexer.index(name=self.index_name, collection=collection)
            
            # Save metadata
            self.metadata_store.update(metadata_store)
            metadata_path = self.index_path / f"{self.index_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_store, f, indent=2)
            
            # Load searcher
            self._load_existing_index()
            
            logger.info(f"Successfully indexed {len(table_documents)} documents with ColBERT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents with ColBERT: {e}")
            return False
    
    def _enhance_text_for_indexing(self, text: str, metadata: Dict[str, Any]) -> str:
        """Enhance text with table structure markers"""
        try:
            enhanced_parts = [text]
            
            # Add table type
            if metadata.get('table_type'):
                enhanced_parts.append(f"TABLE_TYPE:{metadata['table_type']}")
            
            # Add numeric indicators
            if metadata.get('numeric_data'):
                enhanced_parts.append("CONTAINS_NUMBERS")
            
            # Add row count
            if metadata.get('row_count', 0) > 0:
                enhanced_parts.append(f"ROWS:{metadata['row_count']}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing text: {e}")
            return text
    
    async def search_precise(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform precise ColBERT search"""
        if not self.is_initialized or not self.searcher:
            logger.warning("ColBERT searcher not available")
            return []
        
        try:
            # Enhance query
            enhanced_query = await self._enhance_query(query)
            
            # Perform search
            results = self.searcher.search(enhanced_query, k=k)
            
            # Format results
            search_results = []
            for passage_id, rank, score in zip(results[0], results[1], results[2]):
                passage_id_str = str(passage_id)
                
                if passage_id_str in self.metadata_store:
                    metadata_info = self.metadata_store[passage_id_str]
                    
                    # Apply filters
                    if filters and not self._apply_filters(metadata_info['metadata'], filters):
                        continue
                    
                    # Extract matched tokens
                    matched_tokens = self._extract_matched_tokens(query, metadata_info['original_text'])
                    
                    result = SearchResult(
                        document_id=metadata_info['document_id'],
                        content=metadata_info['original_text'],
                        score=float(score),
                        metadata=metadata_info['metadata'],
                        search_type="precise",
                        matched_tokens=matched_tokens
                    )
                    search_results.append(result)
            
            logger.info(f"ColBERT search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in ColBERT search: {e}")
            return []
    
    async def _enhance_query(self, query: str) -> str:
        """Enhance query for better matching"""
        try:
            enhanced_parts = [query]
            
            # Add indicators based on query content
            if any(char.isdigit() for char in query):
                enhanced_parts.append("CONTAINS_NUMBERS")
            
            # Add table keywords
            table_keywords = ['total', 'sum', 'revenue', 'sales', 'price', 'quantity']
            found_keywords = [kw for kw in table_keywords if kw.lower() in query.lower()]
            if found_keywords:
                enhanced_parts.extend(found_keywords)
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to metadata"""
        try:
            for key, value in filters.items():
                if key not in metadata:
                    return False
                
                if isinstance(value, list):
                    if metadata[key] not in value:
                        return False
                elif metadata[key] != value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return True
    
    def _extract_matched_tokens(self, query: str, text: str) -> List[str]:
        """Extract matched tokens between query and text"""
        try:
            query_tokens = set(query.lower().split())
            text_tokens = set(text.lower().split())
            return list(query_tokens.intersection(text_tokens))
            
        except Exception as e:
            logger.error(f"Error extracting matched tokens: {e}")
            return []
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for table-specific intent"""
        try:
            intent = "general"
            confidence = 0.0
            entities = []
            filters = {}
            strategy = "semantic"
            
            query_lower = query.lower()
            
            # Detect table-specific patterns
            if any(indicator in query_lower for indicator in ['table', 'row', 'column', 'cell']):
                intent = "table_specific"
                confidence += 0.4
                strategy = "precise"
            
            # Detect numerical queries
            if re.search(r'\d+|>\s*|<\s*|between|greater|less', query_lower):
                intent = "numerical"
                confidence += 0.3
                strategy = "precise"
            
            # Detect temporal queries
            if any(temporal in query_lower for temporal in ['year', 'quarter', 'month', 'q1', 'q2', 'q3', 'q4']):
                intent = "temporal"
                confidence += 0.2
            
            # Extract entities (simplified)
            entities = re.findall(r'\b[A-Z][a-z]+\b', query)
            
            # Extract numbers as potential filters
            numbers = re.findall(r'\d+', query)
            if numbers:
                filters['numbers'] = numbers
            
            confidence = min(confidence, 1.0)
            
            return QueryAnalysis(
                query=query,
                intent=intent,
                confidence=confidence,
                extracted_entities=entities,
                filters=filters,
                suggested_strategy=strategy
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return QueryAnalysis(
                query=query,
                intent="general",
                confidence=0.0,
                extracted_entities=[],
                filters={},
                suggested_strategy="semantic"
            )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get ColBERT index statistics"""
        try:
            stats = {
                "type": "colbert",
                "is_initialized": self.is_initialized,
                "has_searcher": self.searcher is not None,
                "index_name": self.index_name,
                "indexed_documents": len(self.metadata_store),
                "index_path": str(self.index_path),
                "model_name": self.model_name
            }
            
            if self.index_path.exists():
                try:
                    index_size = sum(f.stat().st_size for f in self.index_path.rglob('*') if f.is_file())
                    stats["index_size_mb"] = round(index_size / (1024 * 1024), 2)
                except:
                    stats["index_size_mb"] = "unknown"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting ColBERT stats: {e}")
            return {"error": str(e)}

class BM25Implementation(PreciseRetrievalInterface):
    """BM25 keyword-based precise retrieval implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.k1 = config.get('k1', 1.2)
        self.b = config.get('b', 0.75)
        self.documents = {}
        self.corpus = []
        self.bm25 = None
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25"""
        try:
            from rank_bm25 import BM25Okapi
            logger.info("BM25 implementation initialized")
            
        except ImportError:
            logger.error("rank-bm25 not available. Install with: pip install rank-bm25")
            raise
    
    async def index_documents(self, documents: List[Document]) -> bool:
        """Index documents with BM25"""
        try:
            from rank_bm25 import BM25Okapi
            
            # Store documents
            for doc in documents:
                self.documents[doc.id] = {
                    'content': doc.content,
                    'metadata': doc.metadata
                }
            
            # Prepare corpus
            self.corpus = [doc.content.lower().split() for doc in documents]
            
            # Create BM25 index
            self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
            
            logger.info(f"Indexed {len(documents)} documents with BM25")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents with BM25: {e}")
            return False
    
    async def search_precise(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        try:
            if not self.bm25:
                return []
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            
            # Format results
            search_results = []
            doc_ids = list(self.documents.keys())
            
            for idx in top_indices:
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    doc_info = self.documents[doc_id]
                    
                    # Apply filters
                    if filters and not self._apply_filters(doc_info['metadata'], filters):
                        continue
                    
                    result = SearchResult(
                        document_id=doc_id,
                        content=doc_info['content'],
                        score=float(scores[idx]),
                        metadata=doc_info['metadata'],
                        search_type="keyword",
                        matched_tokens=tokenized_query
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to metadata"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for keyword search"""
        return QueryAnalysis(
            query=query,
            intent="keyword",
            confidence=0.8,
            extracted_entities=query.split(),
            filters={},
            suggested_strategy="keyword"
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get BM25 statistics"""
        return {
            "type": "bm25",
            "document_count": len(self.documents),
            "k1": self.k1,
            "b": self.b,
            "has_index": self.bm25 is not None
        }

# Register implementations
ServiceFactory.register_precise_retriever("colbert", ColBERTImplementation)
ServiceFactory.register_precise_retriever("bm25", BM25Implementation)