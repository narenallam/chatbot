"""
Hybrid Search Service (Phase 2)
Combines ChromaDB semantic search with ColBERT precise table retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

# Local imports
from app.services.vector_service import vector_service, VectorService
from app.services.colbert_service import colbert_service, ColBERTService, TableQuery, ColBERTResult

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Search strategy enumeration"""
    GENERAL_ONLY = "general_only"
    TABLE_ONLY = "table_only"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class HybridSearchResult:
    """Combined search result from multiple sources"""
    text: str
    source: str
    score: float
    search_type: str  # 'general', 'table', 'hybrid'
    metadata: Dict[str, Any]
    chunk_index: int
    table_specific: bool = False
    matched_tokens: Optional[List[str]] = None
    cell_coordinates: Optional[tuple] = None

class HybridSearchService:
    """
    Advanced search service combining ChromaDB and ColBERT for optimal table retrieval
    """
    
    def __init__(self):
        self.vector_service: VectorService = vector_service
        self.colbert_service: ColBERTService = colbert_service
        
        # Search weights and thresholds
        self.table_confidence_threshold = 0.6
        self.hybrid_weight_general = 0.4
        self.hybrid_weight_table = 0.6
        
        # Query classification keywords
        self.table_indicators = {
            'strong': ['table', 'row', 'column', 'cell', 'spreadsheet'],
            'medium': ['total', 'sum', 'average', 'count', 'revenue', 'sales', 'profit'],
            'weak': ['data', 'number', 'amount', 'value', 'year', 'quarter']
        }
        
        logger.info("Hybrid Search Service initialized")
    
    async def search(
        self, 
        query: str, 
        n_results: int = 5,
        strategy: SearchStrategy = SearchStrategy.AUTO,
        file_hashes: Optional[List[str]] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining general and table-specific retrieval
        
        Args:
            query: Search query
            n_results: Number of results to return
            strategy: Search strategy to use
            file_hashes: Optional file hash filter
            
        Returns:
            List of hybrid search results
        """
        try:
            # Analyze query to determine optimal search strategy
            if strategy == SearchStrategy.AUTO:
                strategy = await self._determine_search_strategy(query)
            
            logger.info(f"Using search strategy: {strategy.value} for query: {query[:50]}...")
            
            # Execute search based on strategy
            if strategy == SearchStrategy.GENERAL_ONLY:
                return await self._search_general_only(query, n_results, file_hashes)
            elif strategy == SearchStrategy.TABLE_ONLY:
                return await self._search_table_only(query, n_results, file_hashes)
            elif strategy == SearchStrategy.HYBRID:
                return await self._search_hybrid(query, n_results, file_hashes)
            else:
                # Default to general search
                return await self._search_general_only(query, n_results, file_hashes)
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to general search
            return await self._search_general_only(query, n_results, file_hashes)
    
    async def _determine_search_strategy(self, query: str) -> SearchStrategy:
        """
        Analyze query to determine optimal search strategy
        
        Args:
            query: Search query
            
        Returns:
            Recommended search strategy
        """
        try:
            # Get table query analysis from ColBERT service
            table_analysis = await self.colbert_service.analyze_table_query(query)
            
            # Calculate table relevance score
            table_score = 0.0
            query_lower = query.lower()
            
            # Strong indicators
            for indicator in self.table_indicators['strong']:
                if indicator in query_lower:
                    table_score += 0.4
            
            # Medium indicators  
            for indicator in self.table_indicators['medium']:
                if indicator in query_lower:
                    table_score += 0.2
            
            # Weak indicators
            for indicator in self.table_indicators['weak']:
                if indicator in query_lower:
                    table_score += 0.1
            
            # Add ColBERT analysis confidence
            table_score += table_analysis.confidence * 0.3
            
            # Numerical query detection
            if any(char.isdigit() for char in query) or any(op in query for op in ['>', '<', '=', 'between']):
                table_score += 0.3
            
            # Cap the score
            table_score = min(table_score, 1.0)
            
            logger.debug(f"Table relevance score: {table_score:.2f} for query: {query}")
            
            # Determine strategy based on score
            if table_score >= 0.8:
                return SearchStrategy.TABLE_ONLY
            elif table_score >= 0.4:
                return SearchStrategy.HYBRID
            else:
                return SearchStrategy.GENERAL_ONLY
                
        except Exception as e:
            logger.error(f"Error determining search strategy: {e}")
            return SearchStrategy.GENERAL_ONLY
    
    async def _search_general_only(
        self, 
        query: str, 
        n_results: int,
        file_hashes: Optional[List[str]] = None
    ) -> List[HybridSearchResult]:
        """Execute general semantic search only"""
        try:
            # Use existing vector service hybrid search
            results = await self.vector_service.hybrid_search(query, n_results, file_hashes)
            
            # Convert to HybridSearchResult format
            hybrid_results = []
            for result in results:
                hybrid_result = HybridSearchResult(
                    text=result.get('text', ''),
                    source=result.get('source', 'Unknown'),
                    score=result.get('hybrid_score', result.get('similarity_score', 0)),
                    search_type='general',
                    metadata=result.get('metadata', {}),
                    chunk_index=result.get('chunk_index', 0),
                    table_specific=False
                )
                hybrid_results.append(hybrid_result)
            
            logger.info(f"General search returned {len(hybrid_results)} results")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error in general search: {e}")
            return []
    
    async def _search_table_only(
        self, 
        query: str, 
        n_results: int,
        file_hashes: Optional[List[str]] = None
    ) -> List[HybridSearchResult]:
        """Execute ColBERT table search only"""
        try:
            # Check if ColBERT is available
            if not self.colbert_service.is_initialized:
                logger.warning("ColBERT not available, falling back to general search")
                return await self._search_general_only(query, n_results, file_hashes)
            
            # Execute ColBERT search
            colbert_results = await self.colbert_service.search_tables(query, n_results)
            
            # Convert to HybridSearchResult format
            hybrid_results = []
            for result in colbert_results:
                # Apply file hash filtering if specified
                if file_hashes:
                    file_hash = result.table_metadata.get('file_hash')
                    if file_hash not in file_hashes:
                        continue
                
                hybrid_result = HybridSearchResult(
                    text=result.text,
                    source=result.table_metadata.get('source', 'Unknown'),
                    score=result.score,
                    search_type='table',
                    metadata=result.table_metadata,
                    chunk_index=result.table_metadata.get('chunk_index', 0),
                    table_specific=True,
                    matched_tokens=result.matched_tokens,
                    cell_coordinates=result.cell_coordinates
                )
                hybrid_results.append(hybrid_result)
            
            logger.info(f"Table search returned {len(hybrid_results)} results")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error in table search: {e}")
            return []
    
    async def _search_hybrid(
        self, 
        query: str, 
        n_results: int,
        file_hashes: Optional[List[str]] = None
    ) -> List[HybridSearchResult]:
        """Execute hybrid search combining both approaches"""
        try:
            # Execute both searches concurrently
            general_task = asyncio.create_task(
                self._search_general_only(query, n_results, file_hashes)
            )
            table_task = asyncio.create_task(
                self._search_table_only(query, n_results, file_hashes)
            )
            
            general_results, table_results = await asyncio.gather(
                general_task, table_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(general_results, Exception):
                logger.error(f"General search failed: {general_results}")
                general_results = []
            
            if isinstance(table_results, Exception):
                logger.error(f"Table search failed: {table_results}")
                table_results = []
            
            # Combine and rerank results
            combined_results = await self._combine_and_rerank_results(
                general_results, table_results, query, n_results
            )
            
            logger.info(f"Hybrid search returned {len(combined_results)} combined results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to general search only
            return await self._search_general_only(query, n_results, file_hashes)
    
    async def _combine_and_rerank_results(
        self,
        general_results: List[HybridSearchResult],
        table_results: List[HybridSearchResult],
        query: str,
        n_results: int
    ) -> List[HybridSearchResult]:
        """
        Combine and rerank results from both search methods
        
        Args:
            general_results: Results from general search
            table_results: Results from table search
            query: Original query
            n_results: Number of final results
            
        Returns:
            Combined and reranked results
        """
        try:
            # Create a combined list
            all_results = []
            
            # Add general results with adjusted scores
            for result in general_results:
                result.score *= self.hybrid_weight_general
                result.search_type = 'hybrid_general'
                all_results.append(result)
            
            # Add table results with adjusted scores
            for result in table_results:
                result.score *= self.hybrid_weight_table
                result.search_type = 'hybrid_table'
                all_results.append(result)
            
            # Remove duplicates based on text similarity
            deduplicated_results = await self._deduplicate_results(all_results)
            
            # Sort by adjusted score
            deduplicated_results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top n_results
            return deduplicated_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return (general_results + table_results)[:n_results]
    
    async def _deduplicate_results(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """
        Remove duplicate results based on text similarity
        
        Args:
            results: List of results to deduplicate
            
        Returns:
            Deduplicated results
        """
        try:
            if len(results) <= 1:
                return results
            
            deduplicated = []
            similarity_threshold = 0.85
            
            for result in results:
                is_duplicate = False
                
                for existing in deduplicated:
                    # Simple text similarity check
                    similarity = self._calculate_text_similarity(result.text, existing.text)
                    
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        # Keep the one with higher score
                        if result.score > existing.score:
                            deduplicated.remove(existing)
                            deduplicated.append(result)
                        break
                
                if not is_duplicate:
                    deduplicated.append(result)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")
            return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Simple Jaccard similarity for efficiency
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    async def add_documents_to_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to both ChromaDB and ColBERT indexes
        
        Args:
            documents: List of document chunks
            
        Returns:
            Success status
        """
        try:
            # Add to ChromaDB (existing functionality)
            chroma_success = True
            try:
                from langchain.schema import Document
                doc_objects = [
                    Document(
                        page_content=doc.get('text', ''),
                        metadata=doc.get('metadata', {})
                    )
                    for doc in documents
                ]
                await self.vector_service.add_documents(doc_objects)
            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}")
                chroma_success = False
            
            # Add to ColBERT (table documents only)
            colbert_success = await self.colbert_service.index_table_documents(documents)
            
            success = chroma_success or colbert_success
            logger.info(f"Document indexing - ChromaDB: {chroma_success}, ColBERT: {colbert_success}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding documents to indexes: {e}")
            return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all search services
        
        Returns:
            Combined service statistics
        """
        try:
            stats = {
                "hybrid_search_service": {
                    "is_initialized": True,
                    "table_confidence_threshold": self.table_confidence_threshold,
                    "hybrid_weights": {
                        "general": self.hybrid_weight_general,
                        "table": self.hybrid_weight_table
                    }
                },
                "vector_service": {
                    "collection_count": getattr(self.vector_service.collection, '_collection', {}).get('count', 0) if hasattr(self.vector_service, 'collection') else 0
                },
                "colbert_service": self.colbert_service.get_index_stats()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}

# Global hybrid search service instance
hybrid_search_service = HybridSearchService()