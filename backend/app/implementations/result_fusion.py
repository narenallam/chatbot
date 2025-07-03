"""
Search Result Fusion Implementations
Intelligent fusion of web search and document search results
"""

import logging
from typing import List, Dict, Any, Union, Optional
from datetime import datetime, timedelta
import re

from app.core.web_search_interfaces import (
    SearchResultFusion, WebSearchResult, SearchContext, SourceType,
    WebSearchProviderRegistry
)

logger = logging.getLogger(__name__)

class IntelligentResultFusion(SearchResultFusion):
    """Intelligent fusion engine that combines web and document results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web_weight = config.get('web_weight', 0.6)
        self.document_weight = config.get('document_weight', 0.4)
        self.recency_boost = config.get('recency_boost', 0.2)
        self.authority_boost = config.get('authority_boost', 0.3)
        self.diversity_threshold = config.get('diversity_threshold', 0.7)
        self.max_results = config.get('max_results', 10)
    
    async def fuse_results(
        self,
        web_results: List[WebSearchResult],
        document_results: List[Any],
        context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Fuse web and document search results intelligently"""
        try:
            # Convert document results to unified format
            unified_doc_results = self._convert_document_results(document_results)
            
            # Combine all results
            all_results = []
            
            # Add web results with web source indicator
            for result in web_results:
                unified_result = self._convert_web_result(result)
                unified_result['source_category'] = 'web'
                all_results.append(unified_result)
            
            # Add document results with document source indicator
            for result in unified_doc_results:
                result['source_category'] = 'document'
                all_results.append(result)
            
            # Apply fusion strategy based on context
            fused_results = await self._apply_fusion_strategy(all_results, context)
            
            # Diversify results
            diverse_results = self._diversify_results(fused_results, context)
            
            # Final ranking and selection
            final_results = await self.rank_results(diverse_results, context)
            
            logger.info(f"Fused {len(web_results)} web + {len(document_results)} document results into {len(final_results)} final results")
            return final_results[:self.max_results]
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            # Fallback to web results only
            return [self._convert_web_result(r) for r in web_results[:self.max_results]]
    
    def _convert_web_result(self, web_result: WebSearchResult) -> Dict[str, Any]:
        """Convert web search result to unified format"""
        return {
            'title': web_result.title,
            'content': web_result.snippet,
            'full_content': web_result.content,
            'url': web_result.url,
            'score': web_result.relevance_score,
            'authority_score': web_result.authority_score,
            'source_type': web_result.source_type.value,
            'provider': web_result.provider,
            'published_date': web_result.published_date,
            'is_recent': web_result.is_recent,
            'matched_tokens': getattr(web_result, 'matched_tokens', []),
            'metadata': web_result.raw_result or {},
            'result_type': 'web_search'
        }
    
    def _convert_document_results(self, document_results: List[Any]) -> List[Dict[str, Any]]:
        """Convert document search results to unified format"""
        unified_results = []
        
        try:
            for doc_result in document_results:
                if isinstance(doc_result, dict):
                    # Handle legacy format from existing system
                    unified_result = {
                        'title': doc_result.get('metadata', {}).get('source', 'Document'),
                        'content': doc_result.get('text', ''),
                        'full_content': doc_result.get('text', ''),
                        'url': f"document://{doc_result.get('metadata', {}).get('source', 'unknown')}",
                        'score': doc_result.get('similarity_score', 0.0),
                        'authority_score': 0.8,  # Documents have high authority
                        'source_type': 'document',
                        'provider': 'document_search',
                        'published_date': None,
                        'is_recent': False,
                        'matched_tokens': [],
                        'metadata': doc_result.get('metadata', {}),
                        'result_type': 'document_search'
                    }
                    unified_results.append(unified_result)
                
                # Handle SearchResult objects from new system
                elif hasattr(doc_result, 'content') and hasattr(doc_result, 'score'):
                    unified_result = {
                        'title': getattr(doc_result, 'metadata', {}).get('source', 'Document'),
                        'content': doc_result.content,
                        'full_content': doc_result.content,
                        'url': f"document://{getattr(doc_result, 'document_id', 'unknown')}",
                        'score': doc_result.score,
                        'authority_score': 0.8,
                        'source_type': 'document',
                        'provider': getattr(doc_result, 'search_type', 'document_search'),
                        'published_date': None,
                        'is_recent': False,
                        'matched_tokens': getattr(doc_result, 'matched_tokens', []),
                        'metadata': getattr(doc_result, 'metadata', {}),
                        'result_type': 'document_search'
                    }
                    unified_results.append(unified_result)
            
            return unified_results
            
        except Exception as e:
            logger.error(f"Error converting document results: {e}")
            return []
    
    async def _apply_fusion_strategy(
        self, 
        all_results: List[Dict[str, Any]], 
        context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Apply context-aware fusion strategy"""
        try:
            # Determine optimal mix based on context
            if context.analysis.requires_latest:
                # For latest info queries, heavily favor web results
                web_weight = 0.8
                doc_weight = 0.2
            elif context.analysis.intent.value in ['specific_facts', 'how_to']:
                # For factual queries, balance web and documents
                web_weight = 0.5
                doc_weight = 0.5
            elif context.analysis.intent.value == 'historical_data':
                # For historical queries, favor documents
                web_weight = 0.3
                doc_weight = 0.7
            else:
                # Default balanced approach
                web_weight = self.web_weight
                doc_weight = self.document_weight
            
            # Apply weighted scoring
            for result in all_results:
                base_score = result.get('score', 0.0)
                authority_score = result.get('authority_score', 0.5)
                
                # Apply source category weights
                if result.get('source_category') == 'web':
                    weighted_score = base_score * web_weight
                else:
                    weighted_score = base_score * doc_weight
                
                # Apply authority boost
                weighted_score += authority_score * self.authority_boost
                
                # Apply recency boost for recent web results
                if result.get('is_recent', False):
                    weighted_score += self.recency_boost
                
                # Store the final fusion score
                result['fusion_score'] = min(1.0, weighted_score)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error applying fusion strategy: {e}")
            return all_results
    
    def _diversify_results(
        self, 
        results: List[Dict[str, Any]], 
        context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Diversify results to avoid redundancy"""
        try:
            if len(results) <= 5:
                return results  # No need to diversify small result sets
            
            diverse_results = []
            seen_content = []
            
            # Sort by fusion score first
            sorted_results = sorted(results, key=lambda x: x.get('fusion_score', 0), reverse=True)
            
            for result in sorted_results:
                content = result.get('content', '').lower()
                
                # Check content similarity with already selected results
                is_too_similar = False
                for seen in seen_content:
                    similarity = self._calculate_text_similarity(content, seen)
                    if similarity > self.diversity_threshold:
                        is_too_similar = True
                        break
                
                if not is_too_similar:
                    diverse_results.append(result)
                    seen_content.append(content)
                    
                    # Limit diversity check to avoid performance issues
                    if len(diverse_results) >= self.max_results:
                        break
            
            logger.info(f"Diversified {len(results)} results to {len(diverse_results)} diverse results")
            return diverse_results
            
        except Exception as e:
            logger.error(f"Error diversifying results: {e}")
            return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        try:
            words1 = set(re.findall(r'\w+', text1.lower()))
            words2 = set(re.findall(r'\w+', text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def rank_results(
        self,
        results: List[Union[WebSearchResult, Any]],
        context: SearchContext
    ) -> List[Union[WebSearchResult, Any]]:
        """Final ranking of fused results"""
        try:
            # Apply context-specific ranking adjustments
            for result in results:
                base_score = result.get('fusion_score', result.get('score', 0))
                
                # Boost based on entity matches
                if context.analysis.entity_types:
                    content_lower = result.get('content', '').lower()
                    entity_matches = sum(1 for entity in context.analysis.entity_types 
                                       if entity in content_lower)
                    if entity_matches > 0:
                        base_score += 0.1 * entity_matches
                
                # Boost based on temporal relevance
                if context.analysis.temporal_indicators:
                    content_lower = result.get('content', '').lower()
                    temporal_matches = sum(1 for indicator in context.analysis.temporal_indicators 
                                         if indicator.lower() in content_lower)
                    if temporal_matches > 0:
                        base_score += 0.05 * temporal_matches
                
                # Apply query-specific filters
                filters = context.analysis.filters
                preferred_sources = filters.get('preferred_source_types', [])
                if preferred_sources and result.get('source_type') in preferred_sources:
                    base_score += 0.1
                
                # Store final ranking score
                result['final_score'] = min(1.0, base_score)
            
            # Sort by final score
            ranked_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
            
            # Ensure source diversity in top results
            top_results = self._ensure_source_diversity(ranked_results[:self.max_results * 2])
            
            return top_results[:self.max_results]
            
        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            return results[:self.max_results]
    
    def _ensure_source_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity of sources in top results"""
        try:
            if len(results) <= 3:
                return results
            
            diverse_results = []
            source_counts = {}
            max_per_source = max(1, len(results) // 3)  # Allow max 1/3 from same source
            
            for result in results:
                source = result.get('provider', 'unknown')
                current_count = source_counts.get(source, 0)
                
                if current_count < max_per_source:
                    diverse_results.append(result)
                    source_counts[source] = current_count + 1
                elif len(diverse_results) < len(results) // 2:
                    # Still add if we haven't reached half the results
                    diverse_results.append(result)
                    source_counts[source] = current_count + 1
            
            # Fill remaining slots with best remaining results
            remaining_results = [r for r in results if r not in diverse_results]
            diverse_results.extend(remaining_results[:self.max_results - len(diverse_results)])
            
            return diverse_results
            
        except Exception as e:
            logger.error(f"Error ensuring source diversity: {e}")
            return results

class SimpleResultFusion(SearchResultFusion):
    """Simple fusion engine for basic combining of results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_results = config.get('max_results', 10)
    
    async def fuse_results(
        self,
        web_results: List[WebSearchResult],
        document_results: List[Any],
        context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Simple fusion that combines and sorts by score"""
        try:
            all_results = []
            
            # Add web results
            for result in web_results:
                all_results.append({
                    'title': result.title,
                    'content': result.snippet,
                    'url': result.url,
                    'score': result.relevance_score,
                    'source_type': 'web',
                    'provider': result.provider,
                    'result_type': 'web_search'
                })
            
            # Add document results
            for doc_result in document_results:
                if isinstance(doc_result, dict):
                    all_results.append({
                        'title': doc_result.get('metadata', {}).get('source', 'Document'),
                        'content': doc_result.get('text', ''),
                        'url': f"document://{doc_result.get('metadata', {}).get('source', 'unknown')}",
                        'score': doc_result.get('similarity_score', 0.0),
                        'source_type': 'document',
                        'provider': 'document_search',
                        'result_type': 'document_search'
                    })
            
            # Sort by score and return top results
            sorted_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:self.max_results]
            
        except Exception as e:
            logger.error(f"Simple fusion failed: {e}")
            return []
    
    async def rank_results(
        self,
        results: List[Union[WebSearchResult, Any]],
        context: SearchContext
    ) -> List[Union[WebSearchResult, Any]]:
        """Simple ranking by score"""
        try:
            return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        except Exception:
            return results

# Register fusion engines
WebSearchProviderRegistry.register_fusion_engine("intelligent", IntelligentResultFusion)
WebSearchProviderRegistry.register_fusion_engine("simple", SimpleResultFusion)