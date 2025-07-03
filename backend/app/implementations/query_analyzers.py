"""
Query Analyzer Implementations
Intelligent query analysis for routing and optimization
"""

import re
import logging
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
import asyncio

from app.core.web_search_interfaces import (
    QueryAnalyzer, QueryAnalysis, SearchIntent, SearchStrategy,
    WebSearchProviderRegistry
)

logger = logging.getLogger(__name__)

class RuleBasedQueryAnalyzer(QueryAnalyzer):
    """Rule-based query analyzer using patterns and keywords"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Temporal indicators
        self.temporal_indicators = {
            'latest': ['latest', 'recent', 'current', 'now', 'today', 'this week', 'this month'],
            'historical': ['history', 'past', 'historical', 'before', 'ago', 'yesterday'],
            'specific_time': ['2024', '2023', 'january', 'february', 'march', 'april', 'may', 'june',
                            'july', 'august', 'september', 'october', 'november', 'december',
                            'q1', 'q2', 'q3', 'q4', 'quarter']
        }
        
        # Entity type patterns
        self.entity_patterns = {
            'company': r'\b(?:inc|corp|ltd|llc|company|corporation)\b',
            'financial': r'\b(?:revenue|profit|earnings|stock|price|market|shares|dividend)\b',
            'technology': r'\b(?:software|hardware|tech|api|algorithm|ai|ml|data)\b',
            'news_event': r'\b(?:news|announcement|launch|release|merger|acquisition)\b',
            'comparison': r'\b(?:vs|versus|compare|comparison|better|best|alternative)\b',
            'how_to': r'\b(?:how to|tutorial|guide|steps|instructions|learn)\b'
        }
        
        # Search intent patterns
        self.intent_patterns = {
            SearchIntent.LATEST_INFO: [
                r'\b(?:latest|current|recent|now|today|news|update)\b',
                r'\b(?:what.{0,20}happening|current.{0,20}status)\b',
                r'\b(?:2024|this year|this month|this week)\b'
            ],
            SearchIntent.SPECIFIC_FACTS: [
                r'\b(?:what is|define|definition|meaning|explain)\b',
                r'\b(?:who is|where is|when was|how many|how much)\b',
                r'\b(?:statistics|data|facts|information)\b'
            ],
            SearchIntent.HISTORICAL_DATA: [
                r'\b(?:history|historical|past|before|ago|was|were)\b',
                r'\b(?:timeline|evolution|development|founded|established)\b'
            ],
            SearchIntent.COMPARISON: [
                r'\b(?:vs|versus|compare|comparison|difference|better|best)\b',
                r'\b(?:alternative|option|choice|which|between)\b'
            ],
            SearchIntent.HOW_TO: [
                r'\b(?:how to|tutorial|guide|step|instruction|learn)\b',
                r'\b(?:teach|show|explain|demonstrate)\b'
            ]
        }
        
        # Keywords that suggest web search is needed
        self.web_search_keywords = {
            'temporal': ['latest', 'current', 'recent', 'now', 'today', 'news', '2024', '2023'],
            'entities': ['company', 'stock', 'market', 'price', 'weather', 'sports', 'politics'],
            'events': ['announcement', 'launch', 'release', 'merger', 'acquisition', 'update'],
            'comparisons': ['vs', 'versus', 'compare', 'better', 'best', 'alternative']
        }
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for intent and routing"""
        try:
            query_lower = query.lower().strip()
            
            # Extract temporal indicators
            temporal_indicators = self._extract_temporal_indicators(query_lower)
            
            # Extract entity types
            entity_types = self._extract_entity_types(query_lower)
            
            # Determine intent
            intent, confidence = self._determine_intent(query_lower)
            
            # Check if latest information is required
            requires_latest = self._requires_latest_info(query_lower, temporal_indicators)
            
            # Suggest search strategy
            suggested_strategy = self._suggest_strategy(intent, requires_latest, temporal_indicators)
            
            # Extract search terms
            search_terms = self._extract_search_terms(query)
            
            # Generate filters
            filters = self._generate_filters(query_lower, entity_types, temporal_indicators)
            
            return QueryAnalysis(
                query=query,
                intent=intent,
                confidence=confidence,
                temporal_indicators=temporal_indicators,
                entity_types=entity_types,
                requires_latest=requires_latest,
                suggested_strategy=suggested_strategy,
                search_terms=search_terms,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query '{query}': {e}")
            # Return default analysis
            return QueryAnalysis(
                query=query,
                intent=SearchIntent.GENERAL,
                confidence=0.5,
                temporal_indicators=[],
                entity_types=[],
                requires_latest=False,
                suggested_strategy=SearchStrategy.AUTO,
                search_terms=[query],
                filters={}
            )
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query"""
        indicators = []
        
        for category, keywords in self.temporal_indicators.items():
            for keyword in keywords:
                if keyword in query:
                    indicators.append(keyword)
        
        # Check for year patterns
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        indicators.extend([f"{year[0]}{year[1:] if len(year) > 1 else ''}" for year in years])
        
        return list(set(indicators))
    
    def _extract_entity_types(self, query: str) -> List[str]:
        """Extract entity types from query"""
        entity_types = []
        
        for entity_type, pattern in self.entity_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                entity_types.append(entity_type)
        
        return entity_types
    
    def _determine_intent(self, query: str) -> tuple[SearchIntent, float]:
        """Determine search intent and confidence"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score / len(patterns)
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            
            if confidence > 0:
                return best_intent, confidence
        
        return SearchIntent.GENERAL, 0.3
    
    def _requires_latest_info(self, query: str, temporal_indicators: List[str]) -> bool:
        """Determine if query requires latest information"""
        # Check for temporal indicators suggesting current information
        latest_keywords = ['latest', 'current', 'recent', 'now', 'today', 'news', 'update']
        
        for keyword in latest_keywords:
            if keyword in query:
                return True
        
        # Check for current year
        current_year = str(datetime.now().year)
        if current_year in temporal_indicators:
            return True
        
        # Check for recent time periods
        recent_periods = ['this week', 'this month', 'this year']
        for period in recent_periods:
            if period in query:
                return True
        
        return False
    
    def _suggest_strategy(
        self, 
        intent: SearchIntent, 
        requires_latest: bool, 
        temporal_indicators: List[str]
    ) -> SearchStrategy:
        """Suggest optimal search strategy"""
        
        # If latest information is required, prioritize web search
        if requires_latest:
            return SearchStrategy.WEB_ONLY if intent == SearchIntent.LATEST_INFO else SearchStrategy.HYBRID
        
        # For specific facts that might be in documents, try documents first
        if intent in [SearchIntent.SPECIFIC_FACTS, SearchIntent.HOW_TO]:
            return SearchStrategy.HYBRID
        
        # For comparisons and current events, prefer web search
        if intent in [SearchIntent.COMPARISON, SearchIntent.LATEST_INFO]:
            return SearchStrategy.WEB_ONLY
        
        # For historical data, documents might be sufficient
        if intent == SearchIntent.HISTORICAL_DATA:
            return SearchStrategy.DOCUMENTS_ONLY
        
        # Default to auto routing
        return SearchStrategy.AUTO
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract key search terms from query"""
        # Remove common stop words but keep important ones
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Keep these important words
        keep_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'latest', 'current', 'recent',
            'best', 'better', 'compare', 'vs', 'versus'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        terms = []
        
        for word in words:
            if len(word) > 2 and (word not in stop_words or word in keep_words):
                terms.append(word)
        
        # If no terms, return the original query
        return terms if terms else [query]
    
    def _generate_filters(
        self, 
        query: str, 
        entity_types: List[str], 
        temporal_indicators: List[str]
    ) -> Dict[str, Any]:
        """Generate search filters based on analysis"""
        filters = {}
        
        # Time-based filters
        if temporal_indicators:
            filters['temporal_indicators'] = temporal_indicators
            
            # Set recency preference
            if any(keyword in query for keyword in ['latest', 'current', 'recent', 'now', 'today']):
                filters['recency_weight'] = 0.8
            else:
                filters['recency_weight'] = 0.3
        
        # Entity-based filters
        if entity_types:
            filters['entity_types'] = entity_types
        
        # Source type preferences
        if 'news' in query or 'announcement' in query:
            filters['preferred_source_types'] = ['news', 'official']
        elif 'research' in query or 'study' in query:
            filters['preferred_source_types'] = ['academic', 'official']
        elif 'tutorial' in query or 'how to' in query:
            filters['preferred_source_types'] = ['blog', 'official']
        
        return filters

class LLMQueryAnalyzer(QueryAnalyzer):
    """LLM-powered query analyzer for more sophisticated analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_model = config.get('llm_model')  # Reference to LLM from AI Service Manager
        self.fallback_analyzer = RuleBasedQueryAnalyzer(config)
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query using LLM with rule-based fallback"""
        try:
            if self.llm_model:
                return await self._llm_analysis(query)
            else:
                logger.warning("No LLM model available, using rule-based fallback")
                return await self.fallback_analyzer.analyze_query(query)
                
        except Exception as e:
            logger.error(f"LLM query analysis failed: {e}, falling back to rule-based")
            return await self.fallback_analyzer.analyze_query(query)
    
    async def _llm_analysis(self, query: str) -> QueryAnalysis:
        """Perform LLM-based query analysis"""
        analysis_prompt = f"""
        Analyze this search query and provide structured analysis:
        
        Query: "{query}"
        
        Please analyze:
        1. Search intent (latest_info, specific_facts, historical_data, comparison, how_to, general)
        2. Confidence level (0.0 to 1.0)
        3. Temporal indicators (words suggesting time sensitivity)
        4. Entity types (company, financial, technology, etc.)
        5. Whether latest information is required (true/false)
        6. Suggested search strategy (web_only, documents_only, hybrid, auto)
        7. Key search terms
        
        Respond in JSON format with these fields:
        {{
            "intent": "intent_name",
            "confidence": 0.8,
            "temporal_indicators": ["latest", "2024"],
            "entity_types": ["company", "financial"],
            "requires_latest": true,
            "suggested_strategy": "hybrid",
            "search_terms": ["key", "terms"],
            "filters": {{"recency_weight": 0.8}}
        }}
        """
        
        try:
            # Use the LLM to analyze the query
            response = await self.llm_model.generate(analysis_prompt)
            
            # Parse JSON response
            import json
            analysis_data = json.loads(response.strip())
            
            # Convert to QueryAnalysis object
            intent = SearchIntent(analysis_data.get('intent', 'general'))
            strategy = SearchStrategy(analysis_data.get('suggested_strategy', 'auto'))
            
            return QueryAnalysis(
                query=query,
                intent=intent,
                confidence=analysis_data.get('confidence', 0.5),
                temporal_indicators=analysis_data.get('temporal_indicators', []),
                entity_types=analysis_data.get('entity_types', []),
                requires_latest=analysis_data.get('requires_latest', False),
                suggested_strategy=strategy,
                search_terms=analysis_data.get('search_terms', [query]),
                filters=analysis_data.get('filters', {})
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM analysis response: {e}")
            # Fall back to rule-based analysis
            return await self.fallback_analyzer.analyze_query(query)

# Register analyzers
WebSearchProviderRegistry.register_analyzer("rule_based", RuleBasedQueryAnalyzer)
WebSearchProviderRegistry.register_analyzer("llm_powered", LLMQueryAnalyzer)