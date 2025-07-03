"""
Web Search Interfaces for Agentic Search System
Technology-agnostic interfaces for multiple web search providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class SearchIntent(Enum):
    """Search intent classification"""
    LATEST_INFO = "latest_info"          # Current events, news, real-time data
    SPECIFIC_FACTS = "specific_facts"    # Company info, definitions, statistics
    HISTORICAL_DATA = "historical_data"  # Past events, historical analysis
    COMPARISON = "comparison"            # Product comparisons, alternatives
    HOW_TO = "how_to"                   # Instructions, tutorials
    GENERAL = "general"                 # General knowledge queries

class SearchStrategy(Enum):
    """Search execution strategy"""
    DOCUMENTS_ONLY = "documents_only"   # Use only local documents
    WEB_ONLY = "web_only"              # Use only web search
    HYBRID = "hybrid"                   # Combine web + documents
    AUTO = "auto"                       # Intelligent routing

class SourceType(Enum):
    """Source type classification"""
    NEWS = "news"
    ACADEMIC = "academic"
    OFFICIAL = "official"
    BLOG = "blog"
    FORUM = "forum"
    SOCIAL = "social"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"

@dataclass
class WebSearchResult:
    """Standardized web search result"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    published_date: Optional[datetime] = None
    source_type: SourceType = SourceType.UNKNOWN
    authority_score: float = 0.0
    relevance_score: float = 0.0
    is_recent: bool = False
    provider: str = "unknown"
    raw_result: Dict[str, Any] = None

@dataclass
class QueryAnalysis:
    """Query analysis results"""
    query: str
    intent: SearchIntent
    confidence: float
    temporal_indicators: List[str]
    entity_types: List[str]
    requires_latest: bool
    suggested_strategy: SearchStrategy
    search_terms: List[str]
    filters: Dict[str, Any]

@dataclass
class SearchContext:
    """Search execution context"""
    original_query: str
    analysis: QueryAnalysis
    max_results: int = 10
    include_content: bool = False
    recency_weight: float = 0.3
    quality_weight: float = 0.7
    preferred_provider: Optional[str] = None

class WebSearchInterface(ABC):
    """Abstract interface for web search providers"""
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        max_results: int = 10,
        include_content: bool = False
    ) -> List[WebSearchResult]:
        """
        Perform web search
        
        Args:
            query: Search query
            max_results: Maximum number of results
            include_content: Whether to fetch full page content
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and capabilities"""
        pass
    
    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if provider is available and has quota"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass

class QueryAnalyzer(ABC):
    """Abstract interface for query analysis"""
    
    @abstractmethod
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query for intent and routing
        
        Args:
            query: User query
            
        Returns:
            Query analysis results
        """
        pass

class ContentProcessor(ABC):
    """Abstract interface for content processing"""
    
    @abstractmethod
    async def extract_content(self, url: str) -> Optional[str]:
        """Extract clean content from URL"""
        pass
    
    @abstractmethod
    async def summarize_content(self, content: str, max_length: int = 500) -> str:
        """Summarize content to specified length"""
        pass
    
    @abstractmethod
    async def extract_key_facts(self, content: str) -> List[str]:
        """Extract key facts from content"""
        pass

class WebSearchAgent(ABC):
    """Abstract interface for intelligent web search agents"""
    
    @abstractmethod
    async def search_with_context(self, context: SearchContext) -> List[WebSearchResult]:
        """
        Perform context-aware search
        
        Args:
            context: Search execution context
            
        Returns:
            Processed and ranked search results
        """
        pass
    
    @abstractmethod
    async def refine_query(self, query: str, intent: SearchIntent) -> str:
        """Refine query for better web search results"""
        pass

class SearchResultFusion(ABC):
    """Abstract interface for search result fusion"""
    
    @abstractmethod
    async def fuse_results(
        self,
        web_results: List[WebSearchResult],
        document_results: List[Any],
        context: SearchContext
    ) -> List[Dict[str, Any]]:
        """
        Fuse web and document search results
        
        Args:
            web_results: Results from web search
            document_results: Results from document search
            context: Search context
            
        Returns:
            Unified and ranked results
        """
        pass
    
    @abstractmethod
    async def rank_results(
        self,
        results: List[Union[WebSearchResult, Any]],
        context: SearchContext
    ) -> List[Union[WebSearchResult, Any]]:
        """Rank results based on relevance, recency, and authority"""
        pass

# Search Provider Registry
class WebSearchProviderRegistry:
    """Registry for web search providers"""
    
    _providers = {}
    _analyzers = {}
    _processors = {}
    _agents = {}
    _fusion_engines = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """Register a web search provider"""
        cls._providers[name] = provider_class
    
    @classmethod
    def register_analyzer(cls, name: str, analyzer_class):
        """Register a query analyzer"""
        cls._analyzers[name] = analyzer_class
    
    @classmethod
    def register_processor(cls, name: str, processor_class):
        """Register a content processor"""
        cls._processors[name] = processor_class
    
    @classmethod
    def register_agent(cls, name: str, agent_class):
        """Register a web search agent"""
        cls._agents[name] = agent_class
    
    @classmethod
    def register_fusion_engine(cls, name: str, fusion_class):
        """Register a result fusion engine"""
        cls._fusion_engines[name] = fusion_class
    
    @classmethod
    def create_provider(cls, name: str, config: Dict[str, Any] = None) -> WebSearchInterface:
        """Create web search provider instance"""
        if name not in cls._providers:
            raise ValueError(f"Unknown web search provider: {name}")
        return cls._providers[name](config or {})
    
    @classmethod
    def create_analyzer(cls, name: str, config: Dict[str, Any] = None) -> QueryAnalyzer:
        """Create query analyzer instance"""
        if name not in cls._analyzers:
            raise ValueError(f"Unknown query analyzer: {name}")
        return cls._analyzers[name](config or {})
    
    @classmethod
    def create_processor(cls, name: str, config: Dict[str, Any] = None) -> ContentProcessor:
        """Create content processor instance"""
        if name not in cls._processors:
            raise ValueError(f"Unknown content processor: {name}")
        return cls._processors[name](config or {})
    
    @classmethod
    def create_agent(cls, name: str, config: Dict[str, Any] = None) -> WebSearchAgent:
        """Create web search agent instance"""
        if name not in cls._agents:
            raise ValueError(f"Unknown web search agent: {name}")
        return cls._agents[name](config or {})
    
    @classmethod
    def create_fusion_engine(cls, name: str, config: Dict[str, Any] = None) -> SearchResultFusion:
        """Create result fusion engine instance"""
        if name not in cls._fusion_engines:
            raise ValueError(f"Unknown fusion engine: {name}")
        return cls._fusion_engines[name](config or {})
    
    @classmethod
    def list_available_components(cls) -> Dict[str, List[str]]:
        """List all available components"""
        return {
            "providers": list(cls._providers.keys()),
            "analyzers": list(cls._analyzers.keys()),
            "processors": list(cls._processors.keys()),
            "agents": list(cls._agents.keys()),
            "fusion_engines": list(cls._fusion_engines.keys())
        }