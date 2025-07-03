"""
Technology-Agnostic AI Service Manager
Manages all AI components with swappable implementations
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from app.core.interfaces import (
    EmbeddingModelInterface,
    VectorDatabaseInterface,
    PreciseRetrievalInterface,
    LLMInterface,
    SearchServiceInterface,
    ServiceFactory,
    Document,
    SearchResult,
    QueryAnalysis,
    SearchStrategy,
)
from app.core.ai_config import AIConfig, ai_config_manager
from app.core.web_search_interfaces import (
    WebSearchAgent,
    QueryAnalyzer,
    SearchResultFusion,
    ContentProcessor,
    SearchContext,
    WebSearchProviderRegistry,
    SearchIntent,
    SearchStrategy as WebSearchStrategy,
)

logger = logging.getLogger(__name__)


class AIServiceManager:
    """
    Manages all AI services with technology-agnostic interface
    Allows hot-swapping of implementations
    """

    def __init__(self):
        self.embedding_model: Optional[EmbeddingModelInterface] = None
        self.vector_database: Optional[VectorDatabaseInterface] = None
        self.precise_retriever: Optional[PreciseRetrievalInterface] = None
        self.llm_model: Optional[LLMInterface] = None
        self.search_service: Optional[SearchServiceInterface] = None

        # Web search components
        self.web_search_agent: Optional[WebSearchAgent] = None
        self.query_analyzer: Optional[QueryAnalyzer] = None
        self.result_fusion: Optional[SearchResultFusion] = None
        self.content_processor: Optional[ContentProcessor] = None

        self.is_initialized = False
        self.current_config: Optional[AIConfig] = None

        # Performance metrics
        self.metrics = {
            "searches_performed": 0,
            "web_searches_performed": 0,
            "documents_indexed": 0,
            "config_switches": 0,
            "errors": 0,
        }

    async def initialize(self, config: Optional[AIConfig] = None) -> bool:
        """
        Initialize AI services with given configuration

        Args:
            config: AI configuration (if None, loads recommended config)

        Returns:
            Success status
        """
        try:
            # Load configuration
            if config is None:
                config = await ai_config_manager.get_recommended_config()

            self.current_config = config

            # Initialize all components
            await self._initialize_embedding_model(config)
            await self._initialize_vector_database(config)
            await self._initialize_precise_retriever(config)
            await self._initialize_llm_model(config)
            await self._initialize_web_search_components(config)
            await self._initialize_search_service(config)

            self.is_initialized = True
            logger.info("AI Service Manager initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize AI Service Manager: {e}")
            self.is_initialized = False
            self.metrics["errors"] += 1
            return False

    async def _initialize_embedding_model(self, config: AIConfig):
        """Initialize embedding model"""
        try:
            self.embedding_model = ServiceFactory.create_embedding_model(
                config.embedding_model, config.embedding_config
            )
            logger.info(f"Initialized embedding model: {config.embedding_model}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    async def _initialize_vector_database(self, config: AIConfig):
        """Initialize vector database"""
        try:
            self.vector_database = ServiceFactory.create_vector_database(
                config.vector_database, config.vector_config
            )
            logger.info(f"Initialized vector database: {config.vector_database}")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise

    async def _initialize_precise_retriever(self, config: AIConfig):
        """Initialize precise retriever"""
        try:
            if config.precise_retriever:
                self.precise_retriever = ServiceFactory.create_precise_retriever(
                    config.precise_retriever, config.precise_config or {}
                )
                logger.info(
                    f"Initialized precise retriever: {config.precise_retriever}"
                )
            else:
                logger.info("No precise retriever configured")

        except Exception as e:
            logger.warning(f"Failed to initialize precise retriever: {e}")
            # Don't raise, as precise retriever is optional

    async def _initialize_llm_model(self, config: AIConfig):
        """Initialize LLM model"""
        try:
            self.llm_model = ServiceFactory.create_llm_model(
                config.llm_model, config.llm_config
            )
            logger.info(f"Initialized LLM model: {config.llm_model}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise

    async def _initialize_web_search_components(self, config: AIConfig):
        """Initialize web search components"""
        try:
            # Get web search configuration
            web_search_config = getattr(config, "search_config", {})

            if not web_search_config.get("enabled", True):
                logger.info("Web search disabled in configuration")
                return

            # Initialize query analyzer
            analyzer_type = web_search_config.get("analyzer_type", "rule_based")
            analyzer_config = web_search_config.get("analyzer_config", {})

            # Pass LLM model to analyzer if using LLM-powered analyzer
            if analyzer_type == "llm_powered" and self.llm_model:
                analyzer_config["llm_model"] = self.llm_model

            self.query_analyzer = WebSearchProviderRegistry.create_analyzer(
                analyzer_type, analyzer_config
            )
            logger.info(f"Initialized query analyzer: {analyzer_type}")

            # Initialize content processor
            processor_type = web_search_config.get("processor_type", "advanced")
            processor_config = web_search_config.get("processor_config", {})
            self.content_processor = WebSearchProviderRegistry.create_processor(
                processor_type, processor_config
            )
            logger.info(f"Initialized content processor: {processor_type}")

            # Initialize result fusion engine
            fusion_type = web_search_config.get("fusion_type", "intelligent")
            fusion_config = web_search_config.get("fusion_config", {})
            self.result_fusion = WebSearchProviderRegistry.create_fusion_engine(
                fusion_type, fusion_config
            )
            logger.info(f"Initialized result fusion: {fusion_type}")

            # Initialize web search agent
            agent_type = web_search_config.get("agent_type", "multi_provider")
            agent_config = web_search_config.get("agent_config", {})
            self.web_search_agent = WebSearchProviderRegistry.create_agent(
                agent_type, agent_config
            )
            logger.info(f"Initialized web search agent: {agent_type}")

            logger.info("Web search components initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize web search components: {e}")
            # Don't raise, as web search is optional

    async def _initialize_search_service(self, config: AIConfig):
        """Initialize search service"""
        try:
            # Create a custom search service that uses our components
            self.search_service = TechnologyAgnosticSearchService(
                embedding_model=self.embedding_model,
                vector_database=self.vector_database,
                precise_retriever=self.precise_retriever,
                config=config.search_config,
            )
            logger.info("Initialized technology-agnostic search service")

        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            raise

    async def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to all relevant indexes

        Args:
            documents: List of documents to index

        Returns:
            Success status
        """
        try:
            if not self.is_initialized:
                logger.error("Service manager not initialized")
                return False

            # Generate embeddings if not present
            documents_with_embeddings = await self._ensure_embeddings(documents)

            # Add to vector database
            vector_success = await self.vector_database.add_documents(
                documents_with_embeddings
            )

            # Add to precise retriever if available
            precise_success = True
            if self.precise_retriever:
                precise_success = await self.precise_retriever.index_documents(
                    documents_with_embeddings
                )

            success = vector_success and precise_success
            if success:
                self.metrics["documents_indexed"] += len(documents)

            logger.info(
                f"Added {len(documents)} documents - Vector: {vector_success}, Precise: {precise_success}"
            )
            return success

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            self.metrics["errors"] += 1
            return False

    async def _ensure_embeddings(self, documents: List[Document]) -> List[Document]:
        """Ensure all documents have embeddings"""
        try:
            documents_needing_embeddings = [
                doc for doc in documents if not doc.embedding
            ]

            if documents_needing_embeddings:
                texts = [doc.content for doc in documents_needing_embeddings]
                embeddings = await self.embedding_model.encode(texts)

                for doc, embedding in zip(documents_needing_embeddings, embeddings):
                    doc.embedding = embedding

            return documents

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.AUTO,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_web_search: bool = True,
        selected_search_engine: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform search using configured services

        Args:
            query: Search query
            strategy: Search strategy
            k: Number of results
            filters: Optional filters
            include_web_search: Whether to include web search

        Returns:
            Search results
        """
        try:
            if not self.is_initialized or not self.search_service:
                logger.error("Search service not available")
                return []

            # Analyze query for web search routing
            web_search_strategy = WebSearchStrategy.AUTO
            if include_web_search and self.query_analyzer:
                query_analysis = await self.query_analyzer.analyze_query(query)
                web_search_strategy = query_analysis.suggested_strategy

            # Determine search approach
            if web_search_strategy == WebSearchStrategy.WEB_ONLY:
                # Web search only
                return await self._web_search_only(
                    query, k, filters, selected_search_engine
                )
            elif web_search_strategy == WebSearchStrategy.DOCUMENTS_ONLY:
                # Document search only
                results = await self.search_service.search(query, strategy, k, filters)
                self.metrics["searches_performed"] += 1
                return results
            elif (
                web_search_strategy
                in [WebSearchStrategy.HYBRID, WebSearchStrategy.AUTO]
                and include_web_search
            ):
                # Combined search
                return await self._hybrid_web_document_search(
                    query, strategy, k, filters, selected_search_engine
                )
            else:
                # Default to document search
                results = await self.search_service.search(query, strategy, k, filters)
                self.metrics["searches_performed"] += 1
                return results

        except Exception as e:
            logger.error(f"Error in search: {e}")
            self.metrics["errors"] += 1
            return []

    async def _web_search_only(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        selected_search_engine: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform web search only"""
        try:
            if not self.web_search_agent or not self.query_analyzer:
                logger.warning("Web search components not available")
                return []

            # Analyze query
            query_analysis = await self.query_analyzer.analyze_query(query)

            # Create search context
            search_context = SearchContext(
                original_query=query,
                analysis=query_analysis,
                max_results=k,
                include_content=True,
                quality_weight=0.3,
                recency_weight=0.7 if query_analysis.requires_latest else 0.3,
                preferred_provider=selected_search_engine,
            )

            # Perform web search
            web_results = await self.web_search_agent.search_with_context(
                search_context
            )

            # Convert to SearchResult format
            converted_results = []
            for web_result in web_results:
                search_result = SearchResult(
                    content=web_result.snippet or web_result.content or "",
                    metadata={
                        "title": web_result.title,
                        "url": web_result.url,
                        "source": web_result.provider,
                        "source_type": "web_search",
                        "authority_score": web_result.authority_score,
                        "published_date": web_result.published_date,
                        "is_recent": web_result.is_recent,
                    },
                    score=web_result.relevance_score,
                    search_type="web_search",
                )
                converted_results.append(search_result)

            self.metrics["web_searches_performed"] += 1
            self.metrics["searches_performed"] += 1

            logger.info(f"Web search returned {len(converted_results)} results")
            return converted_results

        except Exception as e:
            logger.error(f"Error in web search only: {e}")
            return []

    async def _hybrid_web_document_search(
        self,
        query: str,
        strategy: SearchStrategy,
        k: int,
        filters: Optional[Dict[str, Any]],
        selected_search_engine: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform hybrid web and document search"""
        try:
            # Get document results
            doc_results = await self.search_service.search(query, strategy, k, filters)

            # Get web results if components available
            web_results = []
            if self.web_search_agent and self.query_analyzer:
                query_analysis = await self.query_analyzer.analyze_query(query)

                search_context = SearchContext(
                    original_query=query,
                    analysis=query_analysis,
                    max_results=k // 2,  # Split results between web and docs
                    include_content=True,
                    quality_weight=0.3,
                    recency_weight=0.7 if query_analysis.requires_latest else 0.3,
                    preferred_provider=selected_search_engine,
                )

                web_search_results = await self.web_search_agent.search_with_context(
                    search_context
                )

                # Convert web results to SearchResult format
                for web_result in web_search_results:
                    search_result = SearchResult(
                        content=web_result.snippet or web_result.content or "",
                        metadata={
                            "title": web_result.title,
                            "url": web_result.url,
                            "source": web_result.provider,
                            "source_type": "web_search",
                            "authority_score": web_result.authority_score,
                            "published_date": web_result.published_date,
                            "is_recent": web_result.is_recent,
                        },
                        score=web_result.relevance_score,
                        search_type="web_search",
                    )
                    web_results.append(search_result)

            # Fuse results if fusion engine available
            if self.result_fusion and web_results and doc_results:
                try:
                    # Convert document results to web search format for fusion
                    web_search_results = []
                    for web_result in web_results:
                        from app.core.web_search_interfaces import (
                            WebSearchResult,
                            SourceType,
                        )

                        web_search_result = WebSearchResult(
                            title=web_result.metadata.get("title", ""),
                            url=web_result.metadata.get("url", ""),
                            snippet=web_result.content,
                            content=web_result.content,
                            relevance_score=web_result.score,
                            authority_score=web_result.metadata.get(
                                "authority_score", 0.5
                            ),
                            source_type=SourceType.NEWS,
                            provider=web_result.metadata.get("source", "unknown"),
                            published_date=web_result.metadata.get("published_date"),
                            is_recent=web_result.metadata.get("is_recent", False),
                            raw_result=web_result.metadata,
                        )
                        web_search_results.append(web_search_result)

                    # Use query analysis from earlier
                    if not "query_analysis" in locals():
                        query_analysis = await self.query_analyzer.analyze_query(query)

                    search_context = SearchContext(
                        original_query=query,
                        analysis=query_analysis,
                        max_results=k,
                        include_content=True,
                        quality_weight=0.3,
                        recency_weight=0.7 if query_analysis.requires_latest else 0.3,
                    )

                    # Fuse results
                    fused_results = await self.result_fusion.fuse_results(
                        web_search_results, doc_results, search_context
                    )

                    # Convert fused results back to SearchResult format
                    final_results = []
                    for result in fused_results:
                        if result.get("result_type") == "web_search":
                            search_result = SearchResult(
                                content=result.get("content", ""),
                                metadata=result.get("metadata", {}),
                                score=result.get("final_score", result.get("score", 0)),
                                search_type="hybrid_web",
                            )
                        else:
                            search_result = SearchResult(
                                content=result.get("content", ""),
                                metadata=result.get("metadata", {}),
                                score=result.get("final_score", result.get("score", 0)),
                                search_type="hybrid_document",
                            )
                        final_results.append(search_result)

                    self.metrics["web_searches_performed"] += 1
                    self.metrics["searches_performed"] += 1

                    logger.info(
                        f"Hybrid search with fusion returned {len(final_results)} results"
                    )
                    return final_results[:k]

                except Exception as e:
                    logger.warning(
                        f"Result fusion failed, falling back to simple combination: {e}"
                    )

            # Simple combination fallback
            all_results = doc_results + web_results

            # Simple deduplication and scoring
            combined_results = []
            seen_content = set()

            for result in all_results:
                content_hash = hash(result.content.lower()[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    combined_results.append(result)

            # Sort by score
            combined_results.sort(key=lambda x: x.score, reverse=True)

            self.metrics["web_searches_performed"] += 1
            self.metrics["searches_performed"] += 1

            logger.info(f"Hybrid search returned {len(combined_results[:k])} results")
            return combined_results[:k]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to document search only
            return await self.search_service.search(query, strategy, k, filters)

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for intent and optimal strategy"""
        try:
            if self.search_service:
                return await self.search_service.analyze_query(query)
            elif self.precise_retriever:
                return await self.precise_retriever.analyze_query(query)
            else:
                # Fallback analysis
                return QueryAnalysis(
                    query=query,
                    intent="general",
                    confidence=0.5,
                    extracted_entities=[],
                    filters={},
                    suggested_strategy="semantic",
                )

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return QueryAnalysis(
                query=query,
                intent="general",
                confidence=0.0,
                extracted_entities=[],
                filters={},
                suggested_strategy="semantic",
            )

    async def switch_configuration(self, new_config: AIConfig) -> bool:
        """
        Switch to a new configuration at runtime

        Args:
            new_config: New AI configuration

        Returns:
            Success status
        """
        try:
            logger.info("Switching AI configuration...")

            # Store old config for rollback
            old_config = self.current_config
            old_components = {
                "embedding_model": self.embedding_model,
                "vector_database": self.vector_database,
                "precise_retriever": self.precise_retriever,
                "llm_model": self.llm_model,
                "search_service": self.search_service,
                "web_search_agent": self.web_search_agent,
                "query_analyzer": self.query_analyzer,
                "result_fusion": self.result_fusion,
                "content_processor": self.content_processor,
            }

            try:
                # Initialize with new config
                await self._initialize_embedding_model(new_config)
                await self._initialize_vector_database(new_config)
                await self._initialize_precise_retriever(new_config)
                await self._initialize_llm_model(new_config)
                await self._initialize_web_search_components(new_config)
                await self._initialize_search_service(new_config)

                self.current_config = new_config
                self.metrics["config_switches"] += 1

                logger.info("Configuration switched successfully")
                return True

            except Exception as e:
                # Rollback on failure
                logger.error(f"Configuration switch failed, rolling back: {e}")

                self.current_config = old_config
                self.embedding_model = old_components["embedding_model"]
                self.vector_database = old_components["vector_database"]
                self.precise_retriever = old_components["precise_retriever"]
                self.llm_model = old_components["llm_model"]
                self.search_service = old_components["search_service"]
                self.web_search_agent = old_components["web_search_agent"]
                self.query_analyzer = old_components["query_analyzer"]
                self.result_fusion = old_components["result_fusion"]
                self.content_processor = old_components["content_processor"]

                return False

        except Exception as e:
            logger.error(f"Error switching configuration: {e}")
            self.metrics["errors"] += 1
            return False

    async def migrate_data(self, from_config: AIConfig, to_config: AIConfig) -> bool:
        """
        Migrate data between different technology stacks

        Args:
            from_config: Source configuration
            to_config: Target configuration

        Returns:
            Success status
        """
        try:
            logger.info("Starting data migration...")

            # Initialize source services
            source_vector_db = ServiceFactory.create_vector_database(
                from_config.vector_database, from_config.vector_config
            )

            # Initialize target services
            target_embedding_model = ServiceFactory.create_embedding_model(
                to_config.embedding_model, to_config.embedding_config
            )
            target_vector_db = ServiceFactory.create_vector_database(
                to_config.vector_database, to_config.vector_config
            )

            # Export data from source
            # Note: This is a simplified approach. In production, you'd need
            # proper data export/import methods for each database type
            logger.info(
                "Data migration is a complex operation that requires database-specific implementations"
            )

            # For now, we'll just log the migration attempt
            logger.info(
                f"Migration from {from_config.vector_database} to {to_config.vector_database} initiated"
            )

            return True

        except Exception as e:
            logger.error(f"Error migrating data: {e}")
            return False

    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        try:
            stats = {
                "is_initialized": self.is_initialized,
                "current_config": {
                    "embedding_model": (
                        self.current_config.embedding_model
                        if self.current_config
                        else None
                    ),
                    "vector_database": (
                        self.current_config.vector_database
                        if self.current_config
                        else None
                    ),
                    "precise_retriever": (
                        self.current_config.precise_retriever
                        if self.current_config
                        else None
                    ),
                    "llm_model": (
                        self.current_config.llm_model if self.current_config else None
                    ),
                },
                "metrics": self.metrics.copy(),
                "component_stats": {},
            }

            # Get stats from individual components
            if self.embedding_model:
                stats["component_stats"][
                    "embedding_model"
                ] = self.embedding_model.get_model_info()

            if self.vector_database:
                stats["component_stats"][
                    "vector_database"
                ] = self.vector_database.get_stats()

            if self.precise_retriever:
                stats["component_stats"][
                    "precise_retriever"
                ] = self.precise_retriever.get_index_stats()

            if self.llm_model:
                stats["component_stats"]["llm_model"] = self.llm_model.get_model_info()

            # Web search components stats
            if self.web_search_agent:
                stats["component_stats"][
                    "web_search_agent"
                ] = self.web_search_agent.get_provider_stats()

            stats["web_search_enabled"] = bool(
                self.web_search_agent
                and self.query_analyzer
                and self.result_fusion
                and self.content_processor
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}


class TechnologyAgnosticSearchService(SearchServiceInterface):
    """Technology-agnostic search service implementation"""

    def __init__(
        self,
        embedding_model: EmbeddingModelInterface,
        vector_database: VectorDatabaseInterface,
        precise_retriever: Optional[PreciseRetrievalInterface],
        config: Dict[str, Any],
    ):
        self.embedding_model = embedding_model
        self.vector_database = vector_database
        self.precise_retriever = precise_retriever
        self.config = config

        self.table_confidence_threshold = config.get("table_confidence_threshold", 0.6)
        self.hybrid_weight_general = config.get("hybrid_weight_general", 0.4)
        self.hybrid_weight_table = config.get("hybrid_weight_table", 0.6)

    async def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.AUTO,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform search with specified strategy"""
        try:
            if strategy == SearchStrategy.AUTO:
                # Analyze query to determine best strategy
                analysis = await self.analyze_query(query)
                if (
                    analysis.confidence > self.table_confidence_threshold
                    and self.precise_retriever
                ):
                    strategy = SearchStrategy.PRECISE
                else:
                    strategy = SearchStrategy.SEMANTIC

            if strategy == SearchStrategy.PRECISE and self.precise_retriever:
                return await self.precise_retriever.search_precise(query, k, filters)
            elif strategy == SearchStrategy.SEMANTIC:
                query_embedding = await self.embedding_model.encode_single(query)
                return await self.vector_database.search(query_embedding, k, filters)
            elif strategy == SearchStrategy.HYBRID:
                return await self._hybrid_search(query, k, filters)
            else:
                # Fallback to semantic search
                query_embedding = await self.embedding_model.encode_single(query)
                return await self.vector_database.search(query_embedding, k, filters)

        except Exception as e:
            logger.error(f"Error in technology-agnostic search: {e}")
            return []

    async def _hybrid_search(
        self, query: str, k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and precise methods"""
        try:
            # Get semantic results
            query_embedding = await self.embedding_model.encode_single(query)
            semantic_results = await self.vector_database.search(
                query_embedding, k, filters
            )

            # Get precise results if available
            precise_results = []
            if self.precise_retriever:
                precise_results = await self.precise_retriever.search_precise(
                    query, k, filters
                )

            # Combine and rerank results
            combined_results = await self._combine_results(
                semantic_results, precise_results, k
            )

            return combined_results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def _combine_results(
        self,
        semantic_results: List[SearchResult],
        precise_results: List[SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """Combine and rerank results from different search methods"""
        try:
            # Adjust scores
            for result in semantic_results:
                result.score *= self.hybrid_weight_general
                result.search_type = "hybrid_semantic"

            for result in precise_results:
                result.score *= self.hybrid_weight_table
                result.search_type = "hybrid_precise"

            # Combine and deduplicate
            all_results = semantic_results + precise_results

            # Simple deduplication based on content similarity
            deduplicated = []
            for result in all_results:
                is_duplicate = False
                for existing in deduplicated:
                    if (
                        self._calculate_similarity(result.content, existing.content)
                        > 0.8
                    ):
                        if result.score > existing.score:
                            deduplicated.remove(existing)
                            deduplicated.append(result)
                        is_duplicate = True
                        break

                if not is_duplicate:
                    deduplicated.append(result)

            # Sort by score and return top k
            deduplicated.sort(key=lambda x: x.score, reverse=True)
            return deduplicated[:k]

        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return (semantic_results + precise_results)[:k]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to all relevant indexes"""
        # This would be handled by the service manager
        return True

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for intent and strategy"""
        if self.precise_retriever:
            return await self.precise_retriever.analyze_query(query)
        else:
            # Simple fallback analysis
            return QueryAnalysis(
                query=query,
                intent="general",
                confidence=0.5,
                extracted_entities=[],
                filters={},
                suggested_strategy="semantic",
            )

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "type": "technology_agnostic",
            "has_precise_retriever": self.precise_retriever is not None,
            "table_confidence_threshold": self.table_confidence_threshold,
            "hybrid_weights": {
                "general": self.hybrid_weight_general,
                "table": self.hybrid_weight_table,
            },
        }


# Global service manager instance
ai_service_manager = AIServiceManager()
