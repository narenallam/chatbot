"""
Web Search Agent Implementations
Intelligent agents for orchestrating web search across multiple providers
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random

from app.core.web_search_interfaces import (
    WebSearchAgent,
    WebSearchInterface,
    WebSearchResult,
    SearchContext,
    SearchIntent,
    SourceType,
    WebSearchProviderRegistry,
)

logger = logging.getLogger(__name__)


class MultiProviderSearchAgent(WebSearchAgent):
    """Intelligent multi-provider search agent with fallback and optimization"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = []
        self.provider_priority = config.get(
            "provider_priority", ["duckduckgo", "brave_search", "serpapi"]
        )
        self.max_concurrent_searches = config.get("max_concurrent_searches", 2)
        self.timeout_per_provider = config.get("timeout_per_provider", 15)
        self.fallback_enabled = config.get("fallback_enabled", True)

        # Provider usage tracking
        self.provider_stats = {}

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize search providers based on configuration"""
        try:
            print(f"=== DEBUG: MultiProviderSearchAgent config: {self.config}")
            provider_configs = self.config.get("providers", {})
            print(f"=== DEBUG: provider_configs: {provider_configs}")

            for provider_name in self.provider_priority:
                try:
                    provider_config = provider_configs.get(provider_name, {})
                    print(f"=== DEBUG: {provider_name} config: {provider_config}")
                    provider = WebSearchProviderRegistry.create_provider(
                        provider_name, provider_config
                    )
                    self.providers.append(
                        {
                            "name": provider_name,
                            "instance": provider,
                            "config": provider_config,
                            "priority": len(self.providers) + 1,
                        }
                    )

                    # Initialize stats
                    self.provider_stats[provider_name] = {
                        "requests": 0,
                        "successes": 0,
                        "failures": 0,
                        "avg_response_time": 0,
                        "last_used": None,
                    }

                    logger.info(f"Initialized web search provider: {provider_name}")

                except Exception as e:
                    logger.warning(
                        f"Failed to initialize provider {provider_name}: {e}"
                    )

        except Exception as e:
            logger.error(f"Error initializing providers: {e}")

    async def search_with_context(
        self, context: SearchContext
    ) -> List[WebSearchResult]:
        """Perform context-aware search with intelligent provider selection"""
        try:
            logger.info(f"🔍 search_with_context called - query: '{context.original_query}', preferred_provider: '{context.preferred_provider}'")
            logger.info(f"📋 Available providers: {[p['name'] for p in self.providers]}")
            
            # Refine query based on intent
            logger.info(f"🔧 Refining query...")
            refined_query = await self.refine_query(
                context.original_query, context.analysis.intent
            )
            logger.info(f"✅ Refined query: '{refined_query}'")

            # Select optimal providers based on context
            logger.info(f"🎯 Selecting providers...")
            selected_providers = await self._select_providers(context)
            logger.info(f"✅ Selected providers: {[p['name'] for p in selected_providers]}")

            if not selected_providers:
                logger.warning("❌ No available providers for search")
                return []

            # Execute search with selected providers
            logger.info(f"🚀 Executing search with {len(selected_providers)} provider(s)...")
            results = await self._execute_search(
                refined_query, context, selected_providers
            )
            logger.info(f"✅ _execute_search returned {len(results)} raw results")

            # Post-process and rank results
            logger.info(f"🔀 Post-processing results...")
            processed_results = await self._post_process_results(results, context)

            logger.info(
                f"✅ Multi-provider search returned {len(processed_results)} processed results"
            )
            return processed_results

        except Exception as e:
            logger.error(f"❌ Multi-provider search failed: {e}", exc_info=True)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    async def refine_query(self, query: str, intent: SearchIntent) -> str:
        """Refine query for better web search results"""
        try:
            refined_query = query.strip()

            # Add intent-specific refinements
            if intent == SearchIntent.LATEST_INFO:
                # Add temporal qualifiers for latest information
                current_year = datetime.now().year
                if str(current_year) not in refined_query:
                    refined_query = f"{refined_query} {current_year}"

                if not any(
                    word in refined_query.lower()
                    for word in ["latest", "recent", "current"]
                ):
                    refined_query = f"latest {refined_query}"

            elif intent == SearchIntent.COMPARISON:
                # Ensure comparison keywords are present
                if not any(
                    word in refined_query.lower()
                    for word in ["vs", "versus", "compare", "comparison"]
                ):
                    refined_query = f"{refined_query} comparison"

            elif intent == SearchIntent.HOW_TO:
                # Ensure how-to format
                if not refined_query.lower().startswith("how to"):
                    refined_query = f"how to {refined_query}"

            elif intent == SearchIntent.SPECIFIC_FACTS:
                # Add specificity qualifiers
                if not any(
                    word in refined_query.lower()
                    for word in ["what", "definition", "meaning"]
                ):
                    refined_query = f"what is {refined_query}"

            return refined_query

        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return query

    async def _select_providers(self, context: SearchContext) -> List[Dict[str, Any]]:
        """Select optimal providers based on context and availability"""
        try:
            logger.info(f"🔍 Checking availability for {len(self.providers)} providers...")
            available_providers = []

            # Check provider availability
            for provider_info in self.providers:
                provider_name = provider_info["name"]
                try:
                    logger.debug(f"  Checking availability for '{provider_name}'...")
                    is_available = await provider_info["instance"].check_availability()
                    if is_available:
                        rate_limit_info = provider_info[
                            "instance"
                        ].get_rate_limit_info()
                        provider_info["rate_limit"] = rate_limit_info
                        available_providers.append(provider_info)
                        logger.info(f"  ✅ '{provider_name}' is available")
                    else:
                        logger.warning(f"  ❌ '{provider_name}' is not available")
                except Exception as e:
                    logger.warning(
                        f"  ⚠️ Failed to check availability for {provider_name}: {e}", exc_info=True
                    )

            logger.info(f"📋 Found {len(available_providers)} available providers: {[p['name'] for p in available_providers]}")

            if not available_providers:
                logger.error("❌ No available providers found!")
                return []

            # Apply context-based selection strategy
            logger.info(f"🎯 Applying selection strategy...")
            selected = self._apply_selection_strategy(available_providers, context)

            logger.info(f"✅ Selected providers: {[p['name'] for p in selected]}")
            return selected

        except Exception as e:
            logger.error(f"❌ Error selecting providers: {e}", exc_info=True)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self.providers[:1]  # Fallback to first provider

    def _apply_selection_strategy(
        self, available_providers: List[Dict[str, Any]], context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Apply intelligent provider selection strategy"""
        try:
            # Check if user has a preferred provider
            if context.preferred_provider:
                # Map frontend provider names to backend provider names
                provider_mapping = {
                    "duckduckgo": "duckduckgo",
                    "brave": "brave_search",
                    "bing": "serpapi",  # SerpAPI provides Bing/Google results
                    "serpapi": "serpapi",  # Direct SerpAPI mapping
                    "google": "serpapi",  # SerpAPI provides Google results
                }

                backend_provider_name = provider_mapping.get(context.preferred_provider)
                if backend_provider_name:
                    preferred_provider = next(
                        (
                            p
                            for p in available_providers
                            if p["name"] == backend_provider_name
                        ),
                        None,
                    )
                    if preferred_provider:
                        # Check if preferred provider has quota
                        rate_limit = preferred_provider.get("rate_limit", {})
                        remaining = rate_limit.get("remaining")

                        if remaining is None or remaining > 0:
                            logger.info(
                                f"Using preferred provider: {backend_provider_name}"
                            )
                            return [preferred_provider]
                        else:
                            logger.warning(
                                f"Preferred provider {backend_provider_name} has no quota remaining"
                            )

            # Strategy 1: For latest information, prefer higher quality providers
            if context.analysis.requires_latest:
                # Prioritize providers with better fresh results
                quality_order = ["serpapi", "brave_search", "duckduckgo"]
                selected = []

                for provider_name in quality_order:
                    provider = next(
                        (p for p in available_providers if p["name"] == provider_name),
                        None,
                    )
                    if provider:
                        rate_limit = provider.get("rate_limit", {})
                        remaining = rate_limit.get("remaining")

                        # Check if provider has quota
                        if remaining is None or remaining > 0:
                            selected.append(provider)
                            if (
                                len(selected) >= 2
                            ):  # Use max 2 providers for latest info
                                break

                return selected if selected else available_providers[:1]

            # Strategy 2: For general queries, start with free unlimited providers
            else:
                # Prefer unlimited providers first, then limited ones
                unlimited_providers = [
                    p
                    for p in available_providers
                    if p.get("rate_limit", {}).get("unlimited", False)
                ]
                limited_providers = [
                    p
                    for p in available_providers
                    if not p.get("rate_limit", {}).get("unlimited", False)
                ]

                selected = []

                # Add one unlimited provider
                if unlimited_providers:
                    selected.append(unlimited_providers[0])

                # Add one limited provider if quota available
                for provider in limited_providers:
                    rate_limit = provider.get("rate_limit", {})
                    remaining = rate_limit.get("remaining")

                    if (
                        remaining is None or remaining > 10
                    ):  # Keep some quota for critical searches
                        selected.append(provider)
                        break

                return selected if selected else available_providers[:1]

        except Exception as e:
            logger.error(f"Error in selection strategy: {e}")
            return available_providers[:1]

    async def _execute_search(
        self, query: str, context: SearchContext, providers: List[Dict[str, Any]]
    ) -> List[WebSearchResult]:
        """Execute search across selected providers"""
        try:
            # Create search tasks
            search_tasks = []

            for provider_info in providers[: self.max_concurrent_searches]:
                task = self._search_with_provider(query, context, provider_info)
                search_tasks.append(task)

            # Execute searches concurrently with timeout
            try:
                results_lists = await asyncio.wait_for(
                    asyncio.gather(*search_tasks, return_exceptions=True),
                    timeout=self.timeout_per_provider * 2,
                )
            except asyncio.TimeoutError:
                logger.warning("Search timeout exceeded")
                results_lists = []

            # Combine results
            all_results = []
            for results in results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    logger.warning(f"Search task failed: {results}")

            return all_results

        except Exception as e:
            logger.error(f"Error executing searches: {e}")
            return []

    async def _search_with_provider(
        self, query: str, context: SearchContext, provider_info: Dict[str, Any]
    ) -> List[WebSearchResult]:
        """Execute search with a single provider"""
        provider_name = provider_info["name"]
        provider_instance = provider_info["instance"]

        start_time = datetime.now()

        try:
            logger.info(f"🔍 _search_with_provider called - provider: '{provider_name}', query: '{query}'")
            
            # Update stats
            self.provider_stats[provider_name]["requests"] += 1
            self.provider_stats[provider_name]["last_used"] = start_time

            # Execute search
            logger.info(f"📞 Calling provider_instance.search() for '{provider_name}'...")
            results = await asyncio.wait_for(
                provider_instance.search(
                    query=query,
                    max_results=context.max_results,
                    include_content=context.include_content,
                ),
                timeout=self.timeout_per_provider,
            )

            # Update success stats
            response_time = (datetime.now() - start_time).total_seconds()
            stats = self.provider_stats[provider_name]
            stats["successes"] += 1
            stats["avg_response_time"] = (
                stats["avg_response_time"] * (stats["successes"] - 1) + response_time
            ) / stats["successes"]

            logger.info(
                f"✅ Provider {provider_name} returned {len(results)} results in {response_time:.2f}s"
            )
            return results

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Provider {provider_name} timed out after {self.timeout_per_provider}s")
            self.provider_stats[provider_name]["failures"] += 1
            return []

        except Exception as e:
            logger.error(f"❌ Provider {provider_name} failed: {e}", exc_info=True)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.provider_stats[provider_name]["failures"] += 1
            return []

    async def _post_process_results(
        self, results: List[WebSearchResult], context: SearchContext
    ) -> List[WebSearchResult]:
        """Post-process and rank search results"""
        try:
            if not results:
                return []

            # Remove duplicates
            unique_results = self._deduplicate_results(results)

            # Apply context-based scoring
            scored_results = self._score_results(unique_results, context)

            # Sort by combined score
            scored_results.sort(
                key=lambda r: r.relevance_score
                + (r.authority_score * context.quality_weight)
                + (float(r.is_recent) * context.recency_weight),
                reverse=True,
            )

            # Limit results
            final_results = scored_results[: context.max_results]

            logger.info(
                f"Post-processing reduced {len(results)} to {len(final_results)} results"
            )
            return final_results

        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return results[: context.max_results]

    def _deduplicate_results(
        self, results: List[WebSearchResult]
    ) -> List[WebSearchResult]:
        """Remove duplicate results based on URL and content similarity"""
        try:
            unique_results = []
            seen_urls = set()

            for result in results:
                # Skip if URL already seen
                if result.url in seen_urls:
                    continue

                # Check content similarity with existing results
                is_duplicate = False
                for existing in unique_results:
                    similarity = self._calculate_content_similarity(
                        result.snippet, existing.snippet
                    )
                    if similarity > 0.8:
                        # Keep the one with higher authority score
                        if result.authority_score > existing.authority_score:
                            unique_results.remove(existing)
                            break
                        else:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    unique_results.append(result)
                    seen_urls.add(result.url)

            return unique_results

        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")
            return results

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _score_results(
        self, results: List[WebSearchResult], context: SearchContext
    ) -> List[WebSearchResult]:
        """Apply context-based scoring to results"""
        try:
            for result in results:
                # Base relevance score (from provider)
                base_score = result.relevance_score

                # Recency boost for latest info queries
                if context.analysis.requires_latest and result.is_recent:
                    base_score += 0.2

                # Authority boost for specific source types
                preferred_sources = context.analysis.filters.get(
                    "preferred_source_types", []
                )
                if preferred_sources and result.source_type.value in preferred_sources:
                    base_score += 0.1

                # Entity type relevance
                entity_types = context.analysis.entity_types
                if entity_types:
                    content_lower = result.snippet.lower()
                    entity_matches = sum(
                        1 for entity in entity_types if entity in content_lower
                    )
                    if entity_matches > 0:
                        base_score += 0.05 * entity_matches

                # Update result score
                result.relevance_score = min(1.0, base_score)

            return results

        except Exception as e:
            logger.error(f"Error scoring results: {e}")
            return results

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider performance statistics"""
        return {
            "providers": self.provider_stats.copy(),
            "total_providers": len(self.providers),
            "active_providers": len(
                [
                    p
                    for p in self.providers
                    if self.provider_stats[p["name"]]["successes"] > 0
                ]
            ),
        }


# Register the agent
WebSearchProviderRegistry.register_agent("multi_provider", MultiProviderSearchAgent)
