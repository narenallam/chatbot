"""
Web Search Provider Implementations
Concrete implementations for different web search APIs
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin
import json

from app.core.web_search_interfaces import (
    WebSearchInterface,
    WebSearchResult,
    SourceType,
    WebSearchProviderRegistry,
)

logger = logging.getLogger(__name__)


class DuckDuckGoProvider(WebSearchInterface):
    """DuckDuckGo search provider - Free and unlimited"""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = "https://api.duckduckgo.com/"
        self.instant_answer_url = "https://api.duckduckgo.com/"
        self.search_url = "https://html.duckduckgo.com/html/"
        self.timeout = config.get("timeout", 10)
        self.user_agent = config.get(
            "user_agent", "Mozilla/5.0 (compatible; AI-Mate-Bot/1.0)"
        )
        self.max_retries = config.get("max_retries", 3)

    async def search(
        self, query: str, max_results: int = 10, include_content: bool = False
    ) -> List[WebSearchResult]:
        """Search using DuckDuckGo"""
        try:
            results = []

            # Try instant answer API first
            instant_results = await self._get_instant_answers(query)
            if instant_results:
                results.extend(instant_results[: max_results // 2])

            # Get web search results
            web_results = await self._get_web_results(query, max_results - len(results))
            if web_results:
                results.extend(web_results)

            # Fetch content if requested
            if include_content and results:
                await self._fetch_content(results[:5])  # Limit content fetching

            logger.info(f"DuckDuckGo returned {len(results)} results for: {query}")
            return results[:max_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    async def _get_instant_answers(self, query: str) -> List[WebSearchResult]:
        """Get instant answers from DuckDuckGo"""
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(
                    self.instant_answer_url,
                    params=params,
                    headers={"User-Agent": self.user_agent},
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    results = []

                    # Process abstract/definition
                    if data.get("Abstract"):
                        results.append(
                            WebSearchResult(
                                title=data.get("Heading", "Definition"),
                                url=data.get("AbstractURL", ""),
                                snippet=data.get("Abstract", ""),
                                source_type=SourceType.OFFICIAL,
                                authority_score=0.9,
                                relevance_score=0.95,
                                provider="duckduckgo_instant",
                                raw_result=data,
                            )
                        )

                    # Process answer (direct answer)
                    if data.get("Answer"):
                        results.append(
                            WebSearchResult(
                                title="Direct Answer",
                                url=data.get("AnswerURL", ""),
                                snippet=data.get("Answer", ""),
                                source_type=SourceType.OFFICIAL,
                                authority_score=0.95,
                                relevance_score=1.0,
                                provider="duckduckgo_instant",
                                raw_result=data,
                            )
                        )

                    # Process related topics
                    for topic in data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append(
                                WebSearchResult(
                                    title=topic.get("Text", "").split(" - ")[0],
                                    url=topic.get("FirstURL", ""),
                                    snippet=topic.get("Text", ""),
                                    source_type=SourceType.OFFICIAL,
                                    authority_score=0.8,
                                    relevance_score=0.7,
                                    provider="duckduckgo_instant",
                                    raw_result=topic,
                                )
                            )

                    return results

        except Exception as e:
            logger.warning(f"DuckDuckGo instant answers failed: {e}")
            return []

    async def _get_web_results(
        self, query: str, max_results: int
    ) -> List[WebSearchResult]:
        """Get web search results using DuckDuckGo HTML interface"""
        try:
            # Note: This is a simplified implementation
            # For production, you might want to use a library like duckduckgo-search
            # or implement proper HTML parsing

            params = {
                "q": query,
                "b": "",  # No ads
                "kl": "us-en",  # Language
                "df": "",  # Date filter
            }

            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(
                    self.search_url, params=params, headers=headers
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"DuckDuckGo web search failed with status: {response.status}"
                        )
                        return []

                    # For now, return empty list
                    # In a full implementation, parse HTML results here
                    # or use the duckduckgo-search library
                    return []

        except Exception as e:
            logger.warning(f"DuckDuckGo web search failed: {e}")
            return []

    async def _fetch_content(self, results: List[WebSearchResult]):
        """Fetch full content for results"""

        async def fetch_single_content(result: WebSearchResult):
            try:
                if not result.url or not result.url.startswith("http"):
                    return

                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.get(
                        result.url, headers={"User-Agent": self.user_agent}
                    ) as response:
                        if response.status == 200:
                            content = await response.text()
                            # Simple content extraction (in production, use readability or similar)
                            result.content = self._extract_text_content(content)

            except Exception as e:
                logger.warning(f"Failed to fetch content from {result.url}: {e}")

        # Fetch content concurrently but limit concurrency
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(result):
            async with semaphore:
                await fetch_single_content(result)

        await asyncio.gather(
            *[fetch_with_semaphore(result) for result in results],
            return_exceptions=True,
        )

    def _extract_text_content(self, html: str) -> str:
        """Extract text content from HTML (simplified)"""
        try:
            # Remove script and style elements
            import re

            text = re.sub(
                r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
            text = re.sub(
                r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", "", text)
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text[:2000]  # Limit length
        except Exception:
            return ""

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "DuckDuckGo",
            "type": "free",
            "rate_limit": "unlimited",
            "features": ["instant_answers", "web_search", "privacy_focused"],
            "max_results": 50,
            "supports_content": True,
        }

    async def check_availability(self) -> bool:
        """Check if DuckDuckGo is available"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(
                    self.instant_answer_url,
                    params={"q": "test", "format": "json"},
                    headers={"User-Agent": self.user_agent},
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            "provider": "duckduckgo",
            "daily_limit": None,
            "monthly_limit": None,
            "remaining": None,
            "reset_time": None,
            "unlimited": True,
        }


class BraveSearchProvider(WebSearchInterface):
    """Brave Search API provider - 2000 queries/month free"""

    def __init__(self, config: Dict[str, Any]):
        print(f"=== DEBUG: BraveSearchProvider init with config: {config}")
        self.api_key = config.get("api_key")
        print(f"=== DEBUG: BraveSearchProvider api_key: {repr(self.api_key)}")
        if not self.api_key:
            raise ValueError("Brave Search API key is required")

        self.base_url = "https://api.search.brave.com/res/v1/"
        self.timeout = config.get("timeout", 15)
        self.monthly_quota = config.get("monthly_quota", 2000)
        self.requests_made = 0  # Track usage (in production, store in database)

    async def search(
        self, query: str, max_results: int = 10, include_content: bool = False
    ) -> List[WebSearchResult]:
        """Search using Brave Search API"""
        try:
            if self.requests_made >= self.monthly_quota:
                logger.warning("Brave Search monthly quota exceeded")
                return []

            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            }

            params = {
                "q": query,
                "count": min(max_results, 20),  # Brave max is 20
                "search_lang": "en",
                "country": "US",
                "safesearch": "moderate",
                "freshness": "pw",  # Past week for recent results
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(
                    f"{self.base_url}web/search", params=params, headers=headers
                ) as response:
                    self.requests_made += 1

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Brave Search API error {response.status}: {error_text}"
                        )
                        return []

                    data = await response.json()
                    return self._parse_brave_results(data, include_content)

        except Exception as e:
            logger.error(f"Brave Search failed: {e}")
            return []

    def _parse_brave_results(
        self, data: Dict[str, Any], include_content: bool = False
    ) -> List[WebSearchResult]:
        """Parse Brave Search API response"""
        results = []

        try:
            web_results = data.get("web", {}).get("results", [])

            for item in web_results:
                # Parse publication date if available
                published_date = None
                if item.get("age"):
                    try:
                        # Parse age like "2 days ago", "1 week ago"
                        age_text = item["age"].lower()
                        if "day" in age_text:
                            days = int(re.search(r"(\d+)", age_text).group(1))
                            published_date = datetime.now() - timedelta(days=days)
                        elif "week" in age_text:
                            weeks = int(re.search(r"(\d+)", age_text).group(1))
                            published_date = datetime.now() - timedelta(weeks=weeks)
                        elif "hour" in age_text:
                            hours = int(re.search(r"(\d+)", age_text).group(1))
                            published_date = datetime.now() - timedelta(hours=hours)
                    except:
                        pass

                # Determine source type
                source_type = self._classify_source_type(item.get("url", ""))

                # Calculate scores
                authority_score = self._calculate_authority_score(item.get("url", ""))
                relevance_score = min(item.get("score", 0.5), 1.0)
                is_recent = (
                    published_date and (datetime.now() - published_date).days <= 7
                )

                result = WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    published_date=published_date,
                    source_type=source_type,
                    authority_score=authority_score,
                    relevance_score=relevance_score,
                    is_recent=is_recent,
                    provider="brave_search",
                    raw_result=item,
                )

                results.append(result)

            logger.info(f"Brave Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error parsing Brave Search results: {e}")
            return []

    def _classify_source_type(self, url: str) -> SourceType:
        """Classify source type based on URL"""
        url_lower = url.lower()

        if any(
            domain in url_lower
            for domain in ["news.", "reuters.", "ap.org", "bbc.", "cnn.", "npr.org"]
        ):
            return SourceType.NEWS
        elif any(domain in url_lower for domain in ["edu", "gov", ".org"]):
            return SourceType.OFFICIAL
        elif any(
            domain in url_lower
            for domain in ["arxiv.", "scholar.", "pubmed.", "researchgate."]
        ):
            return SourceType.ACADEMIC
        elif any(
            domain in url_lower
            for domain in ["medium.", "blog", "wordpress.", "substack."]
        ):
            return SourceType.BLOG
        elif any(
            domain in url_lower
            for domain in ["reddit.", "stackoverflow.", "quora.", "forums"]
        ):
            return SourceType.FORUM
        elif any(
            domain in url_lower for domain in ["twitter.", "linkedin.", "facebook."]
        ):
            return SourceType.SOCIAL
        else:
            return SourceType.UNKNOWN

    def _calculate_authority_score(self, url: str) -> float:
        """Calculate authority score based on domain"""
        url_lower = url.lower()

        # High authority domains
        if any(
            domain in url_lower
            for domain in [
                "wikipedia.",
                "gov",
                "edu",
                "reuters.",
                "ap.org",
                "bbc.",
                "nature.",
                "science.",
            ]
        ):
            return 0.9

        # Medium authority domains
        if any(
            domain in url_lower
            for domain in [
                "cnn.",
                "npr.org",
                "wsj.",
                "nytimes.",
                "guardian.",
                "forbes.",
            ]
        ):
            return 0.8

        # Low authority domains
        if any(
            domain in url_lower
            for domain in ["blog", "wordpress.", "medium.", "reddit.", "quora."]
        ):
            return 0.4

        return 0.6  # Default

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "Brave Search",
            "type": "freemium",
            "rate_limit": "2000/month",
            "features": ["web_search", "fresh_results", "privacy_focused"],
            "max_results": 20,
            "supports_content": False,
        }

    async def check_availability(self) -> bool:
        """Check if Brave Search is available and has quota"""
        try:
            if self.requests_made >= self.monthly_quota:
                return False

            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(
                    f"{self.base_url}web/search",
                    params={"q": "test", "count": 1},
                    headers=headers,
                ) as response:
                    return response.status == 200

        except Exception:
            return False

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            "provider": "brave_search",
            "daily_limit": None,
            "monthly_limit": self.monthly_quota,
            "remaining": max(0, self.monthly_quota - self.requests_made),
            "reset_time": None,
            "unlimited": False,
        }


class SerpAPIProvider(WebSearchInterface):
    """SerpAPI provider - Google results, 100 searches/month free"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("SerpAPI key is required")

        self.base_url = "https://serpapi.com/search"
        self.timeout = config.get("timeout", 15)
        self.monthly_quota = config.get("monthly_quota", 100)
        self.requests_made = 0  # Track usage

    async def search(
        self, query: str, max_results: int = 10, include_content: bool = False
    ) -> List[WebSearchResult]:
        """Search using SerpAPI (Google results)"""
        try:
            if self.requests_made >= self.monthly_quota:
                logger.warning("SerpAPI monthly quota exceeded")
                return []

            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": min(max_results, 100),
                "hl": "en",
                "gl": "us",
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(self.base_url, params=params) as response:
                    self.requests_made += 1

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"SerpAPI error {response.status}: {error_text}")
                        return []

                    data = await response.json()
                    return self._parse_serp_results(data, include_content)

        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []

    def _parse_serp_results(
        self, data: Dict[str, Any], include_content: bool = False
    ) -> List[WebSearchResult]:
        """Parse SerpAPI response"""
        results = []

        try:
            # Process organic results
            organic_results = data.get("organic_results", [])

            for item in organic_results:
                # Parse date if available
                published_date = None
                if item.get("date"):
                    try:
                        published_date = datetime.fromisoformat(
                            item["date"].replace("Z", "+00:00")
                        )
                    except:
                        pass

                source_type = self._classify_source_type(item.get("link", ""))
                authority_score = self._calculate_authority_score(item.get("link", ""))

                # SerpAPI provides position which we can use for relevance
                position = item.get("position", 1)
                relevance_score = max(0.1, 1.0 - (position - 1) * 0.1)

                is_recent = (
                    published_date and (datetime.now() - published_date).days <= 7
                )

                result = WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    published_date=published_date,
                    source_type=source_type,
                    authority_score=authority_score,
                    relevance_score=relevance_score,
                    is_recent=is_recent,
                    provider="serpapi_google",
                    raw_result=item,
                )

                results.append(result)

            logger.info(f"SerpAPI returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error parsing SerpAPI results: {e}")
            return []

    def _classify_source_type(self, url: str) -> SourceType:
        """Classify source type based on URL"""
        # Same implementation as BraveSearchProvider
        url_lower = url.lower()

        if any(
            domain in url_lower
            for domain in ["news.", "reuters.", "ap.org", "bbc.", "cnn.", "npr.org"]
        ):
            return SourceType.NEWS
        elif any(domain in url_lower for domain in ["edu", "gov", ".org"]):
            return SourceType.OFFICIAL
        elif any(
            domain in url_lower
            for domain in ["arxiv.", "scholar.", "pubmed.", "researchgate."]
        ):
            return SourceType.ACADEMIC
        elif any(
            domain in url_lower
            for domain in ["medium.", "blog", "wordpress.", "substack."]
        ):
            return SourceType.BLOG
        elif any(
            domain in url_lower
            for domain in ["reddit.", "stackoverflow.", "quora.", "forums"]
        ):
            return SourceType.FORUM
        elif any(
            domain in url_lower for domain in ["twitter.", "linkedin.", "facebook."]
        ):
            return SourceType.SOCIAL
        else:
            return SourceType.UNKNOWN

    def _calculate_authority_score(self, url: str) -> float:
        """Calculate authority score based on domain"""
        # Same implementation as BraveSearchProvider
        url_lower = url.lower()

        if any(
            domain in url_lower
            for domain in [
                "wikipedia.",
                "gov",
                "edu",
                "reuters.",
                "ap.org",
                "bbc.",
                "nature.",
                "science.",
            ]
        ):
            return 0.9

        if any(
            domain in url_lower
            for domain in [
                "cnn.",
                "npr.org",
                "wsj.",
                "nytimes.",
                "guardian.",
                "forbes.",
            ]
        ):
            return 0.8

        if any(
            domain in url_lower
            for domain in ["blog", "wordpress.", "medium.", "reddit.", "quora."]
        ):
            return 0.4

        return 0.6

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "SerpAPI (Google)",
            "type": "freemium",
            "rate_limit": "100/month",
            "features": ["google_results", "high_quality", "fresh_results"],
            "max_results": 100,
            "supports_content": False,
        }

    async def check_availability(self) -> bool:
        """Check if SerpAPI is available and has quota"""
        try:
            if self.requests_made >= self.monthly_quota:
                return False

            params = {
                "q": "test",
                "api_key": self.api_key,
                "engine": "google",
                "num": 1,
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(self.base_url, params=params) as response:
                    return response.status == 200

        except Exception:
            return False

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            "provider": "serpapi_google",
            "daily_limit": None,
            "monthly_limit": self.monthly_quota,
            "remaining": max(0, self.monthly_quota - self.requests_made),
            "reset_time": None,
            "unlimited": False,
        }


# Register all providers
WebSearchProviderRegistry.register_provider("duckduckgo", DuckDuckGoProvider)
WebSearchProviderRegistry.register_provider("brave_search", BraveSearchProvider)
WebSearchProviderRegistry.register_provider("serpapi", SerpAPIProvider)
