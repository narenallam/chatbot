"""
Web scraping service for extracting content from URLs
Compatible with BeautifulSoup without lxml dependency
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class WebScraperService:
    """Web scraper service using BeautifulSoup with html.parser"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def scrape_url(self, url: str, timeout: int = 30) -> Dict[str, str]:
        """
        Scrape content from a URL

        Args:
            url: URL to scrape
            timeout: Request timeout in seconds

        Returns:
            Dictionary with title, content, and metadata
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")

            # Make request
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            # Parse with BeautifulSoup using html.parser (built-in)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract content
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            metadata = self._extract_metadata(soup, url)

            return {
                "title": title,
                "content": content,
                "url": url,
                "metadata": metadata,
            }

        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            raise

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()

        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try h1
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text().strip()

        return "Unknown Title"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find main content areas
        content_selectors = [
            "main",
            "article",
            '[role="main"]',
            ".content",
            ".main-content",
            "#content",
            "#main",
            ".post-content",
            ".entry-content",
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return self._clean_text(content_elem.get_text())

        # Fallback to body content
        body = soup.find("body")
        if body:
            return self._clean_text(body.get_text())

        # Last resort - all text
        return self._clean_text(soup.get_text())

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract metadata from the page"""
        metadata = {"url": url, "domain": urlparse(url).netloc}

        # Description
        description_meta = soup.find("meta", attrs={"name": "description"})
        if description_meta and description_meta.get("content"):
            metadata["description"] = description_meta["content"].strip()

        # Keywords
        keywords_meta = soup.find("meta", attrs={"name": "keywords"})
        if keywords_meta and keywords_meta.get("content"):
            metadata["keywords"] = keywords_meta["content"].strip()

        # Author
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            metadata["author"] = author_meta["content"].strip()

        # Open Graph data
        og_tags = soup.find_all("meta", property=lambda x: x and x.startswith("og:"))
        for tag in og_tags:
            property_name = tag.get("property", "").replace("og:", "")
            content = tag.get("content", "")
            if property_name and content:
                metadata[f"og_{property_name}"] = content

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Replace multiple whitespace with single space
        import re

        text = re.sub(r"\s+", " ", text)

        # Remove extra newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def batch_scrape(self, urls: List[str], delay: float = 1.0) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs with delay between requests

        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds

        Returns:
            List of scraped content dictionaries
        """
        results = []

        for i, url in enumerate(urls):
            try:
                logger.info(f"Scraping {i+1}/{len(urls)}: {url}")
                result = self.scrape_url(url)
                results.append(result)

                # Add delay between requests
                if i < len(urls) - 1:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append(
                    {
                        "title": "Error",
                        "content": f"Failed to scrape: {str(e)}",
                        "url": url,
                        "metadata": {"error": str(e)},
                    }
                )

        return results


# Global web scraper instance
web_scraper = WebScraperService()
