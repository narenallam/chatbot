"""
Content Processor Implementations
Extract, clean, and process content from web search results
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse

from app.core.web_search_interfaces import ContentProcessor, WebSearchProviderRegistry

logger = logging.getLogger(__name__)

class AdvancedContentProcessor(ContentProcessor):
    """Advanced content processor with multiple extraction methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 15)
        self.max_content_length = config.get('max_content_length', 10000)
        self.user_agent = config.get('user_agent', 'Mozilla/5.0 (compatible; AI-Mate-Bot/1.0)')
        self.enable_trafilatura = config.get('enable_trafilatura', True)
        self.enable_newspaper = config.get('enable_newspaper', True)
        self.enable_readability = config.get('enable_readability', True)
    
    async def extract_content(self, url: str) -> Optional[str]:
        """Extract clean content from URL using multiple methods"""
        try:
            if not url or not url.startswith(('http://', 'https://')):
                return None
            
            # Fetch page content
            html_content = await self._fetch_html(url)
            if not html_content:
                return None
            
            # Try multiple extraction methods in order of preference
            extraction_methods = [
                ('trafilatura', self._extract_with_trafilatura),
                ('newspaper', self._extract_with_newspaper),
                ('readability', self._extract_with_readability),
                ('basic', self._extract_basic)
            ]
            
            for method_name, method_func in extraction_methods:
                try:
                    if method_name == 'trafilatura' and not self.enable_trafilatura:
                        continue
                    if method_name == 'newspaper' and not self.enable_newspaper:
                        continue
                    if method_name == 'readability' and not self.enable_readability:
                        continue
                    
                    content = await method_func(html_content, url)
                    if content and len(content.strip()) > 100:  # Minimum content threshold
                        logger.info(f"Successfully extracted content using {method_name} from {url}")
                        return content[:self.max_content_length]
                        
                except Exception as e:
                    logger.warning(f"Content extraction method {method_name} failed for {url}: {e}")
                    continue
            
            logger.warning(f"All content extraction methods failed for {url}")
            return None
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None
    
    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL"""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' in content_type:
                            return await response.text()
                    
                    logger.warning(f"Failed to fetch HTML from {url}, status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed to fetch HTML from {url}: {e}")
            return None
    
    async def _extract_with_trafilatura(self, html: str, url: str) -> Optional[str]:
        """Extract content using trafilatura library"""
        try:
            import trafilatura
            
            # Configure trafilatura
            config = trafilatura.settings.DEFAULT_CONFIG.copy()
            config.set('DEFAULT', 'MIN_OUTPUT_SIZE', '100')
            config.set('DEFAULT', 'MIN_EXTRACTED_SIZE', '200')
            
            # Extract main content
            extracted = trafilatura.extract(
                html,
                config=config,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=True
            )
            
            if extracted:
                # Clean and format the content
                cleaned = self._clean_extracted_content(extracted)
                return cleaned
            
            return None
            
        except ImportError:
            logger.warning("trafilatura not available, install with: pip install trafilatura")
            return None
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            return None
    
    async def _extract_with_newspaper(self, html: str, url: str) -> Optional[str]:
        """Extract content using newspaper3k library"""
        try:
            from newspaper import Article
            
            # Create article object
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            # Get article text
            if article.text:
                # Combine title and text
                content_parts = []
                
                if article.title:
                    content_parts.append(f"# {article.title}")
                
                if article.text:
                    content_parts.append(article.text)
                
                # Add publication date if available
                if article.publish_date:
                    content_parts.insert(-1, f"Published: {article.publish_date.strftime('%Y-%m-%d')}")
                
                # Add authors if available
                if article.authors:
                    content_parts.insert(-1, f"Authors: {', '.join(article.authors)}")
                
                combined_content = '\n\n'.join(content_parts)
                return self._clean_extracted_content(combined_content)
            
            return None
            
        except ImportError:
            logger.warning("newspaper3k not available, install with: pip install newspaper3k")
            return None
        except Exception as e:
            logger.warning(f"Newspaper3k extraction failed: {e}")
            return None
    
    async def _extract_with_readability(self, html: str, url: str) -> Optional[str]:
        """Extract content using readability library"""
        try:
            from readability import Document
            
            # Create readability document
            doc = Document(html)
            
            # Extract content
            content = doc.summary()
            if content:
                # Convert HTML to text
                text_content = await self._html_to_text(content)
                if text_content:
                    return self._clean_extracted_content(text_content)
            
            return None
            
        except ImportError:
            logger.warning("readability not available, install with: pip install readability")
            return None
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
            return None
    
    async def _extract_basic(self, html: str, url: str) -> Optional[str]:
        """Basic content extraction using BeautifulSoup"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Find main content areas
            main_content = None
            content_selectors = [
                'main', 'article', '.content', '.post', '.entry',
                '#content', '#main', '#post', '.article-body'
            ]
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element
                    break
            
            # Fallback to body if no main content found
            if not main_content:
                main_content = soup.find('body')
            
            if main_content:
                # Extract text
                text = main_content.get_text(separator='\n', strip=True)
                return self._clean_extracted_content(text)
            
            return None
            
        except Exception as e:
            logger.warning(f"Basic extraction failed: {e}")
            return None
    
    async def _html_to_text(self, html: str) -> Optional[str]:
        """Convert HTML to clean text"""
        try:
            import html2text
            
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.body_width = 0  # Don't wrap lines
            
            text = h.handle(html)
            return text.strip()
            
        except ImportError:
            # Fallback to BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
            except:
                return None
        except Exception as e:
            logger.warning(f"HTML to text conversion failed: {e}")
            return None
    
    def _clean_extracted_content(self, content: str) -> str:
        """Clean and format extracted content"""
        try:
            # Remove excessive whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            # Remove common junk patterns
            junk_patterns = [
                r'Cookie Policy.*?(?=\n|$)',
                r'Privacy Policy.*?(?=\n|$)',
                r'Terms of Service.*?(?=\n|$)',
                r'Subscribe to.*?(?=\n|$)',
                r'Follow us on.*?(?=\n|$)',
                r'Share this.*?(?=\n|$)',
                r'Advertisement.*?(?=\n|$)',
                r'Loading\.\.\.',
                r'Please enable JavaScript',
                r'This site requires JavaScript'
            ]
            
            for pattern in junk_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Clean up again
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = content.strip()
            
            return content
            
        except Exception as e:
            logger.warning(f"Content cleaning failed: {e}")
            return content
    
    async def summarize_content(self, content: str, max_length: int = 500) -> str:
        """Summarize content to specified length"""
        try:
            if len(content) <= max_length:
                return content
            
            # Simple extractive summarization
            sentences = self._split_into_sentences(content)
            if not sentences:
                return content[:max_length]
            
            # Score sentences by position and content
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                
                # Position score (earlier sentences are more important)
                position_score = 1.0 - (i / len(sentences)) * 0.5
                score += position_score
                
                # Length score (prefer medium-length sentences)
                length = len(sentence.split())
                if 10 <= length <= 30:
                    score += 0.3
                elif 5 <= length < 10 or 30 < length <= 50:
                    score += 0.1
                
                # Content score (look for important words)
                important_words = [
                    'important', 'significant', 'key', 'main', 'primary',
                    'result', 'conclusion', 'finding', 'discovered', 'revealed'
                ]
                
                sentence_lower = sentence.lower()
                for word in important_words:
                    if word in sentence_lower:
                        score += 0.2
                
                scored_sentences.append((sentence, score))
            
            # Sort by score and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Build summary
            summary_parts = []
            current_length = 0
            
            for sentence, score in scored_sentences:
                if current_length + len(sentence) <= max_length:
                    summary_parts.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            # Sort selected sentences by original order
            if summary_parts:
                # Find original positions
                original_positions = []
                for summary_sentence in summary_parts:
                    for i, (original_sentence, _) in enumerate(scored_sentences):
                        if original_sentence == summary_sentence:
                            original_positions.append((i, summary_sentence))
                            break
                
                # Sort by original position
                original_positions.sort(key=lambda x: x[0])
                summary = ' '.join([sentence for _, sentence in original_positions])
                
                return summary[:max_length]
            
            # Fallback to truncation
            return content[:max_length]
            
        except Exception as e:
            logger.warning(f"Content summarization failed: {e}")
            return content[:max_length]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception:
            return [text]
    
    async def extract_key_facts(self, content: str) -> List[str]:
        """Extract key facts from content"""
        try:
            facts = []
            
            # Look for fact patterns
            fact_patterns = [
                r'(\d+(?:\.\d+)?%)',  # Percentages
                r'(\$[\d,]+(?:\.\d{2})?)',  # Money amounts
                r'(\d{1,2}/\d{1,2}/\d{4})',  # Dates
                r'(\d{4}-\d{2}-\d{2})',  # ISO dates
                r'(\d+(?:,\d{3})*\s+(?:people|users|customers|employees))',  # Counts
                r'(\w+\s+(?:announced|revealed|discovered|found|reported))',  # Announcements
            ]
            
            for pattern in fact_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:5]:  # Limit facts per pattern
                    if match.strip() and match not in facts:
                        facts.append(match.strip())
            
            # Look for bullet points or numbered lists
            bullet_patterns = [
                r'•\s*(.+?)(?=\n|$)',
                r'\*\s*(.+?)(?=\n|$)',
                r'\d+\.\s*(.+?)(?=\n|$)'
            ]
            
            for pattern in bullet_patterns:
                matches = re.findall(pattern, content)
                for match in matches[:3]:  # Limit bullet facts
                    if len(match.strip()) > 10 and match.strip() not in facts:
                        facts.append(match.strip())
            
            return facts[:10]  # Return top 10 facts
            
        except Exception as e:
            logger.warning(f"Key fact extraction failed: {e}")
            return []

# Register the content processor
WebSearchProviderRegistry.register_processor("advanced", AdvancedContentProcessor)