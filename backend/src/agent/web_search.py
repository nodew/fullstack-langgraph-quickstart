"""Generic web search utilities that work with any LLM provider."""

import re
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import logging

logger = logging.getLogger(__name__)

class WebSearchResult:
    """Represents a web search result with content and metadata."""
    
    def __init__(self, title: str, url: str, snippet: str, content: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content
        self.short_url = self._create_short_url()
    
    def _create_short_url(self) -> str:
        """Create a shortened URL for citation purposes."""
        domain = urlparse(self.url).netloc
        return f"[{domain}]"

class GenericWebSearcher:
    """Generic web searcher that can work with any LLM provider."""
    
    def __init__(self, max_results: int = 5, max_content_length: int = 2000):
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def search(self, query: str) -> List[WebSearchResult]:
        """
        Perform a web search using DuckDuckGo and return results with content.
        
        Args:
            query: The search query string
            
        Returns:
            List of WebSearchResult objects
        """
        try:
            # DuckDuckGo search is synchronous, so we'll run it in a thread
            search_results = await asyncio.to_thread(
                lambda: list(DDGS().text(query, max_results=self.max_results))
            )
            
            results = []
            # Process content extraction concurrently
            tasks = []
            for result in search_results:
                title = result.get('title', '')
                url = result.get('href', '')
                snippet = result.get('body', '')
                
                # Create a task for content extraction
                task = self._extract_content_async(url, title, snippet)
                tasks.append(task)
            
            # Wait for all content extraction tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return valid results
            valid_results = [r for r in results if isinstance(r, WebSearchResult)]
            return valid_results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
    
    async def _extract_content_async(self, url: str, title: str, snippet: str) -> WebSearchResult:
        """
        Extract text content from a web page asynchronously.
        
        Args:
            url: The URL to extract content from
            title: The title of the search result
            snippet: The snippet of the search result
            
        Returns:
            WebSearchResult with extracted content
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    content_bytes = await response.read()
                    
                    soup = BeautifulSoup(content_bytes, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Truncate if too long
                    if len(text) > self.max_content_length:
                        text = text[:self.max_content_length] + "..."
                    
                    return WebSearchResult(title=title, url=url, snippet=snippet, content=text)
                    
        except Exception as e:
            logger.warning(f"Could not extract content from {url}: {e}")
            return WebSearchResult(title=title, url=url, snippet=snippet, content="")

async def perform_web_search_with_llm(
    search_query: str,
    llm,
    search_prompt_template: str,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Perform web search and process results with any LLM.
    
    Args:
        search_query: The search query
        llm: The LLM instance to use for processing
        search_prompt_template: Template for the search prompt
        max_results: Maximum number of search results to process
        
    Returns:
        Dictionary containing processed results and sources
    """
    # Perform web search
    searcher = GenericWebSearcher(max_results=max_results)
    search_results = await searcher.search(search_query)
    
    if not search_results:
        return {
            "web_research_result": "No search results found.",
            "sources_gathered": [],
            "search_query": search_query
        }
    
    # Prepare context from search results
    search_context = []
    sources_gathered = []
    
    for i, result in enumerate(search_results):
        # Create source entry
        source = {
            "title": result.title,
            "url": result.url,
            "short_url": result.short_url,
            "snippet": result.snippet
        }
        sources_gathered.append(source)
        
        # Add to context for LLM processing
        context_entry = f"""
Source {i+1}: {result.title}
URL: {result.url}
Snippet: {result.snippet}
Content: {result.content[:1000]}...
"""
        search_context.append(context_entry)
    
    # Combine all context
    combined_context = "\n---\n".join(search_context)
    
    # Format the prompt with search results
    formatted_prompt = search_prompt_template.format(
        search_query=search_query,
        search_results=combined_context
    )
    
    # Process with LLM
    try:
        response = llm.invoke(formatted_prompt)
        processed_content = response.content if hasattr(response, 'content') else str(response)
        
        # Insert citations into the processed content
        processed_content = _insert_simple_citations(processed_content, sources_gathered)
        
        return {
            "web_research_result": processed_content,
            "sources_gathered": sources_gathered,
            "search_query": search_query
        }
        
    except Exception as e:
        logger.error(f"Error processing search results with LLM: {e}")
        # Return basic summary if LLM processing fails
        basic_summary = _create_basic_summary(search_results)
        return {
            "web_research_result": basic_summary,
            "sources_gathered": sources_gathered,
            "search_query": search_query
        }

def _insert_simple_citations(content: str, sources: List[Dict[str, Any]]) -> str:
    """
    Insert simple citations into content based on source matching.
    
    Args:
        content: The content to add citations to
        sources: List of source dictionaries
        
    Returns:
        Content with citations added
    """
    # Simple heuristic: add citations at the end of sentences that mention key terms
    for i, source in enumerate(sources):
        citation = f" [{source['short_url']}]({source['url']})"
        
        # Look for mentions of the source title or domain in the content
        title_words = source['title'].lower().split()[:3]  # First 3 words of title
        domain = urlparse(source['url']).netloc.replace('www.', '')
        
        # Add citation if we find relevant matches
        for word in title_words:
            if len(word) > 3 and word in content.lower():
                # Find the sentence containing this word and add citation at the end
                sentences = content.split('.')
                for j, sentence in enumerate(sentences):
                    if word in sentence.lower() and citation not in sentence:
                        sentences[j] = sentence + citation
                        break
                content = '.'.join(sentences)
                break
    
    return content

def _create_basic_summary(search_results: List[WebSearchResult]) -> str:
    """
    Create a basic summary from search results if LLM processing fails.
    
    Args:
        search_results: List of WebSearchResult objects
        
    Returns:
        Basic text summary
    """
    summary_parts = []
    for i, result in enumerate(search_results):
        summary_parts.append(f"{i+1}. {result.title}\n{result.snippet}\nSource: {result.url}\n")
    
    return "Search Results Summary:\n\n" + "\n".join(summary_parts)
