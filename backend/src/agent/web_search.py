"""Generic web search utilities that work with any LLM provider."""

import re
import requests
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
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str) -> List[WebSearchResult]:
        """
        Perform a web search using DuckDuckGo and return results with content.
        
        Args:
            query: The search query string
            
        Returns:
            List of WebSearchResult objects
        """
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=self.max_results))
            
            results = []
            for result in search_results:
                # Extract basic information
                title = result.get('title', '')
                url = result.get('href', '')
                snippet = result.get('body', '')
                
                # Try to fetch and extract content from the page
                content = self._extract_content(url)
                
                web_result = WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content
                )
                results.append(web_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
    
    def _extract_content(self, url: str) -> str:
        """
        Extract text content from a web page.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            Extracted text content, truncated if necessary
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
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
            
            return text
            
        except Exception as e:
            logger.warning(f"Could not extract content from {url}: {e}")
            return ""

def perform_web_search_with_llm(
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
    search_results = searcher.search(search_query)
    
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
