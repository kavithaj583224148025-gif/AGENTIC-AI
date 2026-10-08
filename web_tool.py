"""
Web Search Tool Implementation (No API Key Required)
----------------------------------------------------
Searches the web and extracts relevant search result snippets using
standard HTTP requests and BeautifulSoup.
"""

from typing import Any, Dict, List
import urllib.parse
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def search_duckduckgo(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Fetches search results from DuckDuckGo HTML endpoint."""
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}
    
    response = requests.post(url, data=data, headers=HEADERS, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    # DuckDuckGo HTML classes: .result__body contains .result__title and .result__snippet
    result_elements = soup.select(".result")
    for elem in result_elements:
        title_elem = elem.select_one(".result__title .result__a")
        snippet_elem = elem.select_one(".result__snippet")

        if not title_elem or not snippet_elem:
            continue

        title = title_elem.get_text(strip=True)
        raw_href = title_elem.get("href", "")
        snippet = snippet_elem.get_text(strip=True)

        # Decode actual target URL from duckduckgo redirect link
        if "uddg=" in raw_href:
            parsed = urllib.parse.urlparse(raw_href)
            query_params = urllib.parse.parse_qs(parsed.query)
            clean_url = query_params.get("uddg", [raw_href])[0]
        else:
            clean_url = raw_href

        results.append({
            "title": title,
            "snippet": snippet,
            "url": clean_url
        })

        if len(results) >= max_results:
            break

    return results


def search_wikipedia_fallback(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Fallback search using Wikipedia API when search engines throttle."""
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": "1",
        "srlimit": max_results
    }
    response = requests.get(api_url, params=params, headers=HEADERS, timeout=8)
    response.raise_for_status()
    data = response.json()
    
    items = data.get("query", {}).get("search", [])
    results = []
    for item in items:
        # Strip HTML markup from snippet
        raw_snippet = item.get("snippet", "")
        clean_snippet = BeautifulSoup(raw_snippet, "html.parser").get_text(strip=True)
        title = item.get("title", "")
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        results.append({
            "title": title,
            "snippet": clean_snippet,
            "url": f"https://en.wikipedia.org/wiki/{encoded_title}"
        })
    return results


def execute_web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Main entry point for the web_search tool called by the LLM.
    
    Args:
        query: Search string or keywords from LLM
        max_results: Maximum number of snippets to retrieve (clamped between 1 and 5)
        
    Returns:
        dict: JSON-serializable dictionary with status, query, and search results.
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "status": "error",
            "error": "Search query cannot be empty."
        }

    limit = max(1, min(int(max_results), 5))

    # Try DuckDuckGo first
    try:
        results = search_duckduckgo(clean_query, max_results=limit)
        if results:
            return {
                "status": "success",
                "source": "duckduckgo",
                "query": clean_query,
                "count": len(results),
                "results": results
            }
    except Exception:
        # Silently proceed to fallback if DDG is blocked / times out
        pass

    # Try Wikipedia fallback
    try:
        results = search_wikipedia_fallback(clean_query, max_results=limit)
        if results:
            return {
                "status": "success",
                "source": "wikipedia",
                "query": clean_query,
                "count": len(results),
                "results": results
            }
    except Exception as err:
        return {
            "status": "error",
            "query": clean_query,
            "error": f"Failed to retrieve web search results: {err}"
        }

    return {
        "status": "success",
        "source": "none",
        "query": clean_query,
        "count": 0,
        "results": [],
        "message": f"No web search results found for '{clean_query}'."
    }


if __name__ == "__main__":
    # Test execution
    res = execute_web_search("Python programming language", max_results=2)
    print("Test Search Result:")
    import json
    print(json.dumps(res, indent=2))
