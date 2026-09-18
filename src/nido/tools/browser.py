"""Web browser navigation and search tools with URL validation."""

from __future__ import annotations

import urllib.parse
import webbrowser
from typing import Any, Dict

from nido.logging import get_logger

logger = get_logger("nido.tools.browser")

SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q=",
    "duckduckgo": "https://duckduckgo.com/?q=",
    "bing": "https://www.bing.com/search?q=",
}


def open_url(url: str) -> Dict[str, Any]:
    """Open a web URL in the default browser.

    Args:
        url: The web URL to open. Must begin with http:// or https://.
    """
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = f"https://{clean_url}"

    parsed = urllib.parse.urlparse(clean_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return {
            "success": False,
            "error": f"Invalid or disallowed URL scheme: '{url}'. Only http/https supported.",
        }

    try:
        webbrowser.open(clean_url)
        return {
            "success": True,
            "message": f"Opened URL: {clean_url}",
            "url": clean_url,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to open browser URL: {e}",
        }


def search_web(query: str, engine: str = "google") -> Dict[str, Any]:
    """Perform a web search using the default browser.

    Args:
        query: The search term or question to look up.
        engine: Search engine to use ('google', 'duckduckgo', or 'bing').
    """
    engine_key = engine.lower()
    base_url = SEARCH_ENGINES.get(engine_key, SEARCH_ENGINES["google"])
    encoded_query = urllib.parse.quote_plus(query.strip())
    search_url = f"{base_url}{encoded_query}"

    return open_url(search_url)
