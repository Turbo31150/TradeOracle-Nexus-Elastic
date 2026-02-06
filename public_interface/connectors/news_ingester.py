"""
TradeOracle Nexus - Financial News Ingester
Fetches financial news and indexes them into Elasticsearch for RAG.
"""
import json
import urllib.request
from datetime import datetime
from typing import List, Dict, Optional
from connectors.elasticsearch_client import get_es_client


def fetch_crypto_news(limit: int = 20) -> List[Dict]:
    """Fetch crypto news from free API sources."""
    articles = []

    # CryptoCompare News API (free, no key needed)
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit={limit}"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        for item in data.get("Data", []):
            articles.append({
                "title": item.get("title", ""),
                "content": item.get("body", "")[:2000],
                "source": item.get("source", "unknown"),
                "symbol": _extract_symbol(item.get("title", "") + " " + item.get("categories", "")),
                "published_at": datetime.utcfromtimestamp(item.get("published_on", 0)).isoformat(),
                "url": item.get("url", ""),
                "sentiment": 0.0,
            })
    except Exception as e:
        print(f"[News] CryptoCompare error: {e}")

    return articles


def _extract_symbol(text: str) -> str:
    """Extract crypto symbol from text."""
    symbols = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC", "LINK",
               "UNI", "AAVE", "ARB", "OP", "SUI", "APT", "SEI", "TIA", "JUP", "WIF"]
    text_upper = text.upper()
    for sym in symbols:
        if sym in text_upper:
            return sym
    return "CRYPTO"


def ingest_news_to_elasticsearch(limit: int = 50) -> Dict:
    """Fetch news and index them into Elasticsearch."""
    es = get_es_client()
    if not es:
        return {"error": "Elasticsearch not configured"}

    articles = fetch_crypto_news(limit)
    indexed = 0
    for article in articles:
        try:
            es.index(index="financial-news", document=article)
            indexed += 1
        except Exception as e:
            print(f"[ES] Index error: {e}")

    return {"fetched": len(articles), "indexed": indexed}


if __name__ == "__main__":
    result = ingest_news_to_elasticsearch()
    print(f"Ingested: {result}")
