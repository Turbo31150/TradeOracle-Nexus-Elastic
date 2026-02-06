"""
TradeOracle Nexus - Elasticsearch Connector
RAG-powered search on financial data using Elasticsearch Serverless.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional
from elasticsearch import Elasticsearch
from config.settings import ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY


def get_es_client() -> Optional[Elasticsearch]:
    """Create and return an Elasticsearch client."""
    if not ELASTICSEARCH_URL or not ELASTICSEARCH_API_KEY:
        return None
    return Elasticsearch(ELASTICSEARCH_URL, api_key=ELASTICSEARCH_API_KEY)


def ensure_indices(client: Elasticsearch):
    """Create required indices if they don't exist."""
    indices = {
        "financial-news": {
            "mappings": {
                "properties": {
                    "title": {"type": "text", "analyzer": "standard"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "source": {"type": "keyword"},
                    "symbol": {"type": "keyword"},
                    "published_at": {"type": "date"},
                    "sentiment": {"type": "float"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        },
        "price-history": {
            "mappings": {
                "properties": {
                    "symbol": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "open": {"type": "float"},
                    "high": {"type": "float"},
                    "low": {"type": "float"},
                    "close": {"type": "float"},
                    "volume": {"type": "float"},
                    "rsi": {"type": "float"},
                    "macd": {"type": "float"},
                    "signal_line": {"type": "float"},
                    "atr": {"type": "float"},
                    "ema_status": {"type": "keyword"},
                }
            }
        },
        "trading-signals": {
            "mappings": {
                "properties": {
                    "symbol": {"type": "keyword"},
                    "direction": {"type": "keyword"},
                    "score": {"type": "float"},
                    "confidence": {"type": "float"},
                    "entry_price": {"type": "float"},
                    "tp1": {"type": "float"},
                    "tp2": {"type": "float"},
                    "tp3": {"type": "float"},
                    "sl": {"type": "float"},
                    "reasoning": {"type": "text"},
                    "ai_votes": {"type": "object"},
                    "regime": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "tools_used": {"type": "keyword"},
                }
            }
        },
        "pipeline-runs": {
            "mappings": {
                "properties": {
                    "run_id": {"type": "integer"},
                    "started_at": {"type": "date"},
                    "status": {"type": "keyword"},
                    "total_scanned": {"type": "integer"},
                    "signals_found": {"type": "integer"},
                    "regime": {"type": "keyword"},
                    "duration_ms": {"type": "integer"},
                }
            }
        },
    }

    for index_name, body in indices.items():
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=body)
            print(f"[ES] Created index: {index_name}")


def index_signal(client: Elasticsearch, signal: Dict):
    """Index a trading signal into Elasticsearch."""
    doc = {**signal, "created_at": datetime.utcnow().isoformat()}
    client.index(index="trading-signals", document=doc)


def index_price_data(client: Elasticsearch, symbol: str, ohlcv: List, indicators: Dict):
    """Index OHLCV + indicators into Elasticsearch."""
    for candle in ohlcv[-10:]:  # Index last 10 candles
        doc = {
            "symbol": symbol,
            "timestamp": datetime.utcfromtimestamp(candle[0] / 1000).isoformat() if candle[0] > 1e12 else datetime.utcfromtimestamp(candle[0]).isoformat(),
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5],
            **{k: v for k, v in indicators.items() if isinstance(v, (int, float))},
        }
        client.index(index="price-history", document=doc)


def search_news(client: Elasticsearch, query: str, symbol: Optional[str] = None, limit: int = 5) -> List[Dict]:
    """RAG search on financial news."""
    must = [{"multi_match": {"query": query, "fields": ["title^3", "content"]}}]
    if symbol:
        must.append({"term": {"symbol": symbol.upper()}})

    resp = client.search(
        index="financial-news",
        body={"size": limit, "query": {"bool": {"must": must}}, "sort": [{"_score": "desc"}, {"published_at": "desc"}]},
    )
    return [hit["_source"] | {"score": hit["_score"]} for hit in resp["hits"]["hits"]]


def search_signals(client: Elasticsearch, symbol: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Search recent trading signals from Elasticsearch."""
    query = {"term": {"symbol": symbol}} if symbol else {"match_all": {}}
    resp = client.search(
        index="trading-signals",
        body={"size": limit, "query": query, "sort": [{"created_at": "desc"}]},
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def get_price_history(client: Elasticsearch, symbol: str, limit: int = 100) -> List[Dict]:
    """Get price history from Elasticsearch."""
    resp = client.search(
        index="price-history",
        body={"size": limit, "query": {"term": {"symbol": symbol}}, "sort": [{"timestamp": "desc"}]},
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]
