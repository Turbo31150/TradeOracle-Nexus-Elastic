"""
TradeOracle Nexus - Elasticsearch Connector
Handles connection to Elasticsearch Serverless for RAG operations.
"""

import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()


def get_es_client() -> Elasticsearch:
    """Create and return an Elasticsearch client."""
    url = os.getenv("ELASTICSEARCH_URL")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")

    if not url or not api_key:
        raise ValueError("ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY must be set in .env")

    return Elasticsearch(url, api_key=api_key)


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
                    "created_at": {"type": "date"},
                    "tools_used": {"type": "keyword"},
                }
            }
        },
    }

    for index_name, body in indices.items():
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=body)
            print(f"[ES] Created index: {index_name}")
        else:
            print(f"[ES] Index already exists: {index_name}")
