"""
TradeOracle Nexus - SearchNews Tool
RAG-powered financial news search using Elasticsearch.
"""

from typing import Optional
from elasticsearch import Elasticsearch


class SearchNewsTool:
    """Search financial news indexed in Elasticsearch."""

    name = "SearchNews"
    description = (
        "Search for financial news articles related to a cryptocurrency or market event. "
        "Uses Elasticsearch full-text search and vector similarity for RAG retrieval."
    )

    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.index = "financial-news"

    def run(self, query: str, symbol: Optional[str] = None, limit: int = 5) -> list[dict]:
        """Execute a hybrid search (text + vector) on financial news."""
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content"],
                    "type": "best_fields",
                }
            }
        ]

        if symbol:
            must_clauses.append({"term": {"symbol": symbol.upper()}})

        body = {
            "size": limit,
            "query": {"bool": {"must": must_clauses}},
            "sort": [{"_score": "desc"}, {"published_at": "desc"}],
            "_source": ["title", "content", "source", "symbol", "published_at", "sentiment"],
        }

        response = self.es.search(index=self.index, body=body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["relevance_score"] = hit["_score"]
            results.append(doc)

        return results
