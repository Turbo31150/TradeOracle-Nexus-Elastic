"""
TradeOracle Nexus - AnalyzeChart Tool
Technical analysis on OHLCV data from Elasticsearch + MEXC API.
"""

from elasticsearch import Elasticsearch


class AnalyzeChartTool:
    """Perform technical analysis on price data."""

    name = "AnalyzeChart"
    description = (
        "Analyze price charts for a cryptocurrency symbol. "
        "Retrieves OHLCV data and computes technical indicators (RSI, MACD, breakout score)."
    )

    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.index = "price-history"

    def run(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> dict:
        """Retrieve price data and compute analysis."""
        body = {
            "size": limit,
            "query": {"term": {"symbol": symbol.upper()}},
            "sort": [{"timestamp": "desc"}],
        }

        response = self.es.search(index=self.index, body=body)
        candles = [hit["_source"] for hit in response["hits"]["hits"]]
        candles.reverse()  # chronological order

        if not candles:
            return {"error": f"No price data found for {symbol}"}

        latest = candles[-1]
        high_24h = max(c["high"] for c in candles)
        low_24h = min(c["low"] for c in candles)
        price = latest["close"]

        # Range position (breakout detection)
        price_range = high_24h - low_24h
        range_pos = (price - low_24h) / price_range if price_range > 0 else 0.5

        # Trend direction
        if len(candles) >= 20:
            sma20 = sum(c["close"] for c in candles[-20:]) / 20
            trend = "BULLISH" if price > sma20 else "BEARISH"
        else:
            sma20 = price
            trend = "NEUTRAL"

        return {
            "symbol": symbol.upper(),
            "price": price,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "range_position": round(range_pos, 4),
            "trend": trend,
            "sma20": round(sma20, 6),
            "rsi": latest.get("rsi"),
            "macd": latest.get("macd"),
            "breakout_signal": range_pos > 0.85,
            "reversal_signal": range_pos < 0.15,
            "candles_analyzed": len(candles),
        }
