# TradeOracle Nexus

**Autonomous AI Trading Agent powered by Elasticsearch Agent Builder**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-Agent%20Builder-yellow.svg)](https://www.elastic.co)
[![MCP](https://img.shields.io/badge/MCP-FastMCP-green.svg)](https://modelcontextprotocol.io)
[![Gemini](https://img.shields.io/badge/AI-Gemini%202.5-orange.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Elasticsearch Agent Builder Hackathon 2026** - Built with Elastic Agent Builder, Elasticsearch Serverless, and Multi-AI Consensus

---

## The Problem

Crypto traders must process hundreds of data points across 800+ futures contracts simultaneously - prices, volumes, technical indicators, news, and market sentiment. Manual analysis is slow, incomplete, and emotionally biased. Existing tools provide raw data but lack contextual reasoning and autonomous action.

## The Solution

TradeOracle Nexus is an **autonomous multi-step AI agent** that combines:
1. **Elasticsearch RAG** - Searches financial news and historical data for context
2. **Real-time Market Scanning** - Analyzes 800+ MEXC Futures contracts with composite scoring
3. **Multi-AI Consensus** - 4 AI models (Gemini + 3 local LLMs) vote on each signal
4. **Autonomous Action** - Sends alerts via Telegram when high-confidence signals are found

The agent doesn't just answer questions - it **investigates**, **reasons**, and **acts**.

---

## Architecture

```
TradeOracle Nexus/
├── public_interface/           # Public submission (this code)
│   ├── agent/                  # Gemini Agent with LangChain tool calling
│   │   └── gemini_agent.py     # Chain-of-thought reasoning + 10 tools
│   ├── config/                 # Settings (from .env)
│   ├── connectors/
│   │   ├── elasticsearch_client.py  # ES Serverless: 4 indices, RAG search
│   │   └── news_ingester.py         # Financial news -> ES pipeline
│   ├── database/               # SQLite audit trail (6 tables)
│   ├── mcp_server/             # MCP Server (stdio + SSE)
│   ├── pipeline/
│   │   ├── domino.py           # 6-stage Domino Pipeline
│   │   └── ai_consensus.py     # 4-model parallel consensus (hardened)
│   ├── tools/
│   │   ├── market_scanner.py   # Scan 800+ MEXC Futures contracts
│   │   ├── technical_analysis.py  # 14 indicators + patterns
│   │   ├── portfolio.py        # Position & margin tracking
│   │   └── alerts.py           # Telegram notifications
│   ├── app.py                  # Streamlit Dashboard
│   ├── entrypoint.py           # 3 modes: standalone, mcp, pipeline
│   └── requirements.txt
```

---

## Agent Tools (Elastic Agent Builder)

| Tool | Description | Data Source |
|------|-------------|------------|
| **scan_market** | Scan 800+ contracts with composite scoring (0-100) | MEXC Futures API |
| **get_coin_price** | Real-time price + 24h stats for any symbol | MEXC API |
| **get_top_movers** | Top gainers/losers by 24h change | MEXC API |
| **get_ohlcv_data** | OHLCV candlestick data (1m to 1d) | MEXC API |
| **analyze_coin_technical** | Deep analysis: RSI, MACD, BB, Fibonacci, patterns | MEXC + NumPy |
| **get_market_regime** | BTC-based regime detection (ADX/ATR) | MEXC 4H |
| **multi_timeframe_analysis** | 15m + 1h + 4h alignment check | MEXC API |
| **get_positions** | Open positions with margin health | MEXC Authenticated |
| **check_margin_health** | Margin risk assessment + suggestions | MEXC Authenticated |
| **send_telegram_alert** | Push alerts to Telegram | Telegram Bot API |

### Elasticsearch-Specific Tools (RAG)

| Tool | Description | ES Index |
|------|-------------|----------|
| **search_news** | RAG search on financial news | `financial-news` |
| **search_signals** | Query historical signals | `trading-signals` |
| **get_price_history** | Time-series price data | `price-history` |
| **ingest_news** | Fetch & index crypto news | `financial-news` |

---

## Elasticsearch Integration

### 4 Indices

| Index | Purpose | Key Fields |
|-------|---------|------------|
| `financial-news` | News articles with embeddings for RAG | title, content, symbol, sentiment, embedding (768d) |
| `price-history` | OHLCV + technical indicators | symbol, timestamp, OHLCV, RSI, MACD, ATR |
| `trading-signals` | AI-generated trading signals | symbol, direction, score, confidence, TP/SL, ai_votes |
| `pipeline-runs` | Pipeline execution audit trail | run_id, status, regime, duration, signals_found |

### RAG Workflow

```
User Query -> Elasticsearch Search (news + signals + prices) -> Context Injection -> LLM Reasoning -> Action
```

---

## Domino Pipeline (6 Stages)

```
[SCAN 800+] -> [FILTER] -> [REGIME] -> [ANALYZE] -> [AI CONSENSUS] -> [SIGNAL + ALERT]
     |            |            |           |              |                |
   309ms      Top N by      BTC 4H     14 tech       4 models         SQLite
              score        ADX/ATR    indicators    vote parallel     + Telegram
                                                                     + Elasticsearch
```

### AI Consensus (4 Models in Parallel)

| Model | Server | Role |
|-------|--------|------|
| Gemini 2.5 Flash | Google Cloud | Cloud reasoning with retry + backoff |
| GPT-OSS-20B | LM Studio M1 (6 GPU) | Deep local analysis |
| GPT-OSS-20B | LM Studio M2 (3 GPU) | Fast local validation |
| GPT-OSS-20B | LM Studio M3 (3 GPU) | Consensus tiebreaker |

Features: Retry with exponential backoff, JSON + keyword fallback parsing, graceful degradation, unanimity bonus (+10 confidence).

---

## Quick Start

### Prerequisites

- Python 3.10+
- Elasticsearch Serverless account (or local)
- Google Gemini API Key

### Installation

```bash
git clone https://github.com/Turbo31150/TradeOracle-Nexus-Elastic.git
cd TradeOracle-Nexus-Elastic/public_interface
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

### Run Modes

```bash
# Streamlit Dashboard (interactive)
python entrypoint.py --mode standalone

# MCP Server (for Claude Desktop / Claude Code)
python entrypoint.py --mode mcp

# Pipeline CLI (single scan)
python entrypoint.py --mode pipeline --min-score 70 --top-n 5

# Ingest news into Elasticsearch
python -m connectors.news_ingester
```

### MCP Configuration

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tradeoracle-nexus": {
      "command": "python",
      "args": ["entrypoint.py", "--mode", "mcp"],
      "cwd": "/path/to/public_interface"
    }
  }
}
```

---

## Features We Loved Building

1. **Elasticsearch RAG for Financial News** - The ability to index thousands of news articles and search them with hybrid queries (text + vector) gives the agent real-world context that pure LLMs lack.

2. **Multi-AI Consensus with Graceful Degradation** - Having 4 AI models vote in parallel with automatic fallback when one fails was technically challenging but incredibly satisfying. The unanimity bonus makes the system genuinely more confident.

3. **The Domino Pipeline** - Turning a complex 6-stage analysis pipeline into a single command that scans 800+ contracts, detects market regime, runs technical analysis, gets AI consensus, and sends alerts - all in under 60 seconds.

### Challenges

- Handling Gemini API rate limits (429 errors) during stress testing required implementing exponential backoff and graceful degradation so the system continues with local models.
- Parsing AI responses reliably across 4 different models required a robust JSON extraction + keyword fallback system.

---

## Production Stress Test Results

```
Cycles: 5/5 (100% success)
Avg Duration: 46s
Fatal Crashes: 0
AI Model Health: M1 93% | M2 100% | M3 100% | Gemini 27% (quota managed)
Database: 23 runs, 42 analyses, 132 AI votes
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core AI | Google Gemini 2.5 Flash |
| Local AI | LM Studio (GPT-OSS-20B x3) |
| Agent Framework | LangChain + Tool Calling |
| Search & RAG | Elasticsearch Serverless |
| MCP Server | FastMCP |
| Market Data | MEXC Futures API (800+ contracts) |
| Technical Analysis | NumPy (14 indicators) |
| Frontend | Streamlit |
| Database | SQLite (WAL mode) |
| Alerts | Telegram Bot API |

---

## License

MIT License - Built for the Elasticsearch Agent Builder Hackathon 2026
