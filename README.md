# TradeOracle Nexus

**Autonomous AI Trading Agent powered by Elasticsearch**

> Built for the Elasticsearch Agent Builder Hackathon

## Architecture

```
TradeOracle_Nexus/
├── public_interface/     # Next.js Frontend (Cyberpunk UI)
│   ├── src/
│   │   ├── app/          # Next.js App Router
│   │   ├── components/   # React Components
│   │   └── styles/       # Cyberpunk/Neon CSS
│   └── package.json
│
├── private_core/         # Python Agent Backend
│   ├── agent/            # Autonomous Agent Logic
│   │   ├── tools/        # Agent Tools (SearchNews, AnalyzeChart, SendAlert)
│   │   ├── reasoning/    # Chain-of-thought reasoning engine
│   │   └── orchestrator.py
│   ├── connectors/       # Elasticsearch + Data connectors
│   ├── api/              # FastAPI endpoints
│   └── requirements.txt
│
├── .env.example
├── .gitignore
└── README.md
```

## Agent Tools

| Tool | Description | Data Source |
|------|-------------|------------|
| **SearchNews** | RAG search on financial news indexed in Elasticsearch | Elasticsearch |
| **AnalyzeChart** | Technical analysis on OHLCV price data | MEXC API + ES |
| **SendAlert** | Push trading signals via Telegram | Telegram Bot API |
| **MarketSentiment** | Aggregate sentiment from multiple sources | Elasticsearch |

## Tech Stack

- **Frontend**: Next.js 14, TailwindCSS, Framer Motion
- **Backend**: Python 3.11+, FastAPI, LangChain
- **Search & RAG**: Elasticsearch Serverless
- **AI Models**: Gemini, Claude, Local LLMs (LM Studio)
- **Data**: MEXC Futures API (crypto market data)

## Quick Start

```bash
# Backend
cd private_core
pip install -r requirements.txt
uvicorn api.main:app --reload

# Frontend
cd public_interface
npm install
npm run dev
```

## License

MIT
