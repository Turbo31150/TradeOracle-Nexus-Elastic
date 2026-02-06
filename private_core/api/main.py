"""
TradeOracle Nexus - FastAPI Application
API endpoints for the autonomous trading agent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="TradeOracle Nexus",
    description="Autonomous AI Trading Agent powered by Elasticsearch",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    symbol: str | None = None


class AgentResponse(BaseModel):
    query: str
    status: str
    tools_used: list[str] = []
    reasoning: str = ""
    result: dict = {}


@app.get("/")
def root():
    return {
        "name": "TradeOracle Nexus",
        "version": "1.0.0",
        "status": "online",
        "agent": "ready",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "elasticsearch": "pending_connection"}


@app.post("/agent/query", response_model=AgentResponse)
async def agent_query(request: QueryRequest):
    """Send a query to the autonomous agent."""
    # TODO: Connect to orchestrator
    return AgentResponse(
        query=request.query,
        status="received",
        reasoning="Agent skeleton ready - awaiting Elastic Agent Builder integration",
        result={"message": "TradeOracle Nexus is initializing..."},
    )


@app.get("/agent/tools")
def list_tools():
    """List available agent tools."""
    return {
        "tools": [
            {
                "name": "SearchNews",
                "description": "RAG search on financial news via Elasticsearch",
            },
            {
                "name": "AnalyzeChart",
                "description": "Technical analysis on OHLCV price data",
            },
            {
                "name": "SendAlert",
                "description": "Push trading signals via Telegram",
            },
            {
                "name": "MarketSentiment",
                "description": "Aggregate sentiment analysis from multiple sources",
            },
        ]
    }
