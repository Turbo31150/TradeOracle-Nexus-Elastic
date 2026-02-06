"""
TradeOracle Nexus - Agent Orchestrator
Autonomous agent that reasons and acts using Elasticsearch-powered tools.
"""

from connectors.elasticsearch_client import get_es_client
from agent.tools.search_news import SearchNewsTool
from agent.tools.analyze_chart import AnalyzeChartTool
from agent.tools.send_alert import SendAlertTool


class TradeOracleAgent:
    """Autonomous trading agent with tool-use capabilities."""

    def __init__(self):
        self.es = get_es_client()
        self.tools = {
            "SearchNews": SearchNewsTool(self.es),
            "AnalyzeChart": AnalyzeChartTool(self.es),
            "SendAlert": SendAlertTool(),
        }

    def get_tools_description(self) -> str:
        """Return formatted tool descriptions for the LLM."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- **{name}**: {tool.description}")
        return "\n".join(descriptions)

    async def process_query(self, user_query: str) -> dict:
        """
        Process a user query through the agent loop:
        1. Understand the query
        2. Select appropriate tools
        3. Execute tools
        4. Reason about results
        5. Respond or act
        """
        # TODO: Integrate with LLM for reasoning + tool selection
        # This is the skeleton - will be connected to Elastic Agent Builder
        return {
            "query": user_query,
            "status": "agent_ready",
            "available_tools": list(self.tools.keys()),
            "message": "TradeOracle Nexus agent is initialized. Connect to Elastic Agent Builder to enable autonomous reasoning.",
        }
