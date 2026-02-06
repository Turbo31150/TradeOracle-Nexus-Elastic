"""
TradeOracle Nexus - SendAlert Tool
Push trading signals to Telegram.
"""

import os
import httpx


class SendAlertTool:
    """Send trading alerts via Telegram."""

    name = "SendAlert"
    description = (
        "Send a trading signal or alert message to the user via Telegram. "
        "Used when the agent identifies a high-confidence trading opportunity."
    )

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    async def run(self, message: str) -> dict:
        """Send an alert message to Telegram."""
        if not self.token or not self.chat_id:
            return {"error": "Telegram credentials not configured"}

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)

        if response.status_code == 200:
            return {"status": "sent", "message_preview": message[:100]}
        else:
            return {"error": f"Telegram API error: {response.status_code}"}
