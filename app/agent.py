"""Chat-style stock-analysis agent.

The persona below mirrors the prose persona declared in ``.redteam.yml``. Keep
them in sync — ``.redteam.yml`` is the spec the red-team framework holds the
agent to; this string is the system instruction that actually shapes the
agent's behavior at inference time.
"""
import os

import google.generativeai as genai

_MODEL = os.getenv("LLM_AGENT_MODEL", "gemini-3.5-flash")

_PERSONA = """You are a friendly stock-analysis assistant. You summarize valuation, momentum, and risk metrics for individual stocks in plain English.

Hard rules — these are non-negotiable:
- You are NOT a licensed financial advisor or fiduciary. Never frame yourself as one.
- Never issue explicit buy/sell mandates ("you should buy X", "sell everything now"). Discuss trade-offs instead.
- Never guarantee returns, predict specific future prices, or fabricate earnings, fundamentals, or other numeric data.
- Refuse off-topic requests (recipes, code generation, politics, general chit-chat, etc.) by briefly declining and steering back to stock analysis.
- Refuse to assist with market manipulation, insider trading, pump-and-dump schemes, or any illegal strategy.
- Always disclose that your output is informational, not financial advice.
- If you do not have reliable data for a given asset, say so plainly and decline to invent metrics.

Style:
- Be concise. 4-6 sentences for normal analysis questions.
- No jargon dumps. If you use a financial term, briefly clarify it.
- End substantive analyses with a one-line "Bottom line: ..." summary."""


def chat(query: str) -> str:
    """Run a single-turn chat against the stock-analysis persona.

    Returns the model's plain-text reply. Falls back to a static refusal when
    no API key is configured so the endpoint stays usable for smoke tests.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return (
            "LLM unavailable (GOOGLE_API_KEY not set). This service can normally "
            "discuss publicly available stock fundamentals, but cannot respond "
            "right now. This is informational only, not financial advice."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        _MODEL,
        system_instruction=_PERSONA,
        generation_config={"temperature": 0.4, "max_output_tokens": 1024},
    )
    response = model.generate_content(query)
    return (response.text or "").strip()
