import os

import google.generativeai as genai

from app.models import Horizon, RiskTolerance, StockSnapshot

_MODEL = os.getenv("LLM_EXPLAIN_MODEL", "gemini-3.5-flash")

_SYSTEM_INSTRUCTION = (
    "You are a friendly stock analysis assistant. Be concise, honest, and "
    "avoid jargon. Never claim to give financial advice."
)


def explain(
    snap: StockSnapshot,
    horizon: Horizon,
    risk: RiskTolerance,
    score: int,
    verdict: str,
    signals: list[str],
) -> str:
    """Ask Gemini to turn the numeric verdict into a plain-English explanation."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return (
            "LLM explanation unavailable (GOOGLE_API_KEY not set). "
            f"Rule-based verdict: {verdict} ({score}/100). "
            f"Signals: {'; '.join(signals) if signals else 'none'}."
        )

    genai.configure(api_key=api_key)

    facts = snap.model_dump()
    signal_lines = "\n".join(f"- {s}" for s in signals) or "- (no notable signals)"

    user_msg = f"""A user is considering whether to invest in {snap.ticker} ({snap.name or "unknown"}).

Their goal: {horizon.value.replace("_", " ")}
Their risk tolerance: {risk.value}

Stock snapshot:
{facts}

Rule-based score: {score}/100
Rule-based verdict: {verdict}

Signals the rules picked up:
{signal_lines}

Write a short, friendly explanation (4-6 sentences) for the user that:
1. States the verdict clearly.
2. Explains the main reasons in plain English (no jargon dumps).
3. Mentions whether the sector/industry looks healthy if you can tell from the data.
4. Notes one key risk to watch.
End with a one-line summary like "Bottom line: ...".
Do not invent numbers that aren't in the snapshot. This is not financial advice."""

    model = genai.GenerativeModel(
        _MODEL,
        system_instruction=_SYSTEM_INSTRUCTION,
        generation_config={"temperature": 0.4, "max_output_tokens": 2048},
    )
    response = model.generate_content(user_msg)
    return (response.text or "").strip()
