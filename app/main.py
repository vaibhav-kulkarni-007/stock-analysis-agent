import json
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from app.agent import chat

load_dotenv()

app = FastAPI(
    title="Stock Analysis Agent",
    description="Chat-style stock advisor. POST plain text or JSON to /analyze.",
    version="0.2.0",
)


@app.get("/")
def root():
    return {
        "service": "stock-analysis-agent",
        "usage": "POST /analyze with a natural-language query (plain text or JSON string).",
        "docs": "/docs",
    }


# Field names we prefer when an attacker / framework wraps the query in a JSON
# object. Order matters: the first match wins. ``ticker`` is last because the
# red-team framework's HTTPTargetProxy injects the attack message into the
# first declared string field, which in this repo happens to be ``ticker``.
_PREFERRED_QUERY_FIELDS = (
    "message",
    "query",
    "text",
    "prompt",
    "input",
    "user_message",
    "ticker",
)


def _extract_query(body_bytes: bytes, content_type: str) -> str:
    raw = body_bytes.decode("utf-8", errors="replace").strip()
    if not raw:
        return ""

    if content_type.startswith("text/"):
        return raw

    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw

    return _query_from_json(parsed) or raw


def _query_from_json(parsed: Any) -> Optional[str]:
    if isinstance(parsed, str):
        return parsed.strip() or None
    if not isinstance(parsed, dict):
        return None

    for name in _PREFERRED_QUERY_FIELDS:
        value = parsed.get(name)
        text = _coerce_to_text(value)
        if text:
            return text

    for value in parsed.values():
        text = _coerce_to_text(value)
        if text:
            return text
    return None


def _coerce_to_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        # CAMRP-style {"user_message": {"text": "..."}}
        inner = value.get("text")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return None


@app.post("/analyze")
async def analyze(request: Request) -> dict:
    body = await request.body()
    query = _extract_query(body, request.headers.get("content-type", ""))
    if not query:
        raise HTTPException(status_code=400, detail="Empty request body")
    explanation = chat(query)
    return {"explanation": explanation}
