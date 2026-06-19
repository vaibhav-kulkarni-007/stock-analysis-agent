"""Stock fundamentals via Gemini.

For the hackathon demo we replace live market-data APIs (which are blocked from
the corporate network) with a single Gemini call. Gemini uses its training-data
knowledge of major public companies to produce a plausible StockSnapshot.

This is deliberately not a source of real-time market data — it's enough to
drive the scorer and the red-team demo. Numbers may be stale or estimated; for
red-teaming that's actually useful (the agent's confidence on guessed numbers
is itself a target).
"""
import json
import os
import re
from typing import Optional

import google.generativeai as genai

from app.models import StockSnapshot

_MODEL_NAME = os.getenv("STOCK_DATA_MODEL", "gemini-3.5-flash")
_REQUEST_TIMEOUT = 30
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_PROMPT_TEMPLATE = """You are a financial data assistant. Return the most recent
fundamentals and market-data estimates you know for the stock with ticker symbol "{symbol}".

If the ticker does not exist or you do not recognise it, respond with exactly:
{{"unknown": true}}

Otherwise respond with ONLY a JSON object matching this schema. Use null for any
field you cannot estimate. Pay close attention to the UNITS column — mixing
percentages and decimal fractions will break downstream scoring.

| Field                | Type   | Units / range example                                   |
|----------------------|--------|---------------------------------------------------------|
| name                 | string | "Apple Inc."                                            |
| sector               | string | GICS sector, e.g. "Information Technology"              |
| industry             | string | GICS industry, e.g. "Consumer Electronics"              |
| current_price        | float  | USD, e.g. 185.50                                        |
| market_cap           | int    | USD absolute, e.g. 2900000000000                        |
| pe_ratio             | float  | trailing PE ratio, e.g. 28.5                            |
| forward_pe           | float  | forward PE ratio, e.g. 26.0                             |
| profit_margin        | float  | DECIMAL FRACTION (0.26 = 26%)                           |
| revenue_growth       | float  | DECIMAL FRACTION YoY (0.07 = 7%)                        |
| debt_to_equity       | float  | Yahoo-style scaled ratio (145 means D/E of 1.45)        |
| fifty_day_avg        | float  | USD price                                               |
| two_hundred_day_avg  | float  | USD price                                               |
| fifty_two_week_high  | float  | USD price                                               |
| fifty_two_week_low   | float  | USD price                                               |
| one_year_return_pct  | float  | PERCENTAGE NUMBER (15.0 = 15% return, NOT 0.15)         |
| dividend_yield       | float  | DECIMAL FRACTION (0.005 = 0.5%)                         |

Use your training-data knowledge — do not refuse for "no real-time data". Estimates are fine.
Output ONLY the JSON object. No prose, no markdown code fences, no comments."""


class StockDataUnavailable(RuntimeError):
    """Raised when the upstream data source can't be reached or refuses us."""


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _configure() -> None:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise StockDataUnavailable(
            "GOOGLE_API_KEY is not set. Add it to .env or the runtime environment."
        )
    genai.configure(api_key=key)


def _strip_json_fences(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


def fetch_snapshot(ticker: str) -> StockSnapshot:
    symbol = ticker.upper()
    _configure()

    model = genai.GenerativeModel(
        _MODEL_NAME,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    )

    try:
        response = model.generate_content(
            _PROMPT_TEMPLATE.format(symbol=symbol),
            request_options={"timeout": _REQUEST_TIMEOUT},
        )
    except Exception as e:
        raise StockDataUnavailable(f"Gemini call failed: {e}") from e

    raw = _strip_json_fences(getattr(response, "text", "") or "")
    if not raw:
        raise StockDataUnavailable("Gemini returned an empty response")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise StockDataUnavailable(
            f"Gemini returned non-JSON: {raw[:200]}"
        ) from e

    if data.get("unknown") is True:
        raise ValueError(f"Gemini does not recognise ticker '{ticker}'")

    return StockSnapshot(
        ticker=symbol,
        name=data.get("name"),
        sector=data.get("sector"),
        industry=data.get("industry"),
        current_price=_safe_float(data.get("current_price")),
        market_cap=_safe_float(data.get("market_cap")),
        pe_ratio=_safe_float(data.get("pe_ratio")),
        forward_pe=_safe_float(data.get("forward_pe")),
        profit_margin=_safe_float(data.get("profit_margin")),
        revenue_growth=_safe_float(data.get("revenue_growth")),
        debt_to_equity=_safe_float(data.get("debt_to_equity")),
        fifty_day_avg=_safe_float(data.get("fifty_day_avg")),
        two_hundred_day_avg=_safe_float(data.get("two_hundred_day_avg")),
        fifty_two_week_high=_safe_float(data.get("fifty_two_week_high")),
        fifty_two_week_low=_safe_float(data.get("fifty_two_week_low")),
        one_year_return_pct=_safe_float(data.get("one_year_return_pct")),
        dividend_yield=_safe_float(data.get("dividend_yield")),
    )
