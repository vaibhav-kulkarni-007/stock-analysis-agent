# stock-analysis-agent

A simple FastAPI service that analyzes a stock and tells you whether to invest, based on your goal (long term / short term) and risk tolerance.

How it works:
1. Asks Gemini for a fundamentals snapshot of the stock (sector, P/E, margins, moving averages, 52-week range, dividend yield, etc.). For the hackathon we use Gemini's training-data knowledge in place of a live market-data API — it's a stand-in, not a real data source.
2. Applies plain-English scoring rules — fundamentals matter more for long-term goals, momentum matters more for short-term.
3. Asks Claude to turn the score + signals into a friendly explanation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env # then fill in GOOGLE_API_KEY
```

`GOOGLE_API_KEY` is required — Gemini powers both the fundamentals snapshot and the plain-English explanation. Without it the snapshot call returns 503 and the explanation falls back to a rule-based summary.

Gemini model defaults to `gemini-3.5-flash`. Override via `STOCK_DATA_MODEL` (snapshot) or `LLM_EXPLAIN_MODEL` (explanation) if you want a different model.

## Run

```bash
uvicorn app.main:app --reload --port 8009
```

Then open http://127.0.0.1:8009/docs for the interactive Swagger UI, or:

```bash
curl -X POST http://127.0.0.1:8009/analyze \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AAPL","horizon":"long_term","risk_tolerance":"medium"}'
```

### Request

| field            | type    | values                                  |
| ---------------- | ------- | --------------------------------------- |
| `ticker`         | string  | e.g. `AAPL`, `MSFT`, `NSE: TCS`         |
| `horizon`        | string  | `long_term` \| `short_term`             |
| `risk_tolerance` | string  | `low` \| `medium` \| `high`             |

### Response

```json
{
  "ticker": "AAPL",
  "verdict": "HOLD / CAUTIOUS BUY",
  "score": 62,
  "snapshot": { "...": "key metrics estimated by Gemini" },
  "signals": ["Strong revenue growth (...)", "..."],
  "explanation": "Apple looks solid for a long-term hold..."
}
```

Verdict thresholds: `INVEST` ≥ 70, `HOLD / CAUTIOUS BUY` 50–69, `WAIT` 35–49, `AVOID` < 35.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

The tests monkey-patch `fetch_snapshot` so they don't make real HTTP calls.

## Notes

- **Data is approximate.** Gemini supplies the snapshot from its training data, not from a live market feed. Numbers may be stale or estimated. For the red-team demo this is acceptable — the goal is to exercise the agent's persona and guardrails, not to make trades.
- **Live market data, in production.** Swap `app/stock_data.py` for a real provider (Alpha Vantage / Finnhub / IEX Cloud) when going beyond the demo. The exception contract (`StockDataUnavailable` / `ValueError`) is the only interface to preserve.
- **Not financial advice.** This is a toy. The scoring rules are simple heuristics, not a real model.
