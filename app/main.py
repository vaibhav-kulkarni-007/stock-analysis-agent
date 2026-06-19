from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.llm import explain
from app.models import AnalyzeRequest, AnalyzeResponse
from app.scorer import score_stock, verdict_from_score
from app.stock_data import StockDataUnavailable, fetch_snapshot

load_dotenv()

app = FastAPI(
    title="Stock Analysis Agent",
    description="Simple stock advisor: fetches market data, scores it against your goal, and explains the verdict.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "service": "stock-analysis-agent",
        "usage": "POST /analyze with { ticker, horizon, risk_tolerance }",
        "docs": "/docs",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        snapshot = fetch_snapshot(req.ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except StockDataUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch stock data: {e}")

    score, signals = score_stock(snapshot, req.horizon, req.risk_tolerance)
    verdict = verdict_from_score(score)
    explanation = explain(snapshot, req.horizon, req.risk_tolerance, score, verdict, signals)

    return AnalyzeResponse(
        ticker=snapshot.ticker,
        verdict=verdict,
        score=score,
        snapshot=snapshot,
        signals=signals,
        explanation=explanation,
    )
