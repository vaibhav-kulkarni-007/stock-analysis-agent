"""Offline smoke test: bypass Alpha Vantage, exercise scorer + endpoint wiring."""
from fastapi.testclient import TestClient

from app import main, stock_data
from app.models import Horizon, RiskTolerance, StockSnapshot
from app.scorer import score_stock, verdict_from_score


def fake_snapshot(ticker: str) -> StockSnapshot:
    return StockSnapshot(
        ticker=ticker.upper(),
        name="Acme Corp",
        sector="Technology",
        industry="Software",
        current_price=180.0,
        market_cap=2_500_000_000_000,
        pe_ratio=28.0,
        forward_pe=24.0,
        profit_margin=0.22,
        revenue_growth=0.18,
        debt_to_equity=45.0,
        fifty_day_avg=170.0,
        two_hundred_day_avg=160.0,
        fifty_two_week_high=190.0,
        fifty_two_week_low=140.0,
        one_year_return_pct=22.5,
        dividend_yield=0.005,
    )


def test_scorer_long_term_medium_risk():
    snap = fake_snapshot("ACME")
    score, signals = score_stock(snap, Horizon.long_term, RiskTolerance.medium)
    assert 0 <= score <= 100
    assert signals, "expected at least one signal"
    assert verdict_from_score(score) in {"INVEST", "HOLD / CAUTIOUS BUY", "WAIT", "AVOID"}


def test_scorer_short_term_high_risk_favours_momentum():
    snap = fake_snapshot("ACME")
    score, signals = score_stock(snap, Horizon.short_term, RiskTolerance.high)
    assert any("momentum" in s.lower() or "average" in s.lower() for s in signals)


def test_analyze_endpoint(monkeypatch):
    monkeypatch.setattr(main, "fetch_snapshot", fake_snapshot)
    monkeypatch.setattr(stock_data, "fetch_snapshot", fake_snapshot)

    client = TestClient(main.app)
    resp = client.post(
        "/analyze",
        json={"ticker": "ACME", "horizon": "long_term", "risk_tolerance": "low"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticker"] == "ACME"
    assert body["verdict"] in {"INVEST", "HOLD / CAUTIOUS BUY", "WAIT", "AVOID"}
    assert isinstance(body["score"], int)
    assert body["snapshot"]["sector"] == "Technology"
    assert isinstance(body["signals"], list)
    assert body["explanation"]


def test_analyze_endpoint_data_unavailable(monkeypatch):
    def boom(_ticker):
        raise stock_data.StockDataUnavailable("simulated rate limit")

    monkeypatch.setattr(main, "fetch_snapshot", boom)
    client = TestClient(main.app)
    resp = client.post(
        "/analyze",
        json={"ticker": "ACME", "horizon": "long_term", "risk_tolerance": "low"},
    )
    assert resp.status_code == 503
    assert "rate limit" in resp.json()["detail"]
