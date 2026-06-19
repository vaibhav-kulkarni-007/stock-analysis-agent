from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Horizon(str, Enum):
    short_term = "short_term"
    long_term = "long_term"


class RiskTolerance(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    horizon: Horizon
    risk_tolerance: RiskTolerance


class StockSnapshot(BaseModel):
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    fifty_day_avg: Optional[float] = None
    two_hundred_day_avg: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    one_year_return_pct: Optional[float] = None
    dividend_yield: Optional[float] = None


class AnalyzeResponse(BaseModel):
    ticker: str
    verdict: str
    score: int
    snapshot: StockSnapshot
    signals: list[str]
    explanation: str
