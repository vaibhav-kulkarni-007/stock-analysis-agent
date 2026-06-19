from app.models import Horizon, RiskTolerance, StockSnapshot


def score_stock(
    snap: StockSnapshot, horizon: Horizon, risk: RiskTolerance
) -> tuple[int, list[str]]:
    """Return a 0-100 score and a list of human-readable signals.

    Long-term leans on fundamentals (P/E, revenue growth, margins, debt).
    Short-term leans on momentum (price vs moving averages, 1y return).
    Risk tolerance shifts how strict the thresholds are.
    """
    score = 50
    signals: list[str] = []

    if horizon == Horizon.long_term:
        if snap.revenue_growth is not None:
            rg_pct = snap.revenue_growth * 100
            if rg_pct >= 15:
                score += 12
                signals.append(f"Strong revenue growth ({rg_pct:.1f}% YoY)")
            elif rg_pct >= 5:
                score += 6
                signals.append(f"Moderate revenue growth ({rg_pct:.1f}% YoY)")
            elif rg_pct < 0:
                score -= 10
                signals.append(f"Revenue is shrinking ({rg_pct:.1f}% YoY)")

        if snap.profit_margin is not None:
            pm_pct = snap.profit_margin * 100
            if pm_pct >= 15:
                score += 10
                signals.append(f"Healthy profit margin ({pm_pct:.1f}%)")
            elif pm_pct < 0:
                score -= 12
                signals.append(f"Company is unprofitable ({pm_pct:.1f}% margin)")

        if snap.pe_ratio is not None:
            if 0 < snap.pe_ratio <= 20:
                score += 8
                signals.append(f"Reasonable valuation (P/E {snap.pe_ratio:.1f})")
            elif snap.pe_ratio > 40:
                score -= 8
                signals.append(f"Expensive valuation (P/E {snap.pe_ratio:.1f})")

        if snap.debt_to_equity is not None:
            if snap.debt_to_equity < 50:
                score += 5
                signals.append(f"Low debt (D/E {snap.debt_to_equity:.0f})")
            elif snap.debt_to_equity > 150:
                score -= 8
                signals.append(f"High debt load (D/E {snap.debt_to_equity:.0f})")

    else:  # short_term
        if snap.current_price and snap.fifty_day_avg:
            if snap.current_price > snap.fifty_day_avg:
                score += 10
                signals.append("Price above 50-day average (upward momentum)")
            else:
                score -= 8
                signals.append("Price below 50-day average (downward momentum)")

        if snap.current_price and snap.two_hundred_day_avg:
            if snap.current_price > snap.two_hundred_day_avg:
                score += 8
                signals.append("Price above 200-day average (longer trend up)")
            else:
                score -= 6
                signals.append("Price below 200-day average (longer trend down)")

        if snap.one_year_return_pct is not None:
            if snap.one_year_return_pct >= 20:
                score += 8
                signals.append(f"Strong 1-year return ({snap.one_year_return_pct:.1f}%)")
            elif snap.one_year_return_pct <= -10:
                score -= 10
                signals.append(f"Negative 1-year return ({snap.one_year_return_pct:.1f}%)")

        if snap.current_price and snap.fifty_two_week_high:
            pct_off_high = (
                (snap.fifty_two_week_high - snap.current_price)
                / snap.fifty_two_week_high
                * 100
            )
            if pct_off_high < 5:
                score += 4
                signals.append("Trading near 52-week high")
            elif pct_off_high > 30:
                score -= 4
                signals.append(f"Trading {pct_off_high:.0f}% below 52-week high")

    if risk == RiskTolerance.low:
        if snap.pe_ratio is not None and snap.pe_ratio > 30:
            score -= 5
            signals.append("Adjusted: high P/E penalised for low risk tolerance")
        if snap.dividend_yield and snap.dividend_yield > 0.02:
            score += 4
            signals.append(
                f"Pays dividend ({snap.dividend_yield * 100:.1f}%), good for low risk"
            )
    elif risk == RiskTolerance.high:
        if snap.one_year_return_pct and snap.one_year_return_pct >= 30:
            score += 4
            signals.append("Adjusted: strong momentum favoured for high risk tolerance")

    score = max(0, min(100, score))
    return score, signals


def verdict_from_score(score: int) -> str:
    if score >= 70:
        return "INVEST"
    if score >= 50:
        return "HOLD / CAUTIOUS BUY"
    if score >= 35:
        return "WAIT"
    return "AVOID"
