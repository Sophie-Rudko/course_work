from __future__ import annotations

from datetime import datetime, timedelta, timezone

from news_service import TICKER_META


YFINANCE_ALIASES: dict[str, str] = {
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "GSPC": "^GSPC",
    "^GSPC": "^GSPC",
}


def resolve_yfinance_symbol(sec: str) -> str | None:
    sec = sec.strip().upper()
    if sec in YFINANCE_ALIASES:
        return YFINANCE_ALIASES[sec]
    meta = TICKER_META.get(sec)
    if meta:
        return str(meta.get("yfinance"))
    return None


def fetch_yfinance_candles(sec: str, date_from: str, date_till: str) -> list[dict]:
    symbol = resolve_yfinance_symbol(sec)
    if not symbol:
        return []
    try:
        import yfinance as yf
    except Exception:
        return []

    try:
        start = datetime.fromisoformat(date_from).date()
        end = datetime.fromisoformat(date_till).date()
        hist = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
    except Exception:
        return []

    if hist is None or hist.empty:
        return []

    out: list[dict] = []
    for idx, row in hist.iterrows():
        dt = idx.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        out.append({
            "time": int(dt.timestamp()),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
        })
    return out
