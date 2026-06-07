from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone


def demo_news(ticker: str = "SBER", count: int = 12) -> list[dict]:
    now = datetime.now(timezone.utc)
    samples = [
        ("Markets rally as banking sector shows strong earnings", "positive"),
        ("Investors remain cautious amid inflation concerns", "negative"),
        ("Stock holds steady ahead of central bank decision", "neutral"),
        ("Analysts upgrade outlook citing revenue growth", "positive"),
        ("Volatility rises on geopolitical uncertainty", "negative"),
        ("Company reports quarterly profit above expectations", "positive"),
        ("Trading volume declines in mixed session", "neutral"),
        ("Sector faces headwinds from regulatory changes", "negative"),
        ("Market sentiment improves after policy announcement", "positive"),
        ("Shares fluctuate within narrow range", "neutral"),
        ("Bearish pressure builds on weak macro data", "negative"),
        ("Bullish momentum continues for second week", "positive"),
    ]
    out = []
    for i, (title, _tone) in enumerate(samples[:count]):
        published = now - timedelta(hours=i * 6 + 1)
        out.append({
            "title": f"{title} ({ticker})",
            "summary": f"Demo news item about {ticker}. {title}.",
            "source": "Demo Feed",
            "published_at": published.isoformat(),
            "url": f"https://example.com/news/{ticker.lower()}/{i}",
        })
    return out


def demo_candles(sec: str = "SBER", days: int = 90, start_price: float = 280.0) -> list[dict]:
    random.seed(42)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    price = start_price
    out: list[dict] = []
    for i in range(days, 0, -1):
        dt = now - timedelta(days=i)
        daily_ret = random.gauss(0, 0.015)
        open_p = price
        close_p = max(1.0, price * (1 + daily_ret))
        high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, 0.005)))
        low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, 0.005)))
        out.append({
            "time": int(dt.timestamp()),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
        })
        price = close_p
    return out


def demo_sentiment_series(days: int = 90) -> list[dict]:
    random.seed(7)
    now = datetime.now(timezone.utc).date()
    out = []
    for i in range(days, 0, -1):
        d = now - timedelta(days=i)
        score = max(-1.0, min(1.0, random.gauss(0.05, 0.25)))
        out.append({"date": d.isoformat(), "sentiment_score": round(score, 4)})
    return out
