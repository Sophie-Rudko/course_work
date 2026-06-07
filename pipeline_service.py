from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analytics_service import build_market_analytics, filter_news_by_date
from demo_data import demo_candles, demo_news
from forecast_service import forecast_next_volatility
from market_data import fetch_yfinance_candles, resolve_yfinance_symbol
from news_service import fetch_news, _subsample_for_sentiment
from sentiment_service import analyze_news_items
from services import get_moex_candles, InvestError


def default_date_range(days: int = 90) -> tuple[str, str]:
    till = datetime.now(timezone.utc).date()
    frm = till - timedelta(days=days)
    return frm.isoformat(), till.isoformat()


def load_candles(token: str, sec: str, date_from: str, date_till: str) -> dict:
    sec = sec.strip().upper() or "SBER"
    if not date_from or not date_till:
        date_from, date_till = default_date_range()

    yf_sym = resolve_yfinance_symbol(sec)
    is_moex = bool(yf_sym and str(yf_sym).endswith(".ME"))

    if is_moex and token:
        try:
            data = get_moex_candles(token, sec, "24", date_from, date_till)
            if data.get("candles"):
                return {
                    "security": sec,
                    "candles": data["candles"],
                    "source": data.get("source", "unknown"),
                    "date_from": date_from,
                    "date_till": date_till,
                }
        except InvestError:
            pass
        except Exception:
            pass

    if yf_sym:
        yf_candles = fetch_yfinance_candles(sec, date_from, date_till)
        if yf_candles:
            return {
                "security": sec,
                "candles": yf_candles,
                "source": "Yahoo Finance (yfinance)",
                "date_from": date_from,
                "date_till": date_till,
            }

    if token:
        try:
            data = get_moex_candles(token, sec, "24", date_from, date_till)
            if data.get("candles"):
                return {
                    "security": sec,
                    "candles": data["candles"],
                    "source": data.get("source", "unknown"),
                    "date_from": date_from,
                    "date_till": date_till,
                }
        except InvestError:
            pass
        except Exception:
            pass

    yf_candles = fetch_yfinance_candles(sec, date_from, date_till)
    if yf_candles:
        return {
            "security": sec,
            "candles": yf_candles,
            "source": "Yahoo Finance (yfinance)",
            "date_from": date_from,
            "date_till": date_till,
        }

    days = max(30, (datetime.fromisoformat(date_till) - datetime.fromisoformat(date_from)).days)
    return {
        "security": sec,
        "candles": demo_candles(sec, days=days),
        "source": "demo",
        "date_from": date_from,
        "date_till": date_till,
    }


def load_market_context(token: str, sec: str, date_from: str = "", date_till: str = "") -> dict:
    candle_bundle = load_candles(token, sec, date_from, date_till)
    analytics = build_market_analytics(
        candle_bundle["candles"],
        [],
        allow_demo_sentiment=False,
    )
    return {
        "sec": candle_bundle["security"],
        "date_from": candle_bundle["date_from"],
        "date_till": candle_bundle["date_till"],
        "candles": candle_bundle["candles"],
        "candle_source": candle_bundle["source"],
        "analytics": analytics,
    }


def load_full_context(
    token: str,
    sec: str,
    date_from: str = "",
    date_till: str = "",
    *,
    include_historical: bool = True,
) -> dict:
    candle_bundle = load_candles(token, sec, date_from, date_till)
    sec = candle_bundle["security"]
    date_from = candle_bundle["date_from"]
    date_till = candle_bundle["date_till"]

    news_bundle = fetch_news(
        sec,
        limit=60,
        date_from=date_from,
        date_till=date_till,
        include_historical=include_historical,
    )
    if news_bundle["source_mode"] == "demo" and not news_bundle["items"]:
        news_bundle["items"] = demo_news(sec)

    items_in_range = news_bundle.get("items_in_range") or []
    if not items_in_range:
        items_in_range, news_in_range = filter_news_by_date(news_bundle["items"], date_from, date_till)
    else:
        news_in_range = bool(items_in_range)

    finbert_pool = _subsample_for_sentiment(
        items_in_range if items_in_range else news_bundle["items"],
        max_items=500,
    )
    if not finbert_pool and news_bundle["items"]:
        finbert_pool = _subsample_for_sentiment(news_bundle["items"], max_items=60)

    analyzed = analyze_news_items(finbert_pool)
    items_for_analytics = analyzed["items"]

    allow_demo = news_bundle["source_mode"] == "demo"
    if allow_demo and not items_for_analytics:
        items_for_analytics = analyze_news_items(demo_news(sec))["items"]

    analytics = build_market_analytics(
        candle_bundle["candles"],
        items_for_analytics,
        allow_demo_sentiment=allow_demo,
    )
    forecast = forecast_next_volatility(
        candle_bundle["candles"],
        analytics.get("sentiment_df"),
        news_in_range=news_in_range,
        news_in_range_count=len(items_in_range),
    )

    return {
        "sec": sec,
        "date_from": date_from,
        "date_till": date_till,
        "candles": candle_bundle["candles"],
        "candle_source": candle_bundle["source"],
        "news": news_bundle["items"],
        "news_in_range_count": len(items_in_range),
        "news_date_overlap": news_in_range,
        "news_historical_count": news_bundle.get("historical_count", len(items_in_range)),
        "news_source_mode": news_bundle["source_mode"],
        "sentiment_items": analyzed["items"],
        "sentiment_method": analyzed["method"],
        "analytics": analytics,
        "forecast": forecast,
    }
