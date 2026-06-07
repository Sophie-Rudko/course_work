from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from demo_data import demo_sentiment_series


def candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "log_return", "volatility"])
    res = []
    for c in candles:
        ts = c["time"]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        res.append({
            "date": dt,
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        })
    df = pd.DataFrame(res).sort_values("date").reset_index(drop=True)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    return out


def compute_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = compute_log_returns(df)
    out["volatility"] = out["log_return"].rolling(window=window, min_periods=max(2, window // 2)).std()
    return out


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]

    out["tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(26)
    out["senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    out["chikou"] = close.shift(-26)
    return out


DAILY_SENTIMENT_COLS = [
    "date", "sentiment_score", "news_count",
    "neg_ratio", "pos_ratio", "sentiment_dispersion",
]


def aggregate_daily_sentiment(analyzed_items: list[dict]) -> pd.DataFrame:
    res = []
    for item in analyzed_items:
        sent = item.get("sentiment", {})
        score = float(sent.get("sentiment_score", 0.0))
        conf = float(sent.get("confidence", 0.5))
        label = str(sent.get("label", "neutral")).lower()
        weight = max(0.1, conf)
        pub = item.get("published_at", "")
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            day = dt.date()
        except Exception:
            continue
        res.append({
            "date": day,
            "sentiment_score": score,
            "weight": weight,
            "is_neg": 1.0 if label == "negative" else 0.0,
            "is_pos": 1.0 if label == "positive" else 0.0,
        })
    if not res:
        return pd.DataFrame(columns=DAILY_SENTIMENT_COLS)

    df = pd.DataFrame(res)
    days = []
    for day, gr in df.groupby("date"):
        n = len(gr)
        days.append({
            "date": day,
            "sentiment_score": float(np.average(gr["sentiment_score"], weights=gr["weight"])),
            "news_count": int(n),
            "neg_ratio": float(gr["is_neg"].mean()),
            "pos_ratio": float(gr["is_pos"].mean()),
            "sentiment_dispersion": float(gr["sentiment_score"].std(ddof=0)) if n > 1 else 0.0,
        })
    return pd.DataFrame(days).sort_values("date").reset_index(drop=True)


def expand_sentiment_to_trading_days(
    sentiment_df: pd.DataFrame,
    trading_dates: pd.Series,
    *,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    extra_cols = ["neg_ratio", "pos_ratio", "sentiment_dispersion"]
    cal = pd.DataFrame({"date": pd.Series(trading_dates).drop_duplicates().sort_values().values})
    if sentiment_df.empty:
        cal["sentiment_score"] = 0.0
        cal["news_count"] = 0.0
        for c in extra_cols:
            cal[c] = 0.0
        return cal.reset_index(drop=True)

    sent = sentiment_df[["date", "sentiment_score"]].copy()
    sent["news_count"] = sentiment_df.get("news_count", 1.0)
    for c in extra_cols:
        sent[c] = sentiment_df.get(c, 0.0)

    out = cal.merge(sent, on="date", how="left")
    out["news_count"] = out["news_count"].fillna(0.0)
    had_news = out["news_count"] > 0
    if had_news.any():
        for c in ["sentiment_score"] + extra_cols:
            out.loc[~had_news, c] = np.nan
            out[c] = out[c].ffill(limit=ffill_limit).fillna(0.0)
        nc = out["news_count"].replace(0.0, np.nan)
        out["news_count"] = nc.ffill(limit=ffill_limit).fillna(0.0)
    else:
        out["sentiment_score"] = out["sentiment_score"].fillna(0.0)
        for c in extra_cols:
            out[c] = out[c].fillna(0.0)
    return out.reset_index(drop=True)


def compute_pearson_correlation(
    sentiment_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
    *,
    forward_fill_sentiment: bool = True,
) -> dict:
    if sentiment_df.empty or volatility_df.empty:
        return {"pearson_r": None, "p_value": None, "n_observations": 0, "aligned": []}

    vol = volatility_df[["date", "volatility"]].copy()
    sent = sentiment_df[["date", "sentiment_score"]].copy()

    if forward_fill_sentiment and not sent.empty:
        all_dates = vol[["date"]].drop_duplicates().sort_values("date")
        sent = all_dates.merge(sent, on="date", how="left")
        sent["sentiment_score"] = sent["sentiment_score"].ffill().fillna(0.0)

    df_m = pd.merge(sent, vol, on="date", how="inner")
    df_m = df_m.dropna(subset=["sentiment_score", "volatility"])

    if len(df_m) < 3:
        return {"pearson_r": None, "p_value": None, "n_observations": len(df_m), "aligned": []}

    if df_m["sentiment_score"].std() == 0 or df_m["volatility"].std() == 0:
        return {
            "pearson_r": None,
            "p_value": None,
            "n_observations": len(df_m),
            "aligned": [],
            "note": "constant_sentiment_or_volatility",
        }

    r = float(df_m["sentiment_score"].corr(df_m["volatility"]))
    n = len(df_m)
    p_value = _pearson_p_value(r, n)
    res = [
        {
            "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
            "sentiment_score": round(float(row["sentiment_score"]), 4),
            "volatility": round(float(row["volatility"]), 6),
        }
        for _, row in df_m.iterrows()
    ]
    return {
        "pearson_r": round(r, 4) if pd.notna(r) else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
        "n_observations": n,
        "aligned": res,
    }


def _pearson_p_value(r: float, n: int) -> float | None:
    if n < 3 or pd.isna(r) or abs(r) >= 1.0:
        return None
    df = n - 2
    t_stat = abs(r) * np.sqrt(df / max(1e-12, 1.0 - r * r))
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(t_stat, df))
    except Exception:
        from math import erfc, sqrt
        return float(erfc(t_stat / sqrt(2)))


def filter_news_by_date(items: list[dict], date_from: str, date_till: str) -> tuple[list[dict], bool]:
    if not items or not date_from or not date_till:
        return items, bool(items)
    try:
        frm = datetime.fromisoformat(date_from).date()
        till = datetime.fromisoformat(date_till).date()
    except Exception:
        return items, bool(items)

    res = []
    for item in items:
        pub = item.get("published_at", "")
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            day = dt.date()
            if frm <= day <= till:
                res.append(item)
        except Exception:
            continue
    return res, bool(res)


def build_market_analytics(
    candles: list[dict],
    analyzed_news: list[dict],
    *,
    allow_demo_sentiment: bool = False,
) -> dict:
    df = candles_to_dataframe(candles)
    if df.empty:
        return {
            "series": [], "ichimoku": [], "daily_sentiment": [],
            "correlation": compute_pearson_correlation(pd.DataFrame(), pd.DataFrame()),
            "sentiment_source": "none",
        }

    vol_df = compute_rolling_volatility(df)
    ichimoku_df = compute_ichimoku(vol_df)
    sentiment_df = aggregate_daily_sentiment(analyzed_news)
    sentiment_source = "news"
    if sentiment_df.empty and allow_demo_sentiment and len(df) > 0:
        demo = pd.DataFrame(demo_sentiment_series(len(df)))
        demo["date"] = df["date"].values
        sentiment_df = demo[["date", "sentiment_score"]]
        sentiment_source = "demo"
    elif sentiment_df.empty:
        sentiment_source = "none"
    correlation = compute_pearson_correlation(sentiment_df, vol_df[["date", "volatility"]])

    series = []
    for _, row in vol_df.iterrows():
        series.append({
            "date": row["date"].isoformat(),
            "close": round(float(row["close"]), 4),
            "log_return": round(float(row["log_return"]), 6) if pd.notna(row["log_return"]) else None,
            "volatility": round(float(row["volatility"]), 6) if pd.notna(row["volatility"]) else None,
        })

    ichimoku = []
    for _, row in ichimoku_df.iterrows():
        ichimoku.append({
            "date": row["date"].isoformat(),
            "tenkan": _f(row.get("tenkan")),
            "kijun": _f(row.get("kijun")),
            "senkou_a": _f(row.get("senkou_a")),
            "senkou_b": _f(row.get("senkou_b")),
            "chikou": _f(row.get("chikou")),
        })

    daily_sentiment = [
        {"date": row["date"].isoformat(), "sentiment_score": round(float(row["sentiment_score"]), 4)}
        for _, row in sentiment_df.iterrows()
    ]

    return {
        "series": series,
        "ichimoku": ichimoku,
        "daily_sentiment": daily_sentiment,
        "correlation": correlation,
        "volatility_df": vol_df,
        "sentiment_df": sentiment_df,
        "sentiment_source": sentiment_source,
    }


def _f(val) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 4)
    except Exception:
        return None
