from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analytics_service import (
    aggregate_daily_sentiment,
    candles_to_dataframe,
    compute_ichimoku,
    compute_log_returns,
    compute_pearson_correlation,
    compute_rolling_volatility,
    expand_sentiment_to_trading_days,
)
from backtest_service import run_volatility_backtest
from demo_data import demo_candles, demo_news
from forecast_service import build_forecast_features, forecast_next_volatility, sentiment_coverage
from sentiment_service import analyze_news_items, analyze_text


def test_sentiment_score_in_range():
    result = analyze_text("Markets rally as profits beat expectations")
    assert -1.0 <= result["sentiment_score"] <= 1.0
    assert result["label"] in {"positive", "negative", "neutral"}


def test_analyze_news_items_attaches_sentiment():
    analyzed = analyze_news_items(demo_news("SPX", 6))
    assert analyzed["count"] == 6
    for item in analyzed["items"]:
        assert "sentiment" in item
        assert -1.0 <= item["sentiment"]["sentiment_score"] <= 1.0


def test_daily_sentiment_aggregation_columns():
    analyzed = analyze_news_items(demo_news("SPX", 12))["items"]
    df = aggregate_daily_sentiment(analyzed)
    assert not df.empty
    for col in ("date", "sentiment_score", "news_count", "neg_ratio", "sentiment_dispersion"):
        assert col in df.columns
    assert (df["news_count"] > 0).all()


def test_log_returns_match_manual():
    candles = demo_candles("SPX", days=30)
    df = compute_log_returns(candles_to_dataframe(candles))
    manual = math.log(df["close"].iloc[1] / df["close"].iloc[0])
    assert abs(df["log_return"].iloc[1] - manual) < 1e-9


def test_rolling_volatility_non_negative():
    candles = demo_candles("SPX", days=120)
    df = compute_rolling_volatility(candles_to_dataframe(candles))
    vol = df["volatility"].dropna()
    assert not vol.empty
    assert (vol >= 0).all()


def test_ichimoku_components_present():
    candles = demo_candles("SPX", days=120)
    df = compute_ichimoku(candles_to_dataframe(candles))
    for col in ("tenkan", "kijun", "senkou_a", "senkou_b", "chikou"):
        assert col in df.columns


def test_pearson_perfect_correlation():
    dates = pd.date_range("2024-01-01", periods=10, freq="D").date
    sent = pd.DataFrame({"date": dates, "sentiment_score": np.linspace(-1, 1, 10)})
    vol = pd.DataFrame({"date": dates, "volatility": np.linspace(0.0, 0.02, 10)})
    res = compute_pearson_correlation(sent, vol)
    assert res["pearson_r"] is not None
    assert res["pearson_r"] > 0.99


def test_expand_sentiment_to_trading_days_forward_fills():
    trading = pd.Series(pd.date_range("2024-01-01", periods=6, freq="D").date)
    sent = pd.DataFrame({
        "date": [trading.iloc[0], trading.iloc[3]],
        "sentiment_score": [0.5, -0.5],
        "news_count": [3, 2],
    })
    out = expand_sentiment_to_trading_days(sent, trading)
    assert len(out) == 6
    assert (out["news_count"] >= 0).all()


def test_feature_engineering_produces_target():
    candles = demo_candles("SPX", days=160)
    vol_df = compute_rolling_volatility(candles_to_dataframe(candles))
    sent = aggregate_daily_sentiment(analyze_news_items(demo_news("SPX", 12))["items"])
    feats = build_forecast_features(vol_df, sent)
    assert "target_volatility" in feats.columns
    assert not feats.empty


def test_forecast_returns_valid_prediction():
    candles = demo_candles("SPX", days=200)
    sent = aggregate_daily_sentiment(analyze_news_items(demo_news("SPX", 12))["items"])
    forecast = forecast_next_volatility(candles, sent)
    assert forecast["predicted_volatility"] >= 0
    assert "model" in forecast and forecast["model"]


def test_backtest_runs_and_beats_or_ties_baseline():
    candles = demo_candles("SPX", days=220)
    sent = aggregate_daily_sentiment(analyze_news_items(demo_news("SPX", 12))["items"])
    bt = run_volatility_backtest(candles, sent)
    assert bt["status"] == "ok"
    assert bt["summary"]["best_rmse"] <= bt["summary"]["baseline_rmse"] + 1e-9
    assert bt["summary"]["n_models"] >= 5


def test_sentiment_coverage_bounds():
    candles = demo_candles("SPX", days=160)
    vol_df = compute_rolling_volatility(candles_to_dataframe(candles))
    sent = aggregate_daily_sentiment(analyze_news_items(demo_news("SPX", 12))["items"])
    feats = build_forecast_features(vol_df, sent)
    cov = sentiment_coverage(feats)
    assert 0.0 <= cov <= 1.0
