from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from analytics_service import candles_to_dataframe, compute_rolling_volatility
from forecast_service import (
    FEATURE_COLS,
    MIN_ROWS,
    MIN_SENTIMENT_COVERAGE,
    TEST_RATIO,
    cols_without_sentiment,
    clean_model_label,
    make_random_forest,
    build_forecast_features,
    predict_lstm_test,
    sentiment_coverage,
)


SENTIMENT_SCAN_WINDOWS: list[tuple[str, str, str]] = [
    ("2020-02-01", "2020-12-31", "COVID crash 2020"),
    ("2022-01-01", "2022-12-31", "Bear market 2022"),
    ("2023-01-01", "2023-12-31", "Recovery 2023"),
    ("2024-01-01", "2024-06-30", "2024 H1"),
    ("2024-01-01", "2024-12-01", "2024 full year"),
    ("2025-01-01", "2025-06-30", "2025 H1"),
]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _metrics_row(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    preds: list[dict],
    *,
    uses_sentiment: bool = False,
) -> dict:
    pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return {
        "model_key": name,
        "model_label": clean_model_label(name, uses_sentiment),
        "uses_sentiment": uses_sentiment,
        "rmse": round(_rmse(y_true, pred), 8),
        "mae": round(_mae(y_true, pred), 8),
        "predictions": preds,
    }


def _series_preds(test_df: pd.DataFrame, y_pred: np.ndarray) -> list[dict]:
    out = []
    for (_, row), p in zip(test_df.iterrows(), y_pred):
        out.append({
            "date": row["date"].isoformat(),
            "actual": round(float(row["target_volatility"]), 6),
            "predicted": round(float(max(0.0, p)), 6),
        })
    return out


def run_volatility_backtest(
    candles: list[dict],
    sentiment_df: pd.DataFrame,
    *,
    test_ratio: float = TEST_RATIO,
) -> dict:
    vol_df = compute_rolling_volatility(candles_to_dataframe(candles))
    features_df = build_forecast_features(vol_df, sentiment_df)

    empty = {
        "status": "insufficient_data",
        "message": f"Need at least {MIN_ROWS} trading days with features; got {len(features_df)}.",
        "models": [],
        "summary": {},
        "test_period": {},
        "methodology": _methodology_notes(),
    }

    if len(features_df) < MIN_ROWS:
        return empty

    split_idx = max(int(len(features_df) * (1 - test_ratio)), MIN_ROWS - 1)
    split_idx = min(split_idx, len(features_df) - 3)
    train_df = features_df.iloc[:split_idx].copy()
    test_df = features_df.iloc[split_idx:].copy()
    y_test = test_df["target_volatility"].values

    results: list[dict] = []
    price_cols = cols_without_sentiment()
    full_cols = FEATURE_COLS
    har_cols = ["lag_volatility", "har_weekly", "har_monthly"]
    cov = sentiment_coverage(train_df)
    include_sent = cov >= MIN_SENTIMENT_COVERAGE
    y_train = train_df["target_volatility"].values

    def _fit_pred(make_model, cols: list[str], *, scale: bool = False) -> np.ndarray:
        x_tr, x_te = train_df[cols].values, test_df[cols].values
        if scale:
            sc = StandardScaler()
            x_tr, x_te = sc.fit_transform(x_tr), sc.transform(x_te)
        model = make_model()
        model.fit(x_tr, y_train)
        return model.predict(x_te)

    pred_persist = test_df["lag_volatility"].values
    results.append(_metrics_row(
        "baseline_persistence", y_test, pred_persist, _series_preds(test_df, pred_persist),
    ))

    pred_roll = test_df["rolling_mean_volatility"].values
    results.append(_metrics_row(
        "baseline_rolling_mean", y_test, pred_roll, _series_preds(test_df, pred_roll),
    ))

    pred_har = _fit_pred(lambda: Ridge(alpha=0.1), har_cols, scale=True)
    results.append(_metrics_row(
        "har_ridge", y_test, pred_har, _series_preds(test_df, pred_har),
    ))

    families = [
        ("ridge", lambda: Ridge(alpha=0.5), True),
        ("random_forest", make_random_forest, False),
        ("gradient_boosting",
         lambda: GradientBoostingRegressor(n_estimators=120, random_state=42, max_depth=4, learning_rate=0.05),
         False),
        ("hist_gb",
         lambda: HistGradientBoostingRegressor(max_iter=120, random_state=42, max_depth=6, learning_rate=0.05),
         False),
    ]
    for fam, maker, scale in families:
        pred_p = _fit_pred(maker, price_cols, scale=scale)
        results.append(_metrics_row(
            f"{fam}_price", y_test, pred_p, _series_preds(test_df, pred_p), uses_sentiment=False,
        ))
        if include_sent:
            pred_s = _fit_pred(maker, full_cols, scale=scale)
            results.append(_metrics_row(
                f"{fam}_sent", y_test, pred_s, _series_preds(test_df, pred_s), uses_sentiment=True,
            ))

    pred_lstm_p = predict_lstm_test(train_df, test_df, price_cols)
    if pred_lstm_p is not None:
        results.append(_metrics_row(
            "lstm_price", y_test, pred_lstm_p, _series_preds(test_df, pred_lstm_p), uses_sentiment=False,
        ))
    if include_sent:
        pred_lstm_s = predict_lstm_test(train_df, test_df, full_cols)
        if pred_lstm_s is not None:
            results.append(_metrics_row(
                "lstm_sent", y_test, pred_lstm_s, _series_preds(test_df, pred_lstm_s), uses_sentiment=True,
            ))

    baseline_rmse = results[0]["rmse"]
    for r in results:
        r["beats_baseline"] = r["rmse"] < baseline_rmse
        r["rmse_vs_baseline_pct"] = round(
            (baseline_rmse - r["rmse"]) / baseline_rmse * 100, 2
        ) if baseline_rmse > 0 else 0.0

    ranked = sorted(results, key=lambda x: x["rmse"])
    best = ranked[0]

    news_rows = [r for r in ranked if r.get("uses_sentiment")]
    best_news = news_rows[0] if news_rows else None
    sentiment_improves = None
    best_news_sibling = None
    news_helps = False
    if best_news is not None:
        family = best_news["model_key"].rsplit("_", 1)[0]
        best_news_sibling = next(
            (r for r in results if r["model_key"] == f"{family}_price"), None
        )
        if best_news_sibling and best_news_sibling["rmse"] > 0:
            sentiment_improves = round(
                (best_news_sibling["rmse"] - best_news["rmse"]) / best_news_sibling["rmse"] * 100, 2
            )
            news_helps = best_news["rmse"] < best_news_sibling["rmse"]

    ml_models = [r for r in results if not r["model_key"].startswith("baseline")]
    any_ml_beats = any(r["beats_baseline"] for r in ml_models)

    return {
        "status": "ok",
        "message": "",
        "models": results,
        "summary": {
            "best_model_key": best["model_key"],
            "best_model_label": best["model_label"],
            "best_rmse": best["rmse"],
            "best_uses_sentiment": best.get("uses_sentiment", False),
            "best_news_model_key": (best_news or {}).get("model_key"),
            "best_news_model_label": (best_news or {}).get("model_label"),
            "best_news_rmse": (best_news or {}).get("rmse"),
            "best_news_sibling_label": (best_news_sibling or {}).get("model_label"),
            "best_news_sibling_rmse": (best_news_sibling or {}).get("rmse"),
            "news_helps": news_helps,
            "baseline_rmse": baseline_rmse,
            "any_ml_beats_baseline": any_ml_beats,
            "sentiment_improves_rmse_pct": sentiment_improves,
            "sentiment_coverage_train": round(cov, 4),
            "uses_sentiment_features": include_sent,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "n_models": len(results),
            "feature_count": len(FEATURE_COLS),
        },
        "test_period": {
            "from": test_df["date"].iloc[0].isoformat(),
            "till": test_df["date"].iloc[-1].isoformat(),
        },
        "methodology": _methodology_notes(),
    }


def _methodology_notes() -> list[dict]:
    return [
        {"step": 1, "task": "MVP + MVC", "check": "Flask, services, templates"},
        {"step": 2, "task": "Literature algorithms", "check": "FinBERT, Pearson r, RF/LSTM vol forecast"},
        {"step": 3, "task": "Module testing", "check": "Hold-out backtest 80/20 on historical OHLC"},
        {"step": 4, "task": "Dashboard (Plotly)", "check": "Market, News, Volatility, Correlation, Evaluation"},
        {"step": 5, "task": "Results analysis", "check": "RMSE/MAE table; ML vs baseline; sentiment ablation"},
    ]


def scan_sentiment_periods(
    loader,
    sec: str = "^GSPC",
    *,
    windows: list[tuple[str, str, str]] | None = None,
    top_n: int = 3,
) -> list[dict]:
    windows = windows or SENTIMENT_SCAN_WINDOWS
    ranked: list[dict] = []

    for date_from, date_till, label in windows:
        try:
            ctx = loader(sec, date_from, date_till)
            bt = run_volatility_backtest(
                ctx["candles"],
                ctx.get("analytics", {}).get("sentiment_df"),
            )
            if bt.get("status") != "ok":
                continue
            s = bt["summary"]
            imp = s.get("sentiment_improves_rmse_pct")
            if imp is None:
                continue
            ranked.append({
                "label": label,
                "date_from": date_from,
                "date_till": date_till,
                "sentiment_improves_pct": imp,
                "coverage": s.get("sentiment_coverage_train"),
                "best_model": s.get("best_model_label"),
                "eval_url": f"/evaluation?sec={sec}&from={date_from}&till={date_till}",
            })
        except Exception:
            continue

    ranked.sort(key=lambda x: x["sentiment_improves_pct"], reverse=True)
    return ranked[:top_n]
