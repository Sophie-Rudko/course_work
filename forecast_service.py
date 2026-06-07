from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from analytics_service import (
    candles_to_dataframe,
    compute_rolling_volatility,
    expand_sentiment_to_trading_days,
)

PRICE_FEATURE_COLS = [
    "lag_volatility",
    "lag_volatility_2",
    "lag_volatility_3",
    "lag_return",
    "abs_return",
    "rolling_mean_volatility",
    "ewma_volatility",
    "har_weekly",
    "har_monthly",
]

SENTIMENT_SCORE_COLS = [
    "sentiment_lag_1",
    "sentiment_ma_3",
    "sentiment_shock",
]


NEWS_ACTIVITY_COLS = [
    "neg_ratio_lag_1",
    "neg_ratio_ma_3",
    "sentiment_dispersion_lag_1",
    "news_activity_lag_1",
    "news_activity_ma_5",
    "news_burst",
    "news_count_norm",
]
SENTIMENT_FEATURE_COLS = SENTIMENT_SCORE_COLS + NEWS_ACTIVITY_COLS
FEATURE_COLS = PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS
SENTIMENT_SCORE_WEIGHT = 0.35
MIN_SENTIMENT_COVERAGE = 0.08
SELECT_MARGIN = 0.02

MIN_ROWS = 15
TEST_RATIO = 0.2
LSTM_SEQ_LEN = 10


_BASE_MODEL_NAMES = {
    "baseline_persistence": "Persistence (yesterday to tomorrow)",
    "baseline_rolling_mean": "Rolling mean (5-day avg vol)",
    "har_ridge": "HAR-Ridge",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "hist_gb": "HistGradientBoosting",
    "lstm": "LSTM",
}

_PRICE_ONLY_KEYS = {
    "baseline_persistence",
    "baseline_rolling_mean",
    "har_ridge",
}


def clean_model_label(key: str, uses_sentiment: bool) -> str:
    base_key = key
    for suffix in ("_price", "_sent"):
        if key.endswith(suffix):
            base_key = key[: -len(suffix)]
            break
    base = _BASE_MODEL_NAMES.get(base_key, _BASE_MODEL_NAMES.get(key, base_key))
    if key in _PRICE_ONLY_KEYS:
        return base
    return f"{base} + news sentiment" if uses_sentiment else f"{base} (price only)"


def cols_without_sentiment() -> list[str]:
    return list(PRICE_FEATURE_COLS)


def cols_news_activity_only() -> list[str]:
    return list(PRICE_FEATURE_COLS) + list(NEWS_ACTIVITY_COLS)


def _zero_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SENTIMENT_SCORE_COLS:
        if col in out.columns:
            out[col] = 0.0
    return out


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def _align_sentiment_to_vol(vol_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    out = vol_df.copy()
    aligned = expand_sentiment_to_trading_days(sentiment_df, out["date"])
    out = out.merge(aligned, on="date", how="left", suffixes=("", "_sent"))
    out["sentiment_score"] = out["sentiment_score"].fillna(0.0)
    out["news_count"] = out["news_count"].fillna(0.0)
    for c in ("neg_ratio", "sentiment_dispersion"):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)

    nc_lag = out["news_count"].shift(1).fillna(0.0)
    w = SENTIMENT_SCORE_WEIGHT

    raw_sent = out["sentiment_score"].shift(1).fillna(0.0)
    raw_ma = out["sentiment_score"].rolling(3, min_periods=1).mean().shift(1).fillna(0.0)
    out["sentiment_lag_1"] = raw_sent * w
    out["sentiment_ma_3"] = raw_ma * w
    out["sentiment_shock"] = (raw_sent - raw_ma).abs() * w

    out["neg_ratio_lag_1"] = out["neg_ratio"].shift(1).fillna(0.0)
    out["neg_ratio_ma_3"] = out["neg_ratio"].rolling(3, min_periods=1).mean().shift(1).fillna(0.0)

    out["sentiment_dispersion_lag_1"] = out["sentiment_dispersion"].shift(1).fillna(0.0)

    out["news_activity_lag_1"] = (nc_lag > 0).astype(float)
    out["news_activity_ma_5"] = out["news_activity_lag_1"].rolling(5, min_periods=1).mean()
    nc_mean = nc_lag.rolling(10, min_periods=3).mean()
    nc_std = nc_lag.rolling(10, min_periods=3).std().replace(0, np.nan)
    out["news_burst"] = ((nc_lag - nc_mean) / (nc_std + 1e-6)).clip(-3, 3).fillna(0.0)
    out["news_count_norm"] = np.log1p(nc_lag) / 3.0
    return out


def _add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag_volatility"] = out["volatility"].shift(1)
    out["lag_volatility_2"] = out["volatility"].shift(2)
    out["lag_volatility_3"] = out["volatility"].shift(3)
    out["lag_return"] = out["log_return"].shift(1)
    out["abs_return"] = out["log_return"].shift(1).abs()
    out["rolling_mean_volatility"] = out["volatility"].rolling(5, min_periods=2).mean()
    out["ewma_volatility"] = out["volatility"].ewm(span=10, adjust=False).mean()
    out["har_weekly"] = out["volatility"].rolling(5, min_periods=2).mean().shift(1)
    out["har_monthly"] = out["volatility"].rolling(22, min_periods=5).mean().shift(1)
    out["target_volatility"] = out["volatility"].shift(-1)
    return out


def build_forecast_features(vol_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    df = _align_sentiment_to_vol(vol_df, sentiment_df)
    df = df.dropna(subset=["volatility"]).copy()
    df = _add_feature_columns(df)
    return df.dropna(subset=FEATURE_COLS + ["target_volatility"])


def sentiment_coverage(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    if "news_count" in df.columns:
        return float((df["news_count"].shift(1).fillna(0) > 0).mean())
    if "sentiment_lag_1" in df.columns:
        return float((df["sentiment_lag_1"] != 0).mean())
    return 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _zero_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SENTIMENT_FEATURE_COLS:
        out[col] = 0.0
    return out


def _pick_best_rf_variant(
    train_df: pd.DataFrame,
    *,
    val_ratio: float = 0.15,
) -> tuple[list[str], str, bool]:
    price_cols = cols_without_sentiment()
    news_cols = cols_news_activity_only()
    full_cols = FEATURE_COLS
    cov = sentiment_coverage(train_df)

    if cov < MIN_SENTIMENT_COVERAGE or len(train_df) < 12:
        return price_cols, "price_only", False

    split = max(int(len(train_df) * (1 - val_ratio)), 8)
    split = min(split, len(train_df) - 4)
    tr = train_df.iloc[:split]
    val = train_df.iloc[split:]
    y_val = val["target_volatility"].values

    variants: list[tuple[str, list[str], pd.DataFrame, pd.DataFrame]] = [
        ("price_only", price_cols, _zero_sentiment(tr), _zero_sentiment(val)),
        ("news_activity", news_cols, _zero_sentiment_scores(tr), _zero_sentiment_scores(val)),
        ("full_sentiment", full_cols, tr, val),
    ]

    errs: dict[str, tuple[float, list[str]]] = {}
    for name, cols, tr_x, val_x in variants:
        model = make_random_forest()
        model.fit(tr_x[cols].values, tr["target_volatility"].values)
        pred = model.predict(val_x[cols].values)
        errs[name] = (_rmse(y_val, pred), cols)

    price_rmse = errs["price_only"][0]

    best_name = "price_only"
    best_cols = errs["price_only"][1]
    best_rmse = price_rmse
    for name in ("news_activity", "full_sentiment"):
        err, cols = errs[name]
        if err < price_rmse * (1 - SELECT_MARGIN) and err < best_rmse:
            best_rmse = err
            best_cols = cols
            best_name = name

    uses_sent = best_name != "price_only"
    return best_cols, best_name, uses_sent


def _feature_cols_for_data(train_df: pd.DataFrame) -> tuple[list[str], bool]:
    cols, _, uses = _pick_best_rf_variant(train_df)
    return cols, uses


def latest_feature_row(vol_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.Series | None:
    df = _align_sentiment_to_vol(vol_df, sentiment_df)
    df = df.dropna(subset=["volatility"]).copy()
    df = _add_feature_columns(df)
    ready = df.dropna(subset=FEATURE_COLS)
    if ready.empty:
        return None
    return ready.iloc[-1]


def _lstm_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(features)):
        xs.append(features[i - seq_len : i])
        ys.append(targets[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def predict_lstm_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray | None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    combined = pd.concat([train_df, test_df], ignore_index=True)
    if len(combined) < LSTM_SEQ_LEN + 8:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[feature_cols].values)
    targets = combined["target_volatility"].values.astype(np.float32)
    X, y = _lstm_sequences(scaled, targets, LSTM_SEQ_LEN)
    if len(X) < 12:
        return None

    split = len(train_df) - LSTM_SEQ_LEN
    split = max(split, LSTM_SEQ_LEN)
    split = min(split, len(X) - 2)
    if split < LSTM_SEQ_LEN or len(X) - split < 2:
        return None

    class _VolLSTM(nn.Module):
        def __init__(self, input_size: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    device = torch.device("cpu")
    model = _VolLSTM(len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X[:split], dtype=torch.float32, device=device)
    yt = torch.tensor(y[:split], dtype=torch.float32, device=device)

    model.train()
    for _ in range(100):
        opt.zero_grad()
        loss_fn(model(Xt), yt).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_tail = model(torch.tensor(X[split:], dtype=torch.float32, device=device)).cpu().numpy()

    pred_by_idx: dict[int, float] = {}
    for j, p in enumerate(pred_tail):
        row_idx = split + j + LSTM_SEQ_LEN
        pred_by_idx[row_idx] = float(p)

    train_len = len(train_df)
    out = test_df["lag_volatility"].values.astype(float).copy()
    for i in range(len(test_df)):
        combined_idx = train_len + i
        if combined_idx in pred_by_idx:
            out[i] = max(0.0, pred_by_idx[combined_idx])
    return out


def _evaluate_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    y_test = test_df["target_volatility"].values
    baseline_pred = test_df["lag_volatility"].values
    cov = sentiment_coverage(train_df)

    metrics = {
        "baseline_persistence": {
            "rmse": round(_rmse(y_test, baseline_pred), 8),
            "mae": round(_mae(y_test, baseline_pred), 8),
            "description": "Yesterday vol to today (no ML)",
        },
        "sentiment_coverage_train": round(cov, 4),
    }

    if len(train_df) < 8 or len(test_df) < 3:
        return metrics

    price_cols = cols_without_sentiment()
    train_no = _zero_sentiment(train_df)
    test_no = _zero_sentiment(test_df)

    feat_cols, use_sent = _feature_cols_for_data(train_df)

    rf_sent = make_random_forest()
    rf_sent.fit(train_df[feat_cols].values, train_df["target_volatility"].values)
    pred_sent = rf_sent.predict(test_df[feat_cols].values)
    _, variant_name, _ = _pick_best_rf_variant(train_df)
    sent_label = f"RF + {variant_name.replace('_', ' ')}"
    metrics["random_forest_with_sentiment"] = {
        "rmse": round(_rmse(y_test, pred_sent), 8),
        "mae": round(_mae(y_test, pred_sent), 8),
        "description": sent_label,
    }

    rf_no = make_random_forest()
    rf_no.fit(train_no[price_cols].values, train_no["target_volatility"].values)
    pred_no = rf_no.predict(test_no[price_cols].values)
    metrics["random_forest_without_sentiment"] = {
        "rmse": round(_rmse(y_test, pred_no), 8),
        "mae": round(_mae(y_test, pred_no), 8),
        "description": "RF: price features only (ablation)",
    }

    rmse_sent = metrics["random_forest_with_sentiment"]["rmse"]
    rmse_no = metrics["random_forest_without_sentiment"]["rmse"]
    metrics["sentiment_improves_rmse_pct"] = round(
        (rmse_no - rmse_sent) / rmse_no * 100, 2
    ) if rmse_no > 0 else 0.0

    pred_lstm = predict_lstm_test(train_df, test_df, feat_cols)
    if pred_lstm is not None:
        metrics["lstm_with_sentiment"] = {
            "rmse": round(_rmse(y_test, pred_lstm), 8),
            "mae": round(_mae(y_test, pred_lstm), 8),
            "description": f"LSTM seq={LSTM_SEQ_LEN}, lagged sentiment",
        }

    return metrics


def sentiment_value_from_backtest(backtest: dict, *, turbulent_quantile: float = 0.70) -> dict:
    summary = backtest.get("summary") or {}
    models = backtest.get("models") or []
    news_key = summary.get("best_news_model_key")
    if not news_key:
        return {"available": False, "reason": "no news coverage in this period"}

    best_news = next((m for m in models if m["model_key"] == news_key), None)
    family = news_key.rsplit("_", 1)[0]
    sibling = next((m for m in models if m["model_key"] == f"{family}_price"), None)
    if not best_news or not sibling:
        return {"available": False, "reason": "missing model pair"}

    news_preds = best_news.get("predictions") or []
    price_preds = {p["date"]: p["predicted"] for p in (sibling.get("predictions") or [])}
    if len(news_preds) < 6:
        return {"available": False, "reason": "too few test points"}

    dates = [p["date"] for p in news_preds]
    y = np.array([float(p["actual"]) for p in news_preds])
    pred_sent = np.array([float(p["predicted"]) for p in news_preds])
    pred_price = np.array([float(price_preds.get(d, p["actual"])) for d, p in zip(dates, news_preds)])

    thr = float(np.quantile(y, turbulent_quantile))
    turbulent = y >= thr
    calm = ~turbulent

    def _subset_rmse(mask: np.ndarray, pred: np.ndarray) -> float | None:
        return round(_rmse(y[mask], pred[mask]), 8) if int(mask.sum()) >= 2 else None

    def _improve_pct(rmse_price: float | None, rmse_sent: float | None) -> float | None:
        if rmse_price and rmse_sent and rmse_price > 0:
            return round((rmse_price - rmse_sent) / rmse_price * 100, 2)
        return None

    def _block(mask: np.ndarray | None) -> dict:
        rp = round(_rmse(y, pred_price), 8) if mask is None else _subset_rmse(mask, pred_price)
        rs = round(_rmse(y, pred_sent), 8) if mask is None else _subset_rmse(mask, pred_sent)
        return {"price_only_rmse": rp, "with_sentiment_rmse": rs, "improvement_pct": _improve_pct(rp, rs)}

    history = [
        {
            "date": dates[i],
            "actual": round(float(y[i]), 6),
            "price_only": round(float(pred_price[i]), 6),
            "with_sentiment": round(float(pred_sent[i]), 6),
            "turbulent": bool(turbulent[i]),
        }
        for i in range(len(dates))
    ]

    return {
        "available": True,
        "coverage": summary.get("sentiment_coverage_train"),
        "news_model_label": best_news["model_label"],
        "price_model_label": sibling["model_label"],
        "family_label": _BASE_MODEL_NAMES.get(family, family),
        "turbulent_threshold": round(thr, 6),
        "n_turbulent": int(turbulent.sum()),
        "n_calm": int(calm.sum()),
        "overall": _block(None),
        "turbulent": _block(turbulent),
        "calm": _block(calm),
        "history": history,
    }


def _make_model_for_family(family: str):
    if family == "ridge":
        return Ridge(alpha=0.5), True
    if family == "random_forest":
        return make_random_forest(), False
    if family == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=120, random_state=42, max_depth=4, learning_rate=0.05), False
    if family == "hist_gb":
        return HistGradientBoostingRegressor(max_iter=120, random_state=42, max_depth=6, learning_rate=0.05), False
    return None, False


def _predict_next_day(
    best_key: str,
    train_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> tuple[float, dict, list[str], bool]:
    price_cols = cols_without_sentiment()
    last = latest_feature_row(vol_df, sentiment_df)
    if last is None:
        return 0.01, {}, price_cols, False
    importances: dict = {}
    y = train_df["target_volatility"].values

    if best_key == "baseline_persistence":
        return max(0.0, float(last["lag_volatility"])), {}, ["lag_volatility"], False

    if best_key == "baseline_rolling_mean":
        return max(0.0, float(last["rolling_mean_volatility"])), {}, ["rolling_mean_volatility"], False

    if best_key == "har_ridge":
        har_cols = ["lag_volatility", "har_weekly", "har_monthly"]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train_df[har_cols].values)
        model = Ridge(alpha=0.1)
        model.fit(Xtr, y)
        pred = model.predict(scaler.transform(last[har_cols].values.reshape(1, -1)))[0]
        return max(0.0, float(pred)), importances, har_cols, False

    uses_sent = best_key.endswith("_sent")
    family = best_key.rsplit("_", 1)[0] if best_key.endswith(("_price", "_sent")) else best_key
    cols = FEATURE_COLS if uses_sent else price_cols
    model, needs_scaling = _make_model_for_family(family)
    if model is None:
        return max(0.0, float(last["lag_volatility"])), {}, price_cols, False

    Xtr = train_df[cols].values
    Xlast = last[cols].values.reshape(1, -1)
    if needs_scaling:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xlast = scaler.transform(Xlast)
    model.fit(Xtr, y)
    pred = model.predict(Xlast)[0]
    if hasattr(model, "feature_importances_"):
        importances = {c: round(float(v), 4) for c, v in zip(cols, model.feature_importances_)}
    return max(0.0, float(pred)), importances, cols, uses_sent


def forecast_next_volatility(
    candles: list[dict],
    sentiment_df: pd.DataFrame,
    *,
    news_in_range: bool = False,
    news_in_range_count: int = 0,
) -> dict:
    vol_df = compute_rolling_volatility(candles_to_dataframe(candles))
    features_df = build_forecast_features(vol_df, sentiment_df)

    if len(features_df) < MIN_ROWS:
        last_vol = float(vol_df["volatility"].dropna().iloc[-1]) if not vol_df["volatility"].dropna().empty else 0.01
        return {
            "predicted_volatility": round(last_vol, 6),
            "model": "fallback_last_value",
            "train_samples": len(features_df),
            "test_samples": 0,
            "feature_importances": {},
            "evaluation": {},
            "history": _forecast_history_fallback(vol_df),
            "sentiment_coverage": 0.0,
            "uses_sentiment": False,
            "news_in_range": news_in_range,
        }

    split_idx = max(int(len(features_df) * (1 - TEST_RATIO)), MIN_ROWS - 1)
    split_idx = min(split_idx, len(features_df) - 3)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    evaluation = _evaluate_on_test(train_df, test_df)
    cov = sentiment_coverage(train_df)

    from backtest_service import run_volatility_backtest

    backtest = run_volatility_backtest(candles, sentiment_df)
    production_key = "random_forest_price"
    model_ranking: list[dict] = []
    if backtest.get("status") == "ok":
        summary = backtest["summary"]

        ranked_models = sorted(backtest["models"], key=lambda x: x["rmse"])
        production = next(
            (m for m in ranked_models if not m["model_key"].startswith("lstm")),
            ranked_models[0],
        )
        production_key = production["model_key"]
        model_name = production["model_label"]
        best_m = next((m for m in backtest["models"] if m["model_key"] == production_key), None)
        history = [
            {
                "date": p["date"],
                "actual_volatility": p["actual"],
                "predicted_volatility": p["predicted"],
            }
            for p in ((best_m or {}).get("predictions") or [])
        ]

        model_ranking = [
            {
                "model_key": m["model_key"],
                "model_label": m["model_label"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "uses_sentiment": m.get("uses_sentiment", False),
                "is_best": m["model_key"] == production_key,
            }
            for m in sorted(backtest["models"], key=lambda x: x["rmse"])
        ]
        pred, importances, _, uses_sentiment = _predict_next_day(
            production_key, train_df, vol_df, sentiment_df,
        )
        model_name = clean_model_label(production_key, uses_sentiment)
        evaluation["best_model_key"] = production_key
        evaluation["best_model_rmse"] = summary.get("best_rmse")
        evaluation["best_model_label"] = model_name
    else:
        feat_cols, variant_name, use_sentiment = _pick_best_rf_variant(train_df)
        model = make_random_forest()
        if use_sentiment:
            model.fit(train_df[feat_cols].values, train_df["target_volatility"].values)
            model_name = f"RandomForest + {variant_name.replace('_', ' ')}"
            active_cols = feat_cols
        else:
            price_cols = cols_without_sentiment()
            train_no = _zero_sentiment(train_df)
            model.fit(train_no[price_cols].values, train_no["target_volatility"].values)
            model_name = "RandomForest (price-only)"
            active_cols = price_cols
            use_sentiment = False
        last = latest_feature_row(vol_df, sentiment_df)
        if last is None:
            pred = float(test_df["lag_volatility"].iloc[-1])
        else:
            row = last[active_cols]
            pred = max(0.0, float(model.predict(row.values.reshape(1, -1))[0]))
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = {
                col: round(float(imp), 4)
                for col, imp in zip(active_cols, model.feature_importances_)
            }
        uses_sentiment = use_sentiment
        history = []
        for _, row in test_df.iterrows():
            x = row[active_cols].values.reshape(1, -1)
            history.append({
                "date": row["date"].isoformat(),
                "actual_volatility": round(float(row["target_volatility"]), 6),
                "predicted_volatility": round(float(model.predict(x)[0]), 6),
            })

    sentiment_value = sentiment_value_from_backtest(backtest) if backtest.get("status") == "ok" else {"available": False}
    best_summary = backtest.get("summary", {}) if backtest.get("status") == "ok" else {}

    return {
        "predicted_volatility": round(pred, 6),
        "model": model_name,
        "production_model_key": production_key,
        "best_overall_label": best_summary.get("best_model_label"),
        "best_overall_rmse": best_summary.get("best_rmse"),
        "best_news_label": best_summary.get("best_news_model_label"),
        "best_news_rmse": best_summary.get("best_news_rmse"),
        "news_helps": best_summary.get("news_helps", False),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "feature_importances": importances,
        "evaluation": evaluation,
        "model_ranking": model_ranking,
        "sentiment_value": sentiment_value,
        "history": history,
        "sentiment_coverage": round(cov, 4),
        "uses_sentiment": uses_sentiment,
        "news_in_range": news_in_range,
        "news_in_range_count": news_in_range_count,
    }


def _forecast_history_fallback(vol_df: pd.DataFrame) -> list[dict]:
    out = []
    vals = vol_df.dropna(subset=["volatility"]).tail(30)
    for _, row in vals.iterrows():
        v = float(row["volatility"])
        out.append({
            "date": row["date"].isoformat(),
            "actual_volatility": round(v, 6),
            "predicted_volatility": round(v, 6),
        })
    return out
