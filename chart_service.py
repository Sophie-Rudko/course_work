from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def _fig_html(fig: go.Figure) -> str:
    fig.update_layout(
        autosize=True,
        title=dict(x=0, xanchor="left", font=dict(size=14)),
        margin=dict(l=60, r=20, t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    return pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
    )


def market_chart(candles: list[dict], ichimoku: list[dict]) -> str:
    if not candles:
        fig = go.Figure()
        fig.update_layout(title="No candle data available", height=520)
        return _fig_html(fig)

    cdf = pd.DataFrame(candles)
    cdf["date"] = pd.to_datetime(cdf["time"], unit="s")
    idf = pd.DataFrame(ichimoku) if ichimoku else pd.DataFrame()
    if not idf.empty:
        idf["date"] = pd.to_datetime(idf["date"])

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=cdf["date"],
        open=cdf["open"], high=cdf["high"], low=cdf["low"], close=cdf["close"],
        name="OHLC",
    ))

    if not idf.empty:
        for col, color, name in [
            ("tenkan", "#2962FF", "Tenkan"),
            ("kijun", "#B71C1C", "Kijun"),
            ("senkou_a", "#43A047", "Senkou A"),
            ("senkou_b", "#FF6F00", "Senkou B"),
            ("chikou", "#6A1B9A", "Chikou"),
        ]:
            if col in idf.columns:
                fig.add_trace(go.Scatter(
                    x=idf["date"], y=idf[col],
                    mode="lines", name=name,
                    line=dict(width=1.2, color=color),
                ))

    fig.update_layout(
        title="Candlestick + Ichimoku",
        xaxis_rangeslider_visible=False,
        height=520,
        template="plotly_white",
    )
    return _fig_html(fig)


def news_sentiment_chart(analyzed_items: list[dict]) -> str:
    labels = {"positive": 0, "neutral": 0, "negative": 0}
    for item in analyzed_items:
        lbl = item.get("sentiment", {}).get("label", "neutral")
        labels[lbl] = labels.get(lbl, 0) + 1

    total = sum(labels.values()) or 1
    pie_labels, pie_values, pie_colors = [], [], []
    color_map = {"positive": "#26a69a", "neutral": "#90a4ae", "negative": "#ef5350"}
    for key in ("positive", "neutral", "negative"):
        if labels[key] > 0:
            pie_labels.append(f"{key} ({labels[key]})")
            pie_values.append(labels[key])
            pie_colors.append(color_map[key])

    if not pie_values:
        pie_labels, pie_values, pie_colors = ["no data"], [1], ["#90a4ae"]

    fig = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.35,
        marker_colors=pie_colors,
        textinfo="percent",
    )])
    fig.update_layout(
        title=f"Sentiment distribution (n={total} headlines)",
        height=400,
        template="plotly_white",
    )
    return _fig_html(fig)


def volatility_chart(series: list[dict], forecast: dict) -> str:
    if not series:
        fig = go.Figure()
        fig.update_layout(title="No volatility data", height=480)
        return _fig_html(fig)

    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        subplot_titles=("Historical volatility (20-day)", "Forecast vs actual (in-sample)"))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["volatility"],
        mode="lines", name="Rolling volatility",
        line=dict(color="#1565C0"),
    ), row=1, col=1)

    hist = forecast.get("history", [])
    if hist:
        hdf = pd.DataFrame(hist)
        hdf["date"] = pd.to_datetime(hdf["date"])
        fig.add_trace(go.Scatter(
            x=hdf["date"], y=hdf["actual_volatility"],
            mode="lines+markers", name="Actual (next day)",
            line=dict(color="#43A047"),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=hdf["date"], y=hdf["predicted_volatility"],
            mode="lines+markers", name=f"Predicted ({forecast.get('model', '')})",
            line=dict(color="#EF6C00", dash="dash"),
        ), row=2, col=1)

    pred = forecast.get("predicted_volatility")
    fig.update_layout(
        title=f"Next-day volatility forecast: {pred}",
        height=560,
        template="plotly_white",
    )
    return _fig_html(fig)


def sentiment_contribution_chart(sentiment_value: dict | None) -> str:
    if not sentiment_value or not sentiment_value.get("available"):
        fig = go.Figure()
        fig.update_layout(
            title="Sentiment contribution unavailable (not enough news coverage in this period)",
            height=380,
            template="plotly_white",
        )
        return _fig_html(fig)

    hist = sentiment_value.get("history") or []
    if not hist:
        fig = go.Figure()
        fig.update_layout(title="No data", height=380, template="plotly_white")
        return _fig_html(fig)

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    price_label = sentiment_value.get("price_model_label", "Price-only")
    news_label = sentiment_value.get("news_model_label", "+ news sentiment")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["actual"],
        mode="lines", name="Actual volatility",
        line=dict(color="#1565C0", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price_only"],
        mode="lines", name=price_label,
        line=dict(color="#90A4AE", width=1.8, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["with_sentiment"],
        mode="lines", name=news_label,
        line=dict(color="#EF6C00", width=2.0, dash="dash"),
    ))

    turb = df[df["turbulent"]]
    if not turb.empty:
        fig.add_trace(go.Scatter(
            x=turb["date"], y=turb["actual"],
            mode="markers", name="Turbulent day (top 30% vol)",
            marker=dict(color="#C62828", size=7, symbol="circle-open"),
        ))

    fig.update_layout(
        title="News model vs price-only (test set)",
        xaxis_title="Date",
        yaxis_title="Next-day volatility",
        height=440,
        template="plotly_white",
    )
    return _fig_html(fig)


def backtest_charts(backtest: dict) -> tuple[str, str]:
    models = backtest.get("models") or []
    if not models:
        fig = go.Figure()
        fig.update_layout(title="No backtest data", height=400)
        return _fig_html(fig), _fig_html(fig)

    names = [m["model_label"] for m in models]
    rmse_vals = [m["rmse"] for m in models]
    colors = ["#43A047" if m.get("beats_baseline") else "#EF5350" for m in models]

    fig_bar = go.Figure(go.Bar(
        x=rmse_vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.6f}" for v in rmse_vals],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Model comparison (RMSE on test set, lower is better)",
        xaxis_title="RMSE",
        height=max(360, 40 * len(models)),
        template="plotly_white",
        margin=dict(l=220),
    )

    summary = backtest.get("summary") or {}
    best_key = summary.get("best_model_key", "")
    best = next((m for m in models if m["model_key"] == best_key), models[0])
    preds = best.get("predictions") or []
    if not preds:
        fig_line = go.Figure()
        fig_line.update_layout(title="No predictions", height=400)
        return _fig_html(fig_bar), _fig_html(fig_line)

    pdf = pd.DataFrame(preds)
    pdf["date"] = pd.to_datetime(pdf["date"])

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=pdf["date"], y=pdf["actual"],
        mode="lines+markers", name="Actual volatility",
        line=dict(color="#1565C0"),
    ))
    fig_line.add_trace(go.Scatter(
        x=pdf["date"], y=pdf["predicted"],
        mode="lines+markers", name=f"Predicted ({best['model_label']})",
        line=dict(color="#EF6C00", dash="dash"),
    ))
    fig_line.update_layout(
        title=f"Best model on test period: {best['model_label']}",
        xaxis_title="Date",
        yaxis_title="Next-day volatility",
        height=420,
        template="plotly_white",
    )
    return _fig_html(fig_bar), _fig_html(fig_line)


def backtest_overlay_chart(backtest: dict, model_keys: list[str] | None = None) -> str:
    models = backtest.get("models") or []
    if not models:
        fig = go.Figure()
        fig.update_layout(title="No data", height=400)
        return _fig_html(fig)

    summary = backtest.get("summary") or {}
    best_key = summary.get("best_model_key")
    best_news_key = summary.get("best_news_model_key")
    keys: list[str] = []
    if best_key:
        keys.append(best_key)
    if best_news_key and best_news_key not in keys:
        keys.append(best_news_key)
    if "baseline_persistence" not in keys:
        keys.append("baseline_persistence")
    if model_keys:
        keys = model_keys

    palette = ["#C62828", "#1565C0", "#EF5350", "#43A047", "#6A1B9A", "#FF6F00"]

    fig = go.Figure()
    first = next((m for m in models if m.get("predictions")), None)
    if first and first.get("predictions"):
        pdf = pd.DataFrame(first["predictions"])
        pdf["date"] = pd.to_datetime(pdf["date"])
        fig.add_trace(go.Scatter(
            x=pdf["date"], y=pdf["actual"],
            mode="lines", name="Actual",
            line=dict(color="#1565C0", width=2.5),
        ))

    for i, key in enumerate(keys):
        m = next((x for x in models if x["model_key"] == key), None)
        if not m or not m.get("predictions"):
            continue
        pdf = pd.DataFrame(m["predictions"])
        pdf["date"] = pd.to_datetime(pdf["date"])
        is_best = key == best_key
        is_best_news = key == best_news_key and key != best_key
        label = m["model_label"]
        if is_best:
            label = f"{label} (best overall)"
        elif is_best_news:
            label = f"{label} (best with news)"
        fig.add_trace(go.Scatter(
            x=pdf["date"], y=pdf["predicted"],
            mode="lines", name=label,
            line=dict(
                color=palette[i % len(palette)],
                width=3 if (is_best or is_best_news) else 1.8,
                dash="solid" if is_best else ("dashdot" if is_best_news else "dash"),
            ),
        ))

    fig.update_layout(
        title="Actual vs predicted (best overall and best with news)",
        height=460,
        template="plotly_white",
    )
    return _fig_html(fig)


def correlation_charts(correlation: dict) -> tuple[str, str, str]:
    aligned = correlation.get("aligned", [])
    if not aligned:
        empty = go.Figure()
        empty.update_layout(title="No aligned data", height=360)
        html = _fig_html(empty)
        return html, html, html

    df = pd.DataFrame(aligned)
    df["date"] = pd.to_datetime(df["date"])
    r = correlation.get("pearson_r")

    fig_sent = go.Figure(go.Scatter(
        x=df["date"], y=df["sentiment_score"],
        mode="lines+markers", name="Sentiment",
        line=dict(color="#26a69a"),
    ))
    fig_sent.update_layout(title="Daily sentiment", height=320, template="plotly_white")

    fig_vol = go.Figure(go.Scatter(
        x=df["date"], y=df["volatility"],
        mode="lines+markers", name="Volatility",
        line=dict(color="#1565C0"),
    ))
    fig_vol.update_layout(title="Rolling volatility", height=320, template="plotly_white")

    fig_scatter = go.Figure(go.Scatter(
        x=df["sentiment_score"], y=df["volatility"],
        mode="markers", name="Observations",
        marker=dict(size=10, color="#6A1B9A", opacity=0.7),
    ))
    title = f"Sentiment vs volatility (Pearson r = {r})" if r is not None else "Sentiment vs volatility"
    fig_scatter.update_layout(
        title=title,
        xaxis_title="Sentiment score",
        yaxis_title="Volatility",
        height=360,
        template="plotly_white",
    )

    return (
        _fig_html(fig_sent),
        _fig_html(fig_vol),
        _fig_html(fig_scatter),
    )
