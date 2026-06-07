from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flask import Flask, render_template, request, jsonify

from chart_service import (
    backtest_charts,
    backtest_overlay_chart,
    correlation_charts,
    market_chart,
    news_sentiment_chart,
    sentiment_contribution_chart,
    volatility_chart,
)
from backtest_service import run_volatility_backtest, scan_sentiment_periods
from pipeline_service import default_date_range, load_candles, load_full_context, load_market_context
from news_service import fetch_news
from sentiment_service import analyze_news_items, analyze_text

try:
    from app_secrets import TINKOFF_TOKEN
except Exception:
    TINKOFF_TOKEN = ""

app = Flask(__name__)


def _params() -> tuple[str, str, str]:
    raw = (request.args.get("sec") or request.args.get("ticker") or "SBER").strip()
    sec = raw.upper() if not raw.startswith("^") else raw[:1] + raw[1:].upper()
    date_from = (request.args.get("from") or "").strip()
    date_till = (request.args.get("till") or "").strip()
    if not date_from or not date_till:
        date_from, date_till = default_date_range()
    return sec, date_from, date_till


@app.get("/")
def index():
    return market_page()


@app.get("/market")
def market_page():
    sec, date_from, date_till = _params()
    ctx = load_market_context(TINKOFF_TOKEN, sec, date_from, date_till)
    chart_html = market_chart(ctx["candles"], ctx["analytics"]["ichimoku"])
    return render_template(
        "market.html",
        title="Market",
        sec=sec,
        date_from=date_from,
        date_till=date_till,
        chart_html=chart_html,
        candle_source=ctx["candle_source"],
        n_candles=len(ctx["candles"]),
    )


@app.get("/news")
def news_page():
    sec, date_from, date_till = _params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    chart_html = news_sentiment_chart(ctx["sentiment_items"])
    items_sorted = sorted(
        ctx["sentiment_items"],
        key=lambda it: it.get("published_at") or "",
        reverse=True,
    )
    return render_template(
        "news.html",
        title="News",
        sec=sec,
        date_from=date_from,
        date_till=date_till,
        items=items_sorted,
        chart_html=chart_html,
        sentiment_method=ctx["sentiment_method"],
        news_source_mode=ctx["news_source_mode"],
        news_date_overlap=ctx.get("news_date_overlap", False),
        news_in_range_count=ctx.get("news_in_range_count", 0),
    )


@app.get("/volatility")
def volatility_page():
    sec, date_from, date_till = _params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    chart_html = volatility_chart(ctx["analytics"]["series"], ctx["forecast"])
    sentiment_chart_html = sentiment_contribution_chart(ctx["forecast"].get("sentiment_value"))
    return render_template(
        "volatility.html",
        title="Volatility",
        sec=sec,
        date_from=date_from,
        date_till=date_till,
        chart_html=chart_html,
        sentiment_chart_html=sentiment_chart_html,
        forecast=ctx["forecast"],
    )


def _eval_params() -> tuple[str, str, str]:
    raw = (request.args.get("sec") or request.args.get("ticker") or "^GSPC").strip()
    sec = raw.upper() if not raw.startswith("^") else raw[:1] + raw[1:].upper()
    date_from = (request.args.get("from") or "").strip()
    date_till = (request.args.get("till") or "").strip()
    if not date_from or not date_till:
        till = datetime.now(timezone.utc).date()
        frm = till - timedelta(days=365)
        date_from, date_till = frm.isoformat(), till.isoformat()
    return sec, date_from, date_till


@app.get("/evaluation")
def evaluation_page():
    sec, date_from, date_till = _eval_params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    backtest = run_volatility_backtest(
        ctx["candles"],
        ctx["analytics"].get("sentiment_df"),
    )
    bar_html, best_html = backtest_charts(backtest)
    overlay_html = backtest_overlay_chart(backtest)

    sentiment_scan: list[dict] = []
    if request.args.get("scan") == "1":
        def _scan_loader(ticker: str, frm: str, till: str) -> dict:
            return load_full_context(TINKOFF_TOKEN, ticker, frm, till)

        sentiment_scan = scan_sentiment_periods(_scan_loader, sec, top_n=3)

    return render_template(
        "evaluation.html",
        title="Evaluation",
        sec=sec,
        date_from=date_from,
        date_till=date_till,
        backtest=backtest,
        candle_source=ctx["candle_source"],
        n_candles=len(ctx["candles"]),
        sentiment_method=ctx["sentiment_method"],
        sentiment_scan=sentiment_scan,
        bar_chart_html=bar_html,
        best_chart_html=best_html,
        overlay_chart_html=overlay_html,
    )


@app.get("/correlation")
def correlation_page():
    sec, date_from, date_till = _params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    corr = ctx["analytics"]["correlation"]
    sent_html, vol_html, scatter_html = correlation_charts(corr)
    return render_template(
        "correlation.html",
        title="Correlation",
        sec=sec,
        date_from=date_from,
        date_till=date_till,
        correlation=corr,
        sent_chart_html=sent_html,
        vol_chart_html=vol_html,
        scatter_chart_html=scatter_html,
    )


@app.get("/api/candles")
def api_candles():
    sec = (request.args.get("sec") or "SBER").strip().upper()
    date_from = (request.args.get("from") or "").strip()
    date_till = (request.args.get("till") or "").strip()
    bundle = load_candles(TINKOFF_TOKEN, sec, date_from, date_till)
    return jsonify({
        "security": bundle["security"],
        "candles": bundle["candles"],
        "source": bundle["source"],
    })


@app.get("/api/news")
def api_news():
    sec = (request.args.get("sec") or request.args.get("ticker") or "SBER").strip().upper()
    limit = int(request.args.get("limit") or "30")
    date_from = (request.args.get("from") or "").strip()
    date_till = (request.args.get("till") or "").strip()
    return jsonify(fetch_news(sec, limit=limit, date_from=date_from, date_till=date_till))


@app.get("/api/sentiment")
def api_sentiment():
    text = (request.args.get("text") or "").strip()
    sec = (request.args.get("sec") or "SBER").strip().upper()
    if text:
        return jsonify(analyze_text(text))
    bundle = fetch_news(sec)
    res = analyze_news_items(bundle["items"])
    return jsonify(res)


@app.get("/api/volatility")
def api_volatility():
    sec, date_from, date_till = _params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    return jsonify({
        "security": sec,
        "series": ctx["analytics"]["series"],
        "candle_source": ctx["candle_source"],
    })


@app.get("/api/correlation")
def api_correlation():
    sec, date_from, date_till = _params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    return jsonify({
        "security": sec,
        "correlation": ctx["analytics"]["correlation"],
        "daily_sentiment": ctx["analytics"]["daily_sentiment"],
    })


@app.get("/api/forecast")
def api_forecast():
    sec, date_from, date_till = _params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    return jsonify({
        "security": sec,
        "forecast": ctx["forecast"],
    })


@app.get("/api/backtest")
def api_backtest():
    sec, date_from, date_till = _eval_params()
    ctx = load_full_context(TINKOFF_TOKEN, sec, date_from, date_till)
    backtest = run_volatility_backtest(
        ctx["candles"],
        ctx["analytics"].get("sentiment_df"),
    )

    slim = {**backtest, "models": [
        {k: v for k, v in m.items() if k != "predictions"}
        for m in backtest.get("models", [])
    ]}
    return jsonify({
        "security": sec,
        "date_from": date_from,
        "date_till": date_till,
        "candle_source": ctx["candle_source"],
        "backtest": slim,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
