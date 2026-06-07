from __future__ import annotations

import html
import json
import re
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from demo_data import demo_news

TICKER_META: dict[str, dict[str, str | list[str]]] = {
    "SBER": {
        "yfinance": "SBER.ME",
        "name": "Sberbank",
        "queries": ["Sberbank MOEX", "Sberbank Russia finance"],
        "newsapi_q": "Sberbank OR SBER MOEX",
    },
    "GAZP": {
        "yfinance": "GAZP.ME",
        "name": "Gazprom",
        "queries": ["Gazprom MOEX", "Gazprom Russia oil"],
        "newsapi_q": "Gazprom OR GAZP",
    },
    "LKOH": {
        "yfinance": "LKOH.ME",
        "name": "Lukoil",
        "queries": ["Lukoil MOEX", "Lukoil Russia oil"],
        "newsapi_q": "Lukoil OR LKOH",
    },
    "YNDX": {
        "yfinance": "YNDX",
        "name": "Yandex",
        "queries": ["Yandex stock", "Yandex Russia tech"],
        "newsapi_q": "Yandex stock",
    },
    "SPX": {
        "yfinance": "^GSPC",
        "name": "S&P 500",
        "queries": ['"S&P 500" stock market', "Wall Street volatility"],
        "newsapi_q": '"S&P 500" OR "stock market" OR SPX',
    },
    "GSPC": {
        "yfinance": "^GSPC",
        "name": "S&P 500",
        "queries": ['"S&P 500" index', "US stock market volatility"],
        "newsapi_q": '"S&P 500" OR "stock market" OR SPX',
    },
    "^GSPC": {
        "yfinance": "^GSPC",
        "name": "S&P 500",
        "queries": ['"S&P 500" index', "US stock market volatility"],
        "newsapi_q": '"S&P 500" OR "stock market" OR SPX',
    },
}

CACHE_DIR = Path(__file__).resolve().parent / "data" / "news_cache"

_session = requests.Session()
_retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; volatility-mvp/1.0; coursework)",
    "Accept": "application/rss+xml, application/xml, text/json, */*",
})

_JUNK_TITLE_RE = re.compile(
    r"^(main|home|news|top stories)\s*[-–|:]|"
    r"\b(odessa journal|policy analysis|center for european)\b",
    re.I,
)
_FINANCE_HINTS = frozenset({
    "stock", "stocks", "market", "markets", "earnings", "revenue", "profit",
    "index", "shares", "moex", "finance", "dividend", "trade", "investor",
    "bank", "oil", "gdp", "inflation", "fed", "rally", "volatility", "s&p",
    "nasdaq", "dow", "gazprom", "sberbank", "lukoil", "yandex",
})

GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"


def _newsapi_key() -> str:
    try:
        from app_secrets import NEWSAPI_KEY
        return (NEWSAPI_KEY or "").strip()
    except Exception:
        return ""


def _meta(ticker: str) -> dict:
    t = ticker.strip().upper()
    if t in TICKER_META:
        return {"ticker": t, **TICKER_META[t]}
    name = t.lstrip("^")
    return {
        "ticker": t,
        "yfinance": t,
        "name": name,
        "queries": [f"{name} stock finance", f"{name} market news"],
        "newsapi_q": name,
    }


def _parse_iso_date(raw: str) -> str:
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _item_date(item: dict) -> date | None:
    try:
        return datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")).date()
    except Exception:
        return None


def _clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_item(*, title: str, summary: str, source: str, published_at: str, url: str) -> dict[str, str]:
    return {
        "title": _clean_text(title)[:300],
        "summary": _clean_text(summary)[:500],
        "source": source,
        "published_at": published_at,
        "url": url.strip(),
    }


def _filter_by_range(items: list[dict], date_from: str, date_till: str) -> list[dict]:
    if not date_from or not date_till:
        return items
    try:
        frm = datetime.fromisoformat(date_from).date()
        till = datetime.fromisoformat(date_till).date()
    except Exception:
        return items
    out = []
    for item in items:
        d = _item_date(item)
        if d and frm <= d <= till:
            out.append(item)
    return out


def _relevance_score(item: dict, meta: dict) -> float:
    title = (item.get("title") or "").strip()
    if len(title) < 12:
        return -1.0
    if _JUNK_TITLE_RE.search(title):
        return -1.0
    blob = f"{title} {(item.get('summary') or '')}".lower()
    score = 0.0
    for token in (
        str(meta.get("name", "")).lower(),
        str(meta.get("ticker", "")).lower().lstrip("^"),
        str(meta.get("yfinance", "")).lower().replace(".me", ""),
    ):
        if token and len(token) >= 3 and token in blob:
            score += 4.0
    score += sum(0.5 for hint in _FINANCE_HINTS if hint in blob)
    if "Yahoo Finance" in item.get("source", "") and "RSS" not in item.get("source", ""):
        score += 2.0
    if "Google News" in item.get("source", ""):
        score += 1.0
    return score


def _filter_and_rank(items: list[dict], meta: dict, limit: int) -> list[dict]:
    scored = [(item, _relevance_score(item, meta)) for item in items]
    clean = [item for item, sc in scored if sc >= 0]
    if len(clean) < max(5, limit // 4):
        clean = [item for item, sc in scored if sc >= -0.5]
    clean.sort(key=lambda i: _relevance_score(i, meta), reverse=True)
    return clean[:limit]


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        key = re.sub(r"\s+", " ", item["title"].lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _month_chunks(date_from: str, date_till: str) -> list[tuple[str, str]]:
    start = datetime.fromisoformat(date_from).date()
    end = datetime.fromisoformat(date_till).date()
    chunks: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=31), end)
        chunks.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + timedelta(days=1)
    return chunks


def _fetch_rss_url(source_name: str, url: str, limit: int = 100) -> list[dict]:
    try:
        resp = _session.get(url, timeout=(5, 25))
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:
        return []

    out: list[dict] = []
    for entry in feed.entries[:limit]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        summary = re.sub(r"<[^>]+>", " ", str(summary))
        summary = re.sub(r"\s+", " ", summary).strip()
        out.append(_normalize_item(
            title=title,
            summary=summary,
            source=source_name,
            published_at=_parse_iso_date(entry.get("published") or entry.get("updated") or ""),
            url=(entry.get("link") or "").strip(),
        ))
    return out


def _fetch_google_dated(meta: dict, date_from: str, date_till: str) -> list[dict]:
    name = str(meta.get("name", ""))
    queries = list(meta.get("queries") or [f"{name} finance"])
    out: list[dict] = []

    for chunk_from, chunk_till in _month_chunks(date_from, date_till):
        before = (datetime.fromisoformat(chunk_till).date() + timedelta(days=1)).isoformat()
        for q in queries[:2]:
            dated_q = f'{q} after:{chunk_from} before:{before}'
            url = GOOGLE_RSS.format(query=requests.utils.quote(dated_q))
            out.extend(_fetch_rss_url(f"Google News ({chunk_from})", url, limit=100))

    market_q = f'"{name}" OR {meta["ticker"]} stock after:{date_from} before:{date_till}'
    url = GOOGLE_RSS.format(query=requests.utils.quote(market_q))
    out.extend(_fetch_rss_url("Google News (range)", url, limit=100))
    return out


def _fetch_newsapi(meta: dict, date_from: str, date_till: str, limit: int = 100) -> list[dict]:
    key = _newsapi_key()
    if not key:
        return []
    q = str(meta.get("newsapi_q") or meta.get("name", ""))
    try:
        resp = _session.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": q,
                "from": date_from,
                "to": date_till,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": key,
            },
            timeout=(5, 25),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    out: list[dict] = []
    for art in data.get("articles") or []:
        title = (art.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        out.append(_normalize_item(
            title=title,
            summary=(art.get("description") or art.get("content") or "")[:500],
            source=f"NewsAPI ({art.get('source', {}).get('name', 'news')})",
            published_at=_parse_iso_date(art.get("publishedAt") or ""),
            url=(art.get("url") or "").strip(),
        ))
    return out


def _fetch_yfinance_news(ticker: str, limit: int) -> list[dict]:
    try:
        import yfinance as yf
    except Exception:
        return []
    meta = _meta(ticker)
    symbol = str(meta.get("yfinance") or ticker)
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        return []

    out: list[dict] = []
    for item in raw[:limit]:
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (content.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or content.get("description") or "").strip()
        pub = content.get("pubDate") or content.get("displayTime") or ""
        url = ""
        for key in ("clickThroughUrl", "canonicalUrl"):
            block = content.get(key) or {}
            if isinstance(block, dict) and block.get("url"):
                url = block["url"]
                break
        provider = content.get("provider") or {}
        source_name = provider.get("displayName") or "Yahoo Finance"
        out.append(_normalize_item(
            title=title,
            summary=summary,
            source=f"Yahoo Finance ({source_name})",
            published_at=_parse_iso_date(pub),
            url=url,
        ))
    return out


def _cache_path(ticker: str, date_from: str, date_till: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", ticker)
    return CACHE_DIR / f"{safe}_{date_from}_{date_till}.json"


def _load_cache(ticker: str, date_from: str, date_till: str) -> list[dict] | None:
    path = _cache_path(ticker, date_from, date_till)
    if not path.exists():
        return None
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    for it in items:
        it["title"] = _clean_text(it.get("title", ""))[:300]
        it["summary"] = _clean_text(it.get("summary", ""))[:500]
    return items


def _save_cache(ticker: str, date_from: str, date_till: str, items: list[dict]) -> None:
    path = _cache_path(ticker, date_from, date_till)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")


def _subsample_for_sentiment(items: list[dict], max_items: int = 500) -> list[dict]:
    if len(items) <= max_items:
        return items
    by_date: dict[date, list[dict]] = {}
    for item in items:
        d = _item_date(item)
        if d:
            by_date.setdefault(d, []).append(item)
    if not by_date:
        return items[:max_items]

    dates = sorted(by_date.keys())
    n_days = len(dates)
    per_day = max(1, max_items // n_days)
    out: list[dict] = []
    for d in dates:
        out.extend(by_date[d][:per_day])
    if len(out) < max_items:
        seen = {id(i) for i in out}
        for d in dates:
            for item in by_date[d][per_day:]:
                if id(item) in seen:
                    continue
                out.append(item)
                seen.add(id(item))
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
    return out[:max_items]


def fetch_news(
    ticker: str = "SBER",
    limit: int = 80,
    *,
    date_from: str = "",
    date_till: str = "",
    include_historical: bool = True,
) -> dict:
    raw_ticker = ticker.strip() or "SBER"
    ticker = raw_ticker.upper() if not raw_ticker.startswith("^") else raw_ticker[:1] + raw_ticker[1:].upper()
    meta = _meta(ticker)
    recent_items: list[dict] = []
    historical_items: list[dict] = []
    sources_used: set[str] = set()

    yf_items = _fetch_yfinance_news(ticker, min(limit, 30))
    recent_items.extend(yf_items)
    if yf_items:
        sources_used.update(i["source"] for i in yf_items)

    symbol = str(meta.get("yfinance") or ticker)
    recent_items.extend(_fetch_rss_url("Yahoo Finance RSS", YAHOO_RSS.format(symbol=symbol), limit=20))
    recent_items = _dedupe(recent_items)

    if include_historical and date_from and date_till:
        cached = _load_cache(ticker, date_from, date_till)
        if cached:
            historical_items = cached
            sources_used.add("Cache (historical)")
        else:
            historical_items.extend(_fetch_google_dated(meta, date_from, date_till))
            historical_items.extend(_fetch_newsapi(meta, date_from, date_till, limit=100))
            historical_items = _dedupe(historical_items)
            historical_items = _filter_by_range(historical_items, date_from, date_till)
            if historical_items:
                _save_cache(ticker, date_from, date_till, historical_items)
            for i in historical_items:
                sources_used.add(i["source"])

    recent_in_range = _filter_by_range(recent_items, date_from, date_till)
    items_in_range = _dedupe(historical_items + recent_in_range)
    if not items_in_range:
        items_in_range = recent_in_range
    display_items = _dedupe(recent_items + _subsample_for_sentiment(historical_items, 40))
    display_items = _filter_and_rank(display_items, meta, min(limit, 60))
    display_items.sort(key=lambda x: x["published_at"], reverse=True)

    if display_items or historical_items:
        mode = "historical" if historical_items else "live"
        return {
            "items": display_items,
            "recent_items": recent_items,
            "historical_items": historical_items,
            "items_in_range": items_in_range,
            "source_mode": mode,
            "ticker": ticker,
            "sources_used": sorted(sources_used) or list({i["source"] for i in display_items}),
            "historical_count": len(historical_items),
        }

    return {
        "items": demo_news(ticker, min(limit, 12)),
        "recent_items": demo_news(ticker, min(limit, 12)),
        "historical_items": [],
        "items_in_range": demo_news(ticker, min(limit, 12)),
        "source_mode": "demo",
        "ticker": ticker,
        "sources_used": ["Demo Feed"],
        "historical_count": 0,
    }
