from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).resolve().parent / "data" / "sentiment_cache"
_CACHE_FILE = _CACHE_DIR / "finbert_scores.json"
_score_cache: dict[str, dict[str, Any]] | None = None


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def _load_score_cache() -> dict[str, dict[str, Any]]:
    global _score_cache
    if _score_cache is None:
        try:
            _score_cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _score_cache = {}
    return _score_cache


def _save_score_cache() -> None:
    if _score_cache is None:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_score_cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

_POSITIVE = {
    "gain", "gains", "growth", "profit", "profits", "bullish", "rally", "upgrade",
    "strong", "beat", "beats", "surge", "surges", "rise", "rises", "rising",
    "positive", "optimistic", "outperform", "record", "recovery", "rebound",
    "higher", "up", "soar", "jump", "jumps", "boost", "expansion",
}
_NEGATIVE = {
    "loss", "losses", "decline", "declines", "fall", "falls", "drop", "drops",
    "bearish", "downgrade", "weak", "miss", "misses", "crash", "crisis",
    "negative", "pessimistic", "underperform", "slump", "recession", "risk",
    "volatility", "uncertainty", "concern", "concerns", "pressure",
    "lower", "down", "plunge", "sink", "sinks", "attack", "sanctions", "war",
}

_finbert_pipeline = None
_finbert_checked = False


def _load_finbert():
    global _finbert_pipeline, _finbert_checked
    if _finbert_checked:
        return _finbert_pipeline
    _finbert_checked = True
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    except Exception:
        _finbert_pipeline = None
    return _finbert_pipeline


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def _dictionary_sentiment(text: str) -> dict[str, Any]:
    tokens = _tokenize(text)
    pos = len(tokens & _POSITIVE)
    neg = len(tokens & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return {
            "label": "neutral",
            "positive": 0.33,
            "neutral": 0.34,
            "negative": 0.33,
            "confidence": 0.34,
            "sentiment_score": 0.0,
            "method": "dictionary",
        }
    score = (pos - neg) / total
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"
    conf = min(0.95, 0.5 + abs(score) * 0.4)
    pos_p = max(0.0, (score + 1) / 2)
    neg_p = max(0.0, (-score + 1) / 2)
    neu_p = max(0.0, 1.0 - pos_p - neg_p)
    return {
        "label": label,
        "positive": round(pos_p, 4),
        "neutral": round(neu_p, 4),
        "negative": round(neg_p, 4),
        "confidence": round(conf, 4),
        "sentiment_score": round(max(-1.0, min(1.0, score)), 4),
        "method": "dictionary",
    }


def _finbert_sentiment(text: str) -> dict[str, Any] | None:
    pipe = _load_finbert()
    if pipe is None:
        return None
    try:
        snippet = text[:512]
        raw = pipe(snippet, top_k=3, truncation=True)
        probs = {r["label"].lower(): float(r["score"]) for r in raw}
        pos_p = probs.get("positive", 0.0)
        neg_p = probs.get("negative", 0.0)
        neu_p = probs.get("neutral", max(0.0, 1.0 - pos_p - neg_p))
        score = pos_p - neg_p
        conf = max(pos_p, neg_p, neu_p)
        if score > 0.08:
            label = "positive"
        elif score < -0.08:
            label = "negative"
        else:
            label = "neutral"
        return {
            "label": label,
            "positive": round(pos_p, 4),
            "neutral": round(neu_p, 4),
            "negative": round(neg_p, 4),
            "confidence": round(conf, 4),
            "sentiment_score": round(max(-1.0, min(1.0, score)), 4),
            "method": "finbert",
        }
    except Exception:
        return None


def analyze_text(text: str, *, use_cache: bool = True) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return _dictionary_sentiment("")
    if use_cache:
        cache = _load_score_cache()
        key = _cache_key(text)
        cached = cache.get(key)
        if cached is not None:
            return cached
    finbert = _finbert_sentiment(text)
    res = finbert if finbert is not None else _dictionary_sentiment(text)
    if use_cache and res.get("method") == "finbert":
        _load_score_cache()[_cache_key(text)] = res
    return res


def analyze_news_items(items: list[dict]) -> dict[str, Any]:
    analyzed = []
    new_finbert = False
    for item in items:
        blob = f"{item.get('title', '')}. {item.get('summary', '')}"
        sentiment = analyze_text(blob)
        if sentiment.get("method") == "finbert":
            new_finbert = True
        analyzed.append({**item, "sentiment": sentiment})
    if new_finbert:
        _save_score_cache()
    method = "finbert" if any(a["sentiment"]["method"] == "finbert" for a in analyzed) else "dictionary"
    return {"items": analyzed, "method": method, "count": len(analyzed)}
