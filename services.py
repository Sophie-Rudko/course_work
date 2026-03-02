from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
import base64

import pandas as pd

try:
    from t_tech.invest import Client, CandleInterval
    _HAS_TINVEST = True
except Exception:
    _HAS_TINVEST = False

import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class InvestError(Exception):
    pass

def sdk_name() -> str:
    return "tinkoff.invest" if _HAS_TINVEST else "tinkoff.invest (not installed)"

@dataclass(frozen=True)
class CandleRequest:
    instrument_id: str
    days_back: int = 10
    interval: str = "4h"

_INTERVAL_MAP: dict[str, "CandleInterval"] = {}
if _HAS_TINVEST:
    _INTERVAL_MAP = {
        "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
        "4h": CandleInterval.CANDLE_INTERVAL_4_HOUR,
        "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    }

_MOEX_INTERVAL_MAP: dict[str, "CandleInterval"] = {}
if _HAS_TINVEST:
    _MOEX_INTERVAL_MAP = {
        "1": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "10": CandleInterval.CANDLE_INTERVAL_10_MIN,
        "60": CandleInterval.CANDLE_INTERVAL_HOUR,
        "24": CandleInterval.CANDLE_INTERVAL_DAY,
    }

_MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{sec}/candles.json"
_moex_session = requests.Session()
_retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
_moex_session.mount("https://", HTTPAdapter(max_retries=_retry))
_moex_session.headers.update({"User-Agent": "simpleflask-moex/1.0"})


def _get_moex_candles_iss(sec: str, interval: str, date_from: str, date_till: str) -> dict:
    """Fallback: fetch candles from MOEX ISS (no token needed)."""
    def _to_unix(dt_str: str) -> int:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    params = {"from": date_from, "till": date_till, "interval": interval, "iss.meta": "off", "iss.only": "candles"}
    r = _moex_session.get(_MOEX_URL.format(sec=sec), params=params, timeout=(5, 60))
    r.raise_for_status()
    payload = r.json()
    candles = payload.get("candles", {})
    columns = candles.get("columns", [])
    rows = candles.get("data", [])
    idx = {name: i for i, name in enumerate(columns)}
    out = []
    if rows and "begin" in idx:
        for row in rows:
            out.append({
                "time": _to_unix(row[idx["begin"]]),
                "open": row[idx["open"]], "high": row[idx["high"]],
                "low": row[idx["low"]], "close": row[idx["close"]],
            })
    return {"security": sec, "interval": interval, "candles": out, "source": "MOEX ISS"}


def get_moex_candles(token: str, sec: str, interval: str, date_from: str, date_till: str) -> dict:
    """Fetch MOEX candles. Try Tinkoff first; fallback to MOEX ISS if no data."""
    if not date_from or not date_till:
        raise InvestError("date_from and date_till are required")

    from datetime import datetime as dt
    now_utc = dt.now(timezone.utc)
    frm = dt.fromisoformat(date_from).replace(tzinfo=timezone.utc)
    till = dt.fromisoformat(date_till).replace(tzinfo=timezone.utc)
    if frm > now_utc or till > now_utc:
        raise InvestError("Use past dates only. Future dates have no historical data.")
    if token and _HAS_TINVEST:
        tinkoff_interval = _MOEX_INTERVAL_MAP.get(interval)
        if tinkoff_interval:
            till_adj = till.replace(hour=23, minute=59, second=59) if (till.hour == 0 and till.minute == 0) else till
            try:
                with Client(token) as client:
                    resp = client.instruments.find_instrument(query=sec)
                    figi = None
                    for inv in resp.instruments:
                        if inv.ticker and inv.ticker.upper() == sec.upper():
                            class_code = getattr(inv, "class_code", "") or ""
                            if class_code == "TQBR":
                                figi = inv.figi
                                break
                            if figi is None:
                                figi = inv.figi
                    if figi:
                        cr = client.market_data.get_candles(figi=figi, from_=frm, to=till_adj, interval=tinkoff_interval)
                        out = []
                        for c in cr.candles:
                            out.append({
                                "time": int(c.time.timestamp()),
                                "open": float(c.open.units) + float(c.open.nano) / 1e9,
                                "high": float(c.high.units) + float(c.high.nano) / 1e9,
                                "low": float(c.low.units) + float(c.low.nano) / 1e9,
                                "close": float(c.close.units) + float(c.close.nano) / 1e9,
                            })
                        if out:
                            return {"security": sec, "interval": interval, "candles": out, "source": "Tinkoff"}
            except Exception:
                pass
    try:
        data = _get_moex_candles_iss(sec=sec, interval=interval, date_from=date_from, date_till=date_till)
        if data["candles"]:
            return data
    except Exception:
        pass

    return {"security": sec, "interval": interval, "candles": []}