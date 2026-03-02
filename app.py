from __future__ import annotations

from flask import Flask, render_template, request, jsonify

from services import get_moex_candles, InvestError

try:
    from app_secrets import TINKOFF_TOKEN
except Exception:
    TINKOFF_TOKEN = ""

app = Flask(__name__)


# @app.get("/")
# def index():
#     return render_template(
#         "index.html",
#         default_instrument_id="BBG004730N88",  # SBER FIGI из примера
#         default_days=10,
#         default_interval="4h",
#         sdk=sdk_name(),
#     )


@app.get("/")
def moex_page():
    return render_template("moex.html", default_sec="SBER", default_interval="24")


@app.get("/api/candles")
def moex_candles():
    sec = (request.args.get("sec") or "SBER").strip().upper()
    interval = (request.args.get("interval") or "24").strip()
    date_from = (request.args.get("from") or "").strip()
    date_till = (request.args.get("till") or "").strip()

    if not TINKOFF_TOKEN:
        return jsonify({"error": "Tinkoff token not configured", "hint": "Create app_secrets.py with TINKOFF_TOKEN"}), 500

    try:
        data = get_moex_candles(
            token=TINKOFF_TOKEN,
            sec=sec,
            interval=interval,
            date_from=date_from,
            date_till=date_till,
        )
        return jsonify(data)
    except InvestError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500


# @app.post("/run")
# def run():
#     instrument_id = (request.form.get("instrument_id") or "").strip()
#     days_back = int(request.form.get("days_back") or "10")
#     interval = (request.form.get("interval") or "4h").strip()

#     if not TINKOFF_TOKEN:
#         return (
#             render_template(
#                 "error.html",
#                 message="Не найден токен. Создайте файл app_secrets.py рядом с app.py и задайте TINKOFF_TOKEN.",
#             ),
#             500,
#         )

#     try:
#         candles = fetch_candles(
#             TINKOFF_TOKEN,
#             CandleRequest(instrument_id=instrument_id, days_back=days_back, interval=interval),
#         )
#         df = candles_to_dataframe(candles)
#         chart_uri = plot_candles_base64(df)
#         # покажем первые строки таблицы, чтобы не перегружать страницу
#         table_html = df.tail(30).to_html(classes="table", border=0)
#         return render_template(
#             "result.html",
#             instrument_id=instrument_id,
#             days_back=days_back,
#             interval=interval,
#             chart_uri=chart_uri,
#             table_html=table_html,
#             sdk=sdk_name(),
#             n=len(df),
#         )
#     except InvestError as e:
#         return render_template("error.html", message=str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)