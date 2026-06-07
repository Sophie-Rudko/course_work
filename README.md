# Market Volatility Forecasting from Investor Sentiment

This is my university coursework project. It is a small web app (Flask + Plotly) that
collects financial news, reads how positive or negative they are with a sentiment model,
measures how much the market is moving (volatility), shows the link between news mood and
volatility, and predicts tomorrow's volatility of the S&P 500 with machine-learning models.

It is a coursework MVP, not a real trading system. The goal is to be stable, easy to run,
and easy to explain, so I prefer simple, well-understood models over very heavy ones.

## What you need (requirements)

You need Python 3.10 or newer. Everything else is installed from `requirements.txt`:

- Flask, the web framework that serves the pages and the API
- pandas and numpy for working with the price and news data
- scikit-learn and scipy for the machine-learning models and statistics
- plotly for the interactive charts
- feedparser, yfinance and requests to download news and prices
- pytest to run the tests

The sentiment model FinBERT is optional. If the `transformers` and `torch` libraries are
installed, the app uses FinBERT; if not, it falls back to a simple word-list sentiment, so
the app always runs. To turn FinBERT on, install them yourself:

```bash
pip install transformers torch
```

## Setup

Create a virtual environment and install the packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows activate the environment with `.venv\Scripts\activate` instead.

Market-data and NewsAPI tokens are optional. They live in `app_secrets.py`. If you leave
them empty, the app still works: it downloads prices from yfinance and falls back to demo
data when a source is not reachable.

## Run

Start the app:

```bash
python app.py
```

Then open http://127.0.0.1:5001/market in your browser.

The app uses port 5001 on purpose, because on macOS port 5000 is often taken by the AirPlay
Receiver.

## Pages

The dashboard has five pages. On every page you can set the ticker and the date range and
press Load.

- **Market** shows the price as a candlestick chart together with the Ichimoku Cloud
  indicator, so you can see the trend at a glance.
- **News** lists the collected headlines with their sentiment (positive, neutral or
  negative) and shows how the sentiment is distributed.
- **Volatility** draws the 20-day rolling volatility and the next-day forecast from the best
  model found in the backtest. It also shows whether adding news actually helps the forecast.
- **Correlation** shows the news sentiment and the volatility over time, a scatter plot of the
  two, and the Pearson correlation with its p-value. This page is pure statistics, no model.
- **Evaluation** runs a historical backtest: it trains every model on the first 80% of the
  days and tests them on the last 20%, then compares their errors (RMSE and MAE) and checks
  if the news features help.

## API

Besides the pages, the app exposes a small JSON API. It is handy for quick checks or for
plugging the data into another tool. Each endpoint takes a ticker (`sec`) and an optional
date range (`from`, `till`).

- `GET /api/candles` returns the OHLC price candles.
- `GET /api/news` returns the collected news for a ticker and date range.
- `GET /api/sentiment` returns the sentiment of the news, or of any text you pass as `?text=`.
- `GET /api/volatility` returns the log returns and the 20-day rolling volatility.
- `GET /api/correlation` returns the Pearson correlation between sentiment and volatility.
- `GET /api/forecast` returns the next-day volatility forecast.
- `GET /api/backtest` returns the metrics of all models from the backtest.

## Tests

The unit tests cover the core math: log returns, rolling volatility, sentiment scoring,
daily aggregation, Pearson correlation, feature engineering and the backtest. Run them with:

```bash
python -m pytest test_modules.py -q
```

## How the code is organized

The project follows the MVC idea:

- `app.py` is the controller. It defines the routes and the API and decides which template
  to show.
- the `*_service.py` files are the model layer. `pipeline_service.py` ties everything
  together; `news_service.py` collects news; `sentiment_service.py` scores it with FinBERT;
  `analytics_service.py` computes volatility, Ichimoku and the correlation;
  `forecast_service.py` and `backtest_service.py` build the features and the ML models;
  `chart_service.py` builds the Plotly charts; `market_data.py` and `services.py` download
  prices; `demo_data.py` provides fallback data.
- the `templates/` folder is the view: the five HTML pages of the dashboard.
