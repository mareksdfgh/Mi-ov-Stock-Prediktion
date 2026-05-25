# Mičov Stock Prediktion 📈

A machine learning pipeline that predicts hourly stock price movements for multiple tickers simultaneously. Given live market data, it outputs a **BUY / HOLD / SELL** signal plus a predicted price change for the next 4 hours.

> ⚠️ **Disclaimer:** This project is for educational purposes only. Nothing here constitutes financial advice.

---

## How it works

1. **Data** — Downloads up to 730 days of hourly OHLCV data via `yfinance` (free, no API key needed). Also pulls macro context: S&P 500, NASDAQ, VIX, Gold, Oil, DXY, bonds, and 5 sector ETFs.
2. **Features** — ~60 technical indicators per ticker: RSI (7/14/21), MACD, Bollinger Band Z-score, Stochastic RSI, ATR, momentum, moving average distances, volume spikes, and time features (hour, weekday, month). Look-ahead leakage is prevented via 1-step lags on market data.
3. **Model** — Stacking ensemble (XGBoost + HistGradientBoosting → LogisticRegression/Ridge meta-learner). Two outputs: classification (BUY/HOLD/SELL) and regression (predicted % return).
4. **Prediction** — Downloads fresh live data, aligns features to the trained schema, and outputs a signal with confidence probabilities + price forecast chart.

---

## Project structure

```
├── build_dataset.py   # Download data + build dataset.csv
├── features.py        # Shared technical indicator calculations
├── train.py           # Train classification + regression models
├── run.py             # Live prediction + charts
├── requirements.txt
```

---

## Installation & Usage

### 🍎 macOS

**1. Clone the repo**
```bash
git clone https://github.com/your-username/micov-stock-prediktion.git
cd micov-stock-prediktion
```

**2. Install dependencies**
```bash
pip3 install -r requirements.txt
```

> If XGBoost fails on macOS, install the required native library first:
> ```bash
> brew install libomp
> pip install xgboost
> ```

**3. Build the dataset**
```bash
# Default tickers: AAPL MSFT GOOGL NVDA TSLA AMZN META NFLX AMD INTC
python build_dataset.py

# Custom tickers
python build_dataset.py --tickers AAPL MSFT GOOGL --output dataset.csv
```

**4. Train the model**
```bash
python train.py --data dataset.csv

# Without XGBoost (HistGradientBoosting only)
python train.py --data dataset.csv --no_xgb
```

**5. Run a live prediction**
```bash
python run.py --ticker AAPL

# Text output only (no charts)
python run.py --ticker AAPL --no_chart

# Custom chart window
python run.py --ticker MSFT --lookback 200
```

---

### 🪟 Windows

**1. Clone the repo**
```cmd
git clone https://github.com/your-username/micov-stock-prediktion.git
cd micov-stock-prediktion
```

**2. Install dependencies**
```cmd
pip install -r requirements.txt
```

> XGBoost works on Windows without extra steps. If you see a DLL error, install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

**3. Build the dataset**
```cmd
python build_dataset.py

:: Custom tickers
python build_dataset.py --tickers AAPL MSFT GOOGL --output dataset.csv
```

**4. Train the model**
```cmd
python train.py --data dataset.csv

:: Without XGBoost
python train.py --data dataset.csv --no_xgb
```

**5. Run a live prediction**
```cmd
python run.py --ticker AAPL

:: Text output only
python run.py --ticker AAPL --no_chart

:: Custom chart window
python run.py --ticker MSFT --lookback 200
```

---

## Output example

```
==================================================
  Ticker:            AAPL
  Timestamp:         2025-05-24 19:00:00
  Current price:     189.43$
  ─────────────────────────────────────────
  Signal:            BUY
  Confidence:        61.2%
  SELL / HOLD / BUY: 14.3% / 24.5% / 61.2%
  ─────────────────────────────────────────
  Forecast +4h:      191.12$ (+0.89%)
==================================================
```

Two charts are saved automatically:
- `chart_signals_AAPL.png` — price history with BUY/SELL markers + RSI panel
- `chart_prediction_AAPL.png` — current price + 4h forecast arrow

---

## CLI reference

| Script | Key arguments |
|---|---|
| `build_dataset.py` | `--tickers AAPL MSFT ...` `--output dataset.csv` |
| `train.py` | `--data dataset.csv` `--test_ratio 0.15` `--output_dir .` `--no_xgb` |
| `run.py` | `--ticker AAPL` `--model_dir .` `--lookback 200` `--no_chart` |

---

## Requirements

- Python 3.10+
- Internet connection (live data via yfinance)
- ~2–5 min for dataset build, ~5–15 min for training (depends on hardware)
