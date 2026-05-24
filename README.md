# Mičov Stock Prediktion

ML-powered hourly stock signal scanner. Predicts **BUY / HOLD / SELL** + price forecast for the next 4 hours using an ensemble of XGBoost and HistGradientBoosting.

---

## Architecture

```
features.py       ← shared feature engineering (single source of truth)
build_dataset.py  ← downloads data, builds dataset.csv
train.py          ← trains the model ensemble
run.py            ← CLI: live prediction + charts
run_ui.py         ← Tkinter GUI: multi-ticker scanner
tracker.py        ← SQLite accuracy tracking (1h, 4h)
```

| File | Purpose |
|---|---|
| `features.py` | All technical indicators — imported by every other file |
| `build_dataset.py` | Downloads hourly + daily data for multiple tickers, computes ~150 features, saves `dataset.csv` |
| `train.py` | Trains stacking ensemble (XGB + HGB), saves model files |
| `run.py` | CLI: live prediction + charts for a single ticker |
| `run_ui.py` | Tkinter GUI: multi-ticker scanner with embedded charts |
| `tracker.py` | SQLite-based prediction accuracy tracker (1h, 4h, chart similarity) |

---

## Labels

| Label | Meaning | Condition |
|---|---|---|
| BUY | 2 | future_return > +0.3% in 4h |
| HOLD | 1 | future_return between ±0.3% |
| SELL | 0 | future_return < -0.3% in 4h |

---

## Installation

### Windows 10 / 11

Open the folder in your terminal (right-click → "Open in Terminal" or `cd` to it), then:

```bat
pip install -r requirements.txt
```

> **Troubleshooting Windows:**
> - If `tkinter` is missing: reinstall Python from [python.org](https://python.org) and check **"tcl/tk and IDLE"** during setup.
> - If `xgboost` install fails: `pip install xgboost --pre` or run with `--no_xgb` flag.
> - If matplotlib shows no window: `pip install pyqt5`

---

### macOS (Intel + Apple Silicon)

Open Terminal, `cd` into the project folder, then:

```bash
pip3 install -r requirements.txt
brew install libomp
```

> **Troubleshooting macOS:**
> - `OMP: Error #15` → `brew install libomp` fixes it.
> - No Homebrew? Install it: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
> - `tkinter` missing (Homebrew Python): `brew install python-tk@3.11` (adjust version).
> - Without libomp: use `--no_xgb` — HGB-only mode works everywhere without any brew dependency.

---

## Usage

### Step 1 — Build Dataset

```bash
# Default: 10 tickers (AAPL MSFT GOOGL NVDA TSLA AMZN META NFLX AMD INTC)
python build_dataset.py

# Custom tickers
python build_dataset.py --tickers AAPL MSFT NVDA --output dataset.csv
```

**Output files:**
- `dataset.csv` — main dataset (~50k+ rows)
- `dataset_info.csv` — feature statistics
- `dataset_tickers.json` — ticker ID mapping

> Build takes **5–15 minutes** depending on internet speed. yfinance has rate limits — if downloads fail mid-way, wait a minute and retry.

---

### Step 2 — Train Model

```bash
# With XGBoost (recommended, needs libomp on macOS)
python train.py --data dataset.csv

# Without XGBoost (works everywhere, no brew needed)
python train.py --data dataset.csv --no_xgb
```

**Output files:**
- `model_cls.joblib` — classification model
- `model_reg.joblib` — regression model
- `model_meta.json` — feature list, medians, config

> Stacking with CV=5 takes **10–30 minutes**. Normal — go get a coffee.

---

### Step 3 — Run Predictions

**CLI:**
```bash
# Live prediction + charts
python run.py --ticker AAPL

# Text only (no charts)
python run.py --ticker AAPL --no_chart

# Custom chart window (hours)
python run.py --ticker MSFT --lookback 300
```

**GUI:**
```bash
python run_ui.py

# Custom tickers and model path
python run_ui.py --tickers AAPL MSFT NVDA TSLA --model_dir .
```

---

## Prediction Tracking

Predictions are automatically saved to `micov_tracker.db` (SQLite). The tracker evaluates accuracy after 1h and 4h by re-fetching the actual price from yfinance.

```bash
# Quick stats via SQLite directly:
sqlite3 micov_tracker.db \
  "SELECT ticker, AVG(direction_ok_4h) FROM predictions WHERE evaluated_4h=1 GROUP BY ticker;"
```

The GUI shows full accuracy stats in the **Accuracy Tracker** tab.

---

## Output Example

```
==================================================
  Ticker:            AAPL
  Časový okamžik:    2024-11-15 15:00:00
  Aktuální kurz:     228.52$
  ─────────────────────────────────────────
  Signál:            BUY
  Spolehlivost:      61.3%
  SELL / HOLD / BUY: 18.2% / 20.5% / 61.3%
  ─────────────────────────────────────────
  Předpověď +4h: 230.11$ (+0.70%)
==================================================
```

---

## Features (~150 per ticker)

- **Returns:** 1h, 2h, 4h, 8h, 16h, 32h
- **Moving Averages:** MA5–MA200 distance, EMA9/21/55 distance
- **Oscillators:** RSI(7/14/21), Stochastic RSI, Bollinger Z-score
- **MACD:** line, signal, histogram, histogram change
- **Volatility:** rolling std 8/20/50
- **Volume:** MA ratio, spike detection
- **Market context:** S&P500, NASDAQ, DAX, VIX, DXY, Gold, Oil, Bonds
- **Sector ETFs:** XLK, XLF, XLV, XLE, XLY
- **Daily context:** same indicators on daily timeframe
- **Time features:** hour, day_of_week, month, quarter, is_monday, is_friday
- **Ticker one-hot:** binary column per ticker (no ordinal bias)

---

## Disclaimer

For **educational and research purposes only**. Not financial advice. Do not trade real money based on these signals.