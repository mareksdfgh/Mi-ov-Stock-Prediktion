"""
run.py
======
Živá předpověď pro jednu akcii + 3 typy grafů:
  1. Předpověď ceny (Regrese: příštích N hodin jako číslo)
  2. Signal-Plot (BUY/SELL/HOLD vyznačené na grafu kurzu)
  3. Graf ceny s technickými indikátory

Spuštění:
  python run.py --ticker AAPL
  python run.py --ticker AAPL --no_chart      ← pouze textový výstup
  python run.py --ticker AAPL --chart_only    ← pouze grafy (backtest)
  python run.py --ticker MSFT --lookback 200  ← okno grafu v hodinách
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Sdílený modul pro výpočet příznaků — importujeme, neduplikujeme
from features import add_features, rsi

warnings.filterwarnings("ignore")

MARKET_TICKERS = {
    "sp500":  "^GSPC",
    "nasdaq": "^IXIC",
    "dax":    "^GDAXI",
    "vix":    "^VIX",
    "dxy":    "DX-Y.NYB",
    "gold":   "GC=F",
    "oil":    "CL=F",
    "bonds":  "^TNX",
}

SECTOR_ETFS = {
    "tech_etf":     "XLK",
    "finance_etf":  "XLF",
    "health_etf":   "XLV",
    "energy_etf":   "XLE",
    "consumer_etf": "XLY",
}


def fetch_features(ticker: str, ticker_id: int, meta: dict):
    """
    Stáhne živá data a sestaví příznaky identické s build_dataset.py.
    Vrátí (DataFrame příznaků, Series uzavíracích cen).
    """
    import yfinance as yf

    def dl_h(t: str) -> pd.DataFrame:
        """Stáhne hodinová data."""
        df = yf.download(t, period="730d", interval="1h",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df

    def dl_d(t: str) -> pd.DataFrame:
        """Stáhne denní data za posledních 20 let."""
        start = (datetime.now() - timedelta(days=20 * 365)).strftime("%Y-%m-%d")
        df = yf.download(t, start=start, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    def merge_d(df_h: pd.DataFrame, df_d: pd.DataFrame, pfx: str) -> pd.DataFrame:
        """Merge denních příznaků do hodinových dat bez look-ahead."""
        if df_d.empty:
            return df_h
        orig = df_h.index.copy()
        date_idx = pd.to_datetime(df_h.index.date)
        df_d.index = pd.to_datetime(df_d.index.date)
        overlap = [c for c in df_d.columns if c in df_h.columns]
        df_d = df_d.rename(columns={c: f"{c}__{pfx}" for c in overlap})
        df_h.index = date_idx
        df_h = df_h.join(df_d, how="left")
        df_h.index = orig
        return df_h

    def merge_market_h(df_stock: pd.DataFrame, df_mkt: pd.DataFrame,
                       name: str) -> pd.DataFrame:
        """
        Merge hodinových tržních příznaků s lagem 1 krok.
        Zabraňuje look-ahead leakage při nesouladu timestamps.
        """
        if df_mkt.empty:
            return df_stock
        overlap = [c for c in df_mkt.columns if c in df_stock.columns]
        df_mkt = df_mkt.rename(columns={c: f"{c}__{name}" for c in overlap})
        df_mkt = df_mkt.shift(1)  # lag o 1 svíčku
        df_stock = df_stock.join(df_mkt, how="left")
        return df_stock

    print(f"  Načítám {ticker}...")
    h = dl_h(ticker)
    if h.empty:
        raise ValueError(f"Žádná data pro {ticker}")

    close_series = h["Close"].copy()
    h_feat = add_features(h.copy(), prefix="")

    # Denní kontext samotné akcie
    d = dl_d(ticker)
    if not d.empty:
        d_feat = add_features(d.copy(), prefix="d_")
        h_feat = merge_d(h_feat, d_feat, "d_main")

    # Tržní a sektorové příznaky
    for name, t in {**MARKET_TICKERS, **SECTOR_ETFS}.items():
        try:
            mh = dl_h(t)
            if not mh.empty:
                mh_feat = add_features(mh.copy(), prefix=f"{name}_")
                h_feat = merge_market_h(h_feat, mh_feat, name)

            md = dl_d(t)
            if not md.empty:
                md_feat = add_features(md.copy(), prefix=f"{name}_d_")
                h_feat = merge_d(h_feat, md_feat, name)

        except Exception as e:
            # Explicitní logování chyby — tiché except: pass zakázáno
            print(f"    VAROVÁNÍ: {name} ({t}) selhal: {e}")

    # Časové příznaky
    h_feat["hour"]         = h_feat.index.hour
    h_feat["day_of_week"]  = h_feat.index.dayofweek
    h_feat["month"]        = h_feat.index.month
    h_feat["quarter"]      = h_feat.index.quarter
    h_feat["is_monday"]    = (h_feat.index.dayofweek == 0).astype(int)
    h_feat["is_friday"]    = (h_feat.index.dayofweek == 4).astype(int)
    h_feat["week_of_year"] = h_feat.index.isocalendar().week.astype(int).values

    # One-hot sloupce pro ticker — musíme replikovat schéma z tréninku
    features = meta.get("features", [])
    for feat in features:
        if feat.startswith("ticker_oh_"):
            t_name = feat.replace("ticker_oh_", "")
            h_feat[feat] = int(ticker == t_name)

    return h_feat, close_series


def align_features(row: pd.Series, meta: dict) -> np.ndarray:
    """
    Sestaví vektor příznaků pro jednu svíčku podle pořadí z meta.
    Chybějící příznaky jsou nahrazeny mediánem z tréninku.
    """
    features = meta["features"]
    medians  = pd.Series(meta["medians"])
    X = []
    for f in features:
        val = row.get(f, np.nan)
        if pd.isna(val) or np.isinf(val):
            val = medians.get(f, 0.0)
        X.append(float(val))
    return np.array(X).reshape(1, -1)


# ── Grafy ─────────────────────────────────────────────────────────────────────

def plot_signal_chart(close, signals, proba, ticker: str, lookback: int):
    """Graf 1+2: Kurz s označenými BUY/SELL/HOLD signály."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib není nainstalován: pip install matplotlib")
        return

    df = pd.DataFrame({"close": close, "signal": signals}).iloc[-lookback:]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{ticker} — Analýza signálů", fontsize=14, fontweight="bold")

    # Panel 1: Kurz + signály
    ax = axes[0]
    ax.plot(df.index, df["close"], color="#1f77b4", linewidth=1.2, label="Kurz")

    buy_idx  = df[df["signal"] == 2].index
    sell_idx = df[df["signal"] == 0].index

    ax.scatter(buy_idx,  df.loc[buy_idx,  "close"],
               color="lime",  marker="^", s=60, zorder=5, label="BUY")
    ax.scatter(sell_idx, df.loc[sell_idx, "close"],
               color="red",   marker="v", s=60, zorder=5, label="SELL")

    # Klouzavé průměry pro kontext
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    ax.plot(df.index, ma20, color="orange", linewidth=0.8, alpha=0.7, label="MA20")
    ax.plot(df.index, ma50, color="purple", linewidth=0.8, alpha=0.7, label="MA50")

    ax.set_ylabel("Kurz ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Sloupce signálů
    ax2 = axes[1]
    colors = {0: "red", 1: "gray", 2: "lime"}
    bar_colors = [colors.get(s, "gray") for s in df["signal"]]
    ax2.bar(df.index, df["signal"], color=bar_colors, width=0.03, alpha=0.7)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["SELL", "HOLD", "BUY"], fontsize=8)
    ax2.set_ylabel("Signál")
    ax2.grid(True, alpha=0.3)

    # Panel 3: RSI
    ax3 = axes[2]
    rsi_vals = rsi(df["close"], 14)
    ax3.plot(df.index, rsi_vals, color="darkorange", linewidth=0.9)
    ax3.axhline(70, color="red",  linestyle="--", alpha=0.5, linewidth=0.8)
    ax3.axhline(30, color="lime", linestyle="--", alpha=0.5, linewidth=0.8)
    ax3.fill_between(df.index, rsi_vals, 70,
                      where=(rsi_vals >= 70), alpha=0.2, color="red")
    ax3.fill_between(df.index, rsi_vals, 30,
                      where=(rsi_vals <= 30), alpha=0.2, color="lime")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI(14)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"chart_signals_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  ✓ Signal-chart uložen: {fname}")
    plt.show()


def plot_prediction_chart(close, future_pred_pct: float,
                           ticker: str, n_hours: int = 4):
    """Graf 3: Aktuální kurz + předpovídaná cena za N hodin."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Posledních 100 hodin
    recent = close.iloc[-100:]
    ax.plot(recent.index, recent.values,
            color="#1f77b4", linewidth=1.5, label="Historický kurz")

    last_price = recent.iloc[-1]
    last_time  = recent.index[-1]
    pred_price = last_price * (1 + future_pred_pct)

    # Odhadnutý časový krok (medián rozdílů)
    diffs       = pd.Series(recent.index).diff().dropna()
    median_step = diffs.median()
    pred_time   = last_time + median_step * n_hours

    color = "lime" if future_pred_pct >= 0 else "red"
    ax.plot([last_time, pred_time], [last_price, pred_price],
            color=color, linewidth=2.5, linestyle="--",
            marker="o", markersize=8,
            label=f"Předpověď +{n_hours}h: {pred_price:.2f}$ ({future_pred_pct * 100:+.2f}%)")

    ax.axvline(last_time, color="gray", linestyle=":", alpha=0.6)
    ax.set_title(f"{ticker} — Předpověď ceny: příštích {n_hours} hodin",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Kurz ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    fname = f"chart_prediction_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  ✓ Graf předpovědi uložen: {fname}")
    plt.show()


# ── Hlavní predikční funkce ───────────────────────────────────────────────────

def predict(ticker: str, model_dir: str, lookback: int,
            no_chart: bool, chart_only: bool) -> dict:
    model_dir = Path(model_dir)

    # Načtení metadat a modelů
    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    cls_model = joblib.load(model_dir / "model_cls.joblib")
    reg_model = joblib.load(model_dir / "model_reg.joblib")
    print(f"  Modely načteny (trénováno do {meta['trained_until']})")

    # Zjištění ticker ID (pro one-hot sloupce)
    ticker_map_path = model_dir / "dataset_tickers.json"
    ticker_id = 0
    if ticker_map_path.exists():
        with open(ticker_map_path) as f:
            tmap = json.load(f)
        ticker_id = tmap.get("ticker_id_map", {}).get(ticker, 0)

    # Stažení živých dat a sestavení příznaků
    print(f"\n  Načítám živá data...")
    h_feat, close_series = fetch_features(ticker, ticker_id, meta)

    # Predikce pro všechny svíčky v okně (pro signal chart)
    print(f"  Počítám signály pro posledních {lookback} hodin...")
    X_all    = h_feat.replace([np.inf, -np.inf], np.nan)
    features = meta["features"]
    medians  = pd.Series(meta["medians"])

    # Zarovnání na příznaky z tréninku
    available = [f for f in features if f in X_all.columns]
    missing   = [f for f in features if f not in X_all.columns]

    X_slice = X_all[available].iloc[-lookback:].copy()
    for m in missing:
        X_slice[m] = medians.get(m, 0.0)
    X_slice = X_slice[features].fillna(medians).replace([np.inf, -np.inf], 0.0)

    signals      = cls_model.predict(X_slice)
    proba        = cls_model.predict_proba(X_slice)
    pred_returns = reg_model.predict(X_slice)

    close_window = close_series.iloc[-lookback:].reindex(X_slice.index)

    # Nejnovější predikce
    last_signal = signals[-1]
    last_proba  = proba[-1]
    last_ret    = pred_returns[-1]
    last_price  = close_series.iloc[-1]
    pred_price  = last_price * (1 + last_ret)
    label_map   = meta["label_map"]

    print(f"\n{'='*50}")
    print(f"  Ticker:            {ticker}")
    print(f"  Časový okamžik:    {close_series.index[-1]}")
    print(f"  Aktuální kurz:     {last_price:.2f}$")
    print(f"  ─────────────────────────────────────────")
    print(f"  Signál:            {label_map[str(last_signal)]}")
    print(f"  Spolehlivost:      {last_proba[last_signal]:.1%}")
    print(f"  SELL / HOLD / BUY: "
          f"{last_proba[0]:.1%} / {last_proba[1]:.1%} / {last_proba[2]:.1%}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Předpověď +{meta['label_horizon']}h: "
          f"{pred_price:.2f}$ ({last_ret * 100:+.2f}%)")
    print(f"{'='*50}\n")

    if not no_chart:
        plot_signal_chart(close_window, signals, proba, ticker, lookback)
        plot_prediction_chart(close_series, last_ret, ticker, meta["label_horizon"])

    return {
        "ticker":      ticker,
        "timestamp":   str(close_series.index[-1]),
        "price":       float(last_price),
        "signal":      label_map[str(last_signal)],
        "confidence":  float(last_proba[last_signal]),
        "pred_price":  float(pred_price),
        "pred_return": float(last_ret),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",     default="AAPL")
    parser.add_argument("--model_dir",  default=".")
    parser.add_argument("--lookback",   default=200, type=int,
                        help="Hodin pro signal chart (výchozí: 200)")
    parser.add_argument("--no_chart",   action="store_true")
    parser.add_argument("--chart_only", action="store_true")
    args = parser.parse_args()

    result = predict(args.ticker, args.model_dir, args.lookback,
                     args.no_chart, args.chart_only)
