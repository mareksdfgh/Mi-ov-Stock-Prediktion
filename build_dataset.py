"""
build_dataset.py
================
Vytváří společný dataset pro VÍCE akcií.
Každý řádek = jedna hodina jedné akcie.
ticker_id je zakódován jako one-hot → generalistický model bez ordinálního předpokladu.

Výstupy:
  dataset.csv        — hlavní dataset (klasifikace + regrese)
  dataset_info.csv   — statistiky příznaků
  dataset_tickers.json — seznam tickerů + ID

Spuštění:
  python build_dataset.py
  python build_dataset.py --tickers AAPL MSFT GOOGL NVDA TSLA --output dataset.csv

Zdroje dat (zdarma):
  yfinance hodinově  ~730 dní
  yfinance denně     ~20 let (jako kontextové příznaky)
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Sdílený modul — vždy importujeme odsud, ne duplikujeme
from features import add_features

warnings.filterwarnings("ignore")

# ── Výchozí seznam tickerů ────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
    "AMZN", "META", "NFLX", "AMD", "INTC",
]

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

HOURLY_PERIOD  = "730d"
DAILY_YEARS    = 20
LABEL_HORIZON  = 4       # hodin dopředu
LABEL_THRESH   = 0.003   # 0.3% práh pro BUY/SELL


# ── Stahování dat ─────────────────────────────────────────────────────────────

def dl_hourly(ticker: str) -> pd.DataFrame:
    """Stáhne hodinová OHLCV data pro daný ticker."""
    df = yf.download(ticker, period=HOURLY_PERIOD, interval="1h",
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Normalizace timezone → naive datetime
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def dl_daily(ticker: str) -> pd.DataFrame:
    """Stáhne denní OHLCV data za posledních DAILY_YEARS let."""
    start = (datetime.now() - timedelta(days=DAILY_YEARS * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ── Merge denních příznaků do hodinového indexu ───────────────────────────────

def merge_daily_to_hourly(df_hourly: pd.DataFrame,
                           df_daily: pd.DataFrame,
                           prefix: str) -> pd.DataFrame:
    """
    Přidá denní příznaky k hodinovým datům — bez look-ahead.
    Denní hodnota je dostupná až od začátku daného dne.
    Navíc: aplikujeme .shift(1) na tržní příznaky po mergi,
    aby aktuální denní hodnota nespadla do minulé hodinové svíčky.
    """
    if df_daily.empty:
        return df_hourly

    orig_idx = df_hourly.index.copy()
    date_idx = pd.to_datetime(df_hourly.index.date)
    df_daily.index = pd.to_datetime(df_daily.index.date)

    # Přejmenování překrývajících se sloupců
    overlap = [c for c in df_daily.columns if c in df_hourly.columns]
    df_daily = df_daily.rename(columns={c: f"{c}__{prefix}" for c in overlap})

    df_hourly.index = date_idx
    df_hourly = df_hourly.join(df_daily, how="left")
    df_hourly.index = orig_idx

    return df_hourly


def merge_market_hourly(df_stock: pd.DataFrame,
                         df_market: pd.DataFrame,
                         name: str) -> pd.DataFrame:
    """
    Merge hodinových tržních příznaků do dat akcie.
    Aplikuje .shift(1) → zabrání look-ahead leakage při nesouladu timestamps.
    """
    if df_market.empty:
        return df_stock

    # Přejmenování překrývajících se sloupců
    overlap = [c for c in df_market.columns if c in df_stock.columns]
    df_market = df_market.rename(columns={c: f"{c}__{name}" for c in overlap})

    # Lag o 1 krok — ochrana před look-ahead
    df_market = df_market.shift(1)

    df_stock = df_stock.join(df_market, how="left")
    return df_stock


# ── Vytvoření labelů ──────────────────────────────────────────────────────────

def create_labels(close: pd.Series):
    """
    Klasifikace: BUY=2, HOLD=1, SELL=0
    Regrese:     future_return (float, příštích LABEL_HORIZON hodin)
    Bez look-ahead: shift(-horizon) na BUDOUCÍ ceny.
    """
    future_ret = close.pct_change(LABEL_HORIZON).shift(-LABEL_HORIZON)
    label = pd.Series(1, index=close.index, name="label")
    label[future_ret >  LABEL_THRESH] = 2
    label[future_ret < -LABEL_THRESH] = 0
    return label, future_ret.rename("future_return")


# ── One-hot encoding pro ticker_id ────────────────────────────────────────────

def add_ticker_onehot(df: pd.DataFrame, ticker: str,
                       all_tickers: list) -> pd.DataFrame:
    """
    Přidá one-hot sloupce pro každý ticker (ticker_oh_AAPL, ...).
    Lepší než ordinální ID — model nemá falešné pořadové vztahy.
    """
    for t in all_tickers:
        df[f"ticker_oh_{t}"] = int(ticker == t)
    return df


# ── Zpracování jednoho tickeru ────────────────────────────────────────────────

def process_ticker(ticker: str, ticker_id: int,
                   all_tickers: list,
                   market_h: dict, market_d: dict) -> pd.DataFrame:
    print(f"\n  [{ticker_id}] {ticker}")

    # Stažení hodinových dat
    h = dl_hourly(ticker)
    if h.empty:
        print(f"    PŘESKOČENO: žádná hodinová data")
        return pd.DataFrame()

    close_raw = h["Close"].copy()  # uchováme pro labely a grafy
    label, future_ret = create_labels(close_raw)

    h_feat = add_features(h.copy(), prefix="")

    # Denní kontext samotné akcie
    d = dl_daily(ticker)
    if not d.empty:
        d_feat = add_features(d.copy(), prefix="d_")
        h_feat = merge_daily_to_hourly(h_feat, d_feat, prefix="d_main")

    # Hodinové tržní příznaky (s lagem — ochrana před look-ahead)
    for name, mdf in market_h.items():
        if not mdf.empty:
            h_feat = merge_market_hourly(h_feat, mdf.copy(), name)

    # Denní tržní příznaky
    for name, mdf in market_d.items():
        if not mdf.empty:
            h_feat = merge_daily_to_hourly(h_feat, mdf.copy(), prefix=name)

    # Labely
    h_feat["label"]         = label
    h_feat["future_return"] = future_ret
    h_feat["close_raw"]     = close_raw  # pro grafy v run.py

    # Časové příznaky
    h_feat["hour"]         = h_feat.index.hour
    h_feat["day_of_week"]  = h_feat.index.dayofweek
    h_feat["month"]        = h_feat.index.month
    h_feat["quarter"]      = h_feat.index.quarter
    h_feat["is_monday"]    = (h_feat.index.dayofweek == 0).astype(int)
    h_feat["is_friday"]    = (h_feat.index.dayofweek == 4).astype(int)
    h_feat["week_of_year"] = h_feat.index.isocalendar().week.astype(int).values

    # One-hot encoding tickeru (místo ordinálního ID)
    h_feat = add_ticker_onehot(h_feat, ticker, all_tickers)
    h_feat["ticker"] = ticker  # pouze metadata, ne feature

    # Poslední LABEL_HORIZON řádků zahodíme — future return není dostupný
    h_feat = h_feat.iloc[:-LABEL_HORIZON]
    return h_feat


# ── Hlavní pipeline ───────────────────────────────────────────────────────────

def build_dataset(tickers: list, output_path: str):
    print(f"\n{'='*55}")
    print(f" Multi-Ticker Dataset Builder")
    print(f" Tickery: {tickers}")
    print(f"{'='*55}")

    # Tržní data stáhneme jednou — sdílená pro všechny akcie
    print("\n[1/3] Načítání tržních a sektorových dat...")
    all_market = {**MARKET_TICKERS, **SECTOR_ETFS}
    market_h, market_d = {}, {}

    for name, t in all_market.items():
        print(f"  {name} ({t})")
        try:
            h = dl_hourly(t)
            if not h.empty:
                market_h[name] = add_features(h.copy(), prefix=f"{name}_")
            else:
                print(f"    VAROVÁNÍ: {name} — žádná hodinová data")
        except Exception as e:
            print(f"    CHYBA {name} (hodinové): {e}")

        try:
            d = dl_daily(t)
            if not d.empty:
                market_d[name] = add_features(d.copy(), prefix=f"{name}_d_")
            else:
                print(f"    VAROVÁNÍ: {name} — žádná denní data")
        except Exception as e:
            print(f"    CHYBA {name} (denní): {e}")

    # Zpracování tickerů
    print(f"\n[2/3] Zpracování {len(tickers)} akcií...")
    ticker_map = {t: i for i, t in enumerate(tickers)}
    frames = []
    failed = []

    for ticker, tid in ticker_map.items():
        try:
            df = process_ticker(ticker, tid, tickers, market_h, market_d)
            if not df.empty:
                frames.append(df)
                print(f"    ✓ {ticker}: {len(df)} řádků, {df.shape[1]} sloupců")
        except Exception as e:
            print(f"    ✗ {ticker}: {e}")
            failed.append(ticker)

    if not frames:
        raise ValueError("Žádná data nebyla načtena!")

    print(f"\n[3/3] Spojení a uložení...")
    combined = pd.concat(frames, axis=0, ignore_index=False)
    combined.sort_index(inplace=True)

    # NaN filtr — řádek musí mít alespoň 60 % sloupců vyplněno
    before = len(combined)
    combined.dropna(thresh=int(combined.shape[1] * 0.6), inplace=True)
    print(f"  Řádky: {before} → {len(combined)} (po NaN filtru)")

    # Statistiky
    print(f"\n  Celková statistika:")
    print(f"  Řádků celkem:  {len(combined)}")
    print(f"  Příznaků:      {combined.shape[1] - 3}")  # -label -future_return -ticker
    print(f"  Časové rozmezí: {combined.index.min()} → {combined.index.max()}")
    print(f"\n  Rozdělení labelů:")
    dist = combined["label"].value_counts().sort_index()
    for lbl, name in [(0, "SELL"), (1, "HOLD"), (2, "BUY")]:
        print(f"    {name}: {dist.get(lbl, 0):>7}")

    print(f"\n  Na ticker:")
    for t in tickers:
        sub = combined[combined["ticker"] == t]
        if len(sub):
            dist_t = sub["label"].value_counts().sort_index()
            print(f"    {t:<8} {len(sub):>5} řádků  "
                  f"SELL={dist_t.get(0,0):>4} HOLD={dist_t.get(1,0):>4} BUY={dist_t.get(2,0):>4}")

    if failed:
        print(f"\n  Selhalo: {failed}")

    # Uložení datasetu
    combined.to_csv(output_path, index=True)
    print(f"\n  ✓ Dataset uložen: {output_path}")

    info_path = output_path.replace(".csv", "_info.csv")
    combined.describe().T.to_csv(info_path)
    print(f"  ✓ Info: {info_path}")

    # Uložení mapy tickerů (včetně one-hot sloupců)
    map_path = output_path.replace(".csv", "_tickers.json")
    with open(map_path, "w") as f:
        json.dump({"tickers": tickers, "ticker_id_map": ticker_map}, f, indent=2)
    print(f"  ✓ Mapa tickerů: {map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--output",  default="dataset.csv")
    args = parser.parse_args()
    build_dataset(args.tickers, args.output)
