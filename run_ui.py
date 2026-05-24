"""
run_ui.py — Mičov
==================================
Retro Tkinter UI. Kein customtkinter.
Multithreading: queue-basiert, kein after()-Spam.

Aufruf:
  python3 run_ui.py
  python3 run_ui.py --model_dir .
  python3 run_ui.py --tickers AAPL MSFT NVDA TSLA
"""

import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, font as tkfont

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Načtení tracker.py ze stejné složky
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("tracker", Path(__file__).parent / "tracker.py")
    _tmod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_tmod)
    PredictionTracker = _tmod.PredictionTracker
    HAS_TRACKER = True
except Exception as _te:
    HAS_TRACKER = False
    print(f"  tracker.py nicht geladen: {_te}")

# ── Import modulu run.py ─────────────────────────────────────────────────────────
def _import_run(model_dir="."):
    candidates = [
        Path(__file__).parent / "run.py",
        Path(model_dir) / "run.py",
        Path("run.py"),
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("run_module", p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("run.py nicht gefunden")

# ── Konstanty ─────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN","META","NFLX","AMD","INTC"]

# ── Překlady ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "de": {
        "title":         "Mičov",
        "menu_file":     "Datei",
        "menu_scan":     "Scan starten   F5",
        "menu_stop":     "Scan stoppen   Esc",
        "menu_quit":     "Beenden",
        "menu_view":     "Ansicht",
        "menu_charts":   "Alle Charts",
        "menu_settings": "Einstellungen",
        "btn_scan":      " SCAN [F5]",
        "btn_stop":      " STOP [Esc]",
        "lbl_ticker":    "Ticker:",
        "lbl_lookback":  "Lookback (h):",
        "lbl_autoref":   "Auto-Refresh",
        "lbl_model":     "Modell:",
        "lbl_algo":      "Algo: XGB+HGB Stacking",
        "tbl_title":     "  Signal-Tabelle",
        "col_ticker":    "TICKER",
        "col_price":     "KURS ($)",
        "col_signal":    "SIGNAL",
        "col_conf":      "KONF.",
        "col_forecast":  "+4h %",
        "col_sell":      "SELL%",
        "col_hold":      "HOLD%",
        "col_buy":       "BUY%",
        "col_ts":        "ZEITPUNKT",
        "summary_ready": "  Bereit. F5 drücken um Scan zu starten.",
        "chart_title":   "  Chart-Analyse",
        "tab_signal":    "  Signal-Chart  ",
        "tab_pred":      "  Preis-Prognose  ",
        "tab_proba":     "  Wahrscheinlichkeit  ",
        "tab_acc":       "  Accuracy-Tracker  ",
        "det_title":     "  Detail",
        "det_ticker":    "Ticker:",
        "det_signal":    "Signal:",
        "det_conf":      "Konfidenz:",
        "det_price":     "Kurs ($):",
        "det_pred_p":    "Prognose $:",
        "det_pred_r":    "Prognose %:",
        "det_sell":      "SELL%:",
        "det_hold":      "HOLD%:",
        "det_buy":       "BUY%:",
        "status_ready":  "  Bereit",
        "status_scan":   "Scanne",
        "status_done":   "Scan abgeschlossen.",
        "status_stop":   "Gestoppt.",
        "last_scan":     "Letzter Scan:",
        "settings_title":"Einstellungen",
        "settings_lang": "Sprache:",
        "settings_ref":  "Auto-Refresh Intervall:",
        "settings_30":   "30 Minuten",
        "settings_60":   "1 Stunde",
        "settings_ok":   "OK",
        "acc_title":     "  Richtungs-Genauigkeit (nach 4h ausgewertet)",
        "acc_overall":   "  Gesamtstatistik",
        "acc_ctrl":      "  Steuerung",
        "acc_eval_btn":  "Jetzt auswerten",
        "acc_ref_btn":   "Tabelle refresh",
        "acc_next":      "  Nächste Auswertung:",
        "acc_chart_t":   "  Richtungs-Genauigkeit über Zeit (rolling 10)",
        "acc_log_t":     "  Letzte ausgewerteten Vorhersagen",
        "empty_signal":  "Signal-Chart\nTicker aus Tabelle auswählen",
        "empty_pred":    "Preis-Prognose\n+4h Vorhersage-Pfeil",
        "empty_proba":   "Wahrscheinlichkeit\nSELL / HOLD / BUY über Zeit",
        "empty_acc":     "Accuracy-Chart\nWarte auf erste ausgewerteten Predictions...",
    },
    "en": {
        "title":         "Mičov",
        "menu_file":     "File",
        "menu_scan":     "Start Scan   F5",
        "menu_stop":     "Stop Scan   Esc",
        "menu_quit":     "Quit",
        "menu_view":     "View",
        "menu_charts":   "All Charts",
        "menu_settings": "Settings",
        "btn_scan":      " SCAN [F5]",
        "btn_stop":      " STOP [Esc]",
        "lbl_ticker":    "Tickers:",
        "lbl_lookback":  "Lookback (h):",
        "lbl_autoref":   "Auto-Refresh",
        "lbl_model":     "Model:",
        "lbl_algo":      "Algo: XGB+HGB Stacking",
        "tbl_title":     "  Signal Table",
        "col_ticker":    "TICKER",
        "col_price":     "PRICE ($)",
        "col_signal":    "SIGNAL",
        "col_conf":      "CONF.",
        "col_forecast":  "+4h %",
        "col_sell":      "SELL%",
        "col_hold":      "HOLD%",
        "col_buy":       "BUY%",
        "col_ts":        "TIMESTAMP",
        "summary_ready": "  Ready. Press F5 to start scan.",
        "chart_title":   "  Chart Analysis",
        "tab_signal":    "  Signal Chart  ",
        "tab_pred":      "  Price Forecast  ",
        "tab_proba":     "  Probability  ",
        "tab_acc":       "  Accuracy Tracker  ",
        "det_title":     "  Detail",
        "det_ticker":    "Ticker:",
        "det_signal":    "Signal:",
        "det_conf":      "Confidence:",
        "det_price":     "Price ($):",
        "det_pred_p":    "Forecast $:",
        "det_pred_r":    "Forecast %:",
        "det_sell":      "SELL%:",
        "det_hold":      "HOLD%:",
        "det_buy":       "BUY%:",
        "status_ready":  "  Ready",
        "status_scan":   "Scanning",
        "status_done":   "Scan complete.",
        "status_stop":   "Stopped.",
        "last_scan":     "Last scan:",
        "settings_title":"Settings",
        "settings_lang": "Language:",
        "settings_ref":  "Auto-Refresh Interval:",
        "settings_30":   "30 Minutes",
        "settings_60":   "1 Hour",
        "settings_ok":   "OK",
        "acc_title":     "  Directional Accuracy (evaluated after 4h)",
        "acc_overall":   "  Overall Statistics",
        "acc_ctrl":      "  Controls",
        "acc_eval_btn":  "Evaluate Now",
        "acc_ref_btn":   "Refresh Table",
        "acc_next":      "  Next evaluation:",
        "acc_chart_t":   "  Directional Accuracy over Time (rolling 10)",
        "acc_log_t":     "  Recent Evaluated Predictions",
        "empty_signal":  "Signal Chart\nSelect ticker from table",
        "empty_pred":    "Price Forecast\n+4h prediction arrow",
        "empty_proba":   "Probability\nSELL / HOLD / BUY over time",
        "empty_acc":     "Accuracy Chart\nWaiting for first evaluated predictions...",
    },
    "cs": {
        "title":         "Mičov",
        "menu_file":     "Soubor",
        "menu_scan":     "Spustit sken   F5",
        "menu_stop":     "Zastavit sken   Esc",
        "menu_quit":     "Ukončit",
        "menu_view":     "Zobrazení",
        "menu_charts":   "Všechny grafy",
        "menu_settings": "Nastavení",
        "btn_scan":      " SKEN [F5]",
        "btn_stop":      " STOP [Esc]",
        "lbl_ticker":    "Ticker:",
        "lbl_lookback":  "Zpětný pohled (h):",
        "lbl_autoref":   "Auto-Obnovení",
        "lbl_model":     "Model:",
        "lbl_algo":      "Algo: XGB+HGB Stacking",
        "tbl_title":     "  Tabulka signálů",
        "col_ticker":    "TICKER",
        "col_price":     "KURZ ($)",
        "col_signal":    "SIGNÁL",
        "col_conf":      "KONF.",
        "col_forecast":  "+4h %",
        "col_sell":      "SELL%",
        "col_hold":      "HOLD%",
        "col_buy":       "BUY%",
        "col_ts":        "ČASOVÉ RAZÍTKO",
        "summary_ready": "  Připraven. Stiskněte F5 pro spuštění skenu.",
        "chart_title":   "  Analýza grafů",
        "tab_signal":    "  Signální graf  ",
        "tab_pred":      "  Předpověď ceny  ",
        "tab_proba":     "  Pravděpodobnost  ",
        "tab_acc":       "  Sledování přesnosti  ",
        "det_title":     "  Detail",
        "det_ticker":    "Ticker:",
        "det_signal":    "Signál:",
        "det_conf":      "Konfidence:",
        "det_price":     "Kurz ($):",
        "det_pred_p":    "Předpověď $:",
        "det_pred_r":    "Předpověď %:",
        "det_sell":      "SELL%:",
        "det_hold":      "HOLD%:",
        "det_buy":       "BUY%:",
        "status_ready":  "  Připraven",
        "status_scan":   "Skenuji",
        "status_done":   "Sken dokončen.",
        "status_stop":   "Zastaveno.",
        "last_scan":     "Poslední sken:",
        "settings_title":"Nastavení",
        "settings_lang": "Jazyk:",
        "settings_ref":  "Interval auto-obnovení:",
        "settings_30":   "30 minut",
        "settings_60":   "1 hodina",
        "settings_ok":   "OK",
        "acc_title":     "  Směrová přesnost (vyhodnoceno po 4h)",
        "acc_overall":   "  Celková statistika",
        "acc_ctrl":      "  Ovládání",
        "acc_eval_btn":  "Vyhodnotit nyní",
        "acc_ref_btn":   "Obnovit tabulku",
        "acc_next":      "  Další vyhodnocení:",
        "acc_chart_t":   "  Směrová přesnost v čase (klouzavé 10)",
        "acc_log_t":     "  Poslední vyhodnocené předpovědi",
        "empty_signal":  "Signální graf\nVyberte ticker z tabulky",
        "empty_pred":    "Předpověď ceny\nšipka +4h",
        "empty_proba":   "Pravděpodobnost\nSELL / HOLD / BUY v čase",
        "empty_acc":     "Graf přesnosti\nČekám na první vyhodnocené předpovědi...",
    },
    "sk": {
        "title":         "Mičov",
        "menu_file":     "Súbor",
        "menu_scan":     "Spustiť skenovanie   F5",
        "menu_stop":     "Zastaviť skenovanie   Esc",
        "menu_quit":     "Ukončiť",
        "menu_view":     "Zobrazenie",
        "menu_charts":   "Všetky grafy",
        "menu_settings": "Nastavenia",
        "btn_scan":      " SKEN [F5]",
        "btn_stop":      " STOP [Esc]",
        "lbl_ticker":    "Ticker:",
        "lbl_lookback":  "Spätný pohľad (h):",
        "lbl_autoref":   "Auto-Obnovenie",
        "lbl_model":     "Model:",
        "lbl_algo":      "Algo: XGB+HGB Stacking",
        "tbl_title":     "  Tabuľka signálov",
        "col_ticker":    "TICKER",
        "col_price":     "KURZ ($)",
        "col_signal":    "SIGNÁL",
        "col_conf":      "KONF.",
        "col_forecast":  "+4h %",
        "col_sell":      "SELL%",
        "col_hold":      "HOLD%",
        "col_buy":       "BUY%",
        "col_ts":        "ČASOVÁ PEČIATKA",
        "summary_ready": "  Pripravený. Stlačte F5 pre spustenie skenu.",
        "chart_title":   "  Analýza grafov",
        "tab_signal":    "  Signálny graf  ",
        "tab_pred":      "  Predpoveď ceny  ",
        "tab_proba":     "  Pravdepodobnosť  ",
        "tab_acc":       "  Sledovanie presnosti  ",
        "det_title":     "  Detail",
        "det_ticker":    "Ticker:",
        "det_signal":    "Signál:",
        "det_conf":      "Spoľahlivosť:",
        "det_price":     "Kurz ($):",
        "det_pred_p":    "Predpoveď $:",
        "det_pred_r":    "Predpoveď %:",
        "det_sell":      "SELL%:",
        "det_hold":      "HOLD%:",
        "det_buy":       "BUY%:",
        "status_ready":  "  Pripravený",
        "status_scan":   "Skenujem",
        "status_done":   "Skenovanie dokončené.",
        "status_stop":   "Zastavené.",
        "last_scan":     "Posledný sken:",
        "settings_title":"Nastavenia",
        "settings_lang": "Jazyk:",
        "settings_ref":  "Interval auto-obnovenia:",
        "settings_30":   "30 minút",
        "settings_60":   "1 hodina",
        "settings_ok":   "OK",
        "acc_title":     "  Smerová presnosť (vyhodnotené po 4h)",
        "acc_overall":   "  Celková štatistika",
        "acc_ctrl":      "  Ovládanie",
        "acc_eval_btn":  "Vyhodnotiť teraz",
        "acc_ref_btn":   "Obnoviť tabuľku",
        "acc_next":      "  Ďalšie vyhodnotenie:",
        "acc_chart_t":   "  Smerová presnosť v čase (kĺzavé 10)",
        "acc_log_t":     "  Posledné vyhodnotené predpovede",
        "empty_signal":  "Signálny graf\nVyberte ticker z tabuľky",
        "empty_pred":    "Predpoveď ceny\nšípka +4h",
        "empty_proba":   "Pravdepodobnosť\nSELL / HOLD / BUY v čase",
        "empty_acc":     "Graf presnosti\nČakám na prvé vyhodnotené predpovede...",
    },
    "uk": {
        "title":         "Mičov",
        "menu_file":     "Файл",
        "menu_scan":     "Почати сканування   F5",
        "menu_stop":     "Зупинити сканування   Esc",
        "menu_quit":     "Вийти",
        "menu_view":     "Вигляд",
        "menu_charts":   "Всі графіки",
        "menu_settings": "Налаштування",
        "btn_scan":      " СКАН [F5]",
        "btn_stop":      " СТОП [Esc]",
        "lbl_ticker":    "Тікери:",
        "lbl_lookback":  "Глибина (г):",
        "lbl_autoref":   "Авто-оновлення",
        "lbl_model":     "Модель:",
        "lbl_algo":      "Алго: XGB+HGB Stacking",
        "tbl_title":     "  Таблиця сигналів",
        "col_ticker":    "ТІКЕР",
        "col_price":     "ЦІНА ($)",
        "col_signal":    "СИГНАЛ",
        "col_conf":      "КОНФ.",
        "col_forecast":  "+4г %",
        "col_sell":      "SELL%",
        "col_hold":      "HOLD%",
        "col_buy":       "BUY%",
        "col_ts":        "ЧАСОВА МІТКА",
        "summary_ready": "  Готово. Натисніть F5 для початку сканування.",
        "chart_title":   "  Аналіз графіків",
        "tab_signal":    "  Графік сигналів  ",
        "tab_pred":      "  Прогноз ціни  ",
        "tab_proba":     "  Імовірність  ",
        "tab_acc":       "  Точність прогнозів  ",
        "det_title":     "  Деталі",
        "det_ticker":    "Тікер:",
        "det_signal":    "Сигнал:",
        "det_conf":      "Впевненість:",
        "det_price":     "Ціна ($):",
        "det_pred_p":    "Прогноз $:",
        "det_pred_r":    "Прогноз %:",
        "det_sell":      "SELL%:",
        "det_hold":      "HOLD%:",
        "det_buy":       "BUY%:",
        "status_ready":  "  Готово",
        "status_scan":   "Сканування",
        "status_done":   "Сканування завершено.",
        "status_stop":   "Зупинено.",
        "last_scan":     "Останній скан:",
        "settings_title":"Налаштування",
        "settings_lang": "Мова:",
        "settings_ref":  "Інтервал оновлення:",
        "settings_30":   "30 хвилин",
        "settings_60":   "1 година",
        "settings_ok":   "OK",
        "acc_title":     "  Точність напрямку (оцінюється після 4г)",
        "acc_overall":   "  Загальна статистика",
        "acc_ctrl":      "  Керування",
        "acc_eval_btn":  "Оцінити зараз",
        "acc_ref_btn":   "Оновити таблицю",
        "acc_next":      "  Наступна оцінка:",
        "acc_chart_t":   "  Точність напрямку в часі (ковзне 10)",
        "acc_log_t":     "  Останні оцінені прогнози",
        "empty_signal":  "Графік сигналів\nОберіть тікер із таблиці",
        "empty_pred":    "Прогноз ціни\nстрілка +4г",
        "empty_proba":   "Імовірність\nSELL / HOLD / BUY у часі",
        "empty_acc":     "Графік точності\nОчікую перші оцінені прогнози...",
    },
    "ru": {
        "title":         "Mičov",
        "menu_file":     "Файл",
        "menu_scan":     "Начать сканирование   F5",
        "menu_stop":     "Остановить сканирование   Esc",
        "menu_quit":     "Выход",
        "menu_view":     "Вид",
        "menu_charts":   "Все графики",
        "menu_settings": "Настройки",
        "btn_scan":      " СКАН [F5]",
        "btn_stop":      " СТОП [Esc]",
        "lbl_ticker":    "Тикеры:",
        "lbl_lookback":  "Глубина (ч):",
        "lbl_autoref":   "Авто-обновление",
        "lbl_model":     "Модель:",
        "lbl_algo":      "Алго: XGB+HGB Stacking",
        "tbl_title":     "  Таблица сигналов",
        "col_ticker":    "ТИКЕР",
        "col_price":     "ЦЕНА ($)",
        "col_signal":    "СИГНАЛ",
        "col_conf":      "КОНФ.",
        "col_forecast":  "+4ч %",
        "col_sell":      "SELL%",
        "col_hold":      "HOLD%",
        "col_buy":       "BUY%",
        "col_ts":        "ВРЕМЕННАЯ МЕТКА",
        "summary_ready": "  Готово. Нажмите F5 для начала сканирования.",
        "chart_title":   "  Анализ графиков",
        "tab_signal":    "  График сигналов  ",
        "tab_pred":      "  Прогноз цены  ",
        "tab_proba":     "  Вероятность  ",
        "tab_acc":       "  Точность прогнозов  ",
        "det_title":     "  Детали",
        "det_ticker":    "Тикер:",
        "det_signal":    "Сигнал:",
        "det_conf":      "Уверенность:",
        "det_price":     "Цена ($):",
        "det_pred_p":    "Прогноз $:",
        "det_pred_r":    "Прогноз %:",
        "det_sell":      "SELL%:",
        "det_hold":      "HOLD%:",
        "det_buy":       "BUY%:",
        "status_ready":  "  Готово",
        "status_scan":   "Сканирование",
        "status_done":   "Сканирование завершено.",
        "status_stop":   "Остановлено.",
        "last_scan":     "Последний скан:",
        "settings_title":"Настройки",
        "settings_lang": "Язык:",
        "settings_ref":  "Интервал обновления:",
        "settings_30":   "30 минут",
        "settings_60":   "1 час",
        "settings_ok":   "OK",
        "acc_title":     "  Точность направления (оценивается через 4ч)",
        "acc_overall":   "  Общая статистика",
        "acc_ctrl":      "  Управление",
        "acc_eval_btn":  "Оценить сейчас",
        "acc_ref_btn":   "Обновить таблицу",
        "acc_next":      "  Следующая оценка:",
        "acc_chart_t":   "  Точность направления во времени (скользящее 10)",
        "acc_log_t":     "  Последние оценённые прогнозы",
        "empty_signal":  "График сигналов\nВыберите тикер из таблицы",
        "empty_pred":    "Прогноз цены\nстрелка +4ч",
        "empty_proba":   "Вероятность\nSELL / HOLD / BUY во времени",
        "empty_acc":     "График точности\nОжидаю первых оценённых прогнозов...",
    },
}

LANG_NAMES = {
    "de": "Deutsch",
    "en": "English",
    "cs": "Čeština",
    "sk": "Slovenčina",
    "uk": "Українська",
    "ru": "Русский",
}
LOOKBACK        = 120   # Hodin pro signální graf

# ── Retro paleta (inspirace Win95 / CRT) ──────────────────────────────────────
BG        = "#c0c0c0"   # klassisches Silber
BG_DARK   = "#808080"
BG_PANEL  = "#d4d0c8"   # leicht wärmer
TITLE_BG  = "#000080"   # Marineblau
TITLE_FG  = "#ffffff"
RELIEF_IN = "sunken"
RELIEF_OUT= "raised"
FONT_MONO = ("Courier", 10)
FONT_MONO_B=("Courier", 10, "bold")
FONT_MONO_S=("Courier", 9)
FONT_TITLE= ("Courier", 11, "bold")
FONT_BIG  = ("Courier", 14, "bold")

# Barvy signálů (pro text, ne pozadí)
COL_BUY   = "#006400"   # dunkelgrün
COL_SELL  = "#8b0000"   # dunkelrot
COL_HOLD  = "#333333"
COL_GAIN  = "#006400"
COL_LOSS  = "#8b0000"

# Matplotlib téma laděné k retro stylu
MPL_BG    = "#d4d0c8"
MPL_FG    = "#000000"
MPL_GRID  = "#a0a0a0"
MPL_BUY   = "#006400"
MPL_SELL  = "#8b0000"
MPL_PRICE = "#000080"
MPL_MA20  = "#8b4513"
MPL_MA50  = "#4b0082"
MPL_PRED  = "#006400"


# ══════════════════════════════════════════════════════════════════════════════
#  POMOCNÉ WIDGETY
# ══════════════════════════════════════════════════════════════════════════════

def make_title_bar(parent, text):
    f = tk.Frame(parent, bg=TITLE_BG)
    f.pack(fill="x")
    tk.Label(f, text=text, bg=TITLE_BG, fg=TITLE_FG,
             font=FONT_MONO_B, padx=6, pady=3).pack(side="left")
    return f

def raised_frame(parent, **kw):
    return tk.Frame(parent, relief=RELIEF_OUT, bd=2,
                    bg=BG_PANEL, **kw)

def sunken_frame(parent, **kw):
    return tk.Frame(parent, relief=RELIEF_IN, bd=2,
                    bg=BG_PANEL, **kw)

def sep(parent, orient="h"):
    if orient == "h":
        tk.Frame(parent, height=2, bg=BG_DARK, relief="sunken"
                 ).pack(fill="x", pady=2)
    else:
        tk.Frame(parent, width=2, bg=BG_DARK, relief="sunken"
                 ).pack(fill="y", padx=2, side="left")

def label(parent, text, bold=False, color="#000000", size=10, **kw):
    f = ("Courier", size, "bold") if bold else ("Courier", size)
    return tk.Label(parent, text=text, font=f, bg=BG_PANEL,
                    fg=color, **kw)

def win95_button(parent, text, cmd, width=10):
    return tk.Button(parent, text=text, command=cmd,
                     font=FONT_MONO_B, bg=BG, fg="#000000",
                     activebackground=BG_DARK, activeforeground="#fff",
                     relief=RELIEF_OUT, bd=3, width=width,
                     cursor="arrow")

def win95_entry(parent, width=20, textvariable=None):
    return tk.Entry(parent, font=FONT_MONO,
                    relief=RELIEF_IN, bd=2,
                    bg="#ffffff", fg="#000000",
                    insertbackground="#000000",
                    width=width,
                    textvariable=textvariable)

def sig_color(sig):
    return COL_BUY if sig=="BUY" else COL_SELL if sig=="SELL" else COL_HOLD

def sig_arrow(sig):
    return "▲" if sig=="BUY" else "▼" if sig=="SELL" else "─"


# ══════════════════════════════════════════════════════════════════════════════
#  HLAVNÍ OKNO
# ══════════════════════════════════════════════════════════════════════════════

class MicovUI:
    def __init__(self, root, tickers, model_dir):
        self.root       = root
        self.tickers    = list(tickers)
        self.model_dir  = model_dir
        self.results    = {}      # ticker → dict
        self.close_cache= {}      # ticker → pd.Series
        self.signals_cache={}     # ticker → (signals, proba, pred_returns, close_window)
        self.selected   = None
        self.q          = queue.Queue()
        self.running    = False
        self._run_mod   = None
        self._meta      = None
        self._cls_model = None
        self._reg_model = None
        self._lang      = "de"
        self._ref_mins  = 30    # Auto-Refresh: 30 oder 60 Minuten

        self._load_backend()
        self._tracker = PredictionTracker() if HAS_TRACKER else None
        self._build_ui()
        self._poll_queue()
        self._start_hourly_tracker()

    # ── Načtení backendu ──────────────────────────────────────────────────────────

    def _load_backend(self):
        try:
            self._run_mod = _import_run(self.model_dir)
        except Exception as e:
            print(f"run.py nicht gefunden: {e}")
            return

        meta_p = Path(self.model_dir) / "model_meta.json"
        cls_p  = Path(self.model_dir) / "model_cls.joblib"
        reg_p  = Path(self.model_dir) / "model_reg.joblib"
        if meta_p.exists() and cls_p.exists():
            try:
                import joblib
                with open(meta_p) as f:
                    self._meta = json.load(f)
                self._cls_model = joblib.load(cls_p)
                self._reg_model = joblib.load(reg_p)
                print(f"  Modell geladen (bis {self._meta['trained_until']})")
            except Exception as e:
                print(f"  Modell-Fehler: {e}")

    # ── Překlad ────────────────────────────────────────────────────────────

    def t(self, key):
        """Vrátí přeložený řetězec, záložní jazyk je němčina."""
        return TRANSLATIONS.get(self._lang, TRANSLATIONS["de"]).get(
               key, TRANSLATIONS["de"].get(key, key))

    # ── Sestavení UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.title("Mičov")
        root.configure(bg=BG)
        root.geometry("1380x860")
        root.minsize(1100, 700)

        # ── Nabídkový pruh ─────────────────────────────────────────────────────────
        menubar = tk.Menu(root, bg=BG, fg="#000", font=FONT_MONO,
                          relief="flat", bd=0)
        self._menubar   = menubar
        file_menu = tk.Menu(menubar, tearoff=0, bg=BG, fg="#000",
                             font=FONT_MONO)
        self._file_menu = file_menu
        file_menu.add_command(label=self.t("menu_scan"), command=self._start_scan)
        file_menu.add_command(label=self.t("menu_stop"), command=self._stop_scan)
        file_menu.add_separator()
        file_menu.add_command(label=self.t("menu_quit"), command=root.destroy)
        menubar.add_cascade(label=self.t("menu_file"), menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0, bg=BG, fg="#000", font=FONT_MONO)
        self._view_menu = view_menu
        view_menu.add_command(label=self.t("menu_charts"), command=self._show_all_charts)
        menubar.add_cascade(label=self.t("menu_view"), menu=view_menu)

        set_menu = tk.Menu(menubar, tearoff=0, bg=BG, fg="#000", font=FONT_MONO)
        self._set_menu = set_menu
        set_menu.add_command(label=self.t("menu_settings"), command=self._open_settings)
        menubar.add_cascade(label=self.t("menu_settings"), menu=set_menu)

        root.config(menu=menubar)

        root.bind("<F5>", lambda e: self._start_scan())
        root.bind("<Escape>", lambda e: self._stop_scan())

        # ── Panel nástrojů ────────────────────────────────────────────────────────────
        toolbar = tk.Frame(root, bg=BG, relief=RELIEF_OUT, bd=2)
        toolbar.pack(fill="x", padx=2, pady=(2,0))

        self._btn_scan = win95_button(toolbar, self.t("btn_scan"), self._start_scan, 12)
        self._btn_scan.pack(side="left", padx=4, pady=3)
        self._btn_stop = win95_button(toolbar, self.t("btn_stop"), self._stop_scan, 12)
        self._btn_stop.pack(side="left", padx=2, pady=3)

        tk.Frame(toolbar, width=2, bg=BG_DARK, relief="sunken"
                 ).pack(side="left", fill="y", padx=6, pady=3)

        self._lbl_ticker_tb = tk.Label(toolbar, text=self.t("lbl_ticker"),
                                        font=FONT_MONO, bg=BG)
        self._lbl_ticker_tb.pack(side="left", padx=(4,2))
        self.var_tickers = tk.StringVar(value=" ".join(self.tickers))
        self.entry_tickers = win95_entry(toolbar, width=55,
                                          textvariable=self.var_tickers)
        self.entry_tickers.pack(side="left", padx=2, pady=3)

        tk.Frame(toolbar, width=2, bg=BG_DARK, relief="sunken"
                 ).pack(side="left", fill="y", padx=6, pady=3)

        self.var_auto = tk.BooleanVar(value=False)
        self._cb_auto = tk.Checkbutton(toolbar,
                        text=f"{self.t('lbl_autoref')} ({self._ref_mins} min)",
                        variable=self.var_auto,
                        font=FONT_MONO, bg=BG, activebackground=BG)
        self._cb_auto.pack(side="left", padx=4)

        tk.Frame(toolbar, width=2, bg=BG_DARK, relief="sunken"
                 ).pack(side="left", fill="y", padx=6, pady=3)

        self.var_lookback = tk.StringVar(value="120")
        self._lbl_lookback_tb = tk.Label(toolbar, text=self.t("lbl_lookback"),
                                          font=FONT_MONO_S, bg=BG)
        self._lbl_lookback_tb.pack(side="left")
        tk.Spinbox(toolbar, from_=20, to=500, increment=20, width=5,
                    textvariable=self.var_lookback,
                    font=FONT_MONO, bg="#fff", relief=RELIEF_IN
                    ).pack(side="left", padx=2)

        # Tlačítko nastavení
        tk.Frame(toolbar, width=2, bg=BG_DARK, relief="sunken"
                 ).pack(side="left", fill="y", padx=6, pady=3)
        win95_button(toolbar, " ⚙ Settings", self._open_settings, 12
                     ).pack(side="left", padx=4, pady=3)

        # Informace o modelu vpravo
        trained = self._meta["trained_until"][:10] if self._meta else "---"
        self._lbl_model_info = tk.Label(toolbar,
                 text=f"  {self.t('lbl_model')} {trained}  |  {self.t('lbl_algo')}",
                 font=FONT_MONO_S, bg=BG, fg="#444")
        self._lbl_model_info.pack(side="right", padx=10)

        # ── Hlavní rozdělené okno ────────────────────────────────────────────────────────
        paned = tk.PanedWindow(root, orient="horizontal",
                                bg=BG, relief="flat",
                                sashwidth=6, sashrelief="raised")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Levá strana: tabulka ───────────────────────────────────────────────
        left_outer = raised_frame(paned, width=560)
        paned.add(left_outer, minsize=400)

        make_title_bar(left_outer, "  Signal-Tabelle")

        # Treeview se scrollbarem
        tree_frame = tk.Frame(left_outer, bg=BG_PANEL)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=4)

        cols = ("ticker","price","signal","conf","forecast",
                "sell_p","hold_p","buy_p","ts")
        self.tree = ttk.Treeview(tree_frame, columns=cols,
                                  show="headings", height=22)

        # Styl widgetu
        style = ttk.Style()
        style.theme_use("classic")
        style.configure("Treeview",
                         background="#ffffff",
                         foreground="#000000",
                         rowheight=22,
                         fieldbackground="#ffffff",
                         font=("Courier", 10))
        style.configure("Treeview.Heading",
                         background=BG,
                         foreground="#000000",
                         font=("Courier", 9, "bold"),
                         relief="raised")
        style.map("Treeview",
                   background=[("selected","#000080")],
                   foreground=[("selected","#ffffff")])

        hdrs = {
            "ticker":   ("TICKER",   70),
            "price":    ("KURS ($)", 90),
            "signal":   ("SIGNAL",   90),
            "conf":     ("KONF.",    60),
            "forecast": ("+4h %",    75),
            "sell_p":   ("SELL%",    60),
            "hold_p":   ("HOLD%",    60),
            "buy_p":    ("BUY%",     60),
            "ts":       ("ZEITPUNKT",130),
        }
        for col, (hdr, w) in hdrs.items():
            self.tree.heading(col, text=hdr,
                               command=lambda c=col: self._sort_tree(c))
            self.tree.column(col, width=w, anchor="center", minwidth=40)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Barevné tagy
        self.tree.tag_configure("buy",  foreground=COL_BUY,  font=("Courier",10,"bold"))
        self.tree.tag_configure("sell", foreground=COL_SELL, font=("Courier",10,"bold"))
        self.tree.tag_configure("hold", foreground=COL_HOLD)
        self.tree.tag_configure("even", background="#f0f0f0")
        self.tree.tag_configure("odd",  background="#ffffff")

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_double_click)

        # Souhrnný řádek
        sum_f = sunken_frame(left_outer)
        sum_f.pack(fill="x", padx=4, pady=(0,4))
        self.lbl_summary = tk.Label(sum_f,
                                     text="  Bereit. F5 drücken um Scan zu starten.",
                                     font=FONT_MONO_S, bg=BG_PANEL, anchor="w")
        self.lbl_summary.pack(fill="x", padx=4, pady=3)

        # ── Pravá strana: grafy ───────────────────────────────────────────────
        right_outer = raised_frame(paned)
        paned.add(right_outer, minsize=500)

        make_title_bar(right_outer, "  Chart-Analyse")

        # Notebook (záložky) pro 3 typy grafů
        nb = ttk.Notebook(right_outer)
        nb.pack(fill="both", expand=True, padx=4, pady=4)
        style.configure("TNotebook", background=BG_PANEL)
        style.configure("TNotebook.Tab", font=FONT_MONO, padding=[8,4])

        # Záložka 1: Signální graf
        self.tab_signal = tk.Frame(nb, bg=BG_PANEL)
        nb.add(self.tab_signal, text="  Signal-Chart  ")
        self.fig_sig = Figure(facecolor=MPL_BG)
        self.canvas_sig = FigureCanvasTkAgg(self.fig_sig, master=self.tab_signal)
        self.canvas_sig.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty(self.fig_sig, "Signal-Chart\nTicker aus Tabelle auswählen")

        # Záložka 2: Předpověď ceny
        self.tab_pred = tk.Frame(nb, bg=BG_PANEL)
        nb.add(self.tab_pred, text="  Preis-Prognose  ")
        self.fig_pred = Figure(facecolor=MPL_BG)
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=self.tab_pred)
        self.canvas_pred.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty(self.fig_pred, "Preis-Prognose\n+4h Vorhersage-Pfeil")

        # Záložka 3: Graf pravděpodobnosti
        self.tab_proba = tk.Frame(nb, bg=BG_PANEL)
        nb.add(self.tab_proba, text="  Wahrscheinlichkeit  ")
        self.fig_proba = Figure(facecolor=MPL_BG)
        self.canvas_proba = FigureCanvasTkAgg(self.fig_proba, master=self.tab_proba)
        self.canvas_proba.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty(self.fig_proba, "Wahrscheinlichkeit\nSELL / HOLD / BUY über Zeit")

        # Záložka 4: Živý backtest / přesnost
        self.tab_acc = tk.Frame(nb, bg=BG_PANEL)
        nb.add(self.tab_acc, text="  Accuracy-Tracker  ")
        self._build_accuracy_tab(self.tab_acc)

        self.nb = nb

        # Detail panel pod grafy
        detail_outer = sunken_frame(right_outer)
        detail_outer.pack(fill="x", padx=4, pady=(0,4))

        make_title_bar(detail_outer, "  Detail")

        detail_inner = tk.Frame(detail_outer, bg=BG_PANEL)
        detail_inner.pack(fill="x", padx=6, pady=4)

        # 3 sloupce: signál, ceny, pravděpodobnosti
        c1 = tk.Frame(detail_inner, bg=BG_PANEL)
        c1.pack(side="left", padx=8, anchor="n")
        c2 = tk.Frame(detail_inner, bg=BG_PANEL)
        c2.pack(side="left", padx=20, anchor="n")
        c3 = tk.Frame(detail_inner, bg=BG_PANEL)
        c3.pack(side="left", padx=8, anchor="n")

        self.det = {}
        def drow(frame, key, lbl, val="─", col="#000"):
            tk.Label(frame, text=lbl, font=FONT_MONO_S,
                     bg=BG_PANEL, fg="#444", anchor="w", width=14
                     ).pack(anchor="w")
            v = tk.Label(frame, text=val, font=FONT_MONO_B,
                          bg=BG_PANEL, fg=col, anchor="w")
            v.pack(anchor="w")
            self.det[key] = v

        drow(c1, "ticker",   "Ticker:",    "─")
        drow(c1, "signal",   "Signal:",    "─")
        drow(c1, "conf",     "Konfidenz:", "─")
        drow(c2, "price",    "Kurs ($):",  "─")
        drow(c2, "pred_p",   "Prognose $:","─")
        drow(c2, "pred_r",   "Prognose %:","─")
        drow(c3, "sell_p",   "SELL%:",     "─", COL_SELL)
        drow(c3, "hold_p",   "HOLD%:",     "─", COL_HOLD)
        drow(c3, "buy_p",    "BUY%:",      "─", COL_BUY)

        # ── Stavový řádek ───────────────────────────────────────────────────────
        statusbar = tk.Frame(root, bg=BG_DARK, relief="sunken", bd=1)
        statusbar.pack(fill="x", side="bottom")

        self.lbl_status = tk.Label(statusbar, text="  Bereit",
                                    font=FONT_MONO_S,
                                    bg=BG_DARK, fg="#ffffff", anchor="w")
        self.lbl_status.pack(side="left", fill="x", expand=True, padx=4, pady=2)

        self.lbl_time = tk.Label(statusbar, text="",
                                  font=FONT_MONO_S,
                                  bg=BG_DARK, fg="#aaaaaa", anchor="e")
        self.lbl_time.pack(side="right", padx=8, pady=2)

        self.progressbar = ttk.Progressbar(statusbar, length=200,
                                            mode="determinate")
        self.progressbar.pack(side="right", padx=8, pady=3)
        style.configure("TProgressbar", troughcolor=BG_DARK,
                         background="#00cc00", thickness=14)

    # ── Zpracování fronty zpráv ──────────────────────────────────────────────────────────

    def _poll_queue(self):
        """Verarbeitet Nachrichten aus dem Worker-Thread."""
        try:
            while True:
                msg = self.q.get_nowait()
                mtype = msg[0]

                if mtype == "status":
                    self.lbl_status.configure(text=f"  {msg[1]}")

                elif mtype == "progress":
                    n, total = msg[1], msg[2]
                    self.progressbar["value"] = n / total * 100

                elif mtype == "result":
                    ticker, result, close_s, sig_data = msg[1], msg[2], msg[3], msg[4]
                    self.results[ticker] = result
                    self.close_cache[ticker] = close_s
                    self.signals_cache[ticker] = sig_data
                    self._upsert_row(ticker, result)

                elif mtype == "tracker_eval":
                    evals = msg[1]
                    if evals:
                        self.lbl_status.configure(
                            text=f"  Tracker: {len(evals)} Predictions ausgewertet")
                        threading.Thread(target=self._update_acc_ui, daemon=True).start()

                elif mtype == "tracker_time":
                    next_t = msg[1]
                    try:
                        self.lbl_next_eval.configure(
                            text=f"  Nächste Auswertung: {next_t}")
                    except:
                        pass

                elif mtype == "refresh_acc":
                    threading.Thread(target=self._update_acc_ui, daemon=True).start()

                elif mtype == "acc_stats":
                    self._apply_acc_stats(msg[1])

                elif mtype == "done":
                    self.running = False
                    self.progressbar["value"] = 100
                    self._sort_tree("signal")
                    self._update_summary()
                    now = datetime.now().strftime("%H:%M:%S")
                    self.lbl_time.configure(
                        text=f"Letzter Scan: {now}   ")
                    # Auto-Refresh mit einstellbarem Intervall
                    if self.var_auto.get():
                        ms = self._ref_mins * 60 * 1000
                        self.root.after(ms, self._start_scan)

                elif mtype == "error":
                    self.lbl_status.configure(
                        text=f"  FEHLER: {msg[1]}", fg="#ff0000" if False else "#ff4444")

        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # ── Scan ───────────────────────────────────────────────────────────────────

    def _start_scan(self):
        if self.running:
            return
        raw = self.var_tickers.get().strip().upper().split()
        self.tickers = [t for t in raw if t]
        if not self.tickers:
            return
        self.running = True
        self.progressbar["value"] = 0
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _stop_scan(self):
        self.running = False
        self.q.put(("status", "Gestoppt."))

    def _scan_worker(self):
        total    = len(self.tickers)
        lookback = int(self.var_lookback.get() or 120)

        for i, ticker in enumerate(self.tickers):
            if not self.running:
                break
            self.q.put(("status", f"Scanne {ticker}  ({i+1}/{total})..."))
            self.q.put(("progress", i, total))

            try:
                result, close_s, sig_data = self._compute(ticker, lookback)
                self.q.put(("result", ticker, result, close_s, sig_data))
            except Exception as e:
                self.q.put(("error", f"{ticker}: {e}"))

        self.q.put(("progress", total, total))
        self.q.put(("status",   "Scan abgeschlossen."))
        # Predictions direkt nach Scan loggen
        if HAS_TRACKER and hasattr(self, '_tracker') and self._tracker:
            for ticker, result in self.results.items():
                try:
                    self._tracker.log_prediction(
                        ticker=ticker, price=result["price"],
                        signal=result["signal"],
                        confidence=result["confidence"],
                        pred_return=result["pred_return"],
                    )
                except:
                    pass
            self.q.put(("status", f"Scan + {len(self.results)} Predictions gespeichert"))

        self.q.put(("done",))

    def _compute(self, ticker, lookback):
        """Läuft im Worker-Thread. Gibt (result, close_series, sig_data) zurück."""
        if not self._run_mod or not self._meta:
            raise RuntimeError("Kein Modell geladen")

        # Ticker ID
        ticker_id = 0
        tp = Path(self.model_dir) / "dataset_tickers.json"
        if tp.exists():
            with open(tp) as f:
                tmap = json.load(f)
            ticker_id = tmap.get("ticker_id_map", {}).get(ticker, 0)

        # Načtení příznaků a série uzavíracích cen
        h_feat, close_series = self._run_mod.fetch_features(
            ticker, ticker_id, self._meta)

        features = self._meta["features"]
        medians  = pd.Series(self._meta["medians"])

        available = [f for f in features if f in h_feat.columns]
        X_slice   = h_feat[available].iloc[-lookback:].copy()
        for m in [f for f in features if f not in h_feat.columns]:
            X_slice[m] = medians.get(m, 0.0)
        X_slice = X_slice[features].fillna(medians).replace([np.inf,-np.inf], 0.0)

        signals      = self._cls_model.predict(X_slice)
        proba        = self._cls_model.predict_proba(X_slice)
        pred_returns = self._reg_model.predict(X_slice)
        close_window = close_series.iloc[-lookback:].reindex(X_slice.index)

        last_signal = signals[-1]
        last_proba  = proba[-1]
        last_ret    = float(pred_returns[-1])
        last_price  = float(close_series.iloc[-1])
        pred_price  = last_price * (1 + last_ret)
        lmap        = self._meta["label_map"]

        result = {
            "ticker":      ticker,
            "timestamp":   str(close_series.index[-1]),
            "price":       last_price,
            "signal":      lmap[str(last_signal)],
            "confidence":  float(last_proba[last_signal]),
            "pred_price":  pred_price,
            "pred_return": last_ret,
            "proba":       last_proba.tolist(),
        }

        sig_data = (signals, proba, pred_returns, close_window)
        return result, close_series, sig_data

    # ── Tabellen-Updates ───────────────────────────────────────────────────────

    def _upsert_row(self, ticker, r):
        sig    = r["signal"]
        fc     = r["pred_return"] * 100
        p      = r.get("proba", [0,0,0])
        ts_str = r["timestamp"][:16] if len(r["timestamp"]) >= 16 else r["timestamp"]

        vals = (
            ticker,
            f"{r['price']:.2f}",
            f"{sig_arrow(sig)} {sig}",
            f"{r['confidence']:.0%}",
            f"{fc:+.2f}%",
            f"{p[0]:.0%}",
            f"{p[1]:.0%}",
            f"{p[2]:.0%}",
            ts_str,
        )
        tag = sig.lower()

        # Aktualizace existujícího řádku nebo vložení nového
        existing = self.tree.get_children()
        for iid in existing:
            if self.tree.item(iid, "values")[0] == ticker:
                self.tree.item(iid, values=vals, tags=(tag,))
                return

        row_idx = len(self.tree.get_children())
        even_tag = "even" if row_idx % 2 == 0 else "odd"
        self.tree.insert("", "end", iid=ticker, values=vals, tags=(tag, even_tag))

    def _sort_tree(self, col):
        """Seřadí tabulku podle sloupce. Signál: BUY > HOLD > SELL."""
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        if col == "signal":
            order = {"▲ BUY": 0, "─ HOLD": 1, "▼ SELL": 2}
            items.sort(key=lambda x: order.get(x[0], 3))
        elif col in ("price","conf","forecast","sell_p","hold_p","buy_p"):
            try:
                items.sort(key=lambda x: float(
                    x[0].replace("%","").replace("+","").replace("$","").replace("▲ ","").replace("▼ ","")
                ), reverse=True)
            except:
                items.sort()
        else:
            items.sort()
        for idx, (_, k) in enumerate(items):
            self.tree.move(k, "", idx)
            tag = self.tree.item(k, "tags")[0] if self.tree.item(k,"tags") else "hold"
            even_tag = "even" if idx % 2 == 0 else "odd"
            self.tree.item(k, tags=(tag, even_tag))

    def _update_summary(self):
        buys  = [r for r in self.results.values() if r["signal"]=="BUY"]
        sells = [r for r in self.results.values() if r["signal"]=="SELL"]
        holds = [r for r in self.results.values() if r["signal"]=="HOLD"]
        buys.sort(key=lambda x: -x["confidence"])
        msg = (f"  Ergebnisse: {len(self.results)} Ticker  |  "
               f"BUY: {len(buys)}  SELL: {len(sells)}  HOLD: {len(holds)}")
        if buys:
            b = buys[0]
            msg += f"  |  Stärkstes BUY: {b['ticker']} ({b['confidence']:.0%})"
        self.lbl_summary.configure(text=msg)

    # ── Výběr tickeru → grafy ──────────────────────────────────────────────

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        ticker = sel[0]
        self.selected = ticker
        self._update_detail(ticker)
        self._draw_all_charts(ticker)

    def _on_double_click(self, event):
        sel = self.tree.selection()
        if sel:
            self.nb.select(0)  # Signal-Tab aktivieren

    def _update_detail(self, ticker):
        r = self.results.get(ticker)
        if not r:
            return
        sig = r["signal"]
        p   = r.get("proba", [0,0,0])
        fc  = r["pred_return"] * 100
        sc  = sig_color(sig)

        self.det["ticker"].configure(text=ticker)
        self.det["signal"].configure(
            text=f"{sig_arrow(sig)} {sig}", fg=sc)
        self.det["conf"].configure(
            text=f"{r['confidence']:.1%}", fg=sc)
        self.det["price"].configure(text=f"${r['price']:.2f}")
        self.det["pred_p"].configure(
            text=f"${r['pred_price']:.2f}",
            fg=COL_GAIN if fc >= 0 else COL_LOSS)
        self.det["pred_r"].configure(
            text=f"{fc:+.2f}%",
            fg=COL_GAIN if fc >= 0 else COL_LOSS)
        self.det["sell_p"].configure(text=f"{p[0]:.1%}", fg=COL_SELL)
        self.det["hold_p"].configure(text=f"{p[1]:.1%}", fg=COL_HOLD)
        self.det["buy_p"].configure( text=f"{p[2]:.1%}", fg=COL_BUY)

    def _draw_all_charts(self, ticker):
        t1 = threading.Thread(target=self._draw_signal_chart, args=(ticker,), daemon=True)
        t2 = threading.Thread(target=self._draw_pred_chart,   args=(ticker,), daemon=True)
        t3 = threading.Thread(target=self._draw_proba_chart,  args=(ticker,), daemon=True)
        t1.start(); t2.start(); t3.start()

    # ── Signální graf ──────────────────────────────────────────────────────────

    def _draw_signal_chart(self, ticker):
        close_s  = self.close_cache.get(ticker)
        sig_data = self.signals_cache.get(ticker)
        result   = self.results.get(ticker)
        if close_s is None or sig_data is None:
            return

        signals, proba, pred_returns, close_window = sig_data

        fig = self.fig_sig
        fig.clear()

        gs = gridspec.GridSpec(3, 1, figure=fig,
                                height_ratios=[3,1,1], hspace=0.15)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors=MPL_FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(MPL_GRID)

        df = pd.DataFrame({
            "close":  close_window.values,
            "signal": signals,
        }, index=close_window.index).dropna()

        # Panel 1: Kurz + signály
        ax1.plot(df.index, df["close"], color=MPL_PRICE, linewidth=1.2,
                  label="Kurs", zorder=3)

        buy_idx  = df[df["signal"]==2].index
        sell_idx = df[df["signal"]==0].index

        if len(buy_idx):
            ax1.scatter(buy_idx, df.loc[buy_idx,"close"],
                         color=MPL_BUY, marker="^", s=50, zorder=5, label="BUY")
        if len(sell_idx):
            ax1.scatter(sell_idx, df.loc[sell_idx,"close"],
                         color=MPL_SELL, marker="v", s=50, zorder=5, label="SELL")

        if len(df) >= 20:
            ax1.plot(df.index, df["close"].rolling(20).mean(),
                      color=MPL_MA20, linewidth=0.9, alpha=0.8,
                      linestyle="--", label="MA20")
        if len(df) >= 50:
            ax1.plot(df.index, df["close"].rolling(50).mean(),
                      color=MPL_MA50, linewidth=0.9, alpha=0.8,
                      linestyle=":", label="MA50")

        sig   = result["signal"] if result else "─"
        price = result["price"]  if result else 0
        ax1.set_title(f"{ticker}  ${price:.2f}  [{sig_arrow(sig)} {sig}]",
                       fontsize=10, color=MPL_FG,
                       fontfamily="Courier New", pad=4)
        ax1.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax1.grid(True, color=MPL_GRID, alpha=0.4, linewidth=0.5)
        ax1.set_ylabel("Kurs ($)", fontsize=8, color=MPL_FG)

        # Panel 2: Sloupce signálů
        bar_colors = [MPL_BUY if s==2 else MPL_SELL if s==0 else MPL_GRID
                       for s in df["signal"]]
        ax2.bar(df.index, df["signal"], color=bar_colors, width=0.04, alpha=0.8)
        ax2.set_yticks([0,1,2])
        ax2.set_yticklabels(["SELL","HOLD","BUY"], fontsize=7)
        ax2.set_ylabel("Signal", fontsize=7)
        ax2.grid(True, color=MPL_GRID, alpha=0.3, linewidth=0.5)

        # Panel 3: RSI
        rsi_vals = self._rsi(pd.Series(df["close"].values), 14)
        ax3.plot(df.index, rsi_vals, color="saddlebrown", linewidth=0.9)
        ax3.axhline(70, color=MPL_SELL, linestyle="--", alpha=0.6, linewidth=0.8)
        ax3.axhline(30, color=MPL_BUY,  linestyle="--", alpha=0.6, linewidth=0.8)
        ax3.fill_between(df.index, rsi_vals, 70,
                          where=(rsi_vals >= 70), alpha=0.2, color=MPL_SELL)
        ax3.fill_between(df.index, rsi_vals, 30,
                          where=(rsi_vals <= 30), alpha=0.2, color=MPL_BUY)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI(14)", fontsize=7)
        ax3.grid(True, color=MPL_GRID, alpha=0.3, linewidth=0.5)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        fig.autofmt_xdate(rotation=15)
        fig.set_facecolor(MPL_BG)
        fig.tight_layout()
        self.root.after(0, self.canvas_sig.draw)

    # ── Preis-Prognose-Chart ──────────────────────────────────────────────────

    def _draw_pred_chart(self, ticker):
        close_s = self.close_cache.get(ticker)
        result  = self.results.get(ticker)
        if close_s is None or result is None:
            return

        fig = self.fig_pred
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors=MPL_FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(MPL_GRID)

        recent = close_s.iloc[-100:] if len(close_s) > 100 else close_s
        ax.plot(recent.index, recent.values,
                 color=MPL_PRICE, linewidth=1.4, label="Historisch", zorder=3)

        # MA
        if len(recent) >= 20:
            ma20 = recent.rolling(20).mean()
            ax.plot(recent.index, ma20, color=MPL_MA20, linewidth=0.8,
                     linestyle="--", alpha=0.7, label="MA20")

        # Prognose-Pfeil
        last_price = float(recent.iloc[-1])
        last_time  = recent.index[-1]
        pred_ret   = result["pred_return"]
        pred_price = result["pred_price"]
        arrow_col  = MPL_BUY if pred_ret >= 0 else MPL_SELL

        try:
            diffs = pd.Series(recent.index.astype(np.int64)).diff().dropna()
            step  = pd.Timedelta(int(diffs.median()), unit="ns")
            pred_time = last_time + step * 4
            ax.annotate(
                f"+4h: ${pred_price:.2f}\n({pred_ret*100:+.2f}%)",
                xy=(pred_time, pred_price),
                xytext=(last_time, last_price),
                fontsize=8, color=arrow_col,
                fontfamily="Courier New",
                arrowprops=dict(arrowstyle="-|>", color=arrow_col,
                                lw=2.2, mutation_scale=16),
                bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#ffffcc", edgecolor=arrow_col,
                           alpha=0.9),
            )
            ax.plot([last_time, pred_time],
                     [last_price, pred_price],
                     color=arrow_col, linewidth=1.5,
                     linestyle="--", alpha=0.6, zorder=2)
            ax.scatter([pred_time], [pred_price],
                        color=arrow_col, s=60, zorder=5)
        except Exception as e:
            pass

        ax.axvline(last_time, color=MPL_GRID, linestyle=":",
                    alpha=0.7, linewidth=1.0, label="Jetzt")

        sig = result["signal"]
        ax.set_title(f"{ticker}  Preis-Prognose +4h   "
                      f"[{sig_arrow(sig)} {sig}  {result['confidence']:.0%}]",
                      fontsize=10, color=MPL_FG,
                      fontfamily="Courier New", pad=4)
        ax.set_ylabel("Kurs ($)", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(True, color=MPL_GRID, alpha=0.4, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        fig.autofmt_xdate(rotation=15)
        fig.set_facecolor(MPL_BG)
        fig.tight_layout()
        self.root.after(0, self.canvas_pred.draw)

    # ── Wahrscheinlichkeits-Chart ─────────────────────────────────────────────

    def _draw_proba_chart(self, ticker):
        sig_data = self.signals_cache.get(ticker)
        result   = self.results.get(ticker)
        if sig_data is None:
            return

        signals, proba, pred_returns, close_window = sig_data
        idx = close_window.index

        fig = self.fig_proba
        fig.clear()

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1,2], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        for ax in [ax1, ax2]:
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors=MPL_FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(MPL_GRID)

        # Panel 1: Kurs
        ax1.plot(idx, close_window.values, color=MPL_PRICE,
                  linewidth=1.2, label="Kurs")
        ax1.set_ylabel("Kurs ($)", fontsize=7)
        ax1.grid(True, color=MPL_GRID, alpha=0.4, linewidth=0.5)

        # Panel 2: gestapeltes Wahrscheinlichkeits-Flächendiagramm
        p_sell = proba[:, 0]
        p_hold = proba[:, 1]
        p_buy  = proba[:, 2]

        ax2.stackplot(idx,
                       p_sell, p_hold, p_buy,
                       labels=["SELL%","HOLD%","BUY%"],
                       colors=[MPL_SELL, "#aaaaaa", MPL_BUY],
                       alpha=0.75)
        ax2.axhline(0.5, color=MPL_FG, linewidth=0.6,
                     linestyle="--", alpha=0.4)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Wahrscheinlichkeit", fontsize=7)
        ax2.legend(fontsize=7, loc="upper left", framealpha=0.8)
        ax2.grid(True, color=MPL_GRID, alpha=0.3, linewidth=0.5)

        sig = result["signal"] if result else "─"
        ax1.set_title(f"{ticker}  SELL/HOLD/BUY Wahrscheinlichkeit über Zeit",
                       fontsize=9, color=MPL_FG,
                       fontfamily="Courier New", pad=4)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        fig.autofmt_xdate(rotation=15)
        fig.set_facecolor(MPL_BG)
        fig.tight_layout()
        self.root.after(0, self.canvas_proba.draw)

    # ── Alle Charts auf einmal ─────────────────────────────────────────────────

    def _show_all_charts(self):
        sel = self.tree.selection()
        if sel:
            self._draw_all_charts(sel[0])

    # ── Settings-Dialog ───────────────────────────────────────────────────────

    def _open_settings(self):
        """Öffnet modales Einstellungs-Fenster."""
        dlg = tk.Toplevel(self.root)
        dlg.title(self.t("settings_title"))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()   # modal

        # Zentrieren
        self.root.update_idletasks()
        rx = self.root.winfo_x() + self.root.winfo_width()  // 2
        ry = self.root.winfo_y() + self.root.winfo_height() // 2
        dlg.geometry(f"340x220+{rx-170}+{ry-110}")

        make_title_bar(dlg, f"  {self.t('settings_title')}")

        body = tk.Frame(dlg, bg=BG_PANEL, relief=RELIEF_IN, bd=2)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # Sprache
        tk.Label(body, text=self.t("settings_lang"),
                 font=FONT_MONO_B, bg=BG_PANEL
                 ).grid(row=0, column=0, padx=12, pady=(14,4), sticky="w")

        var_lang = tk.StringVar(value=self._lang)
        lang_frame = tk.Frame(body, bg=BG_PANEL)
        lang_frame.grid(row=0, column=1, padx=8, pady=(14,4), sticky="w")

        lang_options = list(LANG_NAMES.keys())
        lang_display = [f"{LANG_NAMES[k]}" for k in lang_options]
        lang_cb = ttk.Combobox(lang_frame, values=lang_display,
                                width=16, font=FONT_MONO,
                                state="readonly")
        # Aktuellen Index setzen
        try:
            lang_cb.current(lang_options.index(self._lang))
        except:
            lang_cb.current(0)
        lang_cb.pack()

        # Refresh-Intervall
        tk.Label(body, text=self.t("settings_ref"),
                 font=FONT_MONO_B, bg=BG_PANEL
                 ).grid(row=1, column=0, padx=12, pady=8, sticky="w")

        var_ref = tk.StringVar(value=self.t("settings_30") if self._ref_mins==30
                                else self.t("settings_60"))
        ref_frame = tk.Frame(body, bg=BG_PANEL)
        ref_frame.grid(row=1, column=1, padx=8, pady=8, sticky="w")

        rb30 = tk.Radiobutton(ref_frame, text=self.t("settings_30"),
                               variable=var_ref,
                               value=self.t("settings_30"),
                               font=FONT_MONO, bg=BG_PANEL,
                               activebackground=BG_PANEL)
        rb30.pack(anchor="w")
        rb60 = tk.Radiobutton(ref_frame, text=self.t("settings_60"),
                               variable=var_ref,
                               value=self.t("settings_60"),
                               font=FONT_MONO, bg=BG_PANEL,
                               activebackground=BG_PANEL)
        rb60.pack(anchor="w")

        # Trennlinie (grid-kompatibel, kein pack!)
        tk.Frame(body, height=2, bg=BG_DARK, relief="sunken"
                 ).grid(row=2, column=0, columnspan=2,
                        sticky="ew", padx=8, pady=6)

        def _apply():
            # Sprache
            idx = lang_cb.current()
            if 0 <= idx < len(lang_options):
                self._lang = lang_options[idx]

            # Intervall (30 oder 60 min)
            val = var_ref.get()
            # Wert ist in der aktuell gewählten Sprache — prüfe auf 60-min String
            tr = TRANSLATIONS.get(self._lang, TRANSLATIONS["de"])
            if val == tr.get("settings_60") or val == "1 Stunde" or val == "1 Hour" or "60" in val or "1 h" in val.lower():
                self._ref_mins = 60
            else:
                self._ref_mins = 30

            dlg.destroy()
            self._apply_language()

        btn_row = tk.Frame(body, bg=BG_PANEL)
        btn_row.grid(row=3, column=0, columnspan=2, pady=(4,10))
        win95_button(btn_row, self.t("settings_ok"), _apply, 10).pack()

        dlg.bind("<Return>", lambda e: _apply())
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def _apply_language(self):
        """Aktualisiert alle UI-Texte nach Sprachänderung (kein Neustart nötig)."""
        self.root.title(self.t("title"))

        # Toolbar
        self._btn_scan.configure(text=self.t("btn_scan"))
        self._btn_stop.configure(text=self.t("btn_stop"))
        self._lbl_ticker_tb.configure(text=self.t("lbl_ticker"))
        self._lbl_lookback_tb.configure(text=self.t("lbl_lookback"))
        self._cb_auto.configure(
            text=f"{self.t('lbl_autoref')} ({self._ref_mins} min)")
        trained = self._meta["trained_until"][:10] if self._meta else "---"
        self._lbl_model_info.configure(
            text=f"  {self.t('lbl_model')} {trained}  |  {self.t('lbl_algo')}")

        # Menü komplett neu aufbauen (sicherste Methode für macOS)
        try:
            self._file_menu.delete(0, "end")
            self._file_menu.add_command(label=self.t("menu_scan"), command=self._start_scan)
            self._file_menu.add_command(label=self.t("menu_stop"), command=self._stop_scan)
            self._file_menu.add_separator()
            self._file_menu.add_command(label=self.t("menu_quit"), command=self.root.destroy)

            self._view_menu.delete(0, "end")
            self._view_menu.add_command(label=self.t("menu_charts"), command=self._show_all_charts)

            self._set_menu.delete(0, "end")
            self._set_menu.add_command(label=self.t("menu_settings"), command=self._open_settings)

            # Cascade-Labels (Menü-Titel) neu setzen
            self._menubar.entryconfigure(1, label=self.t("menu_file"))
            self._menubar.entryconfigure(2, label=self.t("menu_view"))
            self._menubar.entryconfigure(3, label=self.t("menu_settings"))
        except Exception:
            pass  # Menü-Update optional — Rest-UI trotzdem updaten

        # Tabellen-Header
        hdrs = {
            "ticker":  self.t("col_ticker"),
            "price":   self.t("col_price"),
            "signal":  self.t("col_signal"),
            "conf":    self.t("col_conf"),
            "forecast":self.t("col_forecast"),
            "sell_p":  self.t("col_sell"),
            "hold_p":  self.t("col_hold"),
            "buy_p":   self.t("col_buy"),
            "ts":      self.t("col_ts"),
        }
        for col, hdr in hdrs.items():
            self.tree.heading(col, text=hdr)

        # Statusbar + Summary
        self.lbl_status.configure(text=self.t("status_ready"))
        self.lbl_summary.configure(text=self.t("summary_ready"))

        # Notebook Tabs
        try:
            self.nb.tab(0, text=self.t("tab_signal"))
            self.nb.tab(1, text=self.t("tab_pred"))
            self.nb.tab(2, text=self.t("tab_proba"))
            self.nb.tab(3, text=self.t("tab_acc"))
        except Exception:
            pass

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    # ── Sestavení záložky přesnosti ─────────────────────────────────────────────────

    def _build_accuracy_tab(self, parent):
        """Tab 4: Live-Backtesting — 1h/4h getrennt + Chart-Ähnlichkeit."""

        # ── Tabelle oben ───────────────────────────────────────────────────────
        top = raised_frame(parent)
        top.pack(fill="both", expand=True, padx=4, pady=(4,2))
        make_title_bar(top,
            "  Richtungs-Genauigkeit: 1h / 4h — Chart-Ähnlichkeit (Cosine / MAE%)")

        acc_cols = ("ticker","total","eval_1h","ok_1h","acc_1h",
                    "eval_4h","ok_4h","acc_4h","cosine","mae","avg_ret","pending")
        self.acc_tree = ttk.Treeview(top, columns=acc_cols,
                                      show="headings", height=10)
        acc_hdrs = {
            "ticker":  ("TICKER",    60),
            "total":   ("GESAMT",    55),
            "eval_1h": ("Ausw.1h",   62),
            "ok_1h":   ("OK 1h",     55),
            "acc_1h":  ("Acc. 1h",   65),
            "eval_4h": ("Ausw.4h",   62),
            "ok_4h":   ("OK 4h",     55),
            "acc_4h":  ("Acc. 4h",   65),
            "cosine":  ("Chart-Cos.",75),
            "mae":     ("MAE%",      60),
            "avg_ret": ("Ø Rendite", 75),
            "pending": ("Offen",     50),
        }
        for col, (hdr, w) in acc_hdrs.items():
            self.acc_tree.heading(col, text=hdr)
            self.acc_tree.column(col, width=w, anchor="center", minwidth=36)
        self.acc_tree.tag_configure("good", foreground=COL_BUY,
                                     font=("Courier",10,"bold"))
        self.acc_tree.tag_configure("ok",   foreground=COL_BUY)
        self.acc_tree.tag_configure("bad",  foreground=COL_SELL)
        self.acc_tree.tag_configure("even", background="#f0f0f0")
        self.acc_tree.tag_configure("odd",  background="#ffffff")

        acc_vsb = ttk.Scrollbar(top, orient="vertical",
                                  command=self.acc_tree.yview)
        self.acc_tree.configure(yscrollcommand=acc_vsb.set)
        self.acc_tree.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        acc_vsb.pack(side="right", fill="y", pady=4)

        # ── Mitte: Gesamtstatistik + Steuerung ────────────────────────────────
        mid = tk.Frame(parent, bg=BG_PANEL)
        mid.pack(fill="x", padx=4, pady=2)

        stat_f = sunken_frame(mid)
        stat_f.pack(side="left", padx=(0,4), fill="both", expand=True)
        make_title_bar(stat_f, "  Gesamtstatistik")
        self.lbl_acc_1h = tk.Label(stat_f, text="  1h: warte...",
                                    font=("Courier",10), bg=BG_PANEL,
                                    fg="#000", anchor="w")
        self.lbl_acc_1h.pack(fill="x", padx=8, pady=(4,0))
        self.lbl_acc_4h = tk.Label(stat_f, text="  4h: warte...",
                                    font=("Courier",10), bg=BG_PANEL,
                                    fg="#000", anchor="w")
        self.lbl_acc_4h.pack(fill="x", padx=8, pady=0)
        self.lbl_acc_cos = tk.Label(stat_f, text="  Chart-Ähnlichkeit: warte...",
                                     font=("Courier",10), bg=BG_PANEL,
                                     fg="#000", anchor="w")
        self.lbl_acc_cos.pack(fill="x", padx=8, pady=(0,4))

        ctrl_f = raised_frame(mid)
        ctrl_f.pack(side="left", fill="y", padx=(4,0))
        make_title_bar(ctrl_f, "  Steuerung")
        win95_button(ctrl_f, "Jetzt auswerten", self._force_evaluate, 16
                     ).pack(padx=8, pady=6)
        win95_button(ctrl_f, "Tabelle refresh", self._refresh_acc_table, 16
                     ).pack(padx=8, pady=4)
        self.lbl_next_eval = tk.Label(ctrl_f,
                                       text="  Nächste Auswertung: --:--",
                                       font=FONT_MONO_S, bg=BG_PANEL, fg="#444")
        self.lbl_next_eval.pack(padx=8, pady=4)

        # ── Chart: 1h/4h Accuracy + Cosine getrennt ───────────────────────────
        chart_f = raised_frame(parent)
        chart_f.pack(fill="both", expand=True, padx=4, pady=(2,4))
        make_title_bar(chart_f,
            "  Acc. 1h (blau) / 4h (grün) / Chart-Cosine (orange)  [rolling 10]")
        self.fig_acc = Figure(facecolor=MPL_BG)
        self.canvas_acc = FigureCanvasTkAgg(self.fig_acc, master=chart_f)
        self.canvas_acc.get_tk_widget().pack(fill="both", expand=True,
                                              padx=4, pady=4)
        self._draw_empty(self.fig_acc,
            "Accuracy-Chart\nWarte auf erste ausgewerteten Predictions...")

        # ── Log unten ──────────────────────────────────────────────────────────
        log_f = sunken_frame(parent)
        log_f.pack(fill="x", padx=4, pady=(0,4))
        make_title_bar(log_f, "  Letzte ausgewerteten Vorhersagen (4h)")
        self.acc_log = tk.Text(log_f, height=5, font=FONT_MONO_S,
                                bg="#ffffff", fg="#000", relief=RELIEF_IN,
                                bd=2, state="disabled")
        self.acc_log.pack(fill="x", padx=4, pady=4)
        self.acc_log.tag_configure("ok",   foreground=COL_BUY)
        self.acc_log.tag_configure("fail", foreground=COL_SELL)
        self.acc_log.tag_configure("neut", foreground="#555555")
        self.acc_log.tag_configure("head", foreground="#000080",
                                    font=("Courier",9,"bold"))

    # ── Hodinový tracker-Thread ─────────────────────────────────────────────

    def _start_hourly_tracker(self):
        """Startet den Background-Thread der stündlich predictions logt und auswertet."""
        if not self._tracker:
            return
        self._next_log_time    = None
        self._next_eval_time   = None
        t = threading.Thread(target=self._tracker_loop, daemon=True)
        t.start()

    def _tracker_loop(self):
        """Läuft ewig. Alle 60min neue Predictions loggen + auswerten."""
        INTERVAL = 3600   # 1 Stunde in Sekunden
        EVAL_INTERVAL = 900  # Auswertungs-Check alle 15 min

        last_logged  = 0
        last_eval    = 0

        while True:
            now = time.time()

            # Predictions loggen (stündlich)
            if now - last_logged >= INTERVAL:
                if self.results:
                    for ticker, result in list(self.results.items()):
                        try:
                            self._tracker.log_prediction(
                                ticker     = ticker,
                                price      = result["price"],
                                signal     = result["signal"],
                                confidence = result["confidence"],
                                pred_return= result["pred_return"],
                            )
                        except Exception as e:
                            pass
                    last_logged = now
                    count = len(self.results)
                    self.q.put(("status", f"Tracker: {count} Predictions gespeichert"))

            # Auswertung (alle 15 min prüfen ob etwas fällig)
            if now - last_eval >= EVAL_INTERVAL:
                try:
                    new_evals = self._tracker.evaluate_pending()
                    if new_evals:
                        self.q.put(("tracker_eval", new_evals))
                    last_eval = now
                except Exception as e:
                    pass

            # Nächste Log-Zeit anzeigen
            next_log = datetime.fromtimestamp(last_logged + INTERVAL)
            self.q.put(("tracker_time", next_log.strftime("%H:%M:%S")))

            # Accuracy-Tab alle 5 min refreshen
            self.q.put(("refresh_acc",))
            time.sleep(60)   # alle 60s wecken

    def _force_evaluate(self):
        """Sofortige Auswertung (Button)."""
        if not self._tracker:
            return
        def _do():
            self.q.put(("status", "Werte Predictions aus..."))
            try:
                new_evals = self._tracker.evaluate_pending()
                self.q.put(("tracker_eval", new_evals))
                self.q.put(("refresh_acc",))
                self.q.put(("status", f"Auswertung: {len(new_evals)} neu bewertet"))
            except Exception as e:
                self.q.put(("status", f"Auswertungs-Fehler: {e}"))
        threading.Thread(target=_do, daemon=True).start()

    def _refresh_acc_table(self):
        """Accuracy-Tabelle manuell refreshen."""
        if not self._tracker:
            return
        threading.Thread(target=self._update_acc_ui, daemon=True).start()

    def _update_acc_ui(self):
        """Načte statistiky z DB a odešle do UI."""
        try:
            stats = self._tracker.get_stats()
            self.q.put(("acc_stats", stats))
        except Exception as e:
            pass

    def _apply_acc_stats(self, stats):
        """Zapíše statistiky do UI (hlavní vlákno) — 1h/4h + podobnost grafu."""

        def pct(v): return f"{v:.1%}" if v is not None else "─"
        def flt(v, dec=3): return f"{v:.{dec}f}" if v is not None else "─"

        # Celkové štítky
        self.lbl_acc_1h.configure(
            text=(f"  1h-Richtung:   {stats['eval_1h']} ausgewertet  |  "
                  f"{stats['ok_1h']} korrekt  |  Acc: {pct(stats['acc_1h'])}"))
        self.lbl_acc_4h.configure(
            text=(f"  4h-Richtung:   {stats['eval_4h']} ausgewertet  |  "
                  f"{stats['ok_4h']} korrekt  |  Acc: {pct(stats['acc_4h'])}"))
        cos_str = flt(stats.get("avg_cosine"), 3)
        mae_str = flt(stats.get("avg_mae"), 3)
        self.lbl_acc_cos.configure(
            text=(f"  Chart-Ähnlich: Ø Cosine={cos_str}  |  "
                  f"Ø MAE={mae_str}%  |  "
                  f"Gesamt: {stats['total']} Predictions  |  "
                  f"Offen: {stats['pending']}"))

        # Tabulka per ticker
        self.acc_tree.delete(*self.acc_tree.get_children())
        for i, (ticker, d) in enumerate(sorted(stats["by_ticker"].items())):
            a1 = pct(d["acc_1h"]); a4 = pct(d["acc_4h"])
            cos= flt(d["cosine"], 3); mae=flt(d["mae"], 3)
            avg= f"{d['avg_ret']*100:+.2f}%" if d["avg_ret"] else "─"

            # Barevný tag: zelený pokud 4h přesnost ≥55%, červený <45%, jinak neutrální
            acc4 = d["acc_4h"] or 0
            tag  = "good" if acc4 >= 0.55 else "bad" if (d["eval_4h"]>0 and acc4 < 0.45) else "ok"
            even = "even" if i%2==0 else "odd"
            self.acc_tree.insert("","end", tags=(tag, even), values=(
                ticker,
                d["total"],
                d["eval_1h"], d["ok_1h"], a1,
                d["eval_4h"], d["ok_4h"], a4,
                cos, mae, avg, d["pending"],
            ))

        # Protokol posledních 4h vyhodnocení
        self.acc_log.configure(state="normal")
        self.acc_log.delete("1.0","end")
        hdr = ("  " + f"{'TICKER':<6} {'ZEIT':<12} {'SIG':<5}"
               + f"  {'1h%':>6} {'1h?':>4}"
               + f"  {'4h%':>6} {'4h?':>4}"
               + f"  {'COS':>6} {'MAE':>6}\n")
        self.acc_log.insert("end", hdr, "head")
        self.acc_log.insert("end", "  " + "─"*72 + "\n", "head")
        for row in stats["recent"]:
            (ticker, ts_p, sig, conf, p_pred, p_4h,
             ret_1h, ok_1h, ret_4h, ok_4h, cos, mae) = row
            ts_s  = (ts_p[5:16] if ts_p and len(ts_p)>=16 else ts_p or "─")
            r1s   = f"{(ret_1h or 0)*100:>+5.2f}%" if ret_1h is not None else "  ─   "
            r4s   = f"{(ret_4h or 0)*100:>+5.2f}%" if ret_4h is not None else "  ─   "
            c1    = "✓" if ok_1h else "✗"
            c4    = "✓" if ok_4h else "✗"
            cos_s = f"{cos:.3f}" if cos is not None else "  ─  "
            mae_s = f"{mae:.3f}" if mae is not None else "  ─  "
            line  = ("  " + f"{ticker:<6} {ts_s:<12} {sig:<5}"
                     + f"  {r1s} {c1:>3}"
                     + f"  {r4s} {c4:>3}"
                     + f"  {cos_s:>6} {mae_s:>6}\n")
            tag   = "ok" if ok_4h else "fail"
            self.acc_log.insert("end", line, tag)
        self.acc_log.configure(state="disabled")

        # Graf
        self._draw_acc_chart(stats)

    def _draw_acc_chart(self, stats):
        """3-panelový graf: 1h přesnost / 4h přesnost / kosinová podobnost (vše rolling 10)."""
        if not self._tracker:
            return
        try:
            df_all = self._tracker.get_accuracy_over_time()
        except:
            return
        if df_all.empty:
            return

        fig = self.fig_acc
        fig.clear()

        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35)
        ax1 = fig.add_subplot(gs[0])   # 1h přesnost Accuracy
        ax2 = fig.add_subplot(gs[1], sharex=ax1)   # 4h přesnost Accuracy
        ax3 = fig.add_subplot(gs[2], sharex=ax1)   # Graf-Cosine

        pal = ["navy","darkgreen","darkred","darkorange",
               "purple","teal","brown","olive","steelblue","crimson"]

        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors=MPL_FG, labelsize=6)
            for sp in ax.spines.values():
                sp.set_color(MPL_GRID)
            ax.grid(True, color=MPL_GRID, alpha=0.3, linewidth=0.5)

        tickers_in = df_all["ticker"].unique()
        for i, ticker in enumerate(tickers_in):
            sub = df_all[df_all["ticker"]==ticker]
            if len(sub) < 2:
                continue
            col = pal[i % len(pal)]
            lbl = ticker

            # 1h přesnost
            if sub["roll_1h"].notna().any():
                ax1.plot(sub["ts"], sub["roll_1h"]*100,
                          color=col, linewidth=1.2, label=lbl,
                          marker=".", markersize=2)
            # 4h přesnost
            if sub["roll_4h"].notna().any():
                ax2.plot(sub["ts"], sub["roll_4h"]*100,
                          color=col, linewidth=1.2, label=lbl,
                          marker=".", markersize=2)
            # Kosinová podobnost
            if sub["roll_cos"].notna().any():
                ax3.plot(sub["ts"], sub["roll_cos"],
                          color=col, linewidth=1.2, label=lbl,
                          marker=".", markersize=2)

        for ax in [ax1, ax2]:
            ax.axhline(50, color="gray", linestyle="--", lw=0.8, alpha=0.5)
            ax.axhline(60, color=MPL_BUY, linestyle=":", lw=0.8, alpha=0.4)
            ax.set_ylim(0, 100)

        ax3.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5,
                     label="0.5 Baseline")
        ax3.set_ylim(0, 1)

        a1s = f"{stats['acc_1h']:.1%}" if stats["acc_1h"] else "─"
        a4s = f"{stats['acc_4h']:.1%}" if stats["acc_4h"] else "─"
        cos_s = f"{stats.get('avg_cosine', 0) or 0:.3f}"

        ax1.set_title(f"1h Richtungs-Acc (rolling 10)  Ø {a1s}",
                       fontsize=8, color="navy", fontfamily="Courier New")
        ax2.set_title(f"4h Richtungs-Acc (rolling 10)  Ø {a4s}",
                       fontsize=8, color="darkgreen", fontfamily="Courier New")
        ax3.set_title(f"Chart-Cosine-Ähnlichkeit (rolling 10)  Ø {cos_s}",
                       fontsize=8, color="darkorange", fontfamily="Courier New")

        ax1.set_ylabel("Acc %", fontsize=7)
        ax2.set_ylabel("Acc %", fontsize=7)
        ax3.set_ylabel("Cosine", fontsize=7)

        ax1.legend(fontsize=6, framealpha=0.7, ncol=5, loc="lower right")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        fig.autofmt_xdate(rotation=12)
        fig.set_facecolor(MPL_BG)
        fig.tight_layout()
        self.root.after(0, self.canvas_acc.draw)

    @staticmethod
    def _rsi(s, p=14):
        d = s.diff()
        g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
        rs = g / l.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def _draw_empty(self, fig, msg):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                 fontsize=11, color="#888888",
                 fontfamily="Courier New",
                 transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(MPL_GRID)
        fig.set_facecolor(MPL_BG)


# ── Hlavní vstupní bod ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers",   nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--model_dir", default=".")
    a = p.parse_args()

    root = tk.Tk()
    app  = MicovUI(root, a.tickers, a.model_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
