"""
train.py
========
Ensemble trénink: Stacking z XGBoost + HistGradientBoosting

Model 1: Klasifikace  → BUY / SELL / HOLD
Model 2: Regrese      → future_return (příštích N hodin v %)

Stacking:
  Úroveň 0: XGBoost + HistGradientBoosting (oba)
  Úroveň 1: LogisticRegression / Ridge jako meta-learner

Spuštění:
  python train.py --data dataset.csv
  python train.py --data dataset.csv --no_xgb   ← pokud XGBoost chybí

POZNÁMKA:
  XGBoost potřebuje libomp na Macu (brew install libomp).
  Alternativa: příznak --no_xgb → pouze HistGradientBoosting.
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               HistGradientBoostingRegressor,
                               StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix,
                              mean_absolute_error, r2_score)
import sklearn
sklearn.set_config(enable_metadata_routing=True)

warnings.filterwarnings("ignore")

TEST_RATIO  = 0.15
LABEL_COL   = "label"
RETURN_COL  = "future_return"
# Metadata sloupce — nepoužíváme jako příznaky
META_COLS   = ["ticker", "close_raw"]


def load_data(path: str) -> pd.DataFrame:
    """Načte dataset ze souboru CSV."""
    print(f"Načítám dataset: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  Tvar: {df.shape}")
    if "ticker" in df.columns:
        print(f"  Tickery: {sorted(df['ticker'].unique().tolist())}")
    print(f"  Časové rozmezí: {df.index.min()} → {df.index.max()}")
    return df


def prepare(df: pd.DataFrame):
    """
    Rozdělí DataFrame na příznaky X, labely klasifikace y_cls a regrese y_reg.
    Odstraní metadata sloupce a nekonečné hodnoty.
    """
    drop = [LABEL_COL, RETURN_COL] + [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_cls = df[LABEL_COL].astype(int)
    y_reg = df[RETURN_COL].astype(float)
    print(f"  Počet příznaků: {X.shape[1]}")
    return X, y_cls, y_reg


def time_split(X, y_cls, y_reg, ratio):
    """
    Časový split — testovací data jsou vždy NOVĚJŠÍ než tréninková.
    Žádné náhodné míchání: zachovává kauzalitu časové řady.
    """
    n = int(len(X) * (1 - ratio))
    return (X.iloc[:n], X.iloc[n:],
            y_cls.iloc[:n], y_cls.iloc[n:],
            y_reg.iloc[:n], y_reg.iloc[n:])


def sample_weights(y: pd.Series) -> np.ndarray:
    """
    Vypočítá váhy vzorků pro vyvážení tříd.
    Třída s méně vzorky dostane vyšší váhu.
    """
    counts = y.value_counts()
    total, nc = len(y), len(counts)
    w = {cls: total / (nc * cnt) for cls, cnt in counts.items()}
    print(f"  Váhy tříd: { {k: round(v, 3) for k, v in w.items()} }")
    return y.map(w).values


def build_cls_model(use_xgb: bool) -> StackingClassifier:
    """
    Stacking klasifikátor s Metadata Routing pro sample_weight.
    Pokud use_xgb=False, použije se pouze HistGradientBoosting.
    """
    hgb = HistGradientBoostingClassifier(
        max_iter=600, max_depth=7, learning_rate=0.04,
        min_samples_leaf=15, l2_regularization=0.1,
        max_features=0.8, random_state=42
    )
    hgb.set_fit_request(sample_weight=True)

    estimators = [("hgb", hgb)]

    if use_xgb:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=42,
            n_jobs=-1, verbosity=0,
        )
        xgb.set_fit_request(sample_weight=True)
        estimators.append(("xgb", xgb))

    # Meta-learner — LogisticRegression jako finální estimátor
    meta = LogisticRegression(max_iter=500, C=1.0)
    meta.set_fit_request(sample_weight=True)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def build_reg_model(use_xgb: bool) -> StackingRegressor:
    """
    Stacking regresor pro předpověď future_return.
    Regrese nepotřebuje sample_weight (není třídní nerovnováha).
    """
    hgb = HistGradientBoostingRegressor(
        max_iter=600, max_depth=6, learning_rate=0.04,
        min_samples_leaf=15, l2_regularization=0.1,
        max_features=0.8, random_state=42
    )

    estimators = [("hgb", hgb)]

    if use_xgb:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        estimators.append(("xgb", xgb))

    meta = Ridge(alpha=1.0)

    return StackingRegressor(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


def eval_cls(model, X_test, y_test):
    """Vyhodnocení klasifikačního modelu — report + matice záměn."""
    y_pred = model.predict(X_test)
    print("\n── Klasifikace ────────────────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["SELL", "HOLD", "BUY"],
                                zero_division=0))
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["Skut. SELL", "Skut. HOLD", "Skut. BUY"],
        columns=["Před. SELL", "Před. HOLD", "Před. BUY"]
    )
    print(cm)
    return y_pred


def eval_reg(model, X_test, y_test):
    """Vyhodnocení regresního modelu — MAE, R² a přesnost směru."""
    y_pred = model.predict(X_test)
    mae      = mean_absolute_error(y_test, y_pred)
    r2       = r2_score(y_test, y_pred)
    # Přesnost směru — správné znaménko předpovědi
    dir_acc  = np.mean(np.sign(y_pred) == np.sign(y_test))
    print(f"\n── Regrese (future_return) ────────────────────────")
    print(f"  MAE:             {mae:.5f}  ({mae * 100:.3f}%)")
    print(f"  R²:              {r2:.4f}")
    print(f"  Přesnost směru:  {dir_acc:.1%}  (správné znaménko)")
    return y_pred


def feature_importance(model, feature_names: list, top_n: int = 25):
    """Vypíše top N příznaků podle důležitosti z nejlepšího base estimátoru."""
    named = getattr(model, "named_estimators_", {})
    candidates = list(named.items()) if named else [
        (str(i), est) for i, est in enumerate(model.estimators_)
    ]
    for name, est in candidates:
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=feature_names)
            imp = imp.sort_values(ascending=False).head(top_n)
            print(f"\n── Top {top_n} příznaků ({name}) ─────────────────────────")
            for feat, val in imp.items():
                bar = "█" * int(val * 400)
                print(f"  {feat:<45} {val:.4f} {bar}")
            return
    print("  (Důležitost příznaků není k dispozici)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="dataset.csv")
    parser.add_argument("--test_ratio", default=TEST_RATIO, type=float)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--no_xgb",    action="store_true",
                        help="Pouze HistGradientBoosting (bez XGBoost/brew)")
    args = parser.parse_args()

    use_xgb = not args.no_xgb

    # Ověření dostupnosti XGBoost
    if use_xgb:
        try:
            import xgboost
            print("  XGBoost dostupný ✓")
        except Exception as e:
            print(f"  XGBoost NENÍ dostupný: {e}")
            print("  → Přepínám na --no_xgb režim (pouze HGB)")
            use_xgb = False

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    # Načtení dat
    df = load_data(args.data)
    X, y_cls, y_reg = prepare(df)

    X_train, X_test, y_cls_tr, y_cls_te, y_reg_tr, y_reg_te = \
        time_split(X, y_cls, y_reg, args.test_ratio)

    print(f"\n  Trénink: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Trénink do: {X_train.index[-1]}")
    print(f"  Test od:    {X_test.index[0]}")

    sw = sample_weights(y_cls_tr)

    # ── Klasifikační model ─────────────────────────────────────────────────────
    mode = "XGB+HGB Stacking" if use_xgb else "HGB Only"
    print(f"\n  [1/2] Trénink klasifikace ({mode})...")
    print(f"        (Stacking s CV=5 — může trvat několik minut)")
    cls_model = build_cls_model(use_xgb)
    cls_model.fit(X_train, y_cls_tr, sample_weight=sw)
    print("  ✓ Klasifikace hotova")

    eval_cls(cls_model, X_test, y_cls_te)
    if use_xgb:
        feature_importance(cls_model, X_train.columns.tolist())

    # ── Regresní model ─────────────────────────────────────────────────────────
    print(f"\n  [2/2] Trénink regrese ({mode})...")
    reg_model = build_reg_model(use_xgb)
    reg_model.fit(X_train, y_reg_tr)
    print("  ✓ Regrese hotova")

    eval_reg(reg_model, X_test, y_reg_te)

    # ── Uložení modelů ─────────────────────────────────────────────────────────
    joblib.dump(cls_model, out / "model_cls.joblib")
    joblib.dump(reg_model, out / "model_reg.joblib")
    print(f"\n  ✓ model_cls.joblib uložen")
    print(f"  ✓ model_reg.joblib uložen")

    # Metadata modelu — slouží run.py pro správné sestavení příznaků
    meta = {
        "features":      X_train.columns.tolist(),
        "medians":       X_train.median().to_dict(),
        "use_xgb":       use_xgb,
        "trained_until": str(X_train.index[-1]),
        "label_map":     {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "label_horizon": 4,
        "label_thresh":  0.003,
    }
    with open(out / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  ✓ model_meta.json uložen")


if __name__ == "__main__":
    main()
