"""ROAS forecasting: ensemble + walk-forward validation (REPORT.md §4.3-§5.1).

Reproduces the report's central lesson: pooling structurally different brands
into ONE regressor fails, while fitting per-segment models recovers. Uses
XGBoost when available, otherwise a scikit-learn gradient booster as a drop-in
fallback. A per-campaign SARIMA forecaster (the production time-series model)
is provided separately.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config

try:  # XGBoost is the production model; fall back gracefully if unavailable.
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover - environment-dependent
    _HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%), ignoring near-zero denominators."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-6
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _new_booster():
    if _HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, reg_lambda=1.0, random_state=config.RANDOM_SEED,
        )
    return GradientBoostingRegressor(  # pragma: no cover
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=config.RANDOM_SEED,
    )


def _fit_booster(x_train, y_train, x_test):
    model = _new_booster()
    model.fit(x_train, y_train)
    return model, model.predict(x_test)


def pooled_vs_segmented(features: pd.DataFrame, horizon: int = 14) -> dict:
    """Compare ONE pooled booster against per-brand boosters (REPORT.md §5.1).

    Same data, same model, same walk-forward split — the only difference is
    whether structurally different brands share a model. Returns both sets of
    metrics plus the fitted pooled model (for feature importance / XAI).
    """
    df = features.dropna(subset=["roas"]).copy()
    dates = np.sort(df["date"].unique())
    split = dates[-horizon]
    train, test = df[df["date"] < split], df[df["date"] >= split].reset_index(drop=True)
    cols = config.ALL_FEATURES

    pooled_model, pooled_pred = _fit_booster(
        train[cols].fillna(0.0).to_numpy(), train["roas"].to_numpy(),
        test[cols].fillna(0.0).to_numpy())
    pooled = {"mape": mape(test["roas"], pooled_pred), "r2": r2(test["roas"], pooled_pred)}

    seg_pred = np.full(len(test), np.nan)
    for brand in df["brand"].unique():
        tr_b = train[train["brand"] == brand]
        te_mask = (test["brand"] == brand).to_numpy()
        if tr_b.empty or not te_mask.any():
            continue
        _, pred = _fit_booster(tr_b[cols].fillna(0.0).to_numpy(), tr_b["roas"].to_numpy(),
                               test.loc[te_mask, cols].fillna(0.0).to_numpy())
        seg_pred[te_mask] = pred
    valid = ~np.isnan(seg_pred)
    y = test["roas"].to_numpy()
    segmented = {"mape": mape(y[valid], seg_pred[valid]), "r2": r2(y[valid], seg_pred[valid])}

    return {"pooled": pooled, "segmented": segmented, "pooled_model": pooled_model,
            "features": cols, "n_train": len(train), "n_test": len(test)}


def per_campaign_forecast(features: pd.DataFrame, min_obs: int = 40,
                          horizon: int = 14, max_campaigns: int | None = None) -> pd.DataFrame:
    """Fit a per-campaign SARIMA (the production time-series model); score it."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    results: list[dict] = []
    campaigns = list(features.groupby("campaign_id"))
    if max_campaigns is not None:
        campaigns = campaigns[:max_campaigns]
    for cid, g in campaigns:
        y = g.sort_values("date")["roas"].dropna().to_numpy()
        if len(y) < min_obs:
            continue
        train, test = y[:-horizon], y[-horizon:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7),
                              enforce_stationarity=False, enforce_invertibility=False
                              ).fit(disp=False, maxiter=200)
                forecast = np.asarray(fit.forecast(steps=horizon))
        except Exception:  # pragma: no cover - solver edge cases
            continue
        results.append({"campaign": cid, "mape": mape(test, forecast), "r2": r2(test, forecast)})
    return pd.DataFrame(results)


def feature_importance(model, feature_names) -> pd.Series:
    """Normalized (gain/impurity-based) feature importance, descending."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)
    series = pd.Series(np.asarray(importances, dtype=float),
                       index=feature_names).sort_values(ascending=False)
    total = series.sum()
    return (series / total * 100.0).round(2) if total > 0 else series
