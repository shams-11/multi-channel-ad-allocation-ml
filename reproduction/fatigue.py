"""Creative-fatigue projection (REPORT.md §5.4).

Fits an OLS line to each ad set's frequency time series and extrapolates the
number of days until frequency reaches the saturation threshold (4.0). The
r-squared of the fit is reported as a confidence weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def project_days_to_threshold(frequency: np.ndarray, threshold: float | None = None) -> dict | None:
    """Project days until ``frequency`` reaches ``threshold`` via an OLS trend."""
    threshold = config.FATIGUE_THRESHOLD if threshold is None else threshold
    y = np.asarray(frequency, dtype=float)
    if len(y) < 3:
        return None
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    fitted = slope * t + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope <= 0:
        return {"days_to_threshold": float("inf"), "r2": round(r2, 3), "slope": round(float(slope), 4)}
    days = (threshold - fitted[-1]) / slope
    return {"days_to_threshold": round(max(0.0, float(days)), 1),
            "r2": round(r2, 3), "slope": round(float(slope), 4)}


def fatigue_watchlist(panel: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Rank ad sets *approaching* fatigue (highest-confidence, soonest first).

    The watch list holds ad sets not yet at the threshold but rising toward it,
    so the creative team can plan replacements ahead of the saturation point.
    Ad sets already past the threshold are excluded (they are already fatigued).
    """
    threshold = config.FATIGUE_THRESHOLD if threshold is None else threshold
    rows: list[dict] = []
    for adset_id, g in panel.groupby("adset_id"):
        g = g.sort_values("date")
        current = float(g["frequency"].iloc[-1])
        projection = project_days_to_threshold(g["frequency"].to_numpy(), threshold)
        if projection is None or current >= threshold:
            continue
        rows.append(dict(adset=adset_id, current_freq=round(current, 2), **projection))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[np.isfinite(out["days_to_threshold"])]
    return out.sort_values(["r2", "days_to_threshold"], ascending=[False, True]).reset_index(drop=True)
