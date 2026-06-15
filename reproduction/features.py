"""Feature engineering (REPORT.md §4.2).

Builds the autoregressive/momentum, Turkish-calendar, and creative-quality
feature families on top of a synthetic ad panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _turkish_calendar(dates: pd.Series) -> pd.DataFrame:
    """Construct the Turkish-calendar feature block (REPORT.md §3.2.2)."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    bayram = pd.to_datetime(config.BAYRAM_DATES).normalize()
    nearest = np.array([min(abs((day.normalize() - b).days) for b in bayram) for day in d])
    out = pd.DataFrame({
        "day_of_week": d.dt.dayofweek.to_numpy(),
        "is_weekend": (d.dt.dayofweek >= 5).astype(int).to_numpy(),
        "day_of_month": d.dt.day.to_numpy(),
        "month": d.dt.month.to_numpy(),
        "week_of_year": d.dt.isocalendar().week.astype(int).to_numpy(),
        "is_holiday": d.dt.normalize().isin(bayram).astype(int).to_numpy(),
        "is_ramadan_shopping": (d.dt.month == 3).astype(int).to_numpy(),
        "is_black_friday_week": ((d.dt.month == 11) & (d.dt.day >= 22)).astype(int).to_numpy(),
        "days_to_nearest_holiday": nearest,
    })
    out["is_bayram"] = out["is_holiday"]
    out["is_ecommerce_event"] = ((out["is_black_friday_week"] == 1) |
                                 (d.dt.day == config.SALARY_DAY).to_numpy()).astype(int)
    return out


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the panel with all documented predictors added (per campaign)."""
    frames: list[pd.DataFrame] = []
    for _, g in panel.groupby("campaign_id"):
        g = g.sort_values("date").copy()
        # Autoregressive / momentum
        g["roas_lag1"] = g["roas"].shift(1)
        g["roas_lag7"] = g["roas"].shift(7)
        g["roas_ma7"] = g["roas"].rolling(7, min_periods=1).mean()
        g["roas_ma14"] = g["roas"].rolling(14, min_periods=1).mean()
        g["roas_std7"] = g["roas"].rolling(7, min_periods=2).std().fillna(0.0)
        g["spend_ma7"] = g["spend"].rolling(7, min_periods=1).mean()
        g["spend_change"] = g["spend"].diff()
        g["clicks_per_spend"] = g["clicks"] / g["spend"]
        g["cps_ma7"] = g["clicks_per_spend"].rolling(7, min_periods=1).mean()
        # Creative-quality ratios (derived; REPORT.md §3.2.3)
        plays = g["video_p25"].replace(0, np.nan)
        g["hook_rate_ma7"] = (g["video_p25"] / g["impressions"]).rolling(7, min_periods=1).mean()
        g["hold_rate_ma7"] = (g["video_p75"] / plays).rolling(7, min_periods=1).mean()
        g["video_completion_ma7"] = (g["video_p100"] / plays).rolling(7, min_periods=1).mean()
        g["outbound_ctr_lag1"] = g["outbound_ctr"].shift(1)
        g["frequency_lag1"] = g["frequency"].shift(1)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    calendar = _turkish_calendar(out["date"])
    out = pd.concat([out.reset_index(drop=True), calendar], axis=1)
    return out
