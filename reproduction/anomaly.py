"""MAD-modified z-score anomaly detection (Iglewicz-Hoaglin 1993; REPORT.md §5.3).

The classical z-score is unusable on the heavy-tailed, log-normal ROAS
distribution (Fig. 1): the inflated sigma hides genuine downward shifts. The
modified z-score replaces sigma with the median absolute deviation (MAD),
scaled by 0.6745 (= 1 / 1.4826) so the two methods agree under normality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def modified_zscore(values: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin modified z-score: 0.6745 * (x - median) / MAD."""
    x = np.asarray(values, dtype=float)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:  # degenerate spread: fall back to scaled mean abs deviation
        mean_ad = np.mean(np.abs(x - median))
        if mean_ad == 0:
            return np.zeros_like(x)
        return 0.7979 * (x - median) / mean_ad
    return config.MAD_SCALE * (x - median) / mad


def detect_anomalies(df: pd.DataFrame, metrics: list[str] | None = None,
                     threshold: float | None = None) -> pd.DataFrame:
    """Flag rows where |modified z| >= threshold for any monitored metric."""
    metrics = metrics or config.ANOMALY_METRICS
    threshold = config.ANOMALY_Z_THRESHOLD if threshold is None else threshold
    events: list[dict] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        z = modified_zscore(df[metric].to_numpy())
        for i in np.where(np.abs(z) >= threshold)[0]:
            events.append(dict(
                metric=metric, row=int(i),
                value=round(float(df[metric].iloc[i]), 4),
                modified_z=round(float(z[i]), 2),
                direction="low" if z[i] < 0 else "high",
            ))
    return pd.DataFrame(events)


def detect_latest_anomalies(panel: pd.DataFrame, metrics: list[str] | None = None,
                            threshold: float | None = None) -> pd.DataFrame:
    """Live-audit detection: flag each campaign's *latest* day vs its own history.

    This is the production unit of analysis — a campaign's metric today against
    its own recent distribution — and avoids the false floods that result from
    pooling structurally different campaigns into one reference distribution.
    """
    metrics = metrics or config.ANOMALY_METRICS
    threshold = config.ANOMALY_Z_THRESHOLD if threshold is None else threshold
    events: list[dict] = []
    for campaign_id, g in panel.groupby("campaign_id"):
        g = g.sort_values("date")
        for metric in metrics:
            if metric not in g.columns:
                continue
            z = modified_zscore(g[metric].to_numpy())
            if abs(z[-1]) >= threshold:
                events.append(dict(
                    campaign=campaign_id, metric=metric,
                    value=round(float(g[metric].iloc[-1]), 4),
                    modified_z=round(float(z[-1]), 2),
                    direction="low" if z[-1] < 0 else "high",
                ))
    return pd.DataFrame(events)
