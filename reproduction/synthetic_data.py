"""Generate a synthetic dataset that mimics the production ad schema.

All values are fabricated from a seeded RNG; nothing here derives from real
production data. The generator deliberately bakes in the qualitative structure
the report's methods are meant to detect:

* heavy-tailed, log-normal daily ROAS (REPORT.md Fig. 1);
* structurally different brands with opposing calendar effects (REPORT.md §5.1);
* a mid-window bayram inflection (REPORT.md Fig. 3);
* a zero-revenue Audience Network placement (REPORT.md §5.2);
* ad-set frequency rising toward (but mostly below) the 4.0 threshold (§5.4);
* a bimodal lunch + evening hourly order pattern (REPORT.md §5.5);
* a handful of injected metric outliers for the anomaly detector (§5.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED if seed is None else seed)


def generate_ad_panel(n_campaigns: int = 30, seed: int | None = None) -> pd.DataFrame:
    """Return a campaign × date panel of daily ad insights (synthetic)."""
    rng = _rng(seed)
    dates = pd.date_range(config.START_DATE, periods=config.PER_CAMPAIGN_DAYS, freq="D")
    bayram = {ts.normalize() for ts in pd.to_datetime(config.BAYRAM_DATES)}
    rows: list[dict] = []
    for c in range(n_campaigns):
        brand = config.BRANDS[c % len(config.BRANDS)]
        prof = config.BRAND_PROFILES[brand]
        level = prof["level"] * rng.uniform(0.9, 1.1)
        spend_level = rng.uniform(200, 5000)
        freq0 = prof["freq0"] * rng.uniform(0.9, 1.1)
        freq_slope = prof["freq_slope"] * rng.uniform(0.9, 1.1)
        for t, d in enumerate(dates):
            is_weekend = 1 if d.dayofweek >= 5 else 0
            is_salary = 1 if d.day == config.SALARY_DAY else 0
            # brand-specific (and partly opposing) calendar response
            calendar = 1.0 + prof["weekend"] * is_weekend + prof["salary"] * is_salary
            bay = 1.8 if d.normalize() in bayram else 1.0
            noise = rng.lognormal(mean=0.0, sigma=prof["sigma"])
            roas = max(0.0, level * calendar * bay * noise)
            spend = spend_level * rng.lognormal(0.0, 0.25)
            revenue = spend * roas
            impressions = spend * rng.uniform(8, 14)
            clicks = max(1.0, impressions * rng.uniform(0.008, 0.03))
            outbound = clicks * rng.uniform(0.4, 0.8)
            vp25 = impressions * rng.uniform(0.05, 0.18)
            vp50 = vp25 * rng.uniform(0.6, 0.9)
            vp75 = vp50 * rng.uniform(0.6, 0.9)
            vp100 = vp75 * rng.uniform(0.5, 0.85)
            frequency = max(0.5, freq0 + freq_slope * t + rng.normal(0, 0.04))
            rows.append(dict(
                date=d, brand=brand, campaign_id=f"camp_{c:03d}", adset_id=f"adset_{c:03d}",
                spend=spend, revenue=revenue, impressions=impressions, clicks=clicks,
                outbound_clicks=outbound, video_p25=vp25, video_p50=vp50,
                video_p75=vp75, video_p100=vp100, frequency=frequency,
            ))
    df = pd.DataFrame(rows)
    df["roas"] = df["revenue"] / df["spend"]
    df["cpc"] = df["spend"] / df["clicks"]
    df["ctr"] = df["clicks"] / df["impressions"]
    df["hook_rate"] = df["video_p25"] / df["impressions"]
    df["outbound_ctr"] = df["outbound_clicks"] / df["impressions"]
    df["conversion_value"] = df["revenue"]
    return df.sort_values(["campaign_id", "date"]).reset_index(drop=True)


def inject_anomalies(df: pd.DataFrame, n: int = 8, seed: int | None = None) -> pd.DataFrame:
    """Push the *latest day* of a few campaigns to extreme values.

    Injecting on the most recent day mirrors a live-audit scenario, so the
    per-campaign detector (``anomaly.detect_latest_anomalies``) catches them.
    """
    rng = _rng((config.RANDOM_SEED if seed is None else seed) + 1)
    out = df.copy()
    metrics = ["cpc", "ctr", "hook_rate", "roas", "spend", "outbound_ctr", "frequency"]
    last_rows = out.groupby("campaign_id").tail(1).index.to_numpy()
    chosen = rng.choice(last_rows, size=min(n, len(last_rows)), replace=False)
    for i, row in enumerate(chosen):
        metric = metrics[i % len(metrics)]
        factor = 0.12 if (i % 2 == 0) else 7.0  # alternate extreme low / high
        out.loc[row, metric] = float(out.loc[row, metric]) * factor
    return out


def generate_placement_observations(seed: int | None = None) -> pd.DataFrame:
    """Return a stream of (placement, arm, reward) bandit pulls (synthetic).

    Rewards are Bernoulli draws from each placement's latent success rate
    (``config.PLACEMENTS``). Audience Network is forced to zero reward,
    reproducing the report's headline finding (REPORT.md §5.2).
    """
    rng = _rng((config.RANDOM_SEED if seed is None else seed) + 2)
    rows: list[dict] = []
    for placement, spec in config.PLACEMENTS.items():
        for arm in range(spec["n_arms"]):
            n_pulls = int(rng.integers(2, 5))
            for _ in range(n_pulls):
                reward = 0 if placement == "Audience Network" else int(rng.random() < spec["p_success"])
                rows.append(dict(placement=placement, arm=f"{placement[:2]}_{arm:02d}",
                                 reward=reward, spend_share=spec["spend_share"]))
    return pd.DataFrame(rows)


def generate_hourly_orders(seed: int | None = None) -> pd.DataFrame:
    """Return a bimodal (lunch + evening) hourly order-share table (REPORT.md §5.5)."""
    rng = _rng((config.RANDOM_SEED if seed is None else seed) + 3)
    hours = np.arange(24)

    def shape(lunch_w: float, evening_w: float) -> np.ndarray:
        lunch = lunch_w * np.exp(-0.5 * ((hours - 12.5) / 1.6) ** 2)
        evening = evening_w * np.exp(-0.5 * ((hours - 20.5) / 1.8) ** 2)
        daytime = np.clip((hours - 5) / 3, 0, 1) * np.clip((24 - hours) / 4, 0, 1)
        raw = 0.3 + 1.2 * daytime + lunch + evening + rng.normal(0, 0.05, 24)
        raw = np.clip(raw, 0.05, None)
        return raw / raw.sum() * 100.0

    a = shape(3.0, 1.8)  # lunch-dominant
    b = shape(2.2, 2.6)  # evening-leaning
    rows = [dict(hour=int(h), product_line="Product line A", order_share=float(a[h])) for h in hours]
    rows += [dict(hour=int(h), product_line="Product line B", order_share=float(b[h])) for h in hours]
    return pd.DataFrame(rows)
