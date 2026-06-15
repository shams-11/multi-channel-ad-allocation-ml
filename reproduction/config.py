"""Constants for the reproduction harness.

Thresholds (``ANOMALY_Z_THRESHOLD``, ``FATIGUE_THRESHOLD``) and the documented
feature groups are reported exactly as in the production system; all other
numbers here only steer the *synthetic* data generator.
"""
from __future__ import annotations

RANDOM_SEED = 433  # ECON 433

# --- Study window -----------------------------------------------------------
STUDY_DAYS = 32          # short aggregate window (matches the report's T = 32)
PER_CAMPAIGN_DAYS = 120  # longer per-series history for the per-campaign models
START_DATE = "2026-04-09"

# --- Anonymised identities --------------------------------------------------
BRANDS = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"]
PRODUCT_LINES = ["Product line A", "Product line B"]

# Structurally different brand response surfaces. The opposing weekend/salary
# signs are what a single pooled model cannot capture (no brand feature),
# reproducing the report's "pooling loss" negative result (REPORT.md §5.1).
BRAND_PROFILES = {
    "Brand A": {"level": 4.2, "weekend": 0.45, "salary": 0.50, "sigma": 0.30,
                "freq0": 1.2, "freq_slope": 0.012},
    "Brand B": {"level": 1.7, "weekend": -0.35, "salary": 0.10, "sigma": 0.45,
                "freq0": 0.9, "freq_slope": 0.020},
    "Brand C": {"level": 3.0, "weekend": 0.15, "salary": 0.55, "sigma": 0.35,
                "freq0": 1.5, "freq_slope": 0.008},
    "Brand D": {"level": 2.3, "weekend": -0.25, "salary": -0.20, "sigma": 0.50,
                "freq0": 1.0, "freq_slope": 0.024},
    "Brand E": {"level": 5.0, "weekend": 0.50, "salary": 0.30, "sigma": 0.40,
                "freq0": 1.3, "freq_slope": 0.016},
}

# --- Meta placements --------------------------------------------------------
# Each placement carries a *latent* success probability used only to generate
# synthetic Bernoulli rewards. Audience Network is deliberately zero so the
# bandit reproduces the report's headline finding (REPORT.md §5.2).
PLACEMENTS = {
    "Facebook":         {"p_success": 0.53, "spend_share": 0.400, "n_arms": 43},
    "Instagram":        {"p_success": 0.46, "spend_share": 0.582, "n_arms": 50},
    "Audience Network": {"p_success": 0.00, "spend_share": 0.018, "n_arms": 15},
    "Messenger":        {"p_success": 0.20, "spend_share": 0.0005, "n_arms": 17},
    "Threads":          {"p_success": 0.20, "spend_share": 0.0005, "n_arms": 6},
}

# --- ROAS distribution (heavy-tailed log-normal, REPORT.md §3.3.1) ----------
ROAS_TARGET = 2.0           # bandit success: reward = 1 if ROAS >= target

# --- Thresholds (reported exactly as in production) -------------------------
ANOMALY_Z_THRESHOLD = 3.5   # |modified z| >= 3.5 (Iglewicz-Hoaglin)
MAD_SCALE = 0.6745          # modified z-score constant = 1 / 1.4826
FATIGUE_THRESHOLD = 4.0     # frequency at which a creative is "fatigued"

# --- Feature groups (REPORT.md §3.2) ----------------------------------------
AUTOREGRESSIVE_FEATURES = [
    "roas_lag1", "roas_lag7", "roas_ma7", "roas_ma14", "roas_std7",
    "spend_ma7", "spend_change", "clicks_per_spend", "cps_ma7",
]
CALENDAR_FEATURES = [
    "day_of_week", "is_weekend", "day_of_month", "month", "week_of_year",
    "is_holiday", "is_bayram", "is_ramadan_shopping", "is_black_friday_week",
    "is_ecommerce_event", "days_to_nearest_holiday",
]
CREATIVE_FEATURES = [
    "hook_rate_ma7", "hold_rate_ma7", "video_completion_ma7",
    "outbound_ctr_lag1", "frequency_lag1",
]
ALL_FEATURES = AUTOREGRESSIVE_FEATURES + CALENDAR_FEATURES + CREATIVE_FEATURES

# Anomaly metrics monitored (REPORT.md §5.3)
ANOMALY_METRICS = ["roas", "cpc", "ctr", "spend", "hook_rate",
                   "outbound_ctr", "frequency", "conversion_value"]

# Synthetic Turkish-calendar anchors (illustrative)
BAYRAM_DATES = ["2026-04-26", "2026-04-27"]   # mid-window inflection (REPORT.md Fig. 3)
SALARY_DAY = 15                                # Turkish state salaries paid on the 15th
