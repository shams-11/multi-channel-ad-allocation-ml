"""Synthetic-data reproduction harness for the ECON 433 ad-allocation report.

This package regenerates a synthetic dataset that mimics the production schema
and reproduces the four core methods described in the report:

- ``forecasting``  — SARIMA + XGBoost + OLS ensemble and the deliberate
  negative result of pooling structurally different brands (REPORT.md §4-§5.1).
- ``bandit``       — Beta-Bernoulli Thompson Sampling placement bandit that
  surfaces the zero-reward placement (REPORT.md §5.2).
- ``anomaly``      — MAD-modified z-score detector (REPORT.md §5.3).
- ``fatigue``      — OLS frequency projection to a saturation threshold
  (REPORT.md §5.4).

No proprietary code or data is included. Every number produced by this harness
comes from synthetic data and is illustrative only.
"""

__all__ = [
    "config",
    "synthetic_data",
    "features",
    "forecasting",
    "bandit",
    "anomaly",
    "fatigue",
]
