"""End-to-end reproduction run: synthetic data -> all four methods -> report.

Run from the repository root:

    python -m reproduction.run_all

Everything printed is computed from synthetic data and is illustrative only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import (anomaly, bandit, config, fatigue, features,  # noqa: E402
               forecasting, synthetic_data)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _save_figures(panel, importance) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    # ROAS distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(panel["roas"].clip(upper=10), bins=40, color="#17a589")
    ax.set_title("Synthetic daily ROAS distribution")
    ax.set_xlabel("Daily ROAS (clipped at 10)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roas_distribution.png", dpi=110)
    plt.close(fig)

    # Hourly orders (bimodal)
    hourly = synthetic_data.generate_hourly_orders()
    pivot = hourly.pivot(index="hour", columns="product_line", values="order_share")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot(ax=ax, marker="o")
    ax.set_title("Synthetic hourly order share (bimodal)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of daily orders (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hourly_orders.png", dpi=110)
    plt.close(fig)

    # Feature importance
    if not importance.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        importance.head(10)[::-1].plot.barh(ax=ax, color="#17a589")
        ax.set_title("Synthetic feature importance (% of gain)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=110)
        plt.close(fig)


def main() -> None:
    print(f"Synthetic-data reproduction of REPORT.md  (illustrative; seed={config.RANDOM_SEED})")

    # 1. Data ----------------------------------------------------------------
    panel = synthetic_data.generate_ad_panel()
    panel = synthetic_data.inject_anomalies(panel, n=8)
    feats = features.build_features(panel)
    _section("1. DATA")
    print(f"panel: {panel.shape[0]} campaign-days x {panel.shape[1]} cols | "
          f"{panel['campaign_id'].nunique()} campaigns | {panel['brand'].nunique()} brands")
    print(f"daily ROAS  median={panel['roas'].median():.2f}  mean={panel['roas'].mean():.2f}  "
          f"max={panel['roas'].max():.1f}  (heavy-tailed, log-normal)")

    # 2. Forecast: pooled vs segmented --------------------------------------
    _section("2. ROAS FORECAST  -  pooling loss negative result (REPORT.md sec.5.1)")
    fc = forecasting.pooled_vs_segmented(feats)
    print(f"POOLED XGBoost (all brands share one model):  "
          f"R2={fc['pooled']['r2']:.2f}   MAPE={fc['pooled']['mape']:.1f}%")
    print(f"PER-BRAND XGBoost (one model per brand):      "
          f"R2={fc['segmented']['r2']:.2f}   MAPE={fc['segmented']['mape']:.1f}%")
    print(f"(walk-forward split; n_train={fc['n_train']}, n_test={fc['n_test']})")
    print("-> Brands have opposing calendar responses; a single pooled model cannot")
    print("   capture them, so it underperforms per-segment models. That gap IS the")
    print("   negative result. Production goes further still: per-campaign SARIMA+XGBoost.")

    # 3. Bandit --------------------------------------------------------------
    _section("3. THOMPSON SAMPLING PLACEMENT BANDIT (REPORT.md sec.5.2)")
    observations = synthetic_data.generate_placement_observations()
    posteriors = bandit.run_placement_bandit(observations)
    print(posteriors.to_string(index=False))
    for _, row in bandit.identify_waste(posteriors).iterrows():
        print(f"-> PAUSE candidate: {row['placement']}  reward_sum={row['reward_sum']}  "
              f"P(success)={row['p_success']:.2f}  spend_share={row['spend_share'] * 100:.1f}%")

    # 4. Anomaly -------------------------------------------------------------
    _section("4. MAD-MODIFIED Z-SCORE ANOMALY DETECTION (REPORT.md sec.5.3)")
    events = anomaly.detect_latest_anomalies(panel)
    if events.empty:
        print(f"no anomalies above |z|>={config.ANOMALY_Z_THRESHOLD}")
    else:
        print(f"{len(events)} live-audit events at |z|>={config.ANOMALY_Z_THRESHOLD} "
              f"(latest day per campaign):")
        print(events.groupby(["metric", "direction"]).size().to_string())

    # 5. Fatigue -------------------------------------------------------------
    _section("5. CREATIVE-FATIGUE PROJECTION (REPORT.md sec.5.4)")
    watch = fatigue.fatigue_watchlist(panel)
    print(f"{len(watch)} ad sets projected; top 5 by confidence:")
    print(watch.head(5).to_string(index=False))

    # 6. XAI + figures -------------------------------------------------------
    _section("6. FEATURE IMPORTANCE (XAI, REPORT.md sec.6.1)  +  FIGURES")
    importance = forecasting.feature_importance(fc["pooled_model"], fc["features"])
    print(importance.head(8).to_string())
    _save_figures(panel, importance)
    print(f"\nFigures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
