"""Behavioural tests for the reproduction harness (run with: pytest)."""
from __future__ import annotations

import numpy as np

from reproduction import (anomaly, bandit, config, fatigue, features,
                          forecasting, synthetic_data)


def test_panel_has_expected_shape_and_brands():
    panel = synthetic_data.generate_ad_panel()
    assert panel["campaign_id"].nunique() == 30
    assert panel["brand"].nunique() == len(config.BRANDS)
    assert (panel["spend"] > 0).all()


def test_features_include_all_documented_predictors():
    feats = features.build_features(synthetic_data.generate_ad_panel())
    for column in config.ALL_FEATURES:
        assert column in feats.columns, f"missing feature {column}"


def test_bandit_flags_zero_reward_placement():
    posteriors = bandit.run_placement_bandit(synthetic_data.generate_placement_observations())
    audience_network = posteriors[posteriors["placement"] == "Audience Network"].iloc[0]
    assert audience_network["reward_sum"] == 0
    assert audience_network["p_success"] == posteriors["p_success"].min()
    assert "Audience Network" in set(bandit.identify_waste(posteriors)["placement"])


def test_beta_arm_is_immutable():
    arm = bandit.BetaArm()
    updated = arm.update(1)
    assert (arm.alpha, arm.beta) == (1.0, 1.0)          # original unchanged
    assert (updated.alpha, updated.beta) == (2.0, 1.0)  # new object carries the update


def test_modified_zscore_flags_outlier_only():
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 50.0])
    z = anomaly.modified_zscore(x)
    assert abs(z[-1]) > config.ANOMALY_Z_THRESHOLD
    assert np.all(np.abs(z[:-1]) < config.ANOMALY_Z_THRESHOLD)


def test_anomaly_detector_fires_on_injected_outliers():
    panel = synthetic_data.inject_anomalies(synthetic_data.generate_ad_panel(), n=8)
    assert len(anomaly.detect_anomalies(panel)) >= 1


def test_fatigue_projection_positive_for_rising_series():
    projection = fatigue.project_days_to_threshold(np.linspace(1.0, 3.0, 60))
    assert projection is not None
    assert projection["days_to_threshold"] > 0
    assert projection["r2"] > 0.9


def test_pooled_vs_segmented_metrics_are_finite_and_segmented_helps():
    feats = features.build_features(synthetic_data.generate_ad_panel())
    fc = forecasting.pooled_vs_segmented(feats)
    assert np.isfinite(fc["pooled"]["mape"])
    assert np.isfinite(fc["segmented"]["mape"])
    # Structurally different brands: per-segment models should not be worse.
    assert fc["segmented"]["mape"] <= fc["pooled"]["mape"]
