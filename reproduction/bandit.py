"""Beta-Bernoulli Thompson Sampling placement bandit (REPORT.md §5.2).

Maintains a Beta(alpha, beta) posterior per placement and updates it from a
stream of Bernoulli rewards. A placement that never returns a reward stays at
its prior floor — exactly the Audience Network signal the report acts on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaArm:
    """Immutable Beta posterior for a single arm (conjugate prior alpha=beta=1)."""

    alpha: float = 1.0
    beta: float = 1.0

    def update(self, reward: int) -> "BetaArm":
        """Return a NEW arm with the Bernoulli reward folded in (no mutation)."""
        return BetaArm(self.alpha + reward, self.beta + (1 - reward))

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta))


def run_placement_bandit(observations: pd.DataFrame) -> pd.DataFrame:
    """Fold a reward stream into per-placement Beta posteriors.

    ``observations`` must have columns ``placement``, ``reward`` (0/1) and
    ``spend_share``. Returns one row per placement, sorted by posterior mean.
    """
    posteriors: dict[str, BetaArm] = {p: BetaArm() for p in observations["placement"].unique()}
    for placement, reward in zip(observations["placement"], observations["reward"]):
        posteriors[placement] = posteriors[placement].update(int(reward))

    rows = []
    for placement, arm in posteriors.items():
        sub = observations[observations["placement"] == placement]
        rows.append(dict(
            placement=placement, alpha=arm.alpha, beta=arm.beta,
            p_success=round(arm.mean, 4), pulls=int(len(sub)),
            reward_sum=int(sub["reward"].sum()),
            spend_share=float(sub["spend_share"].iloc[0]),
        ))
    return pd.DataFrame(rows).sort_values("p_success", ascending=False).reset_index(drop=True)


def identify_waste(posteriors: pd.DataFrame, min_spend_share: float = 0.005) -> pd.DataFrame:
    """Placements with zero reward AND non-trivial spend — the 'pause' candidates."""
    mask = (posteriors["reward_sum"] == 0) & (posteriors["spend_share"] >= min_spend_share)
    return posteriors[mask].reset_index(drop=True)
