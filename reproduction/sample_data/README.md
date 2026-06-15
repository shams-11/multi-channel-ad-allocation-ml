# Sample data (synthetic)

These CSVs are a committed snapshot of the dataset produced by
[`../synthetic_data.py`](../synthetic_data.py) with the default `seed = 433`.
They **mimic the production schema** so the harness and report are reproducible,
but contain **no real production data** — every value is fabricated.

| File | Rows | Description |
|---|---|---|
| `ad_panel.csv` | 3,600 | Campaign × date panel: `spend`, `revenue`, `roas`, `impressions`, `clicks`, video metrics, `frequency`, etc. (30 synthetic campaigns × 120 days). |
| `placement_observations.csv` | ~400 | Per-placement Bernoulli reward stream for the Thompson Sampling bandit. |
| `hourly_orders.csv` | 48 | Bimodal hourly order share for two anonymized product lines. |

Regenerate them at any time:

```bash
python - <<'PY'
from reproduction import synthetic_data as s
s.generate_ad_panel().round(4).to_csv("reproduction/sample_data/ad_panel.csv", index=False)
s.generate_placement_observations().to_csv("reproduction/sample_data/placement_observations.csv", index=False)
s.generate_hourly_orders().round(4).to_csv("reproduction/sample_data/hourly_orders.csv", index=False)
PY
```

> Identities are neutral by construction (`Brand A–E`, `Product line A/B`); columns
> match the report's schema (REPORT.md §3.1) so the methods run unchanged.
