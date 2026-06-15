# Multi-Channel Ad Spend Allocation Under Uncertainty

**Thompson Sampling and time-series machine learning for real-time performance-marketing budget optimization.**

ECON 433 — Applied Machine Learning · Spring 2026 · Istanbul 29 Mayıs University
Author: **Abdullah Çopur** · Instructor: Dr. Harun Sencal

---

This repository is the **public, anonymized write-up** of an applied machine-learning system that
runs in production on a real e-commerce advertising operation. It documents how a five-brand
Turkish health-and-beauty operator allocates its Meta advertising budget across placements,
creatives, and hours of the day using:

- **Thompson Sampling** (Beta–Bernoulli multi-armed bandit) for placement budget allocation
- **SARIMA + XGBoost + OLS** ensemble for daily ROAS forecasting
- **MAD-modified z-score** (Iglewicz–Hoaglin) for robust anomaly detection
- **OLS frequency projection** for creative-fatigue forecasting

### 📄 Read the report → [`REPORT.md`](REPORT.md)

The full technical report follows the ECON 433 template: abstract, problem definition,
data description, methodology, empirical results, explainability (XAI), and policy implications.
A formatted copy is also provided as [`REPORT.docx`](REPORT.docx) / `REPORT.pdf`.

### 🖥️ Slides → [`PRESENTATION.pptx`](PRESENTATION.pptx) / `PRESENTATION.pdf`

The 12-slide deck (anonymized, and reconciled with the report's final numbers and
the deliberate-negative-result framing).

### 🧪 Run the methods → [`reproduction/`](reproduction/)

A self-contained, **synthetic-data** harness reproduces every method in the report
(forecast ensemble, Thompson Sampling bandit, MAD-z anomaly detection, fatigue projection)
with no production code or data. See [`reproduction/README.md`](reproduction/README.md):

```bash
pip install -r reproduction/requirements.txt
python -m reproduction.run_all      # console report + figures
pytest reproduction/tests -q        # behavioural tests
```

---

## Headline findings

| # | Finding | Takeaway |
|---|---|---|
| 1 | Aggregate XGBoost forecast **fails** (R² = −1570.85, MAPE = 188.94 %) | A *deliberate negative result* — pooling 5 structurally different brands over 32 days. Per-campaign models converge. |
| 2 | Thompson Sampling flags **Audience Network**: ~1.8 % of spend, **0 revenue** | Beta(15, 60) → P(success) = 0.20 at the prior floor. The single most valuable action is the simplest one: pause it. |
| 3 | MAD-z anomaly detector fires a balanced spread of 10 events | Robust to the heavy-tailed, log-normal ROAS distribution where a classical z-score goes blind. |

The recurring lesson: **in capital allocation under uncertainty, the most useful model is often the
one that makes a single counterfactual visible to a decision-maker who already has the authority to act.**

## Figures

| Figure | File |
|---|---|
| 1 — Daily ROAS distribution (log-normal, heavy-tailed) | [`figures/figure1_roas_distribution.png`](figures/figure1_roas_distribution.png) |
| 2 — Feature correlation matrix | [`figures/figure2_correlation_matrix.png`](figures/figure2_correlation_matrix.png) |
| 3 — Daily ROAS time series | [`figures/figure3_roas_timeseries.png`](figures/figure3_roas_timeseries.png) |
| 4 — XGBoost feature importance | [`figures/figure4_feature_importance.png`](figures/figure4_feature_importance.png) |
| 5 — Hourly order share (bimodal, anonymized) | [`figures/figure5_hourly_order_share.png`](figures/figure5_hourly_order_share.png) |

## Tech stack

`FastAPI` · `PostgreSQL 16` · `SQLAlchemy` · `APScheduler` · `XGBoost` ·
`statsmodels SARIMAX` · `NumPy` / `Pandas` · `React` + `TypeScript` + `Recharts`

## A note on privacy

This is a **public-facing, anonymized** version. To protect the operating firm:

- Brand and company names are replaced with neutral labels (*Brand A–E*, *Product line A/B*).
- Absolute monetary figures are reported as **shares of total platform spend**, never as currency amounts.
- Campaign and ad-set names are anonymized; order volumes are given as orders of magnitude.
- All infrastructure identifiers (server addresses, account IDs, credentials) are removed.
- Every **ratio, statistic, model coefficient, and methodological detail is reported exactly as computed.**

The production source code and data are part of a private, commercially operated system and are
**not** included here. A synthetic-data reproduction harness (mimicking the schema and reproducing
the methods of the report) can be added so that every result is independently runnable without any
proprietary data.

## License

The written report and figures are released under
[Creative Commons Attribution 4.0 (CC BY 4.0)](LICENSE). You are free to share and adapt with attribution.
