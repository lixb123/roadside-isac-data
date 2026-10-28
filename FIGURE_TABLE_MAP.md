# Figure, table and claim map

| Item | Inputs | Calculation | Reproduced output |
|---|---|---|---|
| Figures 1–3 | Schematics | No experimental aggregation | Original PDFs in `manuscript-figures/` |
| Figure 4 | `data/main/grid-0.0025/trajectories.csv` | Five learners within trajectory, equal regimes within seed, 24-seed means and 95% bootstrap CI | `verified-results/figure-4-data.csv`, `figure-4.png` |
| Figure 5 | Same main trajectory CSV | Per-regime mean five-second joint delivery, ×100 | `verified-results/figure-5-data.csv`, `figure-5.png` |
| Table 1 | Same main trajectory CSV | Per-regime compute and five-second joint delivery | `verified-results/table-1.csv` |
| Figure 6 | `data/historical/guarded-hold-grid-0.0025.csv` | Six-seed mean time fractions; derived by scenario 1/2/3, policy 5, step .0025 from full historical CSV | `verified-results/figure-6-data.csv`, `figure-6.png` |
| Compute savings 47.4%, 22.9% | Main trajectory CSV | Ratio of paired mean savings to baseline mean cost | `verified-results/recomputed-results.json`: comparisons |
| Adverse delivery differences +2.78, +7.29 pp | Main trajectory CSV | Five learner mean, then two adverse regimes within seed; 24 paired seeds | Same JSON: comparisons |
| Grid difference 0.000448 | Both main grids, trajectory and decision CSVs | Match (seed, regime, policy[, k]); maximum absolute difference and action equality | Same JSON: grid_check |
| Five learner computation 3.843–3.987 s | Main trajectory CSV | Mean three regimes and 24 seeds for each learner | Same JSON: learning_seeds |
| Initializer acceptances and times | `data/initializer/solver-validation.csv` | 24 calls per initializer, three methods; mean total_s and accepted | Same JSON: initializer |

Historical and pilot figures/statistics are analyzed within their own protocols. Timing measurements are input data, not newly measured by the portable reconstruction.
