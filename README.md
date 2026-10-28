# Roadside ISAC: data supporting resource-aware CSI updates

Version 1.0.0 — 28 October 2026.

Repository: https://github.com/lixb123/roadside-isac-data

Fixed version: https://github.com/lixb123/roadside-isac-data/tree/v1.0.0

Release and downloadable package: https://github.com/lixb123/roadside-isac-data/releases/tag/v1.0.0

Companion data for **Resource-Aware Updates for Low-Altitude ITS: Learning When to Refresh CSI in Roadside ISAC**, by Xiaobao Li and Jinlong Liu, Harbin Institute of Technology. Contact: Xiaobao Li (23b905015@stu.hit.edu.cn; ORCID 0009-0001-5416-3943); corresponding author Jinlong Liu (yq20@hit.edu.cn).

This package supports Figures 4–6, Table 1, and the numerical/statistical statements in the current submission. Figures 1–3 are schematics; their submitted PDFs are included for identification. The data are simulated, not measured field channels. This release is identified by GitHub tag `v1.0.0`. No paper DOI or dataset DOI is assigned.

## Reproduce the reported numbers

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_results.py --mat-check
```

The command verifies file hashes, independently recomputes the reported summaries and paired statistics, checks both integration grids, checks MATLAB/Python physics at five reference times in each of 72 assets, and reconstructs Figures 4–6 as PNGs. Outputs go to `reproduced/`. All assertions must pass. A prepared verified example is in `verified-results/`. Reconstructed plots reproduce the data; exact journal layout is retained in `manuscript-figures/`.

No MATLAB, paid solver, cloud service, or model deserialization is needed for this verification. The optional frozen joblib models were created with scikit-learn 1.9.1 and joblib 1.6.0. They are not loaded by the reproduction script. `requirements.txt` covers numerical verification and plotting; archival source excerpts have additional dependencies.

## Contents and experiment separation

| Directory | Contents |
|---|---|
| `data/main/` | 72 original MAT assets, 432 candidate timings, and two grids of trajectory and decision CSVs |
| `data/historical/` | Full historical metric table and exact 18-row Figure 6 subset |
| `data/initializer/` | Separate initializer pilot, including all 72 solver calls |
| `models/` | Five frozen learners, 2,304 training states/targets, selection metadata and validation results |
| `reference/` | Original complete analysis outputs for numerical comparison |
| `metadata/` | Effective protocol, inherited configuration, frozen hashes and source map |
| `scripts/` | Portable verifier and reconstruction code |
| `source/` | Unmodified implementation excerpts for audit |
| `manuscript-figures/` | Six figures used in the manuscript |
| `verified-results/` | Outputs of the successful packaged-data verification |

The main test has 24 independent physical seed clusters and three correlated regimes per cluster. Seven policies yield 504 trajectory rows and 12,096 decision rows per grid. Five learner seeds are averaged within each trajectory, then regimes within each physical seed; they are neither five additional physical trials nor a deployed ensemble. Main compute comparisons weight the three regimes equally. Adverse service comparisons average turn/blockage and OOD.

Historical Figure 6 uses six seeds in each regime and a fixed 2.2-second replay, separate from the main timing protocol. The initializer pilot uses 24 snapshots from two validation seed clusters; do not pool these observations with the main independent experiment.

## Timing and units

`total_compute_s = setup_s + decision_s + solver_s`. It includes recorded restart, candidate solving, explicit acceptance checks, feature generation, policy selection/inference, common setup, and model loading where used. Setup precedes service origin and is outside the online 12-second ledger. Acquisition delay affects deployment; acquisition airtime and continuous observation overhead affect service. These radio costs are not CPU seconds. Simulation integration, file loading and offline fitting are outside online compute totals. Published workstation timings are preserved; rerunning timing on another machine produces a different measurement.

At 1.25 ms, recorded primary decision/setup costs are reused. Actions and delivery outcomes are unchanged. `compute` in raw trajectory CSV is the capped online ledger; it is not interchangeable with `total_compute_s`. Inherited MAT configuration retains legacy experiment fields; use `metadata/protocol.json`, `DATA_DICTIONARY.md` and the frozen implementation for effective main-test settings.

## Analysis conventions

Bootstrap intervals use 20,000 physical-seed resamples and RNG seed 20261024. Resource tests are exact paired two-sided sign-flip tests with Holm adjustment over two baselines. The verification script evaluates the exact tail algebraically for the observed all-positive difference vectors. The original complete service-loss screen, including its 1/2/5-percentage-point sensitivity and decision chronology, is retained under `reference/` and `metadata/original/`; bootstrap mean differences and this conservative screen are distinct analyses.

See `FIGURE_TABLE_MAP.md` and `DATA_DICTIONARY.md` for the source of each result and the meaning of every released CSV column. `metadata/provenance.json` records original project-relative sources and SHA-256 values. `checksums.sha256` covers release files other than itself, with forward-slash paths.

## Scope of reproduction and reuse

This is a result-data and numerical-reconstruction release. Original MATLAB candidate assets and their measured costs are supplied; the main verification reconstructs statistics from recorded results and crosschecks reference physics, rather than rerunning all solver jobs, training, or the entire channel replay. `source/README.md` describes the archival excerpts. No third-party solver binaries, author photos, cover letters, manuscript sources, credentials or internal working notes are included.

No blanket open-source or Creative Commons reuse license is granted for this repository. Public access is provided for inspection and verification; applicable copyright and third-party rights remain with their respective holders. Please contact the authors about reuse beyond applicable law or platform terms. Cite the fixed version `v1.0.0` using `CITATION.cff` and the release link above.
