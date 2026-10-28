# Data dictionary

All CSVs use comma-separated UTF-8 text with header rows. Fractional service metrics are in [0,1]; multiply by 100 for percent. Seeds and indices may appear as integer-looking numeric values.

## Main `trajectories.csv` (504 rows per grid)

| Column | Meaning / unit |
|---|---|
| seed | Physical seed 16101–16124; independent statistical cluster |
| regime | cruise, turn_blockage or ood |
| policy | periodic-local, sparse-local or ai-31/47/59/71/89 |
| availability | Fraction of mission satisfying all scored feasibility constraints |
| goodput | Mean jointly feasible secrecy-rate contribution after radio overhead, normalized bit/s/Hz |
| qos_violation | Fraction failing communication rate constraints |
| echo_violation | Fraction failing echo SNR constraint |
| sic_violation | Fraction failing receiver/SIC/order/power checks grouped by evaluator |
| requests | Full CSI requests during service; excludes common initial acquisition |
| compute | Online capped resource ledger in seconds; excludes setup, may cap overhead differently from actual decision_s |
| limited | Number of selected local updates (action 1) |
| joint | Number of joint/robust updates; zero in main local-only test |
| rejected | Requests with no acceptance recorded in their launch slot, as counted by inherited result routine; a pending job can complete later, so do not infer total failed solves from this column |
| joint_delivery_1s / joint_delivery_5s / joint_delivery_20s | Fraction of six objects delivered by deadline using rate accumulated only in jointly feasible intervals |
| delivery_1s / delivery_5s / delivery_20s | Corresponding rate-only delivery fractions without joint-feasibility mask |
| decision_s | Sum of actual feature/decision/inference overhead in seconds |
| solver_s | Charged restart+solve+acceptance work in seconds, truncated by ledger |
| setup_s | Common initial setup plus per-model loading where used, seconds; before service origin |
| timeouts | Number of budget exhaustion events recorded by replay |
| total_compute_s | setup_s + decision_s + solver_s; resource comparison metric, seconds |

## Main `decisions.csv` (12,096 rows per grid)

| Column | Meaning / unit |
|---|---|
| policy, seed, regime | Same identifiers as trajectory CSV |
| k | Five-second replay slot index 0–23; business opportunities only when k % 4 == 0 |
| action | 0 hold, 1 local update |
| accepted | A pending or new job completes and is accepted within this slot; can occur in a later slot than launch |
| requests | Cumulative full CSI request count after decision |
| compute | Cumulative capped online ledger, seconds |
| decision_s | Actual decision overhead for this slot, seconds |
| state | Active candidate state identifier after slot; 0 initial, local candidate 1+2*k_launch |
| busy | Pending job remains after this slot (0/1) |

## `candidate-timings.csv` (432 rows)

seed/regime identify asset; k is slot index; request_time_s = 5*k; solve_validation_s is recorded restart+solve+explicit-check timer; accepted and failed are original candidate flags; common_setup_s repeats the trajectory's initial setup timer. All timings are seconds. Candidate library generation is shared; candidates never requested by a policy are not charged to it.

## MAT assets (72 files; MATLAB v7)

`c`: inherited configuration; `meta`: 121 one-second trajectory positions, regime, speeds and event times; `z`: latent diffuse coefficients, phases, directions and BS–IRS channel; `probes`: 121×4 noisy scalar observations. `initial`: initial W/V configuration; `setupTime`: measured setup seconds. `Ws` (6×4×24×2), `Vs` (8×24×2), `accepted`, `failed`, `runtime` hold candidates/flags/times. Python axis action index 0 is the local candidate; unused action slots retain initial values and zero timing, not extra measured solves. Only k=0,4,8,12,16,20 have measured local solves. `testW`, `testV`, `testMetrics` provide MATLAB evaluator references at times [0,20,40,80,120] s; testMetrics columns are secrecy, feasibility, echo, first-user rate, observable margin. Complex numbers remain in MAT form. The explicit effective protocol overrides legacy c fields such as testSeeds, scoreStepSeconds, periodSeconds and featureNames.

## Historical CSVs

`scenario`: 1 cruise, 2 turn/blockage, 3 OOD; `seed`: historical seed; `policy`: historical policy index (Figure 6 selects 5, guarded hold); `step_s`: integration step seconds. `goodput`, `availability`, `qos_violation`, `echo_violation`, `sic_violation` use the definitions above. `certified_impossible_failure`: fraction certified by the necessary bound. `acquired_feasible_aged_failure`: failure fraction after feasible acquired CSI. `unresolved_failure`: remaining failure fraction. `qos_only`, `echo_only`, `qos_echo`, `sic_only`, `qos_sic`, `echo_sic`, `all_three`: mutually classified constraint-failure fractions. `delivery_*` and `joint_delivery_*`: deadline fractions as above. These historic policy numbers are not main policy labels. Full historical results are retained without filtering; the 18-row Figure 6 subset is provided separately.

## Initializer CSVs

`trajectory`: historical regime-seed ID; `epoch`: original epoch index (solver-validation uses the stored MATLAB index, initializers uses Python index); `method`: scan, learned or simple. In solver-validation: `init_s`, `solve_s`, `total_s` (seconds); `initial_feasible` (0/1); `initial_margin` (normalized minimum feasibility margin); `accepted` (0/1); `returned_margin` (margin of returned configuration); `secrecy` (normalized secrecy rate); `fallback`, `failed` (0/1). In initializers: `index` is selected codebook candidate; `feasible`, `margin`, `time_s`, `fallback` describe the selection stage. Do not row-join these tables on epoch without accounting for the original indexing conventions.

## Training/selection files

`counterfactual-training.npz`: X (2304×26 causal features), Y (2304×3 counterfactual scores for hold/local/joint), small (2304×12 base features). `training-states.csv`: seed, regime, episode (behavior rollout 0–47), epoch (business opportunity 0–5), best (argmax action index). Rows align with NPZ arrays. Base features, in order: time/120, CSI age/40, relative probe residual, clipped acquired margin, power-drop proxy, angular array loss, risk proxy, predicted-window/40, remaining requests/6, remaining compute/12, opportunity phase, intercept. Expanded features append four current normalized probes, four historical normalized-probe means, four standard deviations, last acceptance indicator and historical acceptance ratio. Exact formulas are in `source/.../confirmation.py` and `direction_env.py`.

`validation.csv`: historical configuration-selection rollouts. Columns inherited from Env.result use main trajectory definitions, with historical assumed costs. `inference_ms_mean/max`: per-decision milliseconds; `last_request_s`, `sixth_request_s`: request times (-1 means absent); `accepted_requests`: accepted count; `model_key`, `expanded`, `learning_seed`, `depth`, `leaf`: candidate model identifiers/hyperparameters. Historical model policy names differ from the main seven policies. `selected.json` preserves both historical full/base selections; this release includes only the five full-d8-l24 models used in the main paper. Joblib containers hold regressor, expanded, depth, leaf and seed. The original full configuration sweep remains in validation.csv; unused model binaries are not included.

## Derived/reference outputs

`summary.csv` and reconstructed main-summary retain policy/regime means of named metrics. `table-1.csv` reports compute_s and delivery_percent. `figure-4-data.csv` reports compute_mean_s, ci_low_s, ci_high_s. Paired-seed-differences reports compute_saved_s (baseline minus learned) and adverse_delivery_difference (learned minus baseline, fraction). `statistics.json` contains complete paired comparisons, bootstrap intervals, exact p values, effect size, training-seed summaries, and conservative loss-screen values. `grid-check.json` contains maximum absolute metric differences and counts of changed actions/acceptances. `mixture-sensitivity.json` records regime compute differences and cruise break-even fraction. `loss-cases.json` preserves the identified paired service difference with seed/regime. `initializer-summary.json` and `historical-diagnosis.json` are means of their respective raw records. `verification.json` records row counts, checks and PASS status, not a new experiment.
