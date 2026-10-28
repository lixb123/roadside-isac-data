# Frozen prospective resource/service screening

Written before generating seeds 16101–16124. No manuscript results are replaced.

## Fixed design

24 independent physical seed clusters, each replayed in cruise, turn_blockage and ood (72 trajectories). Same historical propagation, traffic, receiver feasibility tests, six full-CSI requests and 12-second online computation cap. Business opportunities every 20 seconds. Generate only the six local solver candidates used by these policies; joint updates are disabled for every comparator in this narrowly scoped screening.

Policies: frozen full ExtraTrees models d8/l24, seeds 31/47/59/71/89; periodic-local; frozen sparse-local (age 120, residual .32, margin .08). No new training/tuning. Five learners are separate deployments and their outcomes are averaged within each physical seed, not five independent observations or a deployed ensemble. CSI at acquisition remains perfect; scalar probes and telemetry retain the previous paid noisy model.

## Complete timing sensitivity

Single sequential MATLAB export process. Candidate timer includes deterministic restart initializer, solver and explicit acceptance evaluation. Initial feasible setup is timed separately and added to each policy's resource total but precedes the service origin. Python timers include causal observation/features, prediction, allowed-action checks and policy choice. Common simulation/propagation integration and file loading are excluded. Online full-CSI acquisition adds .2s to deployment, and .6s airtime debit per request is retained. Pending jobs and a 12s timeout retain the incumbent. Decision cost is charged even on holds. Model loading is measured per learner and included as setup. Offline fitting is disclosed separately, not treated as online savings. This is mixed MATLAB/Python sequential workstation replay, not embedded-device measurement.

Primary service: adverse-regime mean 5s joint-compliant delivery; cluster is the seed, average the two adverse regimes and five learner seeds before comparison. Proposed loss tolerance 1pp (engineering screen, not an actual ITS standard); report sensitivity at 2pp and 5pp too. The tolerance will be locked before scoring, using user response if available.

Conservative noninferiority criterion: for each baseline form per-seed nonnegative loss L=max(0,baseline_delivery-AI_delivery), bounded by one. Upper-bound E[L] by an exact one-sided binomial upper confidence bound for P(L>0), which also upper-bounds mean loss. Bonferroni alpha=.025 per baseline. Declare noninferiority only when this bound is below the chosen tolerance. This deliberately conservative screen can be inconclusive; zero differences are not proof of equivalence. Report paired mean differences and seed bootstrap intervals as descriptive quantities only.

Primary resource: actual online decision+initializer+solve+acceptance time plus setup, against both baselines. Also requests, online-only time, accepted/rejected updates, timeouts and per-regime service. Exact paired sign-flip tests on 24 seed means, Holm correction for two resource comparisons; report intervals and five-learner variability. Resource advantage alone does not license a loss-controlled efficiency claim if noninferiority fails.

Implementation clarification before scoring: the 12s ledger truncates pending solver work; any already incurred decision overhead remains in actual resource totals even if it crosses that ledger. At exhausted budgets, the policy skips feature/model computation and holds. Setup precedes service and is outside the12s online ledger. Unused robust/joint candidates are not evaluated by the replay reader.

Grid .0025s, with .00125s repeat only if timing changes crossing points materially; tiny timing differences that lie below grid resolution must be disclosed. Solver runtime is machine-dependent. Physics evaluator crosschecked against MATLAB at five known points in every asset.

## Learned initialization exploration (separate, not confirmatory)

Only old training seeds12101–12104 and validation13101–13102, cruise and turn_blockage. At each20s acquired-CSI point, reproduce deterministic17-member codebook initialization. Learn a codebook index from current acquired channel features using training trajectories only, fixed ExtraTrees classifier96trees/depth8/leaf2/seed31. Compare full deterministic scan, learned single candidate plus feasibility validation and deterministic fallback, and simple candidate0 plus same fallback. Include feature/inference/validation/fallback cost. No future probes or future feasibility targets in learning. Preserve all feasibility tests. Evaluate initializer margins and resulting single-iteration local solves on validation; a small pilot supports a continue/stop decision, not a publication superiority claim.

## Venue decision

No universal required improvement percentage, no acceptance probability. Assess a single clear tutorial insight/architecture, nearest-work distinction, case-study evidence, independent AI value, task realism and format compliance separately. “First” and “guaranteed unchanged performance” are disallowed unless their separate evidence requirements are met.
