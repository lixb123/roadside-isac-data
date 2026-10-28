# Archival implementation excerpts

The source files preserve the exact relevant code and relative project layout for auditing physics, decision features, timing, selection, solver formulation and statistics. Their hashes can be compared with `metadata/frozen-inputs.json` and `metadata/provenance.json`.

Run the portable `scripts/reproduce_results.py` for the supported package workflow. These archival entry points retain their original directory assumptions and are not a complete standalone solver/training project. In particular, the original main replay validates absolute pre-scoring paths, and the MATLAB exporter requires additional trajectory/channel routines and a configured MATLAB/CVX/MOSEK installation. No full solver rerun or retraining is represented as having been verified by this data release. Only the archived `direction_env.py` physics functions are imported by the optional reference crosscheck.
