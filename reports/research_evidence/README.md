# Research Evidence

This directory is the Git-tracked evidence lane for compact, reviewable outputs.
Do not place checkpoints, latent tables, NPZ datasets, or raw Parquet files here.

For each promoted run, retain:

- experiment and generated manifests;
- source, data, environment, and selected-artifact SHA-256 digests;
- complete job status and runtime summaries;
- pilot and final policy-selection records;
- per-market and per-seed metrics;
- aggregate metrics, PBO, concentration, drawdown, and promotion-gate results;
- compact trade and equity summaries;
- failed-run and exclusion records.

Keep routine snapshots in persistent or object storage. Commit this evidence
after each completed stage and once more when the final candidate is locked.
