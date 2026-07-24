# Data Manifests

Track compact provenance records here, not downloaded market data.

Each frozen dataset should record:

- provider and retrieval timestamp;
- requested ticker universe and date boundaries;
- per-file row counts, date ranges, and SHA-256 digests;
- missing tickers and coverage failures;
- feature-pipeline and source commit identifiers.

Raw Parquet files remain outside Git and should be backed up to durable object
storage when the testbed is frozen.
