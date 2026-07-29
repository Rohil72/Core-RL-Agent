# Phase 3: Latent Market-State Discovery

`[ESTABLISHED] [PARTIAL]`

## Question

Does the learned latent space organize market behavior into stable opportunity
states, or does it merely memorize noise, labels, or ticker identity?

## Recorded Results

- KMeans with `k=8` produced an adjusted-Rand mean of approximately `0.9967`
  across seeds, indicating highly reproducible cluster assignments.
- Latent nearest-neighbor correlations were approximately `0.957` for future
  maximum return, `0.969` for future minimum return, and `0.877` for path
  quality.
- These results supported the existence of stable, outcome-separated latent
  states rather than rigid Stage 1-4 cycles.

## Critical Limitation

Cluster ticker entropy was substantially below a random reference. The latent
space was stable, but ticker identity and company-specific structure remained
entangled with market-state structure. Therefore Phase 3 supports
representation reproducibility and outcome separation, not universal
ticker-invariant transfer.

## Consequence

The project stopped asking whether meaningful latent states exist and began
asking whether those states can retrieve economically useful historical
precedents.

## Source Map

- `scripts/phase3_market_state_discovery.py`
- Earlier Phase 3 reports under `reports/phase3/` when restored.
