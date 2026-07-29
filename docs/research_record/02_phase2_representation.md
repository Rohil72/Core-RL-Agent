# Phase 2: Representation Learning and Latent Audit

`[PARTIAL]`

## Hypothesis

The Transformer should learn a contextual market representation whose geometry
captures future opportunity and risk rather than reproducing manually defined
market stages.

The work audited features, targets, learnability, and latent behavior before
moving to market-state discovery. The representation hypothesis gained support:
the encoder repeatedly produced a structured latent space that separated future
opportunity profiles across random initializations.

## Result

The result was not a profitable policy. It was evidence that representation
learning was not the main bottleneck. The later experiments therefore froze the
encoder and concentrated on retrieval semantics, memory construction, and
decision conversion.

## Limits

- Fundamental coverage and point-in-time alignment remained important risks.
- A meaningful latent space does not prove ticker-invariant transfer.
- Direct prediction-head performance was not sufficient evidence for profitable
  analogue retrieval.

## Source Map

- `scripts/phase2_feature_audit.py`
- `scripts/phase2_target_audit.py`
- `scripts/phase2_latent_audit.py`
- `scripts/phase2_baseline_learnability.py`
- `scripts/phase2_run_all.py`
