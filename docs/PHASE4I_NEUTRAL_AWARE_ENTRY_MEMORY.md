# Phase 4I: Neutral-Aware Entry Memory

Phase 4I is a frozen-encoder follow-up to Phase 4H. It changes only external memory reasoning and policy entry filters. The transformer, features, labels, and A2 exit logic remain unchanged.

## Hypothesis

Phase 4H compared successful paths only against failed paths. Ordinary, sideways, slow, and too-early states were not a memory class, so a query could receive a relative success probability even when it did not resemble a useful historical opportunity.

Phase 4I tests whether a third neutral or wait bank makes entry evidence more selective.

## Memory Contract

Each mature training state is classified using the existing causal labels:

- `success`: +10% within 63 sessions and upside reaches its barrier before the -10% drawdown barrier over 126 sessions.
- `failure`: -10% drawdown arrives before upside and the 63-session minimum is at most -10%.
- `neutral`: every remaining mature state, thinned to one state per ticker per 21 sessions so routine states cannot dominate event memories.

The system retrieves 25 causal, same-ticker-excluded neighbours from every bank. It estimates success probability as:

```text
success evidence / (success + failure + neutral evidence)
```

It also requires absolute support: the closest prototype from any bank must be within one training-calibrated kernel bandwidth. A state far from all memory is ineligible, even if its relative density ratio looks favourable.

## Policies

`I0_a2_base` is unchanged A2. `I1_neutral_aware_rank` and `I2_a2_neutral_aware_gate` require all of:

- sufficient neighbours in success, failure, and neutral banks;
- absolute support pass;
- success probability at least 0.50;
- success evidence greater than combined failure-plus-neutral evidence.

The entry score can change, but all candidates keep A2 `opportunity_score` as the exit score. This isolates entry timing from exit behavior.

## Run

```powershell
python scripts\run_phase4h_rally_start_memory.py --config configs\phase4i_entry_advantage_memory.yaml --run-id phase4i_v1 --stage all
```

Resume an interrupted run with:

```powershell
python scripts\run_phase4h_rally_start_memory.py --config configs\phase4i_entry_advantage_memory.yaml --run-id phase4i_v1 --stage all --resume
```

Validation chooses a candidate. Test and holdout remain sealed unless every predeclared promotion gate passes.
