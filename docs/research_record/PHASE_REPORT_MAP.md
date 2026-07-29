# Phase Report Map

This map tags the original implementation documents and artifacts. The
research record summarizes results; the original files remain the source of
the implementation contract.

| Record | Primary tracked docs/configs | Report/artifact location | Tag |
|---|---|---|---|
| Phase 1 | `scripts/phase1_analysis.py`, `scripts/orchestrate_phase1.py` | `reports/phase1/` when restored | incomplete result record |
| Phase 2 | `scripts/phase2_*.py` | `reports/phase2/` when restored | partial |
| Phase 3 | `scripts/phase3_market_state_discovery.py` | `reports/phase3/` when restored | established with ticker-entropy limit |
| Phase 4A-C | `scripts/run_phase4_loss_sweep.py`, `configs/phase4c_rotating_holdouts.yaml` | `reports/phase4b/`, `reports/phase4c/` | partial; pre-fix causal artifacts |
| Phase 4D | `docs/PHASE4D_RETRIEVAL_SEMANTICS.md`, `configs/phase4d_retrieval_semantics.yaml` | `reports/phase4d/phase4d_v1/` | semantic gain; pre-fix causal artifacts |
| Phase 4E | `docs/PHASE4E_CROSS_MARKET_SHARPE.md`, `configs/phase4e_cross_market_sharpe.yaml` | `reports/phase4e/phase4e_v1/` | rejected; pre-fix causal artifacts |
| Phase 4F | `docs/PHASE4F_CAUSAL_EXPOSURE_CONTROLLER.md`, `configs/phase4f_exposure_controller.yaml` | `reports/phase4f/phase4f_v2/` | risk improvement, no promotion; pre-fix causal artifacts |
| Phase 4G | `docs/PHASE4G_MEMORY_TRANSFER_RELIABILITY.md`, `configs/phase4g_memory_reliability.yaml` | `reports/phase4g/phase4g_v1/` | rejected reliability branch; pre-fix causal artifacts |
| Phase 4H | `docs/PHASE4H_RALLY_START_MEMORY.md`, `configs/phase4h_rally_start_memory.yaml` | `reports/phase4h/phase4h_v1/` | rejected rally entry branch; pre-fix causal artifacts |
| Phase 4I | `docs/PHASE4I_NEUTRAL_AWARE_ENTRY_MEMORY.md`, `configs/phase4i_entry_advantage_memory.yaml` | `reports/phase4i/phase4i_v1/` | lower drawdown, no promotion; pre-fix causal artifacts |
| Phase 5 | `docs/PHASE5_EXECUTION.md`, `configs/phase5_*.yaml` | `reports/phase5/` | corrected causal allocator evidence; rejected candidates |
| Phase 6 RL | `configs/final_research_testbed.yaml`, `scripts/build_final_testbed.py` | `reports/final_testbed/phase6_a30_final_v1/` | rejected RL formulation |
| Phase 6 frozen memory | `configs/phase6_frozen_memory_sweep.yaml`, `configs/final_memory_confirmation.yaml` | `reports/phase6_frozen_memory/`, external confirmation files | development winner rejected; sealed confirmation failed |
| Phase 6 dual regional | `configs/phase6_dual_regional_memory.yaml`, `scripts/run_phase6_dual_regional_memory.py` | `reports/phase6_dual_regional_memory/` | rejected regional internal/external memory |
| Local rank | `configs/local_rank_ensemble.yaml`, `scripts/run_local_rank_ensemble.py` | `reports/local_rank_ensemble/` | rejected; embedding bug found |
| Final memory | `configs/final_memory_study.yaml`, `scripts/run_final_memory_study.py` | `reports/final_memory_study/final_memory_study_v1/` | rejected; needs repair |
| Closing repair | `configs/final_memory_repair.yaml`, `docs/FINAL_MEMORY_REPAIR.md` | `reports/final_memory_repair/final_memory_repair_v1/` | implemented; diagnostic-only until run |

## Related Research References

- `docs/PHASE5_REFERENCES.md`
- `docs/research/PHASE4F_EXPOSURE_CONTROLLER_PAPERS.md`
- `docs/research/PHASE4G_MEMORY_TRANSFER_RELIABILITY_PAPERS.md`
- `docs/FINAL_MEMORY_STUDY.md`
