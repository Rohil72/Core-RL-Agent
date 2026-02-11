# In-Depth Code Review

## Scope and Method

Reviewed core pipeline modules (`src/`, `scripts/`, `tests/`) with emphasis on:
- testability and CI reliability,
- RL training stability,
- evaluation correctness,
- data pipeline robustness.

I also executed the existing test suite to validate findings and identify runtime defects.

---

## High-Priority Findings

### 1) `pytest` was collecting non-test binary/text artifacts
- **Symptom**: test collection crashed on `test_output.txt` due to UTF-8 decode errors.
- **Risk**: CI instability; unrelated artifacts can break test runs.
- **Fix implemented**:
  - Added a root `conftest.py` with `collect_ignore` for `test_output.txt` and `test_result.log`.
  - Added `pytest.ini` to constrain Python test discovery to `tests/test_*.py` and ensure `pythonpath = .`.

### 2) Incorrect `calibration_curve` import path
- **Symptom**: import error in `src/eval/metrics.py` (`calibration_curve` imported from `sklearn.metrics`).
- **Root cause**: `calibration_curve` belongs to `sklearn.calibration`.
- **Fix implemented**: moved import to `from sklearn.calibration import calibration_curve`.

### 3) RL training failed when tensorboard is not installed
- **Symptom**: `stable-baselines3` raises `ImportError` if `tensorboard_log` is set and tensorboard is absent.
- **Risk**: smoke tests and lightweight environments fail despite no core training issue.
- **Fix implemented**: detect tensorboard availability and disable tensorboard logging when missing.

### 4) Timezone bug in delayed reward resolution
- **Symptom**: `TypeError: Cannot compare tz-naive and tz-aware timestamps` during training.
- **Root cause**: end-of-episode `resolve_until` used naive `pd.Timestamp.max`, compared with UTC-aware confirmation timestamps.
- **Fix implemented**: use UTC-aware `pd.Timestamp.max.tz_localize('UTC')`.

---

## Remaining Gaps / Recommendations

### A) Hard dependency on Parquet engine is not enforced in environment setup
- Multiple pipeline and fetcher tests require parquet I/O but environment may not have `pyarrow`/`fastparquet` installed.
- **Observed impact**: end-to-end and fetcher tests fail in this environment.
- **Recommendation**:
  1. Add `pyarrow` to `requirements.txt` (preferred), and/or
  2. Add fallback persistence path (e.g., pickle/csv) behind a config switch for constrained environments.

### B) Training config design issues
- `make_env` currently reloads config from path each env init; tests monkeypatch around this.
- **Recommendation**: pass resolved config object through env factory to reduce I/O and improve determinism.

### C) PPO policy architecture warning
- `policy_kwargs` uses legacy `net_arch=[dict(...)]` format.
- **Recommendation**: switch to `net_arch=dict(pi=[...], vf=[...])` to align with current SB3 API.

### D) Artifact files committed to repo root
- Files like `test_output.txt`, `test_result.log` in project root introduce accidental tool/test coupling.
- **Recommendation**: move generated artifacts under `artifacts/` or `reports/`, and ignore with `.gitignore`.

---

## Overall Assessment

- **Architecture**: clear modular decomposition across data, env, training, and evaluation.
- **Main quality risks**: environment/dependency assumptions and test-discovery hygiene.
- **After fixes in this PR**: training smoke path and evaluation import path are materially more robust; test collection is now safer.
