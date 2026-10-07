# trail-consistency (research repo)

Calibrated, consistent radar-LiDAR-inertial odometry under directional degeneracy — is
adaptive Doppler reweighting a workaround for mis-calibrated noise?

Full guideline: `docs/TRaIL_Consistency_Research_Guideline_v2.md`. Ground rules: `CLAUDE.md`.

## Layout
- `sim/toy_consistency.py` — Phase -1 sanity simulation (falsification only, not evidence
  about real data; see Part B)
- `docs/` — FACTS, DECISIONS, PROGRESS, OPEN_QUESTIONS, phase reports
- `ws/src/TRaIL-Odom` — upstream submodule, pinned at `667d7b4132461db2c79e20521dfe5f51b35ffae8`,
  branch `research/consistency`
- `configs/`, `scripts/`, `tests/` — added as later phases need them

## Reproduce the toy simulation
```
python3 sim/toy_consistency.py nominal
python3 sim/toy_consistency.py "radar noise x2"
python3 sim/toy_consistency.py "radar 10% outliers"
```

This project is independent of `~/Desktop/Trail/ObsTRaIL` (see `docs/DECISIONS.md`).
