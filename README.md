# trail-consistency (research repo)

## What this is

A research project studying [TRaIL-Odom](https://arxiv.org/abs/2609.03561) (RA-L 2026), a
tightly-coupled continuous-time radar-LiDAR-inertial odometry estimator that represents the
trajectory as a B-spline and adaptively reweights radar Doppler measurements when LiDAR
geometry becomes directionally degenerate (e.g. a featureless corridor or tunnel, where
LiDAR alone can't constrain motion along the corridor's own axis).

The core research question: **is that adaptive reweighting actually fixing a real geometric
problem, or is it compensating for mis-calibrated sensor noise that a simpler, principled
noise calibration would fix just as well?** If the latter, the adaptive machinery is a
workaround, not a necessary part of the method.

This repo runs TRaIL-Odom (pinned as a submodule, unmodified by default — every change sits
behind an opt-in flag, see `CLAUDE.md`) against the public Boxi dataset sequences (see
[chiyunnoh.github.io/TRaIL-Odom](https://chiyunnoh.github.io/TRaIL-Odom)), evaluates it
against ground truth, and tests that question through a sequence of falsifiable hypotheses
(H0–H6, see the full guideline). Everything is held to one rule: every number in a report
traces to a script's output in `results/`; nothing is asserted without `path:line` evidence
read directly from the code.

## Status

Phases -1 through 4 of the guideline are done (`docs/PROGRESS.md` has the full log,
`docs/reports/` has the per-phase detail):

- **H0** (reproduction): PASSED — this repo's own runs match the paper's reported numbers
  on Park1/BikeTunnel1 within the expected range.
- **H2** (posterior covariance consistency): SUPPORTED, with caveats — the estimator is
  measurably directionally overconfident on real degenerate data (Park1), though the
  baseline (unmodified upstream) shows a similarly elevated NEES, so this may be partly an
  estimator-architecture property rather than specific to the adaptive weighting itself.
- **H1** (does calibration alone close the gap to the full adaptive method?): REJECTED,
  decisively — a properly-calibrated baseline (CALP) is *worse* than the uncalibrated one on
  both tested sequences (Park1: 4.789m vs 2.536m; BikeTunnel1: 34.807m vs 16.530m ATE), and
  closes none of the gap to the full method (0.819m / 2.067m respectively). Calibration is
  not a substitute for the adaptive reweighting.
- **PROXOBS** (an original fix found during this work, gating one radar factor on whether
  radar already observes the LiDAR-degenerate direction): a real, 3-rep-verified improvement
  — BikeTunnel1 ATE mean 1.419m vs upstream's 2.067m. Park1 is already beaten by
  unmodified upstream (0.819m vs the paper's 1.010m). Airfield1 remains an open,
  documented gap (this repo: ~0.354m vs the paper's 0.183m) after fifteen
  independently-verified investigation angles — logged as `UNKNOWN` in
  `docs/OPEN_QUESTIONS.md` per the project's own rule rather than papered over.

Phases 5–7 are not started.

## Layout
- `sim/toy_consistency.py` — Phase -1 sanity simulation (falsification only, not evidence
  about real data; see Part B)
- `docs/` — FACTS, DECISIONS, PROGRESS, OPEN_QUESTIONS, phase reports
- `ws/src/TRaIL-Odom` — upstream submodule, pinned at `667d7b4132461db2c79e20521dfe5f51b35ffae8`,
  branch `research/consistency`
- `configs/` — per-variant research configs (ablation flags layered on the upstream default)
- `scripts/` — build/run/eval pipeline (`run_seq.sh`, `eval_traj.py`, `docker_build.sh`) and
  the live viewer (`native_viewer/`, `run_live_rviz.sh`, `run_native_viewer.sh`)
- `results/` — per-run outputs (`<variant>/<sequence>/rep<rep>/`, `run.json` provenance) and
  `summaries/` (GT, eval CSVs) — never committed as "evidence" without a `run.json` behind it

## Reproduce the toy simulation
```
python3 sim/toy_consistency.py nominal
python3 sim/toy_consistency.py "radar noise x2"
python3 sim/toy_consistency.py "radar 10% outliers"
```

## Run a research sequence (batch, no GUI)
```
scripts/run_seq.sh <variant> <sequence> <rep>   # e.g. scripts/run_seq.sh V4 Park1 1
python3 scripts/eval_traj.py                    # ATE/RTE against GT, writes results/summaries/eval_traj.csv
```
Sequences: `Airfield1 Airfield2 BikeTunnel1 BikeTunnel2 Park1 Park2`. Writes
`results/<variant>/<sequence>/rep<rep>/` including `run.json` (commit hashes, config
hash, flags — see `CLAUDE.md` rule 6).

## Live visualization — Ra-LIO-DD (native 3D viewer)

A standalone native viewer (Open3D + rclpy; not rviz, not upstream's `trail_rviz.rviz`)
for watching a sequence run live: point cloud colored by age (green=fresh →
red=40s+/drift-affected), estimated path vs. GT overlay, live drift estimate, raw IMU
readout, and a sequence dropdown with Run/Stop to start/switch sequences from inside the
app. Code: `scripts/native_viewer/`.

**One-time setup** (GPU acceleration; the viewer renders via Vulkan and needs this, not
just a driver):
```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```
Without this it still runs, just on software rendering (slower, periodic stutter under
a large map).

**Build the image** (once, or after editing `scripts/native_viewer/Dockerfile`):
```
docker build -t trail-odom-native-viewer:latest scripts/native_viewer
```

**Run** (needs an X11 desktop session):
```
scripts/run_native_viewer.sh [sequence] [start_time]   # e.g. scripts/run_native_viewer.sh Park1 20
```
Defaults to `Park1`, `start_time=20`, auto-starts that sequence. Pick a different one
from the panel's dropdown and click "Run selected sequence" any time, including while
one is already running (it stops the old one first). `MAX_RANGE` env var controls how
far back (in meters) the point cloud/path persist before being pruned (default 150).

**Alternative**: `scripts/run_live_rviz.sh [sequence] [start_time]` runs upstream's own
`rviz2` with a recolored copy of their `trail_rviz.rviz` instead — needs only OpenGL
(`--gpus all`, not the CDI setup above), no dropdown/GT overlay/drift/IMU panel.

## Ground rules

Full list in `CLAUDE.md` / `docs/TRaIL_Consistency_Research_Guideline_v2.md` Part J —
most load-bearing: every reported number traces to a `results/` script output; upstream
stays unmodified by default (changes sit behind opt-in flags); one change per commit;
unknowns get written down as `UNKNOWN — needs Ken` in `docs/OPEN_QUESTIONS.md` rather than
guessed at or quietly dropped.

This project is independent of `~/Desktop/Trail/ObsTRaIL` (see `docs/DECISIONS.md`).
