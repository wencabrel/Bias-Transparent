# trail-consistency (research repo)

## What this is

A research project studying [TRaIL-Odom](https://arxiv.org/abs/2609.03561) (RA-L 2026), a
tightly-coupled continuous-time radar-LiDAR-inertial odometry estimator that represents the
trajectory as a B-spline and adaptively reweights radar Doppler measurements when LiDAR
geometry becomes directionally degenerate (e.g. a featureless corridor or tunnel, where
LiDAR alone can't constrain motion along the corridor's own axis).

This repo runs TRaIL-Odom (pinned as a submodule, unmodified by default — every change sits
behind an opt-in flag) against the public Boxi dataset sequences (see
[chiyunnoh.github.io/TRaIL-Odom](https://chiyunnoh.github.io/TRaIL-Odom)) and the held-out
GaRLILEO sequences, evaluates it against ground truth, and tests a series of falsifiable
hypotheses about that adaptive reweighting. Every reported number traces to a script's
output in `results/`.

## Findings

The running, dated account of what's been tried, what passed/failed, and why is in:
- `docs/PROGRESS.md` — phase-by-phase log
- `docs/DECISIONS.md` — decisions and their reasoning, including pre-registered experiment
  criteria (written before running)
- `docs/OPEN_QUESTIONS.md` — open questions
- `docs/reports/` — detailed per-phase writeups

## Project structure

```
.
├── configs/base/           per-variant research configs (ablation flags on upstream default)
├── data/                   datasets (Boxi, GaRLILEO) — not committed
├── docs/
│   ├── FACTS.md            verified path:line facts about the codebase
│   ├── DECISIONS.md        decision log with reasoning
│   ├── PROGRESS.md         phase-by-phase research log
│   ├── OPEN_QUESTIONS.md   open questions
│   ├── reports/            detailed per-phase writeups
│   ├── TRaIL_Consistency_Research_Guideline_v2.md   the full research guideline
│   └── paper_latex/        paper draft; figs/ is generated, gitignored
├── results/                per-run outputs: <variant>/<sequence>/rep<rep>/, run.json
│                           provenance, and summaries/ (GT + eval CSVs) — never committed
│                           as "evidence" without a run.json behind it
├── scripts/
│   ├── run_seq.sh          run one (variant, sequence, rep)
│   ├── eval_traj.py        ATE/RTE against GT
│   ├── docker_build.sh     build the estimator's Docker image
│   ├── run_matrix.py       batch-run a grid of variants/sequences/reps
│   ├── gt_extract.py / gt_velocity.py / gt_latency.py   ground-truth extraction/checks
│   ├── analyze_p1_drift.py / analyze_p3.py / analyze_rt.py   mechanism-test analysis
│   ├── check_clock_sync.py / check_lever_arm.py / check_local_alignment.py   diagnostics
│   ├── paper_figures/      paper figure scripts — gitignored,
│   │                       regenerated from results/ on demand, never committed
│   ├── native_viewer/      standalone live 3D viewer (see below)
│   ├── run_native_viewer.sh / run_live_rviz.sh
│   └── rviz/ , launch/     ROS launch/rviz assets
├── sim/toy_consistency.py  Phase -1 sanity simulation — falsification only, never evidence
│                           about real data (see Part B of the guideline)
└── ws/src/TRaIL-Odom       upstream submodule, pinned at 667d7b4132461db2c79e20521dfe5f51b35ffae8,
                            branch research/consistency
```

## Reproduce the toy simulation
```
python3 sim/toy_consistency.py nominal
python3 sim/toy_consistency.py "radar noise x2"
python3 sim/toy_consistency.py "radar 10% outliers"
```

## Build

```
scripts/docker_build.sh
```

## Run a research sequence (batch, no GUI)
```
scripts/run_seq.sh <variant> <sequence> <rep>   # e.g. scripts/run_seq.sh V4 Park1 1
python3 scripts/eval_traj.py                    # ATE/RTE against GT, writes results/summaries/eval_traj.csv
```
Sequences: `Airfield1 Airfield2 BikeTunnel1 BikeTunnel2 Park1 Park2` (Boxi/TRaIL-Odom) plus
the GaRLILEO sequences (`Downstair CorriLoop BiCorridor SlopeStair Tunnel Overpass Quad`).
Writes `results/<variant>/<sequence>/rep<rep>/` including `run.json` (commit hashes, config
hash, flags).

## Live visualization — Ra-LIO-DD (native 3D viewer)

A standalone native viewer (Open3D + rclpy; not rviz, not upstream's `trail_rviz.rviz`)
for watching a sequence run live: point cloud colored by age (green=fresh →
red=40s+/drift-affected), estimated path vs. GT overlay, live error and "ATE so far"
(same prism-point, full-data rigid alignment as `eval_traj.py --gt-point prism`), raw IMU
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
scripts/run_native_viewer.sh [sequence] [start_time] [method]   # e.g. ... BikeTunnel1 20 PROXOBSEG
```
Defaults to `Park1`, `start_time=20`, method `V4` (unmodified upstream), auto-starts that
sequence. Pick a different variant from the panel's dropdown and click "Run selected
sequence" any time, including while one is already running (it stops the old one first).
`MAX_RANGE` env var controls how far back (in meters) the point cloud/path persist before
being pruned (default 150).

**Alternative**: `scripts/run_live_rviz.sh [sequence] [start_time]` runs upstream's own
`rviz2` with a recolored copy of their `trail_rviz.rviz` instead — needs only OpenGL
(`--gpus all`, not the CDI setup above), no dropdown/GT overlay/drift/IMU panel.

This project is independent of `~/Desktop/Trail/ObsTRaIL` (see `docs/DECISIONS.md`).
