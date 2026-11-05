#!/usr/bin/env python3
"""Fig: accumulated LiDAR map + GT/estimate trajectories for all 13 sequences.

Grid: row 1 = 6 TRaIL-Odom sequences, rows 2-3 = 7 GaRLILEO sequences (4+3).
Each panel: top view (x-y) of the run's local_map.pcd point cloud (colour = height,
i.e. z in the estimator's own frame), GT trajectory (black) and the estimate's
trajectory (teal) after a rigid (no-scale) Umeyama alignment of the estimate onto
GT -- the same alignment evo's `-a` flag performs for the project's own ATE numbers
(scripts/eval_traj.py:105-117). Windows with kappa > 5 and degenerate_dim > 0
(docs/FACTS.md F2, scan_log.csv columns) are drawn in red along the estimated path.
Start is a GT-frame marker; each panel gets its own scale bar (panels differ in scale
by >2x). Sequence length in the title is the GT path length (scripts/eval_traj.py
path_length(), the same quantity reported in results/summaries/gt_geometry.csv).

Usage: python3 scripts/plot_env_maps.py [--variant PROXOBSEGWF] [--rep 1]
                                         [--out figures/env_maps.png]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import eval_traj as E  # load_tum, path_length

TRAIL_SEQS = ['BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2', 'Airfield1', 'Airfield2']
GARLILEO_SEQS = ['Downstair', 'CorriLoop', 'BiCorridor', 'SlopeStair', 'Tunnel', 'Overpass', 'Quad']
KAPPA_THRESH = 5.0  # docs/FACTS.md F2: src/core/state_manager.cpp:1379 kappa_thresh


def run_dir(variant, seq, rep):
    if seq in TRAIL_SEQS:
        return REPO_ROOT / 'results' / variant / seq / f'rep{rep}'
    return REPO_ROOT / 'results' / 'explore_garlileo' / variant / seq / f'rep{rep}'


def gt_path(seq):
    if seq in TRAIL_SEQS:
        return REPO_ROOT / 'results' / 'summaries' / 'gt' / f'{seq}_gt.tum'
    return REPO_ROOT / 'data' / 'GaRLILEO' / 'GT_Trajectory' / f'{seq}_gt.txt'


def umeyama_no_scale(src, dst):
    """Rigid R, t minimising ||dst - (R @ src + t)||, no scale (as evo's `-a`)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = cd - r @ cs
    return r, t


def align_to_gt(gt_t, gt_p, est_t, est_p, max_dt):
    k = np.clip(np.searchsorted(est_t, gt_t), 1, len(est_t) - 1)
    k = np.where(np.abs(est_t[k - 1] - gt_t) < np.abs(est_t[k] - gt_t), k - 1, k)
    ok = np.abs(est_t[k] - gt_t) < max_dt
    r, t = umeyama_no_scale(est_p[k[ok]], gt_p[ok])
    return r, t


def degenerate_mask(scan_log_path, est_t):
    """True where est_t falls in a window with kappa > KAPPA_THRESH and degenerate_dim > 0."""
    windows = []
    with open(scan_log_path, newline='') as f:
        for row in csv.DictReader(f):
            windows.append((float(row['window_end_time']),
                             float(row['kappa']) > KAPPA_THRESH and int(row['degenerate_dim']) > 0))
    windows.sort(key=lambda w: w[0])
    ends = np.array([w[0] for w in windows])

    # scan_log.csv written before submodule commit df99cad stores window_end_time in the
    # data manager's relative clock (starts near 0), not the absolute clock of poses.txt/GT;
    # matching such a log against absolute timestamps silently maps every time to the last
    # window. Refuse it.
    if len(ends) and ends.max() < 1e8:
        raise RuntimeError(f'{scan_log_path}: window_end_time is in the pre-df99cad relative '
                           'clock; rerun with the fixed build')
    flags = np.array([w[1] for w in windows])
    idx = np.searchsorted(ends, est_t)
    idx = np.clip(idx, 0, len(flags) - 1)
    return flags[idx] if len(flags) else np.zeros_like(est_t, dtype=bool)


def load_map_points(pcd_path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    return np.asarray(pcd.points)


def nice_scale_bar_length(span):
    for L in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if L <= span / 4:
            best = L
    return best if 'best' in dir() else 1


def plot_panel(ax, seq, variant, rep):
    rdir = run_dir(variant, seq, rep)
    est_t, est_p = E.load_tum(rdir / 'trail_output' / 'poses.txt')
    gtp = gt_path(seq)
    gt_t, gt_p = E.load_tum(gtp)
    max_dt = 0.01 if seq in TRAIL_SEQS else 0.05
    r, t = align_to_gt(gt_t, gt_p, est_t, est_p, max_dt)

    est_p_aligned = est_p @ r.T + t
    pts = load_map_points(rdir / 'trail_output' / 'local_map.pcd')
    pts_aligned = pts @ r.T + t

    deg = degenerate_mask(rdir / 'scan_log.csv', est_t)

    order = np.argsort(pts_aligned[:, 2])
    ax.scatter(pts_aligned[order, 0], pts_aligned[order, 1], c=pts_aligned[order, 2],
               cmap='viridis', s=0.15, linewidths=0, alpha=0.6, rasterized=True)

    ax.plot(gt_p[:, 0], gt_p[:, 1], color='black', linewidth=0.9, zorder=4, label='GT')

    pts_xy = est_p_aligned[:, :2]
    segs = np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)
    seg_deg = deg[:-1] | deg[1:]
    lc = LineCollection(segs[~seg_deg], colors='teal', linewidths=1.1, zorder=5)
    ax.add_collection(lc)
    lc_deg = LineCollection(segs[seg_deg], colors='red', linewidths=1.3, zorder=6)
    ax.add_collection(lc_deg)

    ax.plot(gt_p[0, 0], gt_p[0, 1], marker='o', markersize=4, markerfacecolor='white',
            markeredgecolor='black', zorder=7)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    bar_len = nice_scale_bar_length(span)
    bar = AnchoredSizeBar(ax.transData, bar_len, f'{bar_len} m', 'lower right',
                           pad=0.3, color='black', frameon=False, size_vertical=span * 0.004,
                           fontproperties={'size': 6})
    ax.add_artist(bar)

    length = E.path_length(gt_p)
    ax.set_title(f'{seq} ({length:.0f} m)', fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='PROXOBSEGWF')
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--out', default=str(REPO_ROOT / 'figures' / 'env_maps.png'))
    args = ap.parse_args()

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 12, hspace=0.35, wspace=0.05)

    for i, seq in enumerate(TRAIL_SEQS):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        plot_panel(ax, seq, args.variant, args.rep)

    for i, seq in enumerate(GARLILEO_SEQS[:4]):
        ax = fig.add_subplot(gs[1, i * 3:(i + 1) * 3])
        plot_panel(ax, seq, args.variant, args.rep)

    for i, seq in enumerate(GARLILEO_SEQS[4:]):
        ax = fig.add_subplot(gs[2, i * 4:(i + 1) * 4])
        plot_panel(ax, seq, args.variant, args.rep)

    fig.suptitle('Environment maps and trajectories, full method (Ours-F)', fontsize=10, y=0.995)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
