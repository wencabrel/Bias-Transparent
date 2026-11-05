#!/usr/bin/env python3
"""tab:p1 (P1, localization): drift rate in degenerate vs well-constrained windows.

Offline analysis of runs that already exist in results/ -- no new odometry runs needed.
For each 1 m GT segment (same definition as eval_traj.py's position_rpe: for GT point i,
j is the first point >= 1 m further along the GT path), the segment error is
|(p_est_j - p_est_i) - (p_gt_j - p_gt_i)| after a rigid Umeyama alignment (no scale, as
ATE), and the segment is labelled degenerate (D) iff its start time falls in a scan_log.csv
window with kappa > 5 and degenerate_dim > 0 (docs/FACTS.md F2 -- the same criterion used
for the red segments in figures/env_maps.png). Drift rate per class = mean(segment error /
segment length) * 100, i.e. mean % position error per metre travelled in that class.
"Share of distance in D" is the fraction of segments (each ~1 m by construction) labelled D,
pooled across all 6 sequences, using V4's own window classification (representative; the
classification is a property of the scene degeneracy, not of which variant is run).

Usage: python3 scripts/analyze_p1_drift.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import eval_traj as E

SEQS = ['BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2', 'Airfield1', 'Airfield2']
KAPPA_THRESH = 5.0


def run_dir(variant, seq, rep):
    return REPO_ROOT / 'results' / variant / seq / f'rep{rep}'


def degenerate_mask_at(scan_log_path, t, est_t0=None):
    """est_t0: the run's first poses.txt time, used only for logs written before submodule
    commit df99cad, whose window_end_time is in the data manager's relative clock. Those are
    shifted so their first window ends at est_t0 (the true offset is not recoverable; this is
    within ~1 s). On TRaIL-Odom only the first window of a run is ever non-degenerate, and
    tab:p1 is unchanged with the first 0/2/5 s excluded, so this approximation does not move
    the table."""
    windows = []
    with open(scan_log_path, newline='') as f:
        for row in csv.DictReader(f):
            windows.append((float(row['window_end_time']),
                             float(row['kappa']) > KAPPA_THRESH and int(row['degenerate_dim']) > 0))
    windows.sort(key=lambda w: w[0])
    ends = np.array([w[0] for w in windows])
    flags = np.array([w[1] for w in windows])
    if len(flags) == 0:
        return np.zeros_like(t, dtype=bool)
    if ends.max() < 1e8:
        if est_t0 is None:
            raise RuntimeError(f'{scan_log_path}: pre-df99cad relative clock and no est_t0 given')
        print(f'WARNING {scan_log_path}: pre-df99cad relative clock, aligned approximately (~1 s)')
        ends = ends + (est_t0 - ends[0])
    idx = np.clip(np.searchsorted(ends, t), 0, len(flags) - 1)
    return flags[idx]


def segment_errors(gt_tum, est_tum, delta_m=1.0, max_dt=0.01):
    """Like eval_traj.position_rpe, but returns per-segment error/length and start index."""
    gt_t, gt_p = E.load_tum(gt_tum)
    est_t, est_p = E.load_tum(est_tum)
    k = np.clip(np.searchsorted(est_t, gt_t), 1, len(est_t) - 1)
    k = np.where(np.abs(est_t[k - 1] - gt_t) < np.abs(est_t[k] - gt_t), k - 1, k)
    ok = np.abs(est_t[k] - gt_t) < max_dt
    g, e = gt_p[ok], est_p[k[ok]]
    gt_t_ok = gt_t[ok]
    cg, ce = g.mean(0), e.mean(0)
    u, _, vt = np.linalg.svd((e - ce).T @ (g - cg))
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    e = (e - ce) @ r.T + cg
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))])
    j = np.searchsorted(s, s + delta_m)
    i = np.nonzero(j < len(g))[0]
    j = j[i]
    err = np.linalg.norm((e[j] - e[i]) - (g[j] - g[i]), axis=1)
    seg_len = s[j] - s[i]
    return err, seg_len, gt_t_ok[i]


def analyze(variant):
    d_err, w_err = [], []
    d_frac_num, d_frac_den = 0, 0
    for seq in SEQS:
        gt_tum = REPO_ROOT / 'results' / 'summaries' / 'gt' / f'{seq}_gt.tum'
        for rep in (1, 2, 3):
            rdir = run_dir(variant, seq, rep)
            est = rdir / 'trail_output' / 'poses.txt'
            scan_log = rdir / 'scan_log.csv'
            if not est.exists() or not scan_log.exists():
                continue
            err, seg_len, t0 = segment_errors(gt_tum, est)
            deg = degenerate_mask_at(scan_log, t0, est_t0=float(np.loadtxt(est, max_rows=1)[0]))
            rate_pct = err / seg_len * 100.0
            d_err.extend(rate_pct[deg].tolist())
            w_err.extend(rate_pct[~deg].tolist())
            d_frac_num += int(deg.sum())
            d_frac_den += len(deg)
    return (float(np.mean(d_err)) if d_err else float('nan'),
            float(np.mean(w_err)) if w_err else float('nan'),
            d_frac_num / d_frac_den if d_frac_den else float('nan'))


if __name__ == '__main__':
    for variant, label in [('V4', 'Released'), ('PROXOBSEGW', 'Ours-W'), ('PROXOBSEGWF', 'Ours-F')]:
        d, w, share = analyze(variant)
        print(f'{label:12s} D={d:6.2f}%  W={w:6.2f}%  share_in_D={share:.3f}')
