#!/usr/bin/env python3
"""Diagnostic (Phase 3 follow-up): does a LOCAL (sliding-window) SE(3) alignment, rather
than one global rotation, substantially reduce the velocity NEES inflation seen in
docs/reports/phase3.md?

For each estimator window, computes a fresh Umeyama rotation using only poses within
+/- --local-window seconds of that time (instead of one rotation fit over the whole
trajectory), then applies it to that window's velocity/covariance/weak-axis before NEES.
If this meaningfully drops NEES, local orientation drift (not GT noise or clock sync,
both separately checked) is the dominant confound.

Usage: python3 scripts/check_local_alignment.py <cov_log.csv> <gt_tum> <poses.txt>
                                                  [--local-window 5.0] [--sigma-pos 0.003]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from evo.core import geometry

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from gt_velocity import fit_gt_velocity
from nees import anees_with_bounds


def local_alignment_rotation(est_t, est_pos, gt_t, gt_pos, t_center, window, max_dt=0.1):
    mask_est = np.abs(est_t - t_center) <= window
    if mask_est.sum() < 10:
        return None
    sub_est_t = est_t[mask_est]
    sub_est_pos = est_pos[mask_est]

    idx = np.searchsorted(gt_t, sub_est_t)
    matched_est, matched_gt = [], []
    for i, t, p in zip(idx, sub_est_t, sub_est_pos):
        candidates = [j for j in (i - 1, i) if 0 <= j < len(gt_t)]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(gt_t[j] - t))
        if abs(gt_t[best] - t) <= max_dt:
            matched_est.append(p)
            matched_gt.append(gt_pos[best])
    if len(matched_est) < 10:
        return None
    x = np.array(matched_est).T
    y = np.array(matched_gt).T
    try:
        r, _, _ = geometry.umeyama_alignment(x, y, with_scale=False)
    except Exception:
        return None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cov_log')
    ap.add_argument('gt_tum')
    ap.add_argument('poses_txt')
    ap.add_argument('--local-window', type=float, default=5.0)
    ap.add_argument('--sigma-pos', type=float, default=0.003)
    ap.add_argument('--max-dt', type=float, default=0.1)
    args = ap.parse_args()

    est = np.loadtxt(args.poses_txt)
    est_t, est_pos = est[:, 0], est[:, 1:4]
    gt_raw = np.loadtxt(args.gt_tum)
    gt_t, gt_pos = gt_raw[:, 0], gt_raw[:, 1:4]

    df = pd.read_csv(args.cov_log)
    df = df[df['ok'] == 1].reset_index(drop=True)

    gt = fit_gt_velocity(args.gt_tum, sigma_pos=args.sigma_pos)
    gt_t_v, gt_v, gt_Sigma, gt_valid = gt['t'], gt['v'], gt['Sigma_v'], gt['valid']
    gt_t_valid, gt_v_valid, gt_Sigma_valid = gt_t_v[gt_valid], gt_v[gt_valid], gt_Sigma[gt_valid]

    overall_eps, dir_eps = [], []
    n_matched = 0
    n_no_local_align = 0
    for _, row in df.iterrows():
        t_est = row['t']
        idx = np.searchsorted(gt_t_valid, t_est)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(gt_t_valid)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(gt_t_valid[i] - t_est))
        if abs(gt_t_valid[best] - t_est) > args.max_dt:
            continue

        r_local = local_alignment_rotation(est_t, est_pos, gt_t, gt_pos, t_est,
                                            args.local_window, args.max_dt)
        if r_local is None:
            n_no_local_align += 1
            continue
        n_matched += 1

        v_est_raw = np.array([row['vx'], row['vy'], row['vz']])
        Sigma_est_raw = np.array([[row['sxx'], row['sxy'], row['sxz']],
                                   [row['sxy'], row['syy'], row['syz']],
                                   [row['sxz'], row['syz'], row['szz']]])
        v_gt = gt_v_valid[best]
        Sigma_gt = gt_Sigma_valid[best]

        v_est = r_local @ v_est_raw
        Sigma_est = r_local @ Sigma_est_raw @ r_local.T
        e_v = v_est - v_gt
        Sigma_total = Sigma_est + Sigma_gt

        try:
            overall_eps.append(e_v @ np.linalg.solve(Sigma_total, e_v))
        except np.linalg.LinAlgError:
            pass

        if row.get('has_weak_axis', 0) == 1:
            u_raw = np.array([row['ux'], row['uy'], row['uz']])
            u = r_local @ u_raw
            u = u / np.linalg.norm(u)
            den = u @ Sigma_total @ u
            if den > 0:
                dir_eps.append((e_v @ u) ** 2 / den)

    print(f"matched + local-aligned: {n_matched} ({n_no_local_align} had no local alignment)")
    if overall_eps:
        overall_eps = np.array(overall_eps)
        print(f"Overall NEES (chi^2_3, expected 3.0): mean={overall_eps.mean():.3f}, n={len(overall_eps)}")
    if dir_eps:
        dir_eps = np.array(dir_eps)
        anees, (lo, hi) = anees_with_bounds(dir_eps)
        print(f"Directional NEES (chi^2_1, expected 1.0): ANEES={anees:.3f}, "
              f"n={len(dir_eps)}, 95% bounds=[{lo:.3f},{hi:.3f}]")


if __name__ == '__main__':
    main()
