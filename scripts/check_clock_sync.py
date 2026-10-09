#!/usr/bin/env python3
"""Diagnostic (Phase 3 follow-up): is there a constant clock offset between the
estimator's timestamps and the GT (total station) timestamps?

Sweeps a constant time shift applied to the GT timestamps, re-matches to the estimated
trajectory at each shift, and reports the position-residual RMSE (after a fresh SE(3)
Umeyama alignment at each shift) as a function of shift. If residual RMSE has a clear
minimum away from shift=0, that is evidence of a real clock offset. If the minimum is at
(or very close to) 0, clock sync is not the dominant confound in the Phase 3 NEES
inflation (docs/reports/phase3.md).

Usage: python3 scripts/check_clock_sync.py <poses.txt> <gt_tum> [--range 1.0] [--step 0.02]
"""
import argparse
import numpy as np
from evo.core import geometry


def residual_rmse_at_shift(est_t, est_pos, gt_t, gt_pos, shift):
    """Match est to (gt_t + shift), align via Umeyama, return RMSE of aligned residuals."""
    shifted_gt_t = gt_t + shift
    idx = np.searchsorted(shifted_gt_t, est_t)
    matched_est, matched_gt = [], []
    for i, t in zip(idx, est_t):
        candidates = [j for j in (i - 1, i) if 0 <= j < len(shifted_gt_t)]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(shifted_gt_t[j] - t))
        if abs(shifted_gt_t[best] - t) <= 0.1:
            matched_est.append(est_pos[np.searchsorted(est_t, t)])
            matched_gt.append(gt_pos[best])
    if len(matched_est) < 10:
        return np.nan, 0
    x = np.array(matched_est).T
    y = np.array(matched_gt).T
    r, t_vec, c = geometry.umeyama_alignment(x, y, with_scale=False)
    x_aligned = c * (r @ x) + t_vec[:, None]
    residuals = x_aligned - y
    rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=0)))
    return rmse, x.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('poses_txt')
    ap.add_argument('gt_tum')
    ap.add_argument('--range', type=float, default=1.0, help='sweep +/- this many seconds')
    ap.add_argument('--step', type=float, default=0.02)
    args = ap.parse_args()

    est = np.loadtxt(args.poses_txt)
    gt = np.loadtxt(args.gt_tum)
    est_t, est_pos = est[:, 0], est[:, 1:4]
    gt_t, gt_pos = gt[:, 0], gt[:, 1:4]

    shifts = np.arange(-args.range, args.range + args.step / 2, args.step)
    results = []
    for shift in shifts:
        rmse, n = residual_rmse_at_shift(est_t, est_pos, gt_t, gt_pos, shift)
        results.append((shift, rmse, n))

    results = [(s, r, n) for s, r, n in results if not np.isnan(r)]
    if not results:
        print("No shift produced enough matches.")
        return

    best = min(results, key=lambda x: x[1])
    zero = min(results, key=lambda x: abs(x[0]))
    print(f"{'shift':>8s} {'rmse':>10s} {'n':>6s}")
    for s, r, n in results:
        marker = ' <-- best' if (s, r, n) == best else (' <-- zero' if abs(s) < args.step / 2 else '')
        print(f"{s:8.3f} {r:10.5f} {n:6d}{marker}")

    print(f"\nBest shift: {best[0]:.3f}s (RMSE {best[1]:.5f}, n={best[2]})")
    print(f"Zero shift: {zero[0]:.3f}s (RMSE {zero[1]:.5f}, n={zero[2]})")
    improvement = (zero[1] - best[1]) / zero[1] * 100
    print(f"Improvement at best vs zero shift: {improvement:.1f}%")
    if abs(best[0]) < 2 * args.step:
        print("-> Minimum is at (or very near) zero shift: clock sync is NOT the dominant confound.")
    elif improvement < 5:
        print("-> Minimum away from zero but improvement is small: clock sync is likely NOT a major factor.")
    else:
        print(f"-> Minimum at nonzero shift with meaningful improvement: "
              f"a ~{best[0]:.2f}s clock offset is plausible and should be investigated further.")


if __name__ == '__main__':
    main()
