#!/usr/bin/env python3
"""Phase 3 (Part E.5/H2): combine a run's cov_log.csv with GT velocity, compute NEES.

For each estimator window (from cov_log.csv), finds the nearest valid GT velocity sample
(within --max-dt), forms e_v = v_est - v_gt, Sigma_v_total = Sigma_v_est + Sigma_v_gt, and
computes:
  - overall NEES: e_v^T Sigma_v_total^-1 e_v  ~  chi^2_3
  - directional NEES along the window's own world-frame weak axis (when one existed that
    window) -- the real test of H2, since isotropic NEES can look fine while a specific
    direction is badly overconfident.

Does not report absolute-position NEES (Part E.5).

IMPORTANT: the estimator's world frame (gravity-aligned at init, arbitrary yaw) and the
total station's survey frame are different, unaligned coordinate systems -- exactly why
evo_ape performs an SE(3) Umeyama alignment before comparing positions. Velocity vectors
need the same rotation applied before any comparison, or e_v is meaningless (confirmed
empirically: without it, ANEES came out ~2000 instead of ~1). We compute that rotation
from the full trajectory (poses.txt vs the GT tum) via evo's own umeyama_alignment and
apply it to every v_est before anything else.

Usage: python3 scripts/compute_nees.py <cov_log.csv> <gt_tum> <poses.txt>
                                        [--max-dt 0.1] [--sigma-pos 0.003]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
from evo.core import geometry

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from gt_velocity import fit_gt_velocity
from nees import anees_with_bounds


def compute_alignment_rotation(poses_txt, gt_tum, max_dt=0.1):
    """Returns the 3x3 rotation that maps estimator-world vectors into the GT frame,
    via Umeyama alignment of the two position trajectories at matched timestamps."""
    est = np.loadtxt(poses_txt)
    gt = np.loadtxt(gt_tum)
    est_t, est_pos = est[:, 0], est[:, 1:4]
    gt_t, gt_pos = gt[:, 0], gt[:, 1:4]

    idx = np.searchsorted(gt_t, est_t)
    matched_est, matched_gt = [], []
    for i, t in zip(idx, est_t):
        candidates = [j for j in (i - 1, i) if 0 <= j < len(gt_t)]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(gt_t[j] - t))
        if abs(gt_t[best] - t) <= max_dt:
            matched_est.append(est_pos[np.searchsorted(est_t, t)])
            matched_gt.append(gt_pos[best])
    matched_est = np.array(matched_est).T  # 3xN
    matched_gt = np.array(matched_gt).T
    r, _, _ = geometry.umeyama_alignment(matched_est, matched_gt, with_scale=False)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cov_log')
    ap.add_argument('gt_tum')
    ap.add_argument('poses_txt')
    ap.add_argument('--max-dt', type=float, default=0.1)
    ap.add_argument('--sigma-pos', type=float, default=0.003)
    args = ap.parse_args()

    r_align = compute_alignment_rotation(args.poses_txt, args.gt_tum, args.max_dt)

    df = pd.read_csv(args.cov_log)
    df = df[df['ok'] == 1].reset_index(drop=True)
    n_total = len(pd.read_csv(args.cov_log))
    n_fail = n_total - len(df)

    gt = fit_gt_velocity(args.gt_tum, sigma_pos=args.sigma_pos)
    gt_t, gt_v, gt_Sigma, gt_valid = gt['t'], gt['v'], gt['Sigma_v'], gt['valid']
    gt_t_valid, gt_v_valid, gt_Sigma_valid = gt_t[gt_valid], gt_v[gt_valid], gt_Sigma[gt_valid]

    overall_eps = []
    dir_eps = []
    n_matched = 0
    for _, row in df.iterrows():
        t_est = row['t']
        idx = np.searchsorted(gt_t_valid, t_est)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(gt_t_valid)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(gt_t_valid[i] - t_est))
        if abs(gt_t_valid[best] - t_est) > args.max_dt:
            continue
        n_matched += 1

        v_est_raw = np.array([row['vx'], row['vy'], row['vz']])
        Sigma_est_raw = np.array([[row['sxx'], row['sxy'], row['sxz']],
                                   [row['sxy'], row['syy'], row['syz']],
                                   [row['sxz'], row['syz'], row['szz']]])
        v_gt = gt_v_valid[best]
        Sigma_gt = gt_Sigma_valid[best]

        # Rotate the estimator's (world-frame-of-its-own) velocity and covariance into
        # the GT frame before comparing -- see module docstring.
        v_est = r_align @ v_est_raw
        Sigma_est = r_align @ Sigma_est_raw @ r_align.T

        e_v = v_est - v_gt
        Sigma_total = Sigma_est + Sigma_gt

        try:
            eps = e_v @ np.linalg.solve(Sigma_total, e_v)
            overall_eps.append(eps)
        except np.linalg.LinAlgError:
            pass

        if row.get('has_weak_axis', 0) == 1:
            u_raw = np.array([row['ux'], row['uy'], row['uz']])
            u = r_align @ u_raw
            u = u / np.linalg.norm(u)
            num = (e_v @ u) ** 2
            den = u @ Sigma_total @ u
            if den > 0:
                dir_eps.append(num / den)

    print(f"cov_log: {n_total} windows ({n_fail} ceres::Covariance failures, "
          f"{100*n_fail/n_total:.1f}%)")
    print(f"matched to GT within {args.max_dt}s: {n_matched}/{len(df)}")

    if overall_eps:
        overall_eps = np.array(overall_eps)
        anees, (lo, hi) = anees_with_bounds(overall_eps)
        # overall NEES is chi^2_3 (not chi^2_1); bounds must use 3*N dof.
        n = len(overall_eps)
        lo3 = stats.chi2.ppf(0.025, 3 * n) / n
        hi3 = stats.chi2.ppf(0.975, 3 * n) / n
        mean_chi2_3 = 3.0
        print(f"\nOverall NEES (chi^2_3, expected mean={mean_chi2_3}): "
              f"mean={overall_eps.mean():.3f}, ANEES/3={overall_eps.mean()/3:.3f}, "
              f"n={n}, 95% bounds on mean=[{lo3:.3f}, {hi3:.3f}], "
              f"{'IN BOUNDS' if lo3 <= overall_eps.mean() <= hi3 else 'OUT OF BOUNDS'}")

    if dir_eps:
        dir_eps = np.array(dir_eps)
        anees, (lo, hi) = anees_with_bounds(dir_eps)
        print(f"\nDirectional NEES along weak axis (chi^2_1, expected mean=1.0): "
              f"ANEES={anees:.3f}, n={len(dir_eps)}, 95% bounds=[{lo:.3f}, {hi:.3f}], "
              f"{'IN BOUNDS' if lo <= anees <= hi else 'OUT OF BOUNDS'}")
    else:
        print("\nNo windows had a weak axis -- no directional NEES computed.")


if __name__ == '__main__':
    main()
