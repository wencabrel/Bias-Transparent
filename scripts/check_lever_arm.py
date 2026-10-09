#!/usr/bin/env python3
"""Diagnostic (Phase 3 follow-up): does the uncorrected prism-to-body lever arm explain
the NEES inflation in docs/reports/phase3.md?

GT is PRISM velocity: v_gt = v_body + omega_body x r_lever (r_lever ~ (0.309,-0.038,-0.235)
in body frame, from tf_static box_base->prism, Phase 0). This was never corrected for
(no orientation GT existed in Phase 0 to do it properly) -- but during turns, this is a
genuine physical cross term, not a pipeline bug, and with a 0.3m lever arm could be
comparable to the velocity errors driving the NEES inflation.

Approximates omega_body(t) via finite differences of the ESTIMATOR's own quaternions in
poses.txt (not GT -- there is no orientation GT), purely to test the correction's
magnitude/effect, then recomputes directional NEES with v_gt corrected by -omega x r.

Usage: python3 scripts/check_lever_arm.py <cov_log.csv> <gt_tum> <poses.txt>
                                           [--lever 0.309 -0.038 -0.235] [--sigma-pos 0.003]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from evo.core import geometry

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from gt_velocity import fit_gt_velocity
from nees import anees_with_bounds


def body_angular_velocity(poses_txt):
    """Finite-difference angular velocity (body frame) from poses.txt quaternions."""
    data = np.loadtxt(poses_txt)
    t = data[:, 0]
    quat_xyzw = data[:, 4:8]
    rots = Rotation.from_quat(quat_xyzw)

    omega_body = np.zeros((len(t), 3))
    for i in range(1, len(t) - 1):
        dt = t[i + 1] - t[i - 1]
        if dt <= 0:
            continue
        # relative rotation body(i-1) -> body(i+1), expressed in body(i) frame
        r_rel = rots[i].inv() * rots[i + 1] * rots[i].inv() * rots[i - 1]
        # simpler + more standard: dq = R(i)^-1 R(i+1) over dt gives omega in body(i) frame
        r_rel2 = rots[i].inv() * rots[i + 1]
        omega_body[i] = r_rel2.as_rotvec() / (t[i + 1] - t[i])
    return t, omega_body, rots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cov_log')
    ap.add_argument('gt_tum')
    ap.add_argument('poses_txt')
    ap.add_argument('--lever', type=float, nargs=3, default=[0.309, -0.038, -0.235])
    ap.add_argument('--sigma-pos', type=float, default=0.003)
    ap.add_argument('--max-dt', type=float, default=0.1)
    args = ap.parse_args()

    r_lever = np.array(args.lever)

    r_align = None
    est = np.loadtxt(args.poses_txt)
    gt_raw = np.loadtxt(args.gt_tum)
    est_t, est_pos = est[:, 0], est[:, 1:4]
    gt_t, gt_pos = gt_raw[:, 0], gt_raw[:, 1:4]
    idx = np.searchsorted(gt_t, est_t)
    matched_est, matched_gt = [], []
    for i, t in zip(idx, est_t):
        candidates = [j for j in (i - 1, i) if 0 <= j < len(gt_t)]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(gt_t[j] - t))
        if abs(gt_t[best] - t) <= args.max_dt:
            matched_est.append(est_pos[np.searchsorted(est_t, t)])
            matched_gt.append(gt_pos[best])
    r_align, _, _ = geometry.umeyama_alignment(np.array(matched_est).T, np.array(matched_gt).T, with_scale=False)

    pose_t, omega_body, rots = body_angular_velocity(args.poses_txt)

    df = pd.read_csv(args.cov_log)
    df = df[df['ok'] == 1].reset_index(drop=True)

    gt = fit_gt_velocity(args.gt_tum, sigma_pos=args.sigma_pos)
    gt_t_v, gt_v, gt_Sigma, gt_valid = gt['t'], gt['v'], gt['Sigma_v'], gt['valid']
    gt_t_valid, gt_v_valid, gt_Sigma_valid = gt_t_v[gt_valid], gt_v[gt_valid], gt_Sigma[gt_valid]

    dir_eps_raw, dir_eps_corrected = [], []
    corrections = []
    for _, row in df.iterrows():
        t_est = row['t']
        idx = np.searchsorted(gt_t_valid, t_est)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(gt_t_valid)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(gt_t_valid[i] - t_est))
        if abs(gt_t_valid[best] - t_est) > args.max_dt:
            continue
        if row.get('has_weak_axis', 0) != 1:
            continue

        pidx = np.searchsorted(pose_t, t_est)
        if pidx <= 0 or pidx >= len(pose_t):
            continue
        omega = omega_body[pidx]
        R_BtoW = rots[pidx].as_matrix()
        correction_world = R_BtoW @ np.cross(omega, r_lever)  # v_prism - v_body, world frame
        corrections.append(np.linalg.norm(correction_world))

        v_est_raw = np.array([row['vx'], row['vy'], row['vz']])
        Sigma_est_raw = np.array([[row['sxx'], row['sxy'], row['sxz']],
                                   [row['sxy'], row['syy'], row['syz']],
                                   [row['sxz'], row['syz'], row['szz']]])
        v_gt = gt_v_valid[best]
        Sigma_gt = gt_Sigma_valid[best]

        v_est = r_align @ v_est_raw
        Sigma_est = r_align @ Sigma_est_raw @ r_align.T
        Sigma_total = Sigma_est + Sigma_gt

        u_raw = np.array([row['ux'], row['uy'], row['uz']])
        u = r_align @ u_raw
        u = u / np.linalg.norm(u)
        den = u @ Sigma_total @ u
        if den <= 0:
            continue

        e_v_raw = v_est - v_gt
        dir_eps_raw.append((e_v_raw @ u) ** 2 / den)

        # v_gt (prism) = v_body + correction  =>  v_body = v_gt - correction
        e_v_corrected = v_est - (v_gt - correction_world)
        dir_eps_corrected.append((e_v_corrected @ u) ** 2 / den)

    corrections = np.array(corrections)
    print(f"Lever-arm correction magnitude: mean={corrections.mean():.4f} m/s, "
          f"max={corrections.max():.4f} m/s, median={np.median(corrections):.4f} m/s")

    for label, eps in [('raw (no lever-arm correction)', dir_eps_raw),
                        ('lever-arm corrected', dir_eps_corrected)]:
        eps = np.array(eps)
        anees, (lo, hi) = anees_with_bounds(eps)
        print(f"{label}: ANEES={anees:.3f}, n={len(eps)}, bounds=[{lo:.3f},{hi:.3f}]")


if __name__ == '__main__':
    main()
