#!/usr/bin/env python3
"""Phase 3 (Part E.5): GT velocity from prism positions, via a smoothing spline.

"Fit a smoothing spline to the prism positions, differentiate, and add the GT velocity
variance to Sigma_v" (Part E.5). This module:
  1. Fits a cubic smoothing spline per axis (scipy UnivariateSpline), with a smoothing
     factor derived from an assumed total-station position noise sigma_pos. [HYP] No
     independent measurement of this total station's actual noise was available this
     session -- sigma_pos=0.003m (3mm) is a literature-typical value for high-end total
     stations, not verified against this specific instrument. Flagged in
     docs/OPEN_QUESTIONS.md.
  2. Differentiates the spline analytically for velocity.
  3. Estimates the GT velocity's own covariance via a parametric bootstrap: resample the
     GT positions with synthetic N(0, sigma_pos^2) noise, refit + redifferentiate, and
     take the empirical covariance of the resulting velocity at each query time. This
     avoids deriving the spline basis's error-propagation Jacobian by hand and is exact
     in the large-bootstrap limit.
  4. Skips (returns invalid) any query time falling in a GT dropout gap wider than
     `max_gap` seconds -- the total station loses lock on turns/occlusions (Park1 has a
     ~2s gap), and spline-interpolating across a multi-second gap is not GT, it's a prior.

GT is position-only (identity quaternion): this is prism velocity in the world frame, not
corrected for the prism-to-IMU lever arm (see scripts/gt_extract.py's docstring for why).
"""
import numpy as np
from scipy.interpolate import UnivariateSpline


def fit_gt_velocity(gt_tum_path, sigma_pos=0.003, n_bootstrap=30, max_gap=0.5, seed=0):
    """Returns a dict with:
      t: (N,) query times (the GT's own sample times, excluding large-gap points)
      v: (N,3) GT velocity at each t
      Sigma_v: (N,3,3) bootstrap covariance of v at each t
      valid: (N,) bool, True where no gap > max_gap straddles that point
    """
    data = np.loadtxt(gt_tum_path)
    t_raw, pos_raw = data[:, 0], data[:, 1:4]
    t0 = t_raw[0]
    t = t_raw - t0

    gaps = np.diff(t)
    gap_before = np.concatenate([[0.0], gaps])
    gap_after = np.concatenate([gaps, [0.0]])
    valid = (gap_before <= max_gap) & (gap_after <= max_gap)

    m = len(t)
    s = m * sigma_pos ** 2  # scipy's convention: s ~ sum of squared residuals under IID noise

    def fit_and_differentiate(pos):
        splines = [UnivariateSpline(t, pos[:, d], s=s, k=3) for d in range(3)]
        return np.stack([sp.derivative(1)(t) for sp in splines], axis=1)

    v = fit_and_differentiate(pos_raw)

    rng = np.random.default_rng(seed)
    v_boot = np.zeros((n_bootstrap, m, 3))
    for b in range(n_bootstrap):
        noisy = pos_raw + rng.normal(scale=sigma_pos, size=pos_raw.shape)
        v_boot[b] = fit_and_differentiate(noisy)

    v_centered = v_boot - v_boot.mean(axis=0, keepdims=True)
    Sigma_v = np.einsum('bni,bnj->nij', v_centered, v_centered) / (n_bootstrap - 1)

    return dict(t=t + t0, v=v, Sigma_v=Sigma_v, valid=valid)


if __name__ == '__main__':
    import sys
    result = fit_gt_velocity(sys.argv[1])
    n_valid = result['valid'].sum()
    print(f"{len(result['t'])} GT points, {n_valid} valid (not adjacent to a gap > 0.5s)")
    print(f"velocity std (sqrt diag Sigma_v), median over valid points: "
          f"{np.sqrt(np.diagonal(result['Sigma_v'][result['valid']], axis1=1, axis2=2)).mean(axis=0)}")
