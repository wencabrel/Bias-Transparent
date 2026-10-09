#!/usr/bin/env python3
"""Phase 3 (Part E.5/H): directional NEES for velocity estimates with known covariance.

    eps_i = (u_i^T e_v)^2 / (u_i^T Sigma_v u_i)  ~  chi^2_1   (per axis u_i, per window)

ANEES is the mean of eps_i over windows; under a correctly-calibrated (consistent)
estimator, ANEES should sit near 1.0 (chi^2_1 mean), with a 95% two-sided bound on the
mean of N draws at chi^2_1(0.025)/N and chi^2_1(0.975)/N... but since we report ANEES over
N windows, the natural check is: ANEES's N*ANEES ~ chi^2_N, so the 95% bound on ANEES
itself is [chi2.ppf(0.025,N)/N, chi2.ppf(0.975,N)/N].

Does NOT report absolute-position NEES (Part E.5: global drift is not represented in a
sliding-window posterior) -- only directional velocity NEES.

Usage as a library: ane es, bounds = anees(errors, variances)
Usage as a script: runs the Part H Phase 3 synthetic validation (correct covariance,
x0.1, x10) and prints PASS/FAIL for each case.
"""
import numpy as np
from scipy import stats


def directional_nees(e_v, Sigma_v, axis):
    """e_v: (N,3) velocity errors. Sigma_v: (N,3,3) covariances. axis: (3,) unit vector.
    Returns eps: (N,) array of per-window chi^2_1 statistics along `axis`."""
    axis = axis / np.linalg.norm(axis)
    num = (e_v @ axis) ** 2
    den = np.einsum('i,nij,j->n', axis, Sigma_v, axis)
    return num / den


def anees_with_bounds(eps, confidence=0.95):
    """eps: (N,) chi^2_1 statistics. Returns (anees, (lower, upper)) using the exact
    N*ANEES ~ chi^2_N distribution under the null hypothesis of correct calibration."""
    n = len(eps)
    anees = float(np.mean(eps))
    alpha = 1 - confidence
    lower = stats.chi2.ppf(alpha / 2, n) / n
    upper = stats.chi2.ppf(1 - alpha / 2, n) / n
    return anees, (lower, upper)


def _synthetic_case(true_scale, n=2000, seed=0):
    """Generate N synthetic velocity errors ~ N(0, true_scale^2 * Sigma_true), report
    directional NEES using the (possibly mis-scaled) *assumed* Sigma_v = Sigma_true."""
    rng = np.random.default_rng(seed)
    Sigma_true = np.diag([0.02, 0.01, 0.03]) ** 2
    L = np.linalg.cholesky(Sigma_true)
    e_v = (L @ rng.normal(size=(3, n))).T * true_scale
    Sigma_v = np.tile(Sigma_true, (n, 1, 1))
    axis = np.array([1.0, 0.0, 0.0])
    eps = directional_nees(e_v, Sigma_v, axis)
    return anees_with_bounds(eps)


def main():
    print("Part H Phase 3 synthetic validation: nees.py on known covariance\n")
    cases = [
        ("correct covariance (scale=1.0)", 1.0, True),
        ("overconfident (true noise x0.1 of assumed -> ANEES << 1)", 0.1, False),
        ("underconfident (true noise x10 of assumed -> ANEES >> 1)", 10.0, False),
    ]
    all_pass = True
    for name, scale, expect_in_bounds in cases:
        anees, (lo, hi) = _synthetic_case(scale)
        in_bounds = lo <= anees <= hi
        ok = (in_bounds == expect_in_bounds)
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: ANEES={anees:.4f}, 95% bounds=[{lo:.4f}, {hi:.4f}], "
              f"in_bounds={in_bounds} (expected {expect_in_bounds})")
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
