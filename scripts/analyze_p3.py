#!/usr/bin/env python3
"""P3 (injected-bias pass-through): compare the measured fraction of an injected radar
forward-velocity bias that reaches the estimate, against Eq. (omega = kappa*s/(1-s+kappa*s)),
on the held-out GaRLILEO sequences.

For each kappa in {1,3,10,30,100}, two runs share the same sensor data and differ only in
AblationConfig::RadarFwdBiasEps (0 for P3BASE<k>, 0.05 for P3BIAS<k>). Their trajectories are
differenced in the radar frame (same central-difference velocity estimator as radar_diag.py)
to get, per degenerate window, delta_v_x(t) = v_bias_x(t) - v_base_x(t). Since the injected
bias is a known fractional error on v_x, delta_v_x(t) / (eps * v_base_x(t)) is a window-level
estimate of the pass-through fraction omega_hat(t) (v_base_x used as the best available proxy
for the true v_x, since the bias affects only the BIAS run).

s per window (the radar's nominal information share at kappa=1, before any gain) is not
logged directly, so it is calibrated empirically from the kappa=1 run: at kappa=1 the formula
gives omega=s exactly, so s_hat(window) := omega_hat measured at kappa=1. The OTHER kappas'
measured omega_hat is then compared with the formula's prediction at that same s_hat -- this
is a self-consistent check (kappa=1 calibrates s, kappa in {3,10,30,100} tests the formula).

Usage: python3 scripts/analyze_p3.py [--kappas 1 3 10 30 100] [--reps 1 2 3]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import eval_traj as E  # noqa: E402
import radar_diag as D  # noqa: E402

GARLILEO = D.GARLILEO
EPS = 0.05
KAPPA_THRESH = 5.0


def run_dir(variant, seq, rep):
    return REPO_ROOT / 'results' / 'explore_garlileo' / variant / seq / f'rep{rep}'


def degenerate_mask(scan_log_path, t):
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
    if len(flags) == 0:
        return np.zeros_like(t, dtype=bool)
    idx = np.clip(np.searchsorted(ends, t), 0, len(flags) - 1)
    return flags[idx]


def radar_fwd_vel(rdir, R_RtoB, p_RinB, ts):
    t, p, q = D.load_pose_file(rdir / 'trail_output' / 'poses.txt')
    return D.Traj(t, p, q, R_RtoB, p_RinB).radar_vel(ts)[:, 0], t


def kappa_str(kappa):
    """Match the shell variant-name convention (P3BASE10, not P3BASE10.0)."""
    return str(int(kappa)) if float(kappa) == int(kappa) else str(kappa)


def pass_through(dir_a, dir_b, R_RtoB, p_RinB):
    """Per-sample fraction of the radar-only forward bias that reaches the estimate.
    The injected model makes the radar-only forward velocity v_x/(1+eps), a bias
    b_x = -eps/(1+eps) v_x, so omega = (v_b_x - v_a_x)/b_x = -(1+eps)(v_b_x - v_a_x)/(eps v_a_x).
    Degenerate windows of run a only, |v_a_x| > 0.3 m/s. Also used for the null control
    (two unbiased runs), where the same statistic should be ~0."""
    if not (dir_a / 'trail_output' / 'poses.txt').exists() or not (dir_b / 'trail_output' / 'poses.txt').exists():
        return None
    ta, _, _ = D.load_pose_file(dir_a / 'trail_output' / 'poses.txt')
    ts = ta[(ta > ta[0] + 2) & (ta < ta[-1] - 2)][::5]  # subsample, skip start/end transients
    v_a, _ = radar_fwd_vel(dir_a, R_RtoB, p_RinB, ts)
    v_b, _ = radar_fwd_vel(dir_b, R_RtoB, p_RinB, ts)
    deg = degenerate_mask(dir_a / 'scan_log.csv', ts)
    ok = deg & np.isfinite(v_a) & np.isfinite(v_b) & (np.abs(v_a) > 0.3)
    return -(1 + EPS) * (v_b[ok] - v_a[ok]) / (EPS * v_a[ok])


def omega_model(kappa, s):
    return kappa * s / (1 - s + kappa * s)


def fit_s(kappas, omegas):
    """One-parameter least-squares fit of s in (0,1) on a fine grid."""
    grid = np.linspace(1e-5, 0.999, 20000)
    k = np.asarray(kappas)[:, None]
    sse = ((omega_model(k, grid[None, :]) - np.asarray(omegas)[:, None]) ** 2).sum(0)
    return float(grid[np.argmin(sse)])


def boot_median_ci(x, rng, n=1000):
    m = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', type=float, nargs='+', default=[1, 3, 10, 30, 100])
    ap.add_argument('--reps', type=int, nargs='+', default=[1])
    ap.add_argument('--sequences', nargs='+', default=GARLILEO)
    ap.add_argument('--null-kappas', type=float, nargs='*', default=[1, 100])
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    pooled, per_seq = {}, {}
    for kappa in args.kappas:
        allv = []
        for seq in args.sequences:
            R_RtoB, p_RinB, _ = D.load_extrinsics(seq)
            vals = []
            for rep in args.reps:
                o = pass_through(run_dir(f'P3BASE{kappa_str(kappa)}', seq, rep),
                                 run_dir(f'P3BIAS{kappa_str(kappa)}', seq, rep), R_RtoB, p_RinB)
                if o is not None and len(o):
                    vals.append(o)
            if vals:
                v = np.concatenate(vals); allv.append(v)
                per_seq[(seq, kappa)] = (float(np.median(v)), len(v))
        if allv:
            pooled[kappa] = np.concatenate(allv)

    print('per sequence: median omega_hat (n samples)')
    for seq in args.sequences:
        print(f'  {seq:12s} ' + '  '.join(
            f'k={kappa_str(k)}: {per_seq[(seq, k)][0]:+.3f} ({per_seq[(seq, k)][1]})' if (seq, k) in per_seq
            else f'k={kappa_str(k)}: --' for k in args.kappas))

    print('\npooled over sequences')
    med = {}
    for k, v in pooled.items():
        med[k] = float(np.median(v)); lo, hi = boot_median_ci(v, rng)
        print(f'  kappa={kappa_str(k):>5s}  median omega_hat {med[k]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(v)}')

    null = []
    for k in args.null_kappas:
        allv = []
        for seq in args.sequences:
            R_RtoB, p_RinB, _ = D.load_extrinsics(seq)
            o = pass_through(run_dir(f'P3BASE{kappa_str(k)}', seq, 1),
                             run_dir(f'P3BASE{kappa_str(k)}', seq, 2), R_RtoB, p_RinB)
            if o is not None and len(o):
                allv.append(o)
        if allv:
            v = np.concatenate(allv); null.append(abs(float(np.median(v))))
            lo, hi = boot_median_ci(v, rng)
            print(f'  NULL (BASE rep1 vs rep2) kappa={kappa_str(k):>5s}  median {np.median(v):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(v)}')

    ks = [k for k in args.kappas if k in med]
    if len(ks) >= 2:
        s = fit_s(ks, [med[k] for k in ks])
        res = [med[k] - omega_model(k, s) for k in ks]
        print(f'\none-parameter fit: s = {s:.4f}')
        for k, r in zip(ks, res):
            print(f'  kappa={kappa_str(k):>5s}  measured {med[k]:+.3f}  model {omega_model(k, s):+.3f}  residual {r:+.3f}')
        mono = all(med[ks[i + 1]] > med[ks[i]] for i in range(len(ks) - 1))
        print('\npre-registered criterion (docs/DECISIONS.md 2026-11-05):')
        print(f'  (a) monotone increase over kappa: {mono}')
        if null and 100 in med:
            print(f'  (b) omega(100) = {med[100]:+.3f} > 3 x max|null| = {3 * max(null):.3f}: {med[100] > 3 * max(null)}')
        else:
            print('  (b) not evaluable (null control or kappa=100 missing)')
        print(f'  (c) max |residual| = {max(abs(r) for r in res):.3f} <= 0.1: {max(abs(r) for r in res) <= 0.1}')


if __name__ == '__main__':
    main()
