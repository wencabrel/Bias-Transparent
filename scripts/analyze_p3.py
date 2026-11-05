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


def omega_hat_for_kappa(seq, kappa, rep, R_RtoB, p_RinB):
    base_d = run_dir(f'P3BASE{kappa_str(kappa)}', seq, rep)
    bias_d = run_dir(f'P3BIAS{kappa_str(kappa)}', seq, rep)
    if not (base_d / 'trail_output' / 'poses.txt').exists() or not (bias_d / 'trail_output' / 'poses.txt').exists():
        return None
    tb, pb, qb = D.load_pose_file(base_d / 'trail_output' / 'poses.txt')
    ts = tb[(tb > tb[0] + 2) & (tb < tb[-1] - 2)][::5]  # subsample, skip start/end transients
    v_base, _ = radar_fwd_vel(base_d, R_RtoB, p_RinB, ts)
    v_bias, _ = radar_fwd_vel(bias_d, R_RtoB, p_RinB, ts)
    deg = degenerate_mask(base_d / 'scan_log.csv', ts)
    ok = deg & np.isfinite(v_base) & np.isfinite(v_bias) & (np.abs(v_base) > 0.3)
    omega = (v_bias[ok] - v_base[ok]) / (EPS * v_base[ok])
    return omega


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', type=float, nargs='+', default=[1, 3, 10, 30, 100])
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--sequences', nargs='+', default=GARLILEO)
    args = ap.parse_args()

    results = {}
    for seq in args.sequences:
        R_RtoB, p_RinB, _ = D.load_extrinsics(seq)
        for kappa in args.kappas:
            vals = []
            for rep in args.reps:
                o = omega_hat_for_kappa(seq, kappa, rep, R_RtoB, p_RinB)
                if o is not None and len(o):
                    vals.append(o)
            if vals:
                pooled = np.concatenate(vals)
                results[(seq, kappa)] = (float(np.median(pooled)), len(pooled))
                print(f'{seq:12s} kappa={kappa:6g}  omega_hat median={np.median(pooled):+.3f}  '
                      f'(IQR {np.percentile(pooled,25):+.3f}..{np.percentile(pooled,75):+.3f}, n={len(pooled)})')
            else:
                print(f'{seq:12s} kappa={kappa:6g}  NO DATA')

    print('\n--- formula check (s calibrated from kappa=1) ---')
    for seq in args.sequences:
        if (seq, 1) not in results:
            continue
        s_hat = results[(seq, 1)][0]
        print(f'{seq:12s} s_hat (from kappa=1) = {s_hat:+.3f}')
        for kappa in args.kappas:
            if kappa == 1 or (seq, kappa) not in results:
                continue
            pred = kappa * s_hat / (1 - s_hat + kappa * s_hat)
            meas = results[(seq, kappa)][0]
            print(f'    kappa={kappa:6g}  predicted omega={pred:+.3f}  measured={meas:+.3f}  diff={meas-pred:+.3f}')


if __name__ == '__main__':
    main()
