#!/usr/bin/env python3
"""Estimate the GT time latency of a sequence from a REFERENCE run (default: V4 rep1,
the unmodified upstream code), independently of the method being evaluated.

Method: cross-correlate the horizontal-speed profiles of the estimate and the GT
(both resampled to 50 Hz, speeds from +/-0.25 s central differences). Speed is
frame-free and insensitive to accumulated drift, unlike a position-RMSE sweep.
The latency is the lag d maximising corr(speed_est(t), speed_gt(t + d)), refined
by a parabola through the peak. Positive d: GT stamps are late by d seconds.

Usage: python3 scripts/gt_latency.py SEQ [SEQ ...] [--variant V4] [--rep 1]
Writes results/summaries/gt_latency.csv.
"""
import argparse, csv, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    a = np.loadtxt(path, usecols=(0, 1, 2, 3))
    return a[np.argsort(a[:, 0])]


def speed(t, p, ts, h=0.25):
    q1 = np.stack([np.interp(ts - h, t, p[:, i]) for i in range(2)], -1)
    q2 = np.stack([np.interp(ts + h, t, p[:, i]) for i in range(2)], -1)
    return np.linalg.norm(q2 - q1, axis=1) / (2 * h)


def latency(est, gt, max_lag=1.0, fs=50.0):
    t0 = max(est[0, 0], gt[0, 0]) + max_lag + 1
    t1 = min(est[-1, 0], gt[-1, 0]) - max_lag - 1
    ts = np.arange(t0, t1, 1 / fs)
    se = speed(est[:, 0], est[:, 1:4], ts)
    lags = np.arange(-max_lag, max_lag + 1e-9, 1 / fs)
    c = []
    for d in lags:
        sg = speed(gt[:, 0], gt[:, 1:4], ts + d)
        c.append(np.corrcoef(se, sg)[0, 1])
    c = np.array(c)
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    else:
        off = 0.0
    return lags[k] + off / fs, c[k], k in (0, len(c) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequences', nargs='+')
    ap.add_argument('--variant', default='V4')
    ap.add_argument('--rep', default='1')
    a = ap.parse_args()
    out = os.path.join(REPO, 'results/summaries/gt_latency.csv')
    rows = {}
    if os.path.exists(out):
        rows = {r['sequence']: r for r in csv.DictReader(open(out))}
    for s in a.sequences:
        est = load(os.path.join(REPO, f'results/{a.variant}/{s}/rep{a.rep}/trail_output/poses.txt'))
        gt = load(os.path.join(REPO, f'results/summaries/gt/{s}_gt.tum'))
        d, peak, edge = latency(est, gt)
        print(f'{s}: latency={d:+.3f}s corr={peak:.4f}{"  (AT EDGE)" if edge else ""}  ref={a.variant} rep{a.rep}')
        rows[s] = dict(sequence=s, latency_s=f'{d:.4f}', corr=f'{peak:.4f}', at_edge=int(edge),
                       ref_variant=a.variant, ref_rep=a.rep)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sequence', 'latency_s', 'corr', 'at_edge', 'ref_variant', 'ref_rep'])
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    print('wrote', out)


if __name__ == '__main__':
    main()
