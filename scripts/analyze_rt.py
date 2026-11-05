#!/usr/bin/env python3
"""tab:rt: real-time behaviour and cost from window_log.csv (research::WindowLogger).

For each (variant, dataset): stalled runs = runs whose run.json validation failed;
per-window processing time (wall_ms of IncrementalOptimization, i.e. T(w)) pooled over all
completed windows of all runs found, mean and 95th percentile; longest completed window;
and the share of windows with wall time above their own duration (T(w) > w).

Only runs that carry window_log.csv are used (the logger was added 2026-11-02), and the
timing numbers are only meaningful for runs made one container at a time.

Usage: python3 scripts/analyze_rt.py --variant V4 --reps 301
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIL = ['BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2', 'Airfield1', 'Airfield2']
GARLILEO = ['Downstair', 'CorriLoop', 'BiCorridor', 'SlopeStair', 'Tunnel', 'Overpass', 'Quad']


def run_dir(variant, seq, rep):
    if seq in TRAIL:
        return REPO_ROOT / 'results' / variant / seq / f'rep{rep}'
    return REPO_ROOT / 'results' / 'explore_garlileo' / variant / seq / f'rep{rep}'


def summarise(variant, seqs, reps):
    wall, win, stalled, runs = [], [], 0, 0
    for seq in seqs:
        for rep in reps:
            d = run_dir(variant, seq, rep)
            if not (d / 'run.json').exists():
                continue
            runs += 1
            if not json.loads((d / 'run.json').read_text())['validation']['ok']:
                stalled += 1
            wl = d / 'window_log.csv'
            if wl.exists():
                with open(wl, newline='') as f:
                    for r in csv.DictReader(f):
                        wall.append(float(r['wall_ms']))
                        win.append(float(r['window_s']))
    wall, win = np.array(wall), np.array(win)
    if len(wall) == 0:
        return dict(runs=runs, stalled=stalled)
    return dict(runs=runs, stalled=stalled, windows=len(wall),
                solve_mean_ms=float(wall.mean()), solve_p95_ms=float(np.percentile(wall, 95)),
                longest_window_s=float(win.max()),
                frac_T_gt_w=float((wall / 1000.0 > win).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--reps', type=int, nargs='+', required=True)
    args = ap.parse_args()
    for name, seqs in (('TRaIL-Odom', TRAIL), ('GaRLILEO', GARLILEO)):
        s = summarise(args.variant, seqs, args.reps)
        print(name, args.variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == '__main__':
    main()
