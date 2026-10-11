#!/usr/bin/env python3
"""Phase 0 (D.3): evaluate estimated trajectories against GT with evo.

For each (variant, sequence, rep) with a poses.txt, runs evo_ape (SE(3) Umeyama alignment,
translation part) and evo_rpe (1m segments) against results/summaries/gt/<sequence>_gt.tum,
and writes one combined CSV.

evo's own console scripts (~/.local/bin/evo_ape etc.) have a shebang pointing at a broken
system python3.10 (numpy ABI mismatch) -- invoke the installed evo package's entry points
directly under the working interpreter instead (same evo install, found via
~/.local/share/mamba/envs/ros_env/lib/python3.11/site-packages per this machine, importable
as plain `evo` under this script's own python3).

Also computes along-axis scale error per leg (Part D.3 of the guideline), for all 6 Boxi
sequences: all 6 are return-to-start (start/end GT position within ~0.1-8.2m of each
other, vs 60-177m of max excursion -- verified 2026-10-11), and plain scale-free ATE
(evo's `-a`, Umeyama without `-s`) can't detect or correct an along-track scale mismatch.
On a return-to-start path this gets amplified rather than averaged out, because a leg-wise
scale error (e.g. outbound compressed, inbound stretched) nets close to zero in total path
length while still producing large positional error at both ends once rigidly aligned --
see docs/DECISIONS.md 2026-10-11 entry for the BikeTunnel1 case that surfaced this.

Usage: python3 scripts/eval_traj.py [--variants V4] [--sequences Seq1 Seq2] [--reps 1 2 3]
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_SEQUENCES = ['Airfield1', 'Airfield2', 'BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2']
ALL_VARIANTS = ['V4']


def load_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append([float(x) for x in parts[:4]])
    rows = np.array(rows)
    order = np.argsort(rows[:, 0])
    return rows[order, 0], rows[order, 1:4]


def path_length(pos):
    if len(pos) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def leg_scale_ratios(gt_t, gt_p, est_t, est_p):
    """Split both trajectories at the GT's turn-around time (max distance from its own
    start) and return (outbound_ratio, inbound_ratio) = est_leg_length / gt_leg_length.
    The split point comes from GT, not from each trajectory's own argmax, so a badly
    drifted estimate (whose own farthest point may not be the true physical turn-around)
    doesn't pick its own, potentially meaningless, split."""
    d_from_start = np.linalg.norm(gt_p - gt_p[0], axis=1)
    turn_time = gt_t[int(np.argmax(d_from_start))]

    def split(t, p):
        idx = int(np.searchsorted(t, turn_time))
        idx = max(1, min(idx, len(t) - 1))
        return p[:idx + 1], p[idx:]

    gt_out, gt_back = split(gt_t, gt_p)
    est_out, est_back = split(est_t, est_p)
    gt_out_len, gt_back_len = path_length(gt_out), path_length(gt_back)
    est_out_len, est_back_len = path_length(est_out), path_length(est_back)
    out_ratio = est_out_len / gt_out_len if gt_out_len > 0 else float('nan')
    back_ratio = est_back_len / gt_back_len if gt_back_len > 0 else float('nan')
    return out_ratio, back_ratio


def run_evo(cmd_module, gt_tum, est_tum, out_zip, extra_args):
    from evo.entry_points import ape, rpe
    entry = {'ape': ape, 'rpe': rpe}[cmd_module]
    argv = sys.argv
    sys.argv = ['evo_' + cmd_module, 'tum', str(gt_tum), str(est_tum),
                '-a', '--pose_relation', 'trans_part',
                '--save_results', str(out_zip), '--no_warnings'] + extra_args
    try:
        entry()
    except SystemExit:
        pass
    finally:
        sys.argv = argv


def read_stats(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read('stats.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=ALL_VARIANTS)
    ap.add_argument('--sequences', nargs='+', default=ALL_SEQUENCES)
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 3])
    args = ap.parse_args()

    eval_dir = REPO_ROOT / 'results' / 'summaries' / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = REPO_ROOT / 'results' / 'summaries' / 'eval_traj.csv'
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                existing[(row['variant'], row['sequence'], int(row['rep']))] = row
    rows_by_key = dict(existing)

    for variant in args.variants:
        for sequence in args.sequences:
            gt_tum = REPO_ROOT / 'results' / 'summaries' / 'gt' / f'{sequence}_gt.tum'
            if not gt_tum.exists():
                print(f"SKIP {sequence}: no GT at {gt_tum}")
                continue
            for rep in args.reps:
                est_tum = REPO_ROOT / 'results' / variant / sequence / f'rep{rep}' / 'trail_output' / 'poses.txt'
                if not est_tum.exists():
                    print(f"SKIP {variant} {sequence} rep{rep}: no poses.txt")
                    continue

                run_dir = eval_dir / variant / sequence / f'rep{rep}'
                run_dir.mkdir(parents=True, exist_ok=True)
                ape_zip = run_dir / 'ape.zip'
                rpe_zip = run_dir / 'rpe.zip'

                run_evo('ape', gt_tum, est_tum, ape_zip, [])
                run_evo('rpe', gt_tum, est_tum, rpe_zip,
                        ['--delta', '1', '--delta_unit', 'm', '--all_pairs'])

                ape_stats = read_stats(ape_zip)
                rpe_stats = read_stats(rpe_zip)

                gt_t, gt_p = load_tum(gt_tum)
                est_t, est_p = load_tum(est_tum)
                out_ratio, back_ratio = leg_scale_ratios(gt_t, gt_p, est_t, est_p)

                rows_by_key[(variant, sequence, rep)] = dict(
                    variant=variant, sequence=sequence, rep=rep,
                    ate_rmse=ape_stats['rmse'], ate_mean=ape_stats['mean'],
                    ate_std=ape_stats['std'], ate_max=ape_stats['max'],
                    rte_rmse=rpe_stats['rmse'], rte_mean=rpe_stats['mean'],
                    rte_std=rpe_stats['std'], rte_max=rpe_stats['max'],
                    outbound_scale_ratio=out_ratio, inbound_scale_ratio=back_ratio,
                )
                print(f"{variant} {sequence} rep{rep}: ATE rmse={ape_stats['rmse']:.4f} "
                      f"RTE rmse={rpe_stats['rmse']:.4f} "
                      f"leg_scale out={out_ratio:.3f} back={back_ratio:.3f}")

    # Merge into existing rows rather than overwrite -- a narrower --sequences/--reps
    # invocation must not silently drop rows for sequences/reps it wasn't asked about.
    # restval='' covers legacy rows written before outbound/inbound_scale_ratio existed --
    # they show blank until that (variant, sequence, rep) is re-evaluated.
    fieldnames = ['variant', 'sequence', 'rep', 'ate_rmse', 'ate_mean', 'ate_std', 'ate_max',
                  'rte_rmse', 'rte_mean', 'rte_std', 'rte_max',
                  'outbound_scale_ratio', 'inbound_scale_ratio']
    sorted_keys = sorted(rows_by_key.keys(), key=lambda k: (k[0], k[1], k[2]))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        for key in sorted_keys:
            writer.writerow(rows_by_key[key])
    print(f"\nwrote {csv_path} ({len(sorted_keys)} rows)")


if __name__ == '__main__':
    main()
