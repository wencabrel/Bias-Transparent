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

Usage: python3 scripts/eval_traj.py [--variants V4] [--sequences Seq1 Seq2] [--reps 1 2 3]
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_SEQUENCES = ['Airfield1', 'Airfield2', 'BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2']
ALL_VARIANTS = ['V4']


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
    rows = []

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
                rows.append(dict(
                    variant=variant, sequence=sequence, rep=rep,
                    ate_rmse=ape_stats['rmse'], ate_mean=ape_stats['mean'],
                    ate_std=ape_stats['std'], ate_max=ape_stats['max'],
                    rte_rmse=rpe_stats['rmse'], rte_mean=rpe_stats['mean'],
                    rte_std=rpe_stats['std'], rte_max=rpe_stats['max'],
                ))
                print(f"{variant} {sequence} rep{rep}: ATE rmse={ape_stats['rmse']:.4f} "
                      f"RTE rmse={rpe_stats['rmse']:.4f}")

    csv_path = REPO_ROOT / 'results' / 'summaries' / 'eval_traj.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == '__main__':
    main()
