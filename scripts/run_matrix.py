#!/usr/bin/env python3
"""Phase 0 (H0): run scripts/run_seq.sh over all variants x sequences x reps, resumable.

Runs are sequential (not parallel): each spins up rmw_zenohd + a docker container bound to
--net=host, so concurrent runs would collide on the zenoh daemon's port.

Skips any (variant, sequence, rep) whose results/<variant>/<sequence>/rep<rep>/run.json
already records exit_code 0, so a killed/resumed matrix doesn't redo finished runs.

Usage: python3 scripts/run_matrix.py [--variants V4] [--sequences Seq1 Seq2] [--reps 3]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_SEQUENCES = ['Airfield1', 'Airfield2', 'BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2']
ALL_VARIANTS = ['V4']


def already_done(variant, sequence, rep):
    run_json = REPO_ROOT / 'results' / variant / sequence / f'rep{rep}' / 'run.json'
    if not run_json.exists():
        return False
    try:
        data = json.loads(run_json.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return data.get('exit_code') == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=ALL_VARIANTS)
    ap.add_argument('--sequences', nargs='+', default=ALL_SEQUENCES)
    ap.add_argument('--reps', type=int, default=3)
    args = ap.parse_args()

    jobs = [
        (variant, sequence, rep)
        for variant in args.variants
        for sequence in args.sequences
        for rep in range(1, args.reps + 1)
    ]

    print(f"{len(jobs)} total jobs", flush=True)
    failures = []
    for i, (variant, sequence, rep) in enumerate(jobs, 1):
        if already_done(variant, sequence, rep):
            print(f"[{i}/{len(jobs)}] SKIP {variant} {sequence} rep{rep} (already done)", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] RUN  {variant} {sequence} rep{rep}", flush=True)
        result = subprocess.run(
            [str(REPO_ROOT / 'scripts' / 'run_seq.sh'), variant, sequence, str(rep)],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"[{i}/{len(jobs)}] FAILED {variant} {sequence} rep{rep} (exit {result.returncode})", flush=True)
            failures.append((variant, sequence, rep))
        else:
            print(f"[{i}/{len(jobs)}] OK   {variant} {sequence} rep{rep}", flush=True)

    print(f"\ndone: {len(jobs) - len(failures)}/{len(jobs)} succeeded", flush=True)
    if failures:
        print("failures:", failures, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
