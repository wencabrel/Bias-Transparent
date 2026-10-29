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


def quat_to_rot(q):
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def corrected_estimate(est_tum, out_tum, lever=None, latency=0.0):
    """Write a corrected copy of est_tum (GT is never modified).
    lever: prism position in the body frame (from the GT .meta.json, prism_in_body); the
      estimate is moved to the prism point p + R_est @ lever, so the estimate's own
      orientation error still counts as error. Added 2026-10-26: the AP20 GT is the prism,
      0.31 m ahead of the body origin; left unapplied, it reads as a ~0.3 m along-track error.
    latency: seconds added to the estimate's timestamps (results/summaries/gt_latency.csv,
      measured from the V4 reference run by speed-profile cross-correlation)."""
    a = np.loadtxt(est_tum)
    a = a[np.argsort(a[:, 0])]
    if lever is not None:
        a[:, 1:4] = a[:, 1:4] + np.einsum('nij,j->ni', quat_to_rot(a[:, 4:8]), np.asarray(lever))
    a[:, 0] = a[:, 0] + latency
    np.savetxt(out_tum, a, fmt='%.9f')
    return out_tum


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


def position_rpe(gt_tum, est_tum, delta_m=1.0, max_dt=0.01):
    """Relative position error for position-only GT (added 2026-10-28).

    The Boxi GT (AP20 prism) carries no orientation -- every quaternion is 0 0 0 1 -- so
    evo_rpe's trans_part, which expresses each segment in the frame of its first pose,
    compares a world-frame GT displacement with a body-frame estimate displacement and
    reads ~sqrt(2) x delta whatever the accuracy. Here the estimate is associated with GT
    (nearest stamp within max_dt, as evo), rigidly aligned (Umeyama, no scale, as the ATE),
    and for every GT point i, j is the first point >= delta_m further along the GT path;
    the error is |(p_est_j - p_est_i) - (p_gt_j - p_gt_i)| in the world frame."""
    gt_t, gt_p = load_tum(gt_tum)
    est_t, est_p = load_tum(est_tum)
    k = np.clip(np.searchsorted(est_t, gt_t), 1, len(est_t) - 1)
    k = np.where(np.abs(est_t[k - 1] - gt_t) < np.abs(est_t[k] - gt_t), k - 1, k)
    ok = np.abs(est_t[k] - gt_t) < max_dt
    g, e = gt_p[ok], est_p[k[ok]]
    cg, ce = g.mean(0), e.mean(0)
    u, _, vt = np.linalg.svd((e - ce).T @ (g - cg))
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    e = (e - ce) @ r.T + cg
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))])
    j = np.searchsorted(s, s + delta_m)
    i = np.nonzero(j < len(g))[0]
    j = j[i]
    err = np.linalg.norm((e[j] - e[i]) - (g[j] - g[i]), axis=1)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mean=float(err.mean()),
                std=float(err.std()), max=float(err.max()))


def read_stats(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read('stats.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=ALL_VARIANTS)
    ap.add_argument('--sequences', nargs='+', default=ALL_SEQUENCES)
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 3])
    # Defaults reproduce the original evaluation exactly (body origin, no time shift).
    ap.add_argument('--gt-point', choices=['body', 'prism'], default='body',
                    help='evaluate the estimate at the body origin (original) or at the GT prism')
    ap.add_argument('--gt-latency', action='store_true',
                    help='shift estimate timestamps by results/summaries/gt_latency.csv')
    ap.add_argument('--rpe', choices=['evo', 'position'], default='evo',
                    help="relative error: evo_rpe trans_part (original; invalid for this "
                         "orientation-free GT, see position_rpe) or world-frame position RPE")
    args = ap.parse_args()

    tag = '_'.join(x for x in (('prism' if args.gt_point == 'prism' else ''),
                               ('lat' if args.gt_latency else ''),
                               ('rpepos' if args.rpe == 'position' else '')) if x)
    suffix = f'_{tag}' if tag else ''
    latencies = {}
    if args.gt_latency:
        with open(REPO_ROOT / 'results' / 'summaries' / 'gt_latency.csv', newline='') as f:
            latencies = {r['sequence']: float(r['latency_s']) for r in csv.DictReader(f)}

    eval_dir = REPO_ROOT / 'results' / 'summaries' / f'eval{suffix}'
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = REPO_ROOT / 'results' / 'summaries' / f'eval_traj{suffix}.csv'
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

                if tag:
                    lever = None
                    if args.gt_point == 'prism':
                        meta = json.loads((gt_tum.parent / f'{sequence}_gt.tum.meta.json').read_text())
                        lever = meta['prism_in_body']
                    if args.gt_latency and sequence not in latencies:
                        print(f"SKIP {variant} {sequence} rep{rep}: no latency for {sequence}")
                        continue
                    est_tum = corrected_estimate(est_tum, run_dir / 'est_corrected.tum', lever,
                                                 latencies.get(sequence, 0.0))

                run_evo('ape', gt_tum, est_tum, ape_zip, [])
                ape_stats = read_stats(ape_zip)
                if args.rpe == 'position':
                    rpe_stats = position_rpe(gt_tum, est_tum)
                else:
                    run_evo('rpe', gt_tum, est_tum, rpe_zip,
                            ['--delta', '1', '--delta_unit', 'm', '--all_pairs'])
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
