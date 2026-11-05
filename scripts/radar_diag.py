#!/usr/bin/env python3
"""Ground-truth-free radar diagnostic (paper Sec. diag, tab:diag; tab:mech floor-gate (a)).

Per radar scan (one PointCloud2 message, all targets share its stamp) the radar-only
least-squares ego-velocity v_scan = argmin sum (v_r + u.v)^2 (one 3-robust-sigma rejection
pass) is compared with the radar-frame velocity of a fixed-weight reference trajectory
(variant REF: reweighting off, results/REF/<seq>/rep1 or results/explore_garlileo/REF/...),
moving scans only (|v_ref| > 0.3 m/s). On GaRLILEO, whose GT carries orientation, v_scan is
also regressed on the GT radar-frame velocity.

Outputs (results/summaries/radar_diag.csv, one row per sequence, plus pooled rows per
dataset): OLS slopes z/x/y (radar on reference) with 95% bootstrap CI over scans; Deming
slope for z with the error-variance ratio estimated from the data (radar: per-scan
s^2 [(U^T U)^-1]_zz; reference: var(v_ref - v_GT) in the radar frame); OLS z slope on GT
(GaRLILEO); mean radar-frame u_z of returns; fraction of returns below the body plane
(h_i < 0); RMS Doppler residual vs reference for h<0 over h>=0; and, for h<0 returns that
point downward, the share whose range exceeds the range at which their line of sight meets
the floor (floor height below the body origin from the LiDAR map under the trajectory).

Usage: python3 scripts/radar_diag.py [--sequences ...] [--boot 1000]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import json

import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import eval_traj as E  # noqa: E402

TRAIL = ['BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2', 'Airfield1', 'Airfield2']
GARLILEO = ['Downstair', 'CorriLoop', 'BiCorridor', 'SlopeStair', 'Tunnel', 'Overpass', 'Quad']
MIN_SPEED = 0.3


def quat_to_rot(q):
    return E.quat_to_rot(np.atleast_2d(q))


def load_extrinsics(seq):
    if seq in TRAIL:
        cfg = yaml.safe_load(open(REPO_ROOT / 'configs' / 'base' / 'boxi.yaml'))
    else:
        cfg = yaml.safe_load(open(REPO_ROOT / 'ws/src/TRaIL-Odom/dataset/GaRLILEO/config.yaml'))
    c = cfg['Configor']['DataStream']['CalibParam']
    q = c['SO3_RtoB']
    R_RtoB = quat_to_rot(np.array([q['qx'], q['qy'], q['qz'], q['qw']], float))[0]
    p = c['POS_RinB']
    return R_RtoB, np.array([p['r0c0'], p['r1c0'], p['r2c0']], float), cfg['Configor']['DataStream']['RadarTopic']


def bag_path(seq):
    return REPO_ROOT / ('data/Boxi_ros2_sqlite' if seq in TRAIL else 'data/GaRLILEO') / seq


def read_radar(seq, topic):
    """List of (t, xyz Nx3, v_r N) per message, NaNs dropped."""
    store = get_typestore(Stores.ROS2_HUMBLE)
    scans = []
    with AnyReader([bag_path(seq)], default_typestore=store) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            off = {f.name: f.offset for f in msg.fields}
            n = msg.width * msg.height
            if n == 0:
                continue
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
            col = lambda name: buf[:, off[name]:off[name] + 4].copy().view('<f4').ravel().astype(float)
            xyz = np.stack([col('x'), col('y'), col('z')], 1)
            vr = col('velocity')
            ok = np.isfinite(xyz).all(1) & np.isfinite(vr) & (np.linalg.norm(xyz, axis=1) > 1e-6)
            if ok.sum() == 0:
                continue
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            scans.append((t, xyz[ok], vr[ok]))
    return scans


def scan_lsq(xyz, vr):
    """Radar-only v (radar frame) for one scan, its z-variance factor, and kept mask."""
    u = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    keep = np.ones(len(vr), bool)
    v = None
    for _ in range(2):
        if keep.sum() < 3:
            return None
        U, b = u[keep], -vr[keep]
        A = U.T @ U
        w = np.linalg.eigvalsh(A)
        if w[0] < 1e-6 * w[2]:
            return None
        v = np.linalg.solve(A, U.T @ b)
        e = np.abs(u @ v + vr)
        thr = max(3 * 1.4826 * np.median(e[keep]), 0.1)
        new = e <= thr
        if (new == keep).all():
            break
        keep = new
    U, b = u[keep], -vr[keep]
    A = U.T @ U
    res = U @ v - b
    s2 = float(res @ res / max(len(b) - 3, 1))
    return v, s2 * np.linalg.inv(A)[2, 2], keep


class Traj:
    """Pose trajectory -> radar-frame velocity at arbitrary times (central differences)."""

    def __init__(self, t, p, q, R_RtoB, p_RinB, half=0.25):
        self.t, self.p, self.R = t, p, quat_to_rot(q)
        self.R_RtoB, self.p_RinB, self.half = R_RtoB, p_RinB, half

    def radar_vel(self, ts):
        out = np.full((len(ts), 3), np.nan)
        for k, ti in enumerate(ts):
            if ti - self.half < self.t[0] or ti + self.half > self.t[-1]:
                continue
            i0 = np.searchsorted(self.t, ti - self.half)
            i1 = np.searchsorted(self.t, ti + self.half)
            i1 = min(i1, len(self.t) - 1)
            if i1 <= i0:
                continue
            dt = self.t[i1] - self.t[i0]
            vW = (self.p[i1] - self.p[i0]) / dt
            Rm = self.R[np.searchsorted(self.t, ti).clip(0, len(self.t) - 1)]
            dR = self.R[i0].T @ self.R[i1]
            ang = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2.0
            wB = ang / dt
            vB = Rm.T @ vW + np.cross(wB, self.p_RinB)
            out[k] = self.R_RtoB.T @ vB
        return out


def load_pose_file(path):
    a = np.loadtxt(path)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def ref_dir(seq):
    if seq in TRAIL:
        return REPO_ROOT / 'results' / 'REF' / seq / 'rep1'
    return REPO_ROOT / 'results' / 'explore_garlileo' / 'REF' / seq / 'rep1'


def gt_path(seq):
    if seq in TRAIL:
        return REPO_ROOT / 'results' / 'summaries' / 'gt' / f'{seq}_gt.tum'
    return REPO_ROOT / 'data' / 'GaRLILEO' / 'GT_Trajectory' / f'{seq}_gt.txt'


def _omega(R, t, j):
    j = int(np.clip(j, 1, len(t) - 2))
    dR = R[j - 1].T @ R[j + 1]
    return np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2.0 / (t[j + 1] - t[j - 1])


def gt_radar_vel(seq, ref, ts, R_RtoB, p_RinB):
    """GT radar-frame velocity at ts. Body-origin velocity from GT positions (central difference
    over +-half, rejected across GT gaps). TRaIL-Odom GT is the survey prism, position only: the
    prism lever arm (meta prism_in_body) is removed with the reference's angular rate, and the
    reference's orientation rotates world -> body (its error is ~1 deg, i.e. ~2% of speed).
    GaRLILEO GT carries the body orientation and is used directly. GT and reference worlds are
    related by a rigid Umeyama fit, as for ATE."""
    gt_t, gt_p, gt_q = load_pose_file(gt_path(seq))
    lever = np.zeros(3)
    if seq in TRAIL:
        lever = np.array(json.loads(Path(str(gt_path(seq)) + '.meta.json').read_text())['prism_in_body'])
    k = np.clip(np.searchsorted(ref.t, gt_t), 1, len(ref.t) - 1)
    k = np.where(np.abs(ref.t[k - 1] - gt_t) < np.abs(ref.t[k] - gt_t), k - 1, k)
    ok = np.abs(ref.t[k] - gt_t) < 0.05
    dst = ref.p[k[ok]] + np.einsum('nij,j->ni', ref.R[k[ok]], lever)
    src = gt_p[ok]
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    Ra = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    gp = (gt_p - cs) @ Ra.T + cd
    gR = Ra @ quat_to_rot(gt_q) if seq in GARLILEO else None
    half = 0.25
    out = np.full((len(ts), 3), np.nan)
    for i, ti in enumerate(ts):
        i0 = np.searchsorted(gt_t, ti - half)
        i1 = min(np.searchsorted(gt_t, ti + half), len(gt_t) - 1)
        if i1 <= i0 or not (1.5 * half <= gt_t[i1] - gt_t[i0] <= 2.5 * half):
            continue
        vP = (gp[i1] - gp[i0]) / (gt_t[i1] - gt_t[i0])
        if gR is not None:
            j = int(np.clip(np.searchsorted(gt_t, ti), 1, len(gt_t) - 2))
            R, w = gR[j], _omega(gR, gt_t, j)
        else:
            j = np.searchsorted(ref.t, ti)
            R, w = ref.R[int(np.clip(j, 0, len(ref.t) - 1))], _omega(ref.R, ref.t, j)
        vB = R.T @ vP - np.cross(w, lever)
        out[i] = R_RtoB.T @ (vB + np.cross(w, p_RinB))
    return out


def floor_height(ref_d, ref):
    """Height of the body origin above the floor, from the LiDAR map: for every 200th pose,
    map points 3-12 m away horizontally and 0.2-3 m below the pose; the mode of their height
    below the pose (5 cm bins) is the floor (the Mid-360's -7 deg lower FOV does not see the
    ground next to the vehicle). Median over poses."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    pts = np.asarray(o3d.io.read_point_cloud(str(ref_d / 'trail_output' / 'local_map.pcd')).points)
    tree = cKDTree(pts[:, :2])
    hs = []
    bins = np.arange(0.2, 3.0001, 0.05)
    for i in range(0, len(ref.t), 200):
        idx = np.array(tree.query_ball_point(ref.p[i, :2], 12.0))
        if len(idx) == 0:
            continue
        d = np.linalg.norm(pts[idx, :2] - ref.p[i, :2], axis=1)
        dz = ref.p[i, 2] - pts[idx[d >= 3.0], 2]
        dz = dz[(dz >= 0.2) & (dz <= 3.0)]
        if len(dz) < 100:
            continue
        hist, edges = np.histogram(dz, bins)
        hs.append(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
    return float(np.median(hs)) if hs else float('nan')


def ols_slope(x, y):
    x0, y0 = x - x.mean(), y - y.mean()
    return float((x0 @ y0) / (x0 @ x0))


def deming(x, y, delta):
    sxx, syy, sxy = np.var(x), np.var(y), np.cov(x, y, bias=True)[0, 1]
    return float((syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy))


def boot_ci(x, y, fn, n, rng):
    m = len(x)
    vals = [fn(x[i], y[i]) for i in (rng.integers(0, m, m) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def slopes_block(prefix, X, Y, G, nboot, rng):
    """X: reference, Y: radar, G: GT (radar frame, Nx3). OLS radar-on-reference and
    radar-on-GT per axis; Deming radar-vs-reference per axis with the error-variance ratio
    delta = var(radar - b_gt GT) / var(reference - GT), i.e. both error variances measured
    against GT (GT velocity error is negligible next to either)."""
    row = {}
    g = np.isfinite(G).all(1)
    for j, ax in enumerate('xyz'):
        row[f'{prefix}ols_{ax}'] = ols_slope(X[:, j], Y[:, j])
        row[f'{prefix}ols_gt_{ax}'] = ols_slope(G[g, j], Y[g, j])
        row[f'{prefix}ref_on_gt_{ax}'] = ols_slope(G[g, j], X[g, j])
        row[f'{prefix}gt_std_{ax}'] = float(G[g, j].std())
        var_rad = float(np.var(Y[g, j] - row[f'{prefix}ols_gt_{ax}'] * G[g, j]))
        var_ref = float(np.var(X[g, j] - G[g, j]))
        delta = var_rad / var_ref
        row[f'{prefix}reliability_ref_{ax}'] = float(np.var(G[g, j]) / (np.var(G[g, j]) + var_ref))
        row[f'{prefix}deming_{ax}'] = deming(X[g, j], Y[g, j], delta)
    xz, yz, gz = X[g, 2], Y[g, 2], G[g, 2]
    row[f'{prefix}ols_z_lo'], row[f'{prefix}ols_z_hi'] = boot_ci(X[:, 2], Y[:, 2], ols_slope, nboot, rng)
    row[f'{prefix}ols_gt_z_lo'], row[f'{prefix}ols_gt_z_hi'] = boot_ci(gz, yz, ols_slope, nboot, rng)
    var_rad = float(np.var(yz - row[f'{prefix}ols_gt_z'] * gz))
    delta = var_rad / float(np.var(xz - gz))
    row[f'{prefix}deming_z_lo'], row[f'{prefix}deming_z_hi'] = boot_ci(xz, yz, lambda a, b: deming(a, b, delta), nboot, rng)
    return row


def analyse(seq, nboot, rng):
    R_RtoB, p_RinB, topic = load_extrinsics(seq)
    rd = ref_dir(seq)
    t, p, q = load_pose_file(rd / 'trail_output' / 'poses.txt')
    ref = Traj(t, p, q, R_RtoB, p_RinB)
    scans = read_radar(seq, topic)

    ts, vs = [], []
    for (ti, xyz, vr) in scans:
        out = scan_lsq(xyz, vr)
        if out is not None:
            ts.append(ti)
            vs.append(out[0])
    ts, vs = np.array(ts), np.array(vs)
    vref = ref.radar_vel(ts)
    vgt = gt_radar_vel(seq, ref, ts, R_RtoB, p_RinB)
    ok = np.isfinite(vref).all(1) & (np.linalg.norm(vref, axis=1) > MIN_SPEED)
    X, Y, G = vref[ok], vs[ok], vgt[ok]

    row = dict(sequence=seq, dataset='TRaIL-Odom' if seq in TRAIL else 'GaRLILEO',
               n_scans=int(ok.sum()), n_scans_gt=int(np.isfinite(G).all(1).sum()))
    row.update(slopes_block('', X, Y, G, nboot, rng))

    # per-return quantities vs reference (all returns of moving scans)
    vref_by_t = dict(zip(ts[ok], vref[ok]))
    uz, h, res, rngs, dBz, pred, meas = [], [], [], [], [], [], []
    for (ti, xyz, vr) in scans:
        v = vref_by_t.get(ti)
        if v is None:
            continue
        u = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        uz.append(u[:, 2])
        h.append((xyz @ R_RtoB.T + p_RinB)[:, 2])
        res.append(vr + u @ v)
        pred.append(-(u @ v))
        meas.append(vr)
        rngs.append(np.linalg.norm(xyz, axis=1))
        dBz.append((u @ R_RtoB.T)[:, 2])
    uz, h, res, rngs, dBz, pred, meas = map(np.concatenate, (uz, h, res, rngs, dBz, pred, meas))
    gated_down = (h < 0) & (dBz < -1e-3)
    per_return = dict(uz=uz, h=h, res=res, pred=pred, meas=meas, gd_range=rngs[gated_down], gd_dBz=dBz[gated_down],
                      gd_zR=np.full(int(gated_down.sum()), p_RinB[2]))
    row['floor_height'] = floor_height(rd, ref)
    return row, (X, Y, G), per_return


def per_return_stats(r):
    below, above = r['res'][r['h'] < 0], r['res'][r['h'] >= 0]
    out = dict(mean_uz=float(r['uz'].mean()),
               mean_elev_deg=float(np.degrees(np.arcsin(np.clip(r['uz'], -1, 1))).mean()),
               frac_h_neg=float((r['h'] < 0).mean()),
               res_ratio_hneg=float(np.sqrt(np.mean(below ** 2)) / np.sqrt(np.mean(above ** 2))) if len(below) else float('nan'),
               n_gated_down=int(len(r['gd_range'])))
    # signed: measured / reference-predicted radial speed, returns with |pred| > 0.3 m/s
    for name, m in (('hneg', r['h'] < 0), ('hpos', r['h'] >= 0)):
        k = m & (np.abs(r['pred']) > 0.3)
        out[f'speed_ratio_{name}'] = float(np.median(r['meas'][k] / r['pred'][k])) if k.sum() > 20 else float('nan')
    return out


def frac_beyond(r, H):
    """Gated, downward-pointing returns classified by range / (range at which their line of
    sight meets a floor H below the body origin): < 0.95 before the floor (low objects),
    0.95-1.05 at the floor (direct ground hit), > 1.05 beyond it (mirror image)."""
    if len(r['gd_range']) == 0:
        return dict(frac_before_floor=float('nan'), frac_at_floor=float('nan'), frac_beyond_floor=float('nan'))
    q = r['gd_range'] / ((r['gd_zR'] + H) / (-r['gd_dBz']))
    return dict(frac_before_floor=float((q < 0.95).mean()), frac_at_floor=float(((q >= 0.95) & (q <= 1.05)).mean()),
                frac_beyond_floor=float((q > 1.05).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequences', nargs='+', default=TRAIL + GARLILEO)
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, pools = [], {'TRaIL-Odom': ([], []), 'GaRLILEO': ([], [])}
    for seq in args.sequences:
        if not (ref_dir(seq) / 'trail_output' / 'poses.txt').exists():
            print(f'SKIP {seq}: no REF run')
            continue
        row, xyg, pr = analyse(seq, args.boot, rng)
        row.update(per_return_stats(pr))
        rows.append(row)
        pools[row['dataset']][0].append(xyg)
        pools[row['dataset']][1].append((row, pr))
    for ds, (xygs, rprs) in pools.items():
        if not xygs:
            continue
        H = float(np.nanmedian([r['floor_height'] for r, _ in rprs]))
        for r, pr in rprs:
            r['platform_floor_height'] = H
            r.update(frac_beyond(pr, H))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        X, Y, G = (np.concatenate([a[i] for a in xygs]) for i in range(3))
        allpr = {k: np.concatenate([p[k] for _, p in rprs]) for k in rprs[0][1]}
        row = dict(sequence='POOLED', dataset=ds, n_scans=len(X), n_scans_gt=int(np.isfinite(G).all(1).sum()),
                   platform_floor_height=H)
        row.update(slopes_block('', X, Y, G, args.boot, rng))
        row.update(per_return_stats(allpr))
        row.update(frac_beyond(allpr, H))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    out = REPO_ROOT / 'results' / 'summaries' / 'radar_diag.csv'
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('sequence', 'dataset'), k))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
