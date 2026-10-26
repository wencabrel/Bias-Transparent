"""Ground-truth overlay + live error for the native viewer.

GT (results/summaries/gt/<Sequence>_gt.tum) is NOT in the same coordinate frame as the
live estimate -- verified directly: at matching timestamps GT starts around
(-0.05, 7.04, 1.34) while trail_node's own world frame starts the run near (0,0,0).
Comparing them without alignment would be meaningless, not "drift".

Mirrors the evaluation behind the reported numbers (scripts/eval_traj.py --gt-point prism):
  * the GT is the total-station PRISM, 0.31 m ahead of the IMU body origin that /axis
    reports, so every estimate sample is moved to the prism point p + R_est @ r, with r
    = prism_in_body from <Sequence>_gt.tum.meta.json (box_base == cpt7_imu, checked by
    scripts/gt_extract.py) -- without this, a 0.3 m along-track offset is shown as error;
  * the alignment is a rigid Kabsch fit (rotation + translation, no scale) over ALL
    samples so far, refit as the run goes; at the end of a run it is the same fit
    (and "ATE so far" the same number, up to evo's association details) as the
    evaluation's full-trajectory alignment.
An earlier version froze a fit from the first few seconds. A small rotation error from
such a short, nearly straight window grows linearly with distance (1 deg over 100 m is
1.7 m), so the GT path drifted away from an estimate that, by the evaluation, sits within
centimetres of it.

GT is drawn only up to the current time: while the trajectory so far is nearly a
straight line, the rotation about that line is not yet constrained, which does not move
GT points on the line but would swing any not-yet-reached part of the GT path.

Live estimate != final trajectory: /axis is the in-window state, poses.txt is the full
spline after the run, so the live numbers are close to, not identical with,
results/summaries/eval_traj_prism.csv.
"""
import json
import os

import numpy as np


def load_gt_tum(path):
    """Returns (times: Nx float64, positions: Nx3 float64), sorted by time."""
    times, pos = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            times.append(float(parts[0]))
            pos.append((float(parts[1]), float(parts[2]), float(parts[3])))
    times = np.asarray(times, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    order = np.argsort(times)
    return times[order], pos[order]


def load_prism_lever(gt_path):
    """prism_in_body from the GT's .meta.json, or None if absent."""
    try:
        with open(gt_path + '.meta.json') as f:
            return np.asarray(json.load(f)['prism_in_body'], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def kabsch(src, dst):
    """R, t minimizing ||R @ src + t - dst|| (no scale), src/dst: Nx3."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = cd - r @ cs
    return r, t


class GtAligner:
    """Collects (abs_time, prism-point estimate) samples inside GT's time range and keeps
    a Kabsch alignment over all of them, refit every REFIT_PERIOD_S of data. `status` is
    'collecting' (not enough forward motion yet), 'ready', or 'unavailable' (no GT file).
    `version` increments on every refit so the viewer knows to redraw the GT path."""

    MIN_SAMPLES = 50           # a floor before the first fit
    MIN_PRIMARY_STD_M = 2.0    # some real forward progress before the first fit
    REFIT_PERIOD_S = 2.0       # seconds of data between refits (rate-independent)
    MAX_FIT_POINTS = 4000      # evenly subsampled for the fit; cheap at this size

    def __init__(self, gt_path):
        self.status = 'collecting'
        self.version = 0
        self.lever = None
        self._gt_times = None
        self._gt_pos = None
        self._samples_t = []
        self._samples_p = []
        self._last_fit_t = None
        self.R = None
        self.t = None
        self.ate_so_far = None
        try:
            self._gt_times, self._gt_pos = load_gt_tum(gt_path)
            self.lever = load_prism_lever(gt_path)
        except OSError:
            self.status = 'unavailable'

    def _in_gt_range(self, abs_t):
        return self._gt_times[0] <= abs_t <= self._gt_times[-1]

    def _gt_at(self, abs_t):
        """GT interpolated at abs_t (times must lie inside GT's range)."""
        abs_t = np.atleast_1d(abs_t)
        return np.stack([np.interp(abs_t, self._gt_times, self._gt_pos[:, i]) for i in range(3)], -1)

    def _prism_point(self, est_pos, est_quat):
        p = np.asarray(est_pos, dtype=np.float64)
        if self.lever is None or est_quat is None:
            return p
        return p + quat_to_rot(est_quat) @ self.lever

    def add_sample(self, abs_t, est_pos, est_quat=None):
        if self.status == 'unavailable' or not self._in_gt_range(abs_t):
            # trail_node's pose stream starts ~8-13 s before GT's own recorded range;
            # samples outside it have no GT to compare with (never clamp to GT's ends)
            return
        self._samples_t.append(abs_t)
        self._samples_p.append(self._prism_point(est_pos, est_quat))
        n = len(self._samples_p)
        if n < self.MIN_SAMPLES or (self.status == 'ready' and abs_t - self._last_fit_t < self.REFIT_PERIOD_S):
            return
        est = np.asarray(self._samples_p, dtype=np.float64)
        if self.status != 'ready':
            sv = np.linalg.svd(est - est.mean(0), compute_uv=False) / np.sqrt(max(n - 1, 1))
            if sv[0] < self.MIN_PRIMARY_STD_M:
                return
        idx = np.linspace(0, n - 1, min(n, self.MAX_FIT_POINTS)).astype(int)
        times = np.asarray(self._samples_t)
        gt_all = self._gt_at(times)
        self.R, self.t = kabsch(est[idx], gt_all[idx])
        err = est @ self.R.T + self.t - gt_all
        self.ate_so_far = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        self._last_fit_t = abs_t
        self.status = 'ready'
        self.version += 1

    def gt_path_in_est_frame(self, upto_abs_t=None):
        """GT up to `upto_abs_t` (default: the latest sample) transformed into the
        estimate's frame, for overlay alongside the live cloud/path."""
        if self.status != 'ready':
            return []
        upto = self._samples_t[-1] if upto_abs_t is None else upto_abs_t
        pts = self._gt_pos[self._gt_times <= upto]
        # est ~= R^T @ (gt - t)  (inverse of est_aligned = R@est + t ~= gt)
        pts = (pts - self.t) @ self.R
        return [tuple(p) for p in pts]

    def live_error(self, abs_t, est_pos, est_quat=None):
        """Distance (m) between the aligned current prism-point estimate and GT at the
        same time; None if not ready or abs_t lies outside GT's recorded range."""
        if self.status != 'ready' or not self._in_gt_range(abs_t):
            return None
        aligned = self.R @ self._prism_point(est_pos, est_quat) + self.t
        return float(np.linalg.norm(aligned - self._gt_at(abs_t)[0]))
