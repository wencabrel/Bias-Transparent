"""Ground-truth overlay + live drift for the native viewer.

GT (results/summaries/gt/<Sequence>_gt.tum) is NOT in the same coordinate frame as the
live estimate -- verified directly: at matching timestamps GT starts around
(-0.05, 7.04, 1.34) while trail_node's own world frame starts the run near (0,0,0).
Comparing them without alignment would be meaningless, not "drift".

Mirrors the project's own evo-based methodology (translation-only, no scale, Umeyama/
Kabsch rotation+translation fit -- see scripts/eval_traj.py), but computed from a single
early window and then FROZEN rather than continuously refit. A continuously-refit
alignment would silently reabsorb drift every frame and could never show it growing;
freezing it after an initial anchor window is what makes the live number a real
diagnostic instead of one that can't help but read near-zero.

This is a live, approximate diagnostic, not the research-grade number in
results/summaries/eval_traj.csv (that uses the full trajectory with evo's own
association/alignment implementation). Labelled as such in the UI.
"""
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
    """Collects (abs_time, est_pos) samples until there's enough data to fit a stable
    one-time alignment to GT, then freezes it. `status` is 'collecting', 'ready', or
    'unavailable' (no GT file for this sequence).

    Readiness requires genuine two-axis (lateral) spread, not just sample count or
    straight-line distance travelled: on Park1, /axis publishes at ~100Hz and the
    vehicle travels in a near-perfectly straight line for the first ~100s (lateral std
    stays under 0.2m until ~50s -- verified directly against
    results/V4/Park1/rep1/trail_output/poses.txt). A straight-line segment is a
    textbook Kabsch degenerate case: it satisfies any simple "enough spread" check while
    leaving the rotation about the direction of travel completely unconstrained, so the
    fit can numerically "succeed" while actually being close to arbitrary. Checking the
    *second* principal-axis standard deviation (from SVD of the centered samples) is
    what actually distinguishes "has turned enough to fix rotation" from "has only gone
    straight so far"."""

    MIN_SAMPLES = 200          # ~2s at 100Hz; a cheap floor before bothering to check spread
    MIN_PRIMARY_STD_M = 3.0    # real forward progress, not just odometry jitter
    MIN_LATERAL_STD_M = 1.0    # genuine curvature/turning, not a straight line
    GIVE_UP_SAMPLES = 20000    # ~200s at 100Hz -- close to a full Boxi run's length

    def __init__(self, gt_path):
        self.status = 'collecting'
        self._gt_times = None
        self._gt_pos = None
        self._samples_t = []
        self._samples_p = []
        self.R = None
        self.t = None
        try:
            self._gt_times, self._gt_pos = load_gt_tum(gt_path)
        except OSError:
            self.status = 'unavailable'

    def _in_gt_range(self, abs_t):
        return self._gt_times[0] <= abs_t <= self._gt_times[-1]

    def _nearest_gt(self, abs_t):
        idx = np.searchsorted(self._gt_times, abs_t)
        idx = min(max(idx, 0), len(self._gt_times) - 1)
        return self._gt_pos[idx]

    def add_sample(self, abs_t, est_pos):
        if self.status != 'collecting':
            return
        if not self._in_gt_range(abs_t):
            # Real bug found this way: trail_node's own pose stream can start before
            # GT's own recorded time range does (verified directly -- this run's
            # poses.txt started ~13s before Park1_gt.tum's first row). Samples from
            # that gap would all silently clamp to GT's first point in _nearest_gt,
            # matching many distinct real positions to one single GT point and biasing
            # the whole fit even after real curvature later came in. Skip them outright
            # rather than including a corrupted correspondence.
            return
        self._samples_t.append(abs_t)
        self._samples_p.append(est_pos)
        n = len(self._samples_p)
        if n < self.MIN_SAMPLES:
            return
        est = np.asarray(self._samples_p, dtype=np.float64)
        centered = est - est.mean(0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        std_per_axis = singular_values / np.sqrt(max(n - 1, 1))
        if (std_per_axis[0] < self.MIN_PRIMARY_STD_M
                or std_per_axis[1] < self.MIN_LATERAL_STD_M):
            # still (close to) a straight line so far -- rotation isn't constrained yet,
            # keep collecting rather than freeze a degenerate fit
            if n > self.GIVE_UP_SAMPLES:
                self.status = 'unavailable'
            return
        gt_matched = np.stack([self._nearest_gt(t) for t in self._samples_t])
        self.R, self.t = kabsch(est, gt_matched)
        self.status = 'ready'

    def gt_path_in_est_frame(self):
        """Whole GT trajectory transformed into the estimate's (unaligned) frame, for
        overlay alongside the live cloud/path. Only valid once status == 'ready'."""
        if self.status != 'ready':
            return []
        # est ~= R^T @ (gt - t)  (inverse of est_aligned = R@est + t ~= gt)
        pts = (self._gt_pos - self.t) @ self.R
        return [tuple(p) for p in pts]

    def live_drift(self, abs_t, est_pos):
        """Distance (m) between the aligned current estimate and time-matched GT.
        None if not ready yet, or if abs_t falls outside GT's own recorded time range
        (clamping to GT's boundary point there is exactly the bug that corrupted the
        alignment fit in the first place -- never show a number built on that same clamp)."""
        if self.status != 'ready' or not self._in_gt_range(abs_t):
            return None
        aligned = self.R @ np.asarray(est_pos) + self.t
        gt_p = self._nearest_gt(abs_t)
        return float(np.linalg.norm(aligned - gt_p))
