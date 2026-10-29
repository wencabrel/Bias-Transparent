#!/usr/bin/env python3
"""Native 3D viewer for TRaIL-Odom live runs: Open3D GUI (point cloud + trajectory +
vehicle marker + a real side control panel with toggles/sliders/stats), fed by ROS2 data
via ros_bridge.RosBridge.

Structure deliberately mirrors ~/Light-3D/lightning-lm's UI (src/ui/ui_cloud.*,
ui_trajectory.*, ui_car.*, imgui_panel.*): a dedicated view class per geometry kind plus
a stats/control panel, driven by a periodic update tick -- same separation of concerns,
built on Open3D's GUI toolkit + rclpy instead of Pangolin + ImGui (that stack isn't in
our Docker image and would need a multi-hour from-source build to add; Open3D gives the
same category of native, real-time, controllable 3D view without that risk).

Usage (inside the trail-odom-native-viewer container, with ROS sourced):
    python3 viewer_app.py
"""
import csv
import os
import sys
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ros_bridge import RosBridge
from gt_tools import GtAligner, load_gt_tum, load_prism_lever, quat_to_rot, kabsch
from sequence_runner import SequenceRunner, SEQUENCES, LIVE_VIZ_DIR

AGE_MAX_S = 40.0  # seconds; matches the drift-crossover measurement in run_live_rviz.sh


def age_to_rgb(ages):
    """Green (fresh) -> yellow -> red (old/likely drifted). Same gradient formula as
    lightning-lm's UiCloud::SetCloudWithUncertainty (src/ui/ui_cloud.cc:74-83):
    u<0.5: (t, 0.8, 0.2*(1-t)); u>=0.5: (1, 0.8*(1-t), 0) -- ported directly rather than
    reusing our own earlier blue->red scheme, since the point was to match theirs."""
    u = np.clip(np.asarray(ages, dtype=np.float32) / AGE_MAX_S, 0.0, 1.0)
    out = np.empty((len(u), 3), dtype=np.float32)
    lo = u < 0.5
    t_lo = u[lo] * 2.0
    out[lo] = np.stack([t_lo, np.full_like(t_lo, 0.8), 0.2 * (1.0 - t_lo)], axis=1)
    t_hi = (u[~lo] - 0.5) * 2.0
    out[~lo] = np.stack([np.ones_like(t_hi), 0.8 * (1.0 - t_hi), np.zeros_like(t_hi)], axis=1)
    return out


class PointCloudView:
    """Owns the /local_map_aged geometry. Mirrors lightning-lm's UiCloud.

    Builds a brand-new MaterialRecord on every update() rather than reusing one object
    across add_geometry calls: reusing one MaterialRecord across many add/remove cycles
    triggered a hard C++ abort in Filament ("destroying MaterialInstance ... which is
    still in use by Renderable") -- a real crash seen in production, not a hypothetical.
    MaterialRecord is a lightweight Python struct, so a fresh one per frame is cheap.
    """

    NAME = "cloud"

    def __init__(self, scene):
        self._scene = scene
        self._point_size = 3.0
        self._present = False

    def set_point_size(self, size):
        self._point_size = size
        if self._present:
            mat = self._make_material()
            self._scene.scene.modify_geometry_material(self.NAME, mat)

    def _make_material(self):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self._point_size
        return mat

    def update(self, points, ages):
        # points is a numpy array (ros_bridge.py's vectorized decode), not a Python
        # list -- `if not points` on a multi-element array raises "truth value of an
        # array is ambiguous"; must check length explicitly.
        if len(points) == 0:
            return
        # Tensor-based geometry (o3d.t.geometry.PointCloud), not the legacy
        # o3d.geometry.PointCloud + Vector3dVector: measured directly, Vector3dVector
        # construction alone cost 32-75ms at ~60k points (verified against
        # Open3DScene.add_geometry, which does accept tensor geometries). The tensor API
        # measured at 1.7-2.2ms for the same point counts live -- roughly 20-30x faster,
        # not a guess. At ~4Hz updates that's the difference between a real GUI-thread
        # stutter several times a second and one that's no longer perceptible.
        pts = o3c.Tensor(np.ascontiguousarray(points, dtype=np.float32))
        colors = o3c.Tensor(np.ascontiguousarray(age_to_rgb(ages), dtype=np.float32))
        pcd = o3d.t.geometry.PointCloud(pts)
        pcd.point.colors = colors
        if self._present:
            self._scene.scene.remove_geometry(self.NAME)
        self._scene.scene.add_geometry(self.NAME, pcd, self._make_material())
        self._present = True

    def clear(self):
        if self._present:
            self._scene.scene.remove_geometry(self.NAME)
            self._present = False


class TrajectoryView:
    """Mirrors lightning-lm's UiTrajectory. Used for both the live estimated path and
    the (separately colored) GT overlay -- one instance each, distinguished by `name`.
    See PointCloudView's docstring for why a fresh MaterialRecord is built every
    update() instead of reused."""

    def __init__(self, scene, name, color, line_width=3.0):
        self._scene = scene
        self.name = name
        self._color = color
        self._line_width = line_width
        self._present = False
        self.visible = True

    def _make_material(self):
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = self._line_width
        return mat

    def update(self, path):
        if len(path) < 2:
            return
        pts = np.asarray(path, dtype=np.float64)
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([self._color] * len(lines))
        if self._present:
            self._scene.scene.remove_geometry(self.name)
        self._scene.scene.add_geometry(self.name, ls, self._make_material())
        self._present = True
        if not self.visible:
            self._scene.scene.show_geometry(self.name, False)

    def set_visible(self, visible):
        self.visible = visible
        if self._present:
            self._scene.scene.show_geometry(self.name, visible)

    def clear(self):
        if self._present:
            self._scene.scene.remove_geometry(self.name)
            self._present = False


class GridView:
    """Ground-plane reference grid, resized to actually contain the current point
    cloud's extent. A fixed small grid (the original version) looked comically tiny
    once the map grew past it -- the grid never adapted to how much the cloud had
    actually accumulated. See PointCloudView's docstring for why a fresh MaterialRecord
    is built every update() instead of reused."""

    NAME = "grid"

    def __init__(self, scene):
        self._scene = scene
        self._present = False
        self._half_extent = None

    def _make_material(self):
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 1.0
        return mat

    def update_for_bounds(self, max_abs_xy):
        # pick a half-extent comfortably larger than the data, snapped to a round step
        # so the grid doesn't get rebuilt on every tiny fluctuation in cloud extent
        target = max(20.0, max_abs_xy * 1.3)
        step = 2.0
        while target / step > 30:
            step *= 2.0
        half_extent = np.ceil(target / step) * step
        if self._present and half_extent == self._half_extent:
            return
        self._half_extent = half_extent
        n = int(half_extent / step)
        pts, lines = [], []
        for i in range(-n, n + 1):
            pts.append([i * step, -n * step, 0])
            pts.append([i * step, n * step, 0])
            lines.append([len(pts) - 2, len(pts) - 1])
            pts.append([-n * step, i * step, 0])
            pts.append([n * step, i * step, 0])
            lines.append([len(pts) - 2, len(pts) - 1])
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(pts)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector([[0.21, 0.23, 0.30]] * len(lines))
        if self._present:
            self._scene.scene.remove_geometry(self.NAME)
        self._scene.scene.add_geometry(self.NAME, grid, self._make_material())
        self._present = True

    def clear(self):
        if self._present:
            self._scene.scene.remove_geometry(self.NAME)
            self._present = False
        self._half_extent = None


class VehicleView:
    """Small oriented cone, mirrors lightning-lm's UiCar. See PointCloudView's
    docstring for why a fresh MaterialRecord is built every update() instead of
    reused."""

    NAME = "vehicle"

    def __init__(self, scene):
        self._scene = scene
        self._base = o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=1.4)
        self._base.compute_vertex_normals()
        # cone's axis is +Z by default; rotate so it points along +X (forward)
        self._base.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)),
                           center=(0, 0, 0))
        self._present = False

    def _make_material(self):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = (AMBER[0], AMBER[1], AMBER[2], 1.0)
        return mat

    def update(self, pos, quat):
        if pos is None:
            return
        mesh = o3d.geometry.TriangleMesh(self._base)
        if quat is not None:
            x, y, z, w = quat
            R = o3d.geometry.get_rotation_matrix_from_quaternion((w, x, y, z))
            mesh.rotate(R, center=(0, 0, 0))
        mesh.translate(pos)
        if self._present:
            self._scene.scene.remove_geometry(self.NAME)
        self._scene.scene.add_geometry(self.NAME, mesh, self._make_material())
        self._present = True

    def clear(self):
        if self._present:
            self._scene.scene.remove_geometry(self.NAME)
            self._present = False


APP_NAME = "Ra-LIO-DD"  # Radar-LiDAR-Inertial Odometry under Directional Degeneracy --
# ours, not upstream's trail_rviz.rviz; matches the project's own research title
# (README.md / docs/TRaIL_Consistency_Research_Guideline_v2.md).

# Baked into our own title string, under our own control, rather than relying on
# mutter's automatic "(on <hostname>)" window-decoration annotation -- that text is
# hardcoded inside GNOME's window manager (not something a client app can reword), and
# run_native_viewer.sh now sets the container's hostname to match the host machine's own
# so that annotation doesn't also get added on top of this one.
HOST_LABEL = "ZJNU"


def _window_title(sequence):
    return f"{APP_NAME} -- {sequence} ({HOST_LABEL})"


# ---------------------------------------------------------------------------
# Design system. One stable accent per data family, used consistently by the
# panel, the legend and the 3D scene so color always means the same thing:
#   cyan = live estimate   magenta = ground truth   teal = run control
#   amber = method / IMU   violet = map stats       red/green = stop / live
# ---------------------------------------------------------------------------
INK      = (0.93, 0.94, 0.97)   # primary text
MUTED    = (0.55, 0.58, 0.67)   # captions
FAINT    = (0.38, 0.41, 0.50)   # footers / notes
PANEL_BG = (0.072, 0.078, 0.108)
SCENE_BG = (0.043, 0.048, 0.072)

CYAN    = (0.13, 0.83, 0.93)
TEAL    = (0.18, 0.80, 0.72)
MAGENTA = (0.95, 0.42, 0.72)
AMBER   = (0.97, 0.73, 0.24)
RED     = (0.93, 0.40, 0.40)
GREEN   = (0.30, 0.85, 0.50)
VIOLET  = (0.60, 0.53, 0.96)


def _txt(text, color=INK, bg=None):
    lbl = gui.Label(text)
    lbl.text_color = gui.Color(*color)
    if bg is not None:
        lbl.background_color = gui.Color(*bg)
    return lbl


# NOTE on rendering support in this Open3D build (0.19, verified on-host): only
# Button honors background_color (Label/Vert/Horiz backgrounds silently don't
# draw), and the default font has no geometric glyphs (U+25CF/U+25A0/U+258D all
# render as "?"). All accents below are therefore plain colored TEXT, plus
# colored buttons -- the only two channels that actually render.

def _status(text, color):
    return _txt(text, color)


def _section(text, accent, em):
    row = gui.Horiz(0.5 * em)
    row.add_child(_txt(text.upper(), accent))
    row.add_stretch()
    return row


def _button(text, color, tooltip=None):
    btn = gui.Button(text)
    btn.background_color = gui.Color(*color)
    if tooltip:
        btn.tooltip = tooltip
    return btn


def _kpi_card(caption, accent, em):
    """Accent-colored caption + value; the value label is returned so the tick
    loop can update it in place."""
    col = gui.Vert(0.1 * em)
    col.add_child(_txt(caption.upper(), accent))
    value = _txt("--")
    col.add_child(value)
    return col, value


def _load_ref_ate(csv_path):
    """The benchmark numbers from results/summaries/eval_traj_prism.csv, keyed
    (variant, sequence) -> (mean_rmse, n_reps, min, max). These come from headless
    run_seq.sh reps evaluated on the final spline -- the numbers the panel's live
    and final readouts are compared against."""
    reps = {}
    try:
        with open(csv_path, newline='') as f:
            for r in csv.DictReader(f):
                reps.setdefault((r['variant'], r['sequence']), []).append(float(r['ate_rmse']))
    except (OSError, KeyError, ValueError):
        return {}
    return {k: (sum(v) / len(v), len(v), min(v), max(v)) for k, v in reps.items()}


class ViewerApp:
    COLOR_VIEW = CYAN
    COLOR_STATS = TEAL
    COLOR_GT = MAGENTA     # matches gt_path_view's magenta
    COLOR_IMU = AMBER
    COLOR_RESET = RED

    def __init__(self, bridge: RosBridge, sequence: str, repo_root: str, start_time: int = 20):
        self.bridge = bridge
        self.sequence = sequence
        self.repo_root = repo_root
        self.start_time = start_time
        self.runner = SequenceRunner()
        self.aligner = GtAligner(self._gt_path_for(sequence))
        self._gt_drawn_version = -1
        self._method = os.environ.get('TRAIL_VIEWER_METHOD', 'V4')
        self._ref_ate = _load_ref_ate(os.path.join(
            repo_root, 'results', 'summaries', 'eval_traj_prism.csv'))
        self._awaiting_final_ate = False
        self.window = gui.Application.instance.create_window(
            _window_title(sequence), 1400, 900)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_window_close)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([SCENE_BG[0], SCENE_BG[1], SCENE_BG[2], 1.0])
        self.scene_widget.scene.scene.set_sun_light(
            [0.3, 0.3, -1.0], [1.0, 1.0, 1.0], 60000)
        self.window.add_child(self.scene_widget)

        self.grid_view = GridView(self.scene_widget)
        self.grid_view.update_for_bounds(20.0)

        self.cloud_view = PointCloudView(self.scene_widget)
        # Bright cyan vs magenta: both read clearly against the cloud's warm
        # green/yellow/red age palette, and are unambiguous against each other.
        self.path_view = TrajectoryView(self.scene_widget, "path", (0.0, 0.85, 1.0),
                                         line_width=5.0)
        self.gt_path_view = TrajectoryView(self.scene_widget, "gt_path", (1.0, 0.1, 0.85),
                                            line_width=5.0)
        self.vehicle_view = VehicleView(self.scene_widget)

        self.scene_widget.setup_camera(
            60.0, o3d.geometry.AxisAlignedBoundingBox([-20, -20, -5], [20, 20, 5]),
            [0, 0, 0])

        self._sequence_combo = gui.Combobox()
        for seq in SEQUENCES:
            self._sequence_combo.add_item(seq)
        if sequence in SEQUENCES:
            self._sequence_combo.selected_index = SEQUENCES.index(sequence)
        self._sequence_combo.tooltip = "Sequence to launch from this panel"
        self._run_status = _txt("idle", MUTED)
        btn_run = _button("Run sequence", self.COLOR_STATS,
                          tooltip="Start the selected sequence with the current method flags")
        btn_run.set_on_clicked(self._on_run_clicked)
        btn_stop = _button("Stop", self.COLOR_RESET, tooltip="Stop the running sequence")
        btn_stop.set_on_clicked(self._on_stop_clicked)

        self._follow = gui.Checkbox("Follow vehicle")
        self._follow.checked = False
        self._follow.tooltip = "Camera chases the vehicle marker"
        self._show_path = gui.Checkbox("Show estimated path")
        self._show_path.checked = True
        self._show_path.set_on_checked(self._on_toggle_path)
        self._show_gt = gui.Checkbox("Show GT path")
        self._show_gt.checked = True
        self._show_gt.set_on_checked(self._on_toggle_gt)
        self._point_size = gui.Slider(gui.Slider.DOUBLE)
        self._point_size.set_limits(1.0, 6.0)
        self._point_size.double_value = 3.0
        self._point_size.set_on_value_changed(self._on_point_size)

        btn_top = _button("Top", self.COLOR_VIEW, tooltip="Orthographic-style top-down view")
        btn_top.set_on_clicked(lambda: self._set_view("top"))
        btn_side = _button("Side", self.COLOR_VIEW, tooltip="Oblique side view")
        btn_side.set_on_clicked(lambda: self._set_view("side"))
        btn_reset = _button("Reset", self.COLOR_RESET, tooltip="Back to the default side view")
        btn_reset.set_on_clicked(self._on_reset_view)

        em = self.window.theme.font_size
        panel = gui.ScrollableVert(0.55 * em,
                                   gui.Margins(0.9 * em, 0.9 * em, 0.9 * em, 0.9 * em))
        panel.background_color = gui.Color(*PANEL_BG)

        # Header: app identity, host tag, live-status + method dots.
        title_row = gui.Horiz(0.5 * em)
        title_row.add_child(_txt(APP_NAME, CYAN))
        title_row.add_stretch()
        title_row.add_child(_txt(HOST_LABEL, CYAN))
        panel.add_child(title_row)
        panel.add_child(_txt("Radar-LiDAR-Inertial Odometry - Directional Degeneracy", MUTED))
        chip_row = gui.Horiz(0.9 * em)
        self._hdr_status = _status("IDLE", FAINT)
        chip_row.add_child(self._hdr_status)
        # set by run_native_viewer.sh's 3rd argument; the research flags it maps to are in
        # this process's env, so every run started from the panel uses that method
        chip_row.add_child(_status(f"METHOD {self._method}", AMBER))
        chip_row.add_stretch()
        self._stat_running = _txt(sequence, MUTED)
        chip_row.add_child(self._stat_running)
        panel.add_child(chip_row)
        panel.add_fixed(0.25 * em)

        # Telemetry KPI cards.
        panel.add_child(_section("Telemetry", VIOLET, em))
        kpi_row1 = gui.Horiz(0.5 * em)
        card, self._stat_elapsed = _kpi_card("elapsed", CYAN, em)
        kpi_row1.add_child(card)
        card, self._stat_path_len = _kpi_card("path length", TEAL, em)
        kpi_row1.add_child(card)
        panel.add_child(kpi_row1)
        kpi_row2 = gui.Horiz(0.5 * em)
        card, self._stat_points = _kpi_card("map points", VIOLET, em)
        kpi_row2.add_child(card)
        card, self._stat_ate = _kpi_card(f"ATE - {self._method}", MAGENTA, em)
        self._stat_ate.tooltip = ("Benchmark ATE for this method+sequence, from "
                                  "eval_traj_prism.csv (headless reps, final spline). "
                                  "This run's own score appears under FINAL ATE once "
                                  "the bag finishes.")
        kpi_row2.add_child(card)
        panel.add_child(kpi_row2)
        panel.add_fixed(0.25 * em)

        # Sequence control.
        panel.add_child(_section("Sequence", self.COLOR_STATS, em))
        panel.add_child(self._sequence_combo)
        run_row = gui.Horiz(0.5 * em)
        run_row.add_child(btn_run)
        run_row.add_child(btn_stop)
        panel.add_child(run_row)
        panel.add_child(self._run_status)
        panel.add_fixed(0.25 * em)

        # View controls.
        panel.add_child(_section("View", self.COLOR_VIEW, em))
        panel.add_child(self._follow)
        panel.add_child(self._show_path)
        panel.add_child(self._show_gt)
        panel.add_child(_txt("Point size", MUTED))
        panel.add_child(self._point_size)
        view_row = gui.Horiz(0.5 * em)
        view_row.add_child(btn_top)
        view_row.add_child(btn_side)
        view_row.add_child(btn_reset)
        panel.add_child(view_row)
        panel.add_fixed(0.25 * em)

        # Ground truth: status, live drift, method note. Value stacked under its
        # caption (right-aligning it in a Horiz let the label collapse to ~1 char
        # wide and wrap vertically -- seen live on Park1).
        panel.add_child(_section("Ground truth", self.COLOR_GT, em))
        self._stat_gt_status = _txt("loading...", MUTED)
        panel.add_child(self._stat_gt_status)
        panel.add_child(_txt("ERROR NOW", self.COLOR_GT))
        self._stat_drift = _txt("--")
        panel.add_child(self._stat_drift)
        # Live filter readout -- a diagnostic, NOT the benchmark number: it scores
        # the in-window filter state, so it runs higher than the final spline.
        panel.add_child(_txt("LIVE ATE - FILTER STATE", MUTED))
        self._stat_live_ate = _txt("--")
        panel.add_child(self._stat_live_ate)
        # The benchmark number for this method+sequence (headless reps, final
        # spline) -- what the run should be compared against.
        panel.add_child(_txt(f"REF ATE - {self._method}", MUTED))
        self._stat_ref_ate = _txt("--")
        panel.add_child(self._stat_ref_ate)
        # Filled in when the bag finishes: this run's own poses.txt scored with
        # the same prism-point rigid-fit recipe as the reference csv.
        panel.add_child(_txt("FINAL ATE - THIS RUN", MUTED))
        self._stat_final_ate = _txt("waiting for run to finish...", FAINT)
        panel.add_child(self._stat_final_ate)
        self._drift_note = _txt(
            "Live numbers: prism point, rigid fit on all data so far --\n"
            "filter state, not the final spline (see REF/FINAL above).",
            FAINT)
        panel.add_child(self._drift_note)
        panel.add_fixed(0.25 * em)

        # Raw IMU. Values stacked under captions like ERROR NOW: right-aligned
        # 3-axis values can hit the panel edge and wrap mid-number (seen live
        # on the accel row).
        panel.add_child(_section("IMU - raw /gt_box/cpt7", self.COLOR_IMU, em))
        panel.add_child(_txt("GYRO - RAD/S", MUTED))
        self._stat_imu_gyro = _txt("--")
        panel.add_child(self._stat_imu_gyro)
        panel.add_child(_txt("ACCEL - M/S2", MUTED))
        self._stat_imu_accel = _txt("--")
        panel.add_child(self._stat_imu_accel)
        panel.add_fixed(0.25 * em)

        # Legend: words carry the exact scene colors (font has no swatch glyphs).
        panel.add_child(_section("Legend", FAINT, em))
        age_row = gui.Horiz(0.4 * em)
        age_row.add_child(_txt("cloud age:", MUTED))
        age_row.add_stretch()
        age_row.add_child(_txt("fresh", (0.0, 0.8, 0.2)))
        age_row.add_child(_txt("aging", (1.0, 0.8, 0.0)))
        age_row.add_child(_txt("40s+ drift risk", (1.0, 0.0, 0.0)))
        panel.add_child(age_row)
        path_row = gui.Horiz(0.4 * em)
        path_row.add_child(_txt("paths:", MUTED))
        path_row.add_stretch()
        path_row.add_child(_txt("estimate", (0.0, 0.85, 1.0)))
        path_row.add_child(_txt("GT", (1.0, 0.1, 0.85)))
        path_row.add_child(_txt("vehicle", AMBER))
        panel.add_child(path_row)
        panel.add_fixed(0.25 * em)
        panel.add_child(_txt("TRaIL consistency research build - ZJNU", FAINT))
        self.panel = panel
        self.window.add_child(panel)

        self._panel_width = 26 * em
        self._last_cloud_version = -1
        self._last_path_version = -1
        self._last_vehicle_version = -1
        self._last_imu_version = -1
        self._update_ref_ate()
        self.window.set_on_tick_event(self._on_tick)

    def _gt_path_for(self, sequence):
        return os.path.join(self.repo_root, "results", "summaries", "gt", f"{sequence}_gt.tum")

    def _on_run_clicked(self):
        seq = self._sequence_combo.selected_text
        if not seq:
            return
        self._run_status.text = f"starting {seq}..."
        # Clear every view and the GT alignment state so nothing from the previous
        # sequence (or previous run of the same sequence) lingers on screen or biases
        # the new run's alignment fit.
        self.cloud_view.clear()
        self.path_view.clear()
        self.gt_path_view.clear()
        self.vehicle_view.clear()
        self.grid_view.clear()
        self.grid_view.update_for_bounds(20.0)
        self._gt_drawn_version = -1
        self.aligner = GtAligner(self._gt_path_for(seq))
        self.bridge.reset_for_new_run()
        self._last_cloud_version = -1
        self._last_path_version = -1
        self._last_vehicle_version = -1
        self._last_imu_version = -1

        self.sequence = seq
        self.window.title = _window_title(seq)
        self._stat_running.text = seq
        self._stat_final_ate.text = "waiting for run to finish..."
        self._stat_final_ate.text_color = gui.Color(*FAINT)
        self._awaiting_final_ate = True
        self._update_ref_ate()
        self.runner.start(seq, start_time=self.start_time)
        self._set_live(True)
        self._run_status.text = f"running {seq}"

    def _on_stop_clicked(self):
        self.runner.stop()
        self._run_status.text = "stopped"
        self._set_live(False)
        self._stat_running.text = "none"

    def _set_live(self, live):
        if live:
            self._hdr_status.text = "LIVE"
            self._hdr_status.text_color = gui.Color(*GREEN)
        else:
            self._hdr_status.text = "IDLE"
            self._hdr_status.text_color = gui.Color(*FAINT)

    def _update_ref_ate(self):
        key = (self._method, self.sequence)
        if key in self._ref_ate:
            mean, n, lo, hi = self._ref_ate[key]
            self._stat_ref_ate.text = f"{mean:.3f} m  (reps {lo:.3f}-{hi:.3f}, n={n})"
            self._stat_ate.text = f"{mean:.3f} m"
            self._stat_ate.text_color = gui.Color(*INK)
        else:
            self._stat_ref_ate.text = "-- (no recorded reps for this method)"
            self._stat_ate.text = "--"

    def _final_ate(self):
        """This run's poses.txt scored with the reference recipe (prism point,
        rigid Kabsch fit, np.interp association). Validated against evo_ape's
        trans_part/SE(3)-alignment result on the same file: 0.404 vs 0.407 m on
        Park1 -- the 3 mm gap is evo's timestamp-association detail."""
        poses = os.path.join(LIVE_VIZ_DIR, self.sequence, 'trail_output', 'poses.txt')
        gt_path = self._gt_path_for(self.sequence)
        try:
            gt_t, gt_p = load_gt_tum(gt_path)
            lever = load_prism_lever(gt_path)
            a = np.loadtxt(poses)
        except OSError:
            return None
        a = a[np.argsort(a[:, 0])]
        est_p = a[:, 1:4]
        if lever is not None:
            rots = np.stack([quat_to_rot(q) for q in a[:, 4:8]])
            est_p = est_p + np.einsum('nij,j->ni', rots, lever)
        in_range = (a[:, 0] >= gt_t[0]) & (a[:, 0] <= gt_t[-1])
        if in_range.sum() < 2:
            return None
        est_p, est_t = est_p[in_range], a[:, 0][in_range]
        gt_at = np.stack([np.interp(est_t, gt_t, gt_p[:, i]) for i in range(3)], -1)
        R, t = kabsch(est_p, gt_at)
        err = est_p @ R.T + t - gt_at
        return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

    def _show_final_ate(self):
        self._awaiting_final_ate = False
        ate = self._final_ate()
        if ate is None:
            self._stat_final_ate.text = "-- (no poses.txt)"
            self._stat_final_ate.text_color = gui.Color(*MUTED)
            return
        self._stat_final_ate.text = f"{ate:.3f} m"
        key = (self._method, self.sequence)
        if key in self._ref_ate:
            _, _, lo, hi = self._ref_ate[key]
            # green = at least as good as the worst recorded rep (beating the
            # benchmark is good, not a warning); amber = within 1.5x of it
            color = GREEN if ate <= hi else (AMBER if ate <= hi * 1.5 else RED)
            self._stat_final_ate.text_color = gui.Color(*color)
        # the headline KPI stays pinned to the benchmark number; this run's own
        # score lives in the FINAL ATE row above
        self._set_live(False)
        self._run_status.text = "finished"

    def _on_window_close(self):
        self.runner.stop()
        return True

    def _on_point_size(self, value):
        self.cloud_view.set_point_size(value)

    def _on_toggle_path(self, checked):
        self.path_view.set_visible(checked)

    def _on_toggle_gt(self, checked):
        self.gt_path_view.set_visible(checked)

    def _set_view(self, mode):
        if mode == "top":
            self.scene_widget.look_at([0, 0, 0], [0.01, 0.01, 40], [0, 1, 0])
        elif mode == "side":
            self.scene_widget.look_at([0, 0, 0], [30, 0, 10], [0, 0, 1])

    def _on_reset_view(self):
        self._follow.checked = False
        self._set_view("side")

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.panel.frame = gui.Rect(r.get_right() - self._panel_width, r.y,
                                     self._panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - self._panel_width, r.height)

    def _on_tick(self):
        snap = self.bridge.state.snapshot()

        # Only rebuild geometry when the underlying ROS data actually changed --
        # set_on_tick_event fires every render frame (tens of times/sec), far faster
        # than new /local_map_aged messages arrive, so rebuilding unconditionally was
        # redoing the same ~90k-point geometry dozens of times per real update and
        # blocking the GUI thread (mouse input is handled on this same thread).
        has_points = snap["points"] is not None and len(snap["points"]) > 0
        if has_points and snap["cloud_version"] != self._last_cloud_version:
            self.cloud_view.update(snap["points"], snap["ages"])
            self._last_cloud_version = snap["cloud_version"]
            xy = np.asarray(snap["points"], dtype=np.float32)[:, :2]
            self.grid_view.update_for_bounds(float(np.abs(xy).max()))

        if snap["path"] and snap["path_version"] != self._last_path_version:
            self.path_view.update(snap["path"])
            self._last_path_version = snap["path_version"]

        if snap["vehicle_pos"] is not None and snap["vehicle_version"] != self._last_vehicle_version:
            self.vehicle_view.update(snap["vehicle_pos"], snap["vehicle_quat"])
            self._last_vehicle_version = snap["vehicle_version"]
            if self._follow.checked:
                p = snap["vehicle_pos"]
                self.scene_widget.look_at(list(p), [p[0] - 15, p[1] - 15, p[2] + 10], [0, 0, 1])

            if snap["vehicle_abs_t"] is not None:
                self.aligner.add_sample(snap["vehicle_abs_t"], snap["vehicle_pos"], snap["vehicle_quat"])
                if self.aligner.status == "ready" and self.aligner.version != self._gt_drawn_version:
                    # redraw on every refit: GT up to now, in the latest alignment
                    self.gt_path_view.update(self.aligner.gt_path_in_est_frame())
                    self._gt_drawn_version = self.aligner.version
                self._update_gt_labels(snap)

        if snap["imu_gyro"] is not None and snap["imu_version"] != self._last_imu_version:
            self._last_imu_version = snap["imu_version"]
            gx, gy, gz = snap["imu_gyro"]
            ax, ay, az = snap["imu_accel"]
            self._stat_imu_gyro.text = f"{gx:+.3f} {gy:+.3f} {gz:+.3f}"
            self._stat_imu_accel.text = f"{ax:+.2f} {ay:+.2f} {az:+.2f}"

        self._stat_elapsed.text = f"{snap['elapsed_s']:.1f} s"
        self._stat_path_len.text = f"{snap['path_length_m']:.1f} m"
        self._stat_points.text = f"{snap['n_points']:,}"

        # Natural end of the bag (+ grace period): poses.txt is complete, score it
        # with the reference recipe so the panel shows the benchmark-comparable ATE.
        if self._awaiting_final_ate and self.runner.finished_naturally():
            self._show_final_ate()
        return True

    def _update_gt_labels(self, snap):
        status = self.aligner.status
        if status == "unavailable":
            self._stat_gt_status.text = "unavailable for this sequence"
            self._stat_gt_status.text_color = gui.Color(*MUTED)
            self._stat_drift.text = "--"
            self._stat_drift.text_color = gui.Color(*MUTED)
            self._stat_live_ate.text = "--"
            return
        if status == "collecting":
            self._stat_gt_status.text = "collecting alignment samples..."
            self._stat_gt_status.text_color = gui.Color(*MUTED)
            self._stat_drift.text = "--"
            self._stat_drift.text_color = gui.Color(*MUTED)
            self._stat_live_ate.text = "--"
            return
        self._stat_gt_status.text = "aligned - refit on all data so far"
        self._stat_gt_status.text_color = gui.Color(*GREEN)
        if self.aligner.ate_so_far is not None:
            self._stat_live_ate.text = f"{self.aligner.ate_so_far:.3f} m"
        drift = self.aligner.live_error(snap["vehicle_abs_t"], snap["vehicle_pos"], snap["vehicle_quat"])
        if drift is not None:
            self._stat_drift.text = f"{drift:.3f} m"
            # traffic-light readout: green is nominal, amber needs attention, red
            # means the estimate has visibly left the GT path
            color = GREEN if drift < 0.25 else (AMBER if drift < 0.5 else RED)
            self._stat_drift.text_color = gui.Color(*color)
        else:
            # Not a bug: the estimate can run past where GT's own recorded time range
            # ends (verified on Park1 -- the estimate continues ~47s beyond it), and
            # showing a clamped drift there would repeat the exact bug that corrupted
            # the alignment fit in the first place.
            self._stat_drift.text = "-- (past GT range)"
            self._stat_drift.text_color = gui.Color(*MUTED)


def main():
    initial_sequence = sys.argv[1] if len(sys.argv) > 1 else SEQUENCES[0]
    if initial_sequence not in SEQUENCES:
        initial_sequence = SEQUENCES[0]
    start_time = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    bridge = RosBridge()
    bridge.start()

    gui.Application.instance.initialize()
    app = ViewerApp(bridge, initial_sequence, repo_root, start_time=start_time)
    # Auto-start the sequence passed on the command line (keeps `run_native_viewer.sh
    # Park1` working the same as before); the dropdown + Run button let the user switch
    # to a different sequence afterward without relaunching the whole app.
    app.runner.start(initial_sequence, start_time=start_time)
    app._stat_running.text = initial_sequence
    app._set_live(True)
    app._awaiting_final_ate = True
    app._run_status.text = f"running {initial_sequence}"
    gui.Application.instance.run()

    app.runner.stop()
    bridge.stop()
    # rmw_zenoh's runtime aborts during static teardown at interpreter exit
    # ("terminate called without an active exception", seen on every window close).
    # The run is fully done at this point -- skip the teardown.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
