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
import os
import sys
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ros_bridge import RosBridge
from gt_tools import GtAligner
from sequence_runner import SequenceRunner, SEQUENCES

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
        grid.colors = o3d.utility.Vector3dVector([[0.25, 0.25, 0.28]] * len(lines))
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
        mat.base_color = (0.95, 0.35, 0.1, 1.0)
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


def _section_header(text, color):
    lbl = gui.Label(text)
    lbl.background_color = gui.Color(*color, 0.9)
    lbl.text_color = gui.Color(1.0, 1.0, 1.0)
    return lbl


def _colored_button(text, color):
    btn = gui.Button(text)
    btn.background_color = gui.Color(*color)
    return btn


class ViewerApp:
    # one accent color per panel section, reused for its header and related buttons
    COLOR_VIEW = (0.20, 0.55, 0.95)
    COLOR_STATS = (0.10, 0.70, 0.55)
    COLOR_GT = (0.95, 0.15, 0.70)      # matches gt_path_view's magenta
    COLOR_IMU = (0.95, 0.60, 0.15)
    COLOR_RESET = (0.90, 0.30, 0.25)

    def __init__(self, bridge: RosBridge, sequence: str, repo_root: str, start_time: int = 20):
        self.bridge = bridge
        self.sequence = sequence
        self.repo_root = repo_root
        self.start_time = start_time
        self.runner = SequenceRunner()
        self.aligner = GtAligner(self._gt_path_for(sequence))
        self._gt_drawn_version = -1
        self.window = gui.Application.instance.create_window(
            _window_title(sequence), 1280, 800)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_window_close)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.07, 0.07, 0.09, 1.0])
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
        self._run_status = gui.Label("not running")
        btn_run = _colored_button("Run selected sequence", self.COLOR_STATS)
        btn_run.set_on_clicked(self._on_run_clicked)
        btn_stop = _colored_button("Stop", self.COLOR_RESET)
        btn_stop.set_on_clicked(self._on_stop_clicked)

        self._follow = gui.Checkbox("Follow vehicle")
        self._follow.checked = False
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

        self._stat_elapsed = gui.Label("elapsed: --")
        self._stat_path_len = gui.Label("path length: --")
        self._stat_points = gui.Label("map points: --")
        self._legend = gui.Label("cloud color: green=fresh -> red=40s+/drift-affected")
        self._path_legend = gui.Label("path color: cyan=estimate, magenta=GT")

        self._stat_gt_status = gui.Label("GT: loading...")
        self._stat_drift = gui.Label("error now: --")
        self._stat_ate = gui.Label("ATE so far: --")
        self._drift_note = gui.Label(
            "(prism point, rigid fit on all data so far -- same method as\n"
            " results/summaries/eval_traj_prism.csv; live state, not final spline)")

        self._stat_imu_gyro = gui.Label("gyro: -- rad/s")
        self._stat_imu_accel = gui.Label("accel: -- m/s2")

        btn_top = _colored_button("Top view", self.COLOR_VIEW)
        btn_top.set_on_clicked(lambda: self._set_view("top"))
        btn_side = _colored_button("Side view", self.COLOR_VIEW)
        btn_side.set_on_clicked(lambda: self._set_view("side"))
        btn_reset = _colored_button("Reset view", self.COLOR_RESET)
        btn_reset.set_on_clicked(self._on_reset_view)

        em = self.window.theme.font_size
        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        title = gui.Label(APP_NAME)
        title.text_color = gui.Color(0.0, 0.85, 1.0)
        panel.add_child(title)
        self._stat_running = gui.Label(f"running: {sequence}")
        self._stat_running.text_color = gui.Color(0.0, 0.85, 1.0)
        panel.add_child(self._stat_running)
        # set by run_native_viewer.sh's 3rd argument; the research flags it maps to are in
        # this process's env, so every run started from the panel uses that method
        method_lbl = gui.Label(f"method: {os.environ.get('TRAIL_VIEWER_METHOD', 'V4')}")
        method_lbl.text_color = gui.Color(1.0, 0.75, 0.2)
        panel.add_child(method_lbl)
        panel.add_fixed(0.5 * em)
        panel.add_child(_section_header("Sequence", self.COLOR_STATS))
        panel.add_child(self._sequence_combo)
        panel.add_child(btn_run)
        panel.add_child(btn_stop)
        panel.add_child(self._run_status)
        panel.add_fixed(0.5 * em)
        panel.add_child(_section_header("View", self.COLOR_VIEW))
        panel.add_child(self._follow)
        panel.add_child(self._show_path)
        panel.add_child(self._show_gt)
        panel.add_child(gui.Label("Point size"))
        panel.add_child(self._point_size)
        panel.add_child(btn_top)
        panel.add_child(btn_side)
        panel.add_child(btn_reset)
        panel.add_fixed(0.5 * em)
        panel.add_child(_section_header("Stats", self.COLOR_STATS))
        panel.add_child(self._stat_elapsed)
        panel.add_child(self._stat_path_len)
        panel.add_child(self._stat_points)
        panel.add_fixed(0.5 * em)
        panel.add_child(_section_header("Ground truth", self.COLOR_GT))
        panel.add_child(self._stat_gt_status)
        panel.add_child(self._stat_drift)
        panel.add_child(self._stat_ate)
        panel.add_child(self._drift_note)
        panel.add_fixed(0.5 * em)
        panel.add_child(_section_header("IMU (raw, /gt_box/cpt7/.../imu)", self.COLOR_IMU))
        panel.add_child(self._stat_imu_gyro)
        panel.add_child(self._stat_imu_accel)
        panel.add_fixed(0.5 * em)
        panel.add_child(self._legend)
        panel.add_child(self._path_legend)
        self.panel = panel
        self.window.add_child(panel)

        self._panel_width = 22 * em
        self._last_cloud_version = -1
        self._last_path_version = -1
        self._last_vehicle_version = -1
        self._last_imu_version = -1
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
        self._stat_running.text = f"running: {seq}"
        self.runner.start(seq, start_time=self.start_time)
        self._run_status.text = f"running {seq}"

    def _on_stop_clicked(self):
        self.runner.stop()
        self._run_status.text = "stopped"
        self._stat_running.text = "running: none"

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
            self._stat_imu_gyro.text = f"gyro: {gx:+.3f} {gy:+.3f} {gz:+.3f} rad/s"
            self._stat_imu_accel.text = f"accel: {ax:+.2f} {ay:+.2f} {az:+.2f} m/s2"

        self._stat_elapsed.text = f"elapsed: {snap['elapsed_s']:.1f} s"
        self._stat_path_len.text = f"path length: {snap['path_length_m']:.1f} m"
        self._stat_points.text = f"map points: {snap['n_points']}"
        return True

    def _update_gt_labels(self, snap):
        status = self.aligner.status
        if status == "unavailable":
            self._stat_gt_status.text = "GT: unavailable for this sequence"
            self._stat_drift.text = "error now: --"
            self._stat_ate.text = "ATE so far: --"
            return
        if status == "collecting":
            self._stat_gt_status.text = "GT: collecting alignment samples..."
            self._stat_drift.text = "error now: --"
            self._stat_ate.text = "ATE so far: --"
            return
        self._stat_gt_status.text = "GT: aligned (refit on all data so far)"
        if self.aligner.ate_so_far is not None:
            self._stat_ate.text = f"ATE so far: {self.aligner.ate_so_far:.3f} m"
        drift = self.aligner.live_error(snap["vehicle_abs_t"], snap["vehicle_pos"], snap["vehicle_quat"])
        if drift is not None:
            self._stat_drift.text = f"error now: {drift:.3f} m"
        else:
            # Not a bug: the estimate can run past where GT's own recorded time range
            # ends (verified on Park1 -- the estimate continues ~47s beyond it), and
            # showing a clamped drift there would repeat the exact bug that corrupted
            # the alignment fit in the first place.
            self._stat_drift.text = "error now: -- (past GT's recorded time range)"


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
    app._stat_running.text = f"running: {initial_sequence}"
    app._run_status.text = f"running {initial_sequence}"
    gui.Application.instance.run()

    app.runner.stop()
    bridge.stop()


if __name__ == "__main__":
    sys.exit(main())
