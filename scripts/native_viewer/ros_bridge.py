"""ROS2 subscriber side of the native viewer: runs rclpy in a background thread,
decodes /path, /axis, /local_map_aged, and hands off thread-safe snapshots to the
Open3D GUI thread via a lock-protected State object (viewer_app.py polls it on a timer).

Mirrors lightning-lm's PangolinWindow::Update* pattern (src/ui/pangolin_window.h) --
a producer (here: ROS callbacks) pushes data into shared state that a separate render
loop consumes, except we use a polled State object instead of their push callbacks,
since Open3D's GUI needs updates issued from its own main thread via
Application.instance.post_to_main_thread.
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import PointCloud2, Imu

# Same for all 6 Boxi sequences, see configs/base/boxi.yaml IMUTopic. Subscribed directly
# to the raw bag topic (not anything trail_node republishes -- it doesn't publish IMU
# data itself) so the panel can show genuine live sensor readings.
IMU_TOPIC = '/gt_box/cpt7/offline_from_novatel_logs/imu'


class State:
    """Thread-safe snapshot of the latest data. Read with .snapshot(), written by
    RosBridge's callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.points = None       # Nx3 float32 list of (x,y,z)
        self.ages = None         # N float32 list, seconds since first seen
        self.path = []           # list of (x,y,z)
        self.vehicle_pos = None  # (x,y,z)
        self.vehicle_quat = None  # (x,y,z,w)
        self.vehicle_abs_t = None  # absolute epoch seconds, for GT time association
        self.elapsed_s = 0.0
        self.path_length_m = 0.0
        self.n_points = 0
        self.imu_gyro = None   # (x,y,z) rad/s
        self.imu_accel = None  # (x,y,z) m/s^2
        self.imu_version = 0
        # Bumped on each ROS-side update; the GUI tick (which fires every render frame,
        # much faster than ROS messages arrive) compares these against what it last
        # rendered and skips rebuilding geometry when nothing actually changed. Without
        # this the full point cloud (tens of thousands of points) was being torn down
        # and rebuilt on every single frame regardless of whether new data had arrived,
        # which is what was blocking the GUI thread and making the mouse unresponsive.
        self.cloud_version = 0
        self.path_version = 0
        self.vehicle_version = 0

    def update_cloud(self, points, ages):
        with self._lock:
            self.points = points
            self.ages = ages
            self.n_points = len(points)
            self.cloud_version += 1

    def update_path(self, path):
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            dz = path[i][2] - path[i - 1][2]
            length += (dx * dx + dy * dy + dz * dz) ** 0.5
        with self._lock:
            self.path = path
            self.path_length_m = length
            self.path_version += 1

    def update_vehicle(self, pos, quat, abs_t, elapsed_s):
        with self._lock:
            self.vehicle_pos = pos
            self.vehicle_quat = quat
            self.vehicle_abs_t = abs_t
            self.elapsed_s = elapsed_s
            self.vehicle_version += 1

    def update_imu(self, gyro, accel):
        with self._lock:
            self.imu_gyro = gyro
            self.imu_accel = accel
            self.imu_version += 1

    def reset(self):
        """Called when the user picks a different sequence to run -- clears stale data
        from the previous run so old geometry doesn't linger on screen until new
        messages arrive."""
        with self._lock:
            self.points = None
            self.ages = None
            self.path = []
            self.vehicle_pos = None
            self.vehicle_quat = None
            self.vehicle_abs_t = None
            self.elapsed_s = 0.0
            self.path_length_m = 0.0
            self.n_points = 0
            self.imu_gyro = None
            self.imu_accel = None
            # version counters intentionally NOT reset -- they only need to keep
            # increasing so the GUI's "did this change since last render" checks still
            # work; resetting them to 0 could collide with the last-rendered values the
            # GUI thread already has and cause a stale frame to go unrefreshed.
            self.cloud_version += 1
            self.path_version += 1
            self.vehicle_version += 1
            self.imu_version += 1

    def snapshot(self):
        with self._lock:
            return dict(
                points=self.points, ages=self.ages, path=list(self.path),
                vehicle_pos=self.vehicle_pos, vehicle_quat=self.vehicle_quat,
                vehicle_abs_t=self.vehicle_abs_t,
                elapsed_s=self.elapsed_s, path_length_m=self.path_length_m,
                n_points=self.n_points,
                cloud_version=self.cloud_version, path_version=self.path_version,
                vehicle_version=self.vehicle_version,
                imu_gyro=self.imu_gyro, imu_accel=self.imu_accel,
                imu_version=self.imu_version,
            )


class _BridgeNode(Node):
    def __init__(self, state: State, path_topic: str, cloud_topic: str):
        super().__init__('native_viewer_bridge')
        self.state = state
        self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, 10)
        self.create_subscription(Path, path_topic, self._on_path, 10)
        self.create_subscription(Odometry, '/axis', self._on_odom, 10)
        self.create_subscription(Imu, IMU_TOPIC, self._on_imu, 10)
        self._t0 = None
        # /axis's header.stamp comes from trail_node's internal state->timestamp
        # (ws/src/TRaIL-Odom/src/core/trail.cpp:390), which is relative (starts near 0
        # at run start) -- verified directly against a live run (first value was
        # 0.70s). GT's .tum file uses absolute Unix epoch time, and trail_node's own
        # poses.txt is epoch-scaled only when written at shutdown (Save()), not in the
        # live-published topics. Rather than guess the exact relative->absolute
        # relationship or touch upstream code, this computes it empirically: the raw
        # IMU topic (subscribed directly from the bag, not anything trail_node
        # republishes) keeps the bag's real original epoch timestamps, and both streams
        # advance in lockstep off the same bag playback clock, so the offset between
        # them is effectively constant. Computed once from the first available pair and
        # frozen, not continuously recomputed.
        self._latest_imu_abs_t = None
        self._epoch_offset = None

    def _on_cloud(self, msg: PointCloud2):
        # Vectorized decode: the previous version did a per-point Python loop with
        # struct.unpack_from, which measured at ~20ms for ~38k points (and climbing
        # linearly with map size) just to build a Python list of tuples -- that list
        # then cost another ~20ms downstream in viewer_app.py converting it back into a
        # numpy array (np.asarray on a list of tuples isn't a fast bulk operation). Both
        # ran on the GUI thread inside the render tick, which is what was blocking mouse
        # input for tens of milliseconds several times a second. Reading the raw buffer
        # straight into a numpy array removes both costs: this is a zero-copy view plus
        # a strided reshape, no per-point Python-level work at all.
        offs = {f.name: f.offset for f in msg.fields}
        if not {'x', 'y', 'z'} <= offs.keys():
            return
        step = msg.point_step
        n = msg.width * msg.height
        if n == 0:
            self.state.update_cloud(np.empty((0, 3), dtype=np.float32),
                                     np.empty(0, dtype=np.float32))
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)

        def field(name):
            off = offs[name]
            # .copy() forces a contiguous buffer before the dtype-changing .view():
            # raw[:, off:off+4] is a column-slice across strided rows, and whether
            # numpy accepts a dtype-size-changing .view() directly on that without an
            # explicit copy turned out to depend on exact buffer/offset details that a
            # synthetic test didn't reproduce -- verified live, not assumed, after it
            # raised "ValueError: To change to a dtype of a different size, the array
            # must be C-contiguous" on an uncopied slice. The copy is a small n*4-byte
            # buffer, nowhere near the cost of the per-point Python loop this replaced.
            return raw[:, off:off + 4].copy().view('<f4').reshape(-1)

        points = np.stack([field('x'), field('y'), field('z')], axis=1)
        ages = field('intensity') if 'intensity' in offs else np.zeros(n, dtype=np.float32)
        self.state.update_cloud(points, ages)

    def _on_path(self, msg: Path):
        path = [(p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in msg.poses]
        self.state.update_path(path)

    def _on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = t
        if self._epoch_offset is None and self._latest_imu_abs_t is not None:
            self._epoch_offset = self._latest_imu_abs_t - t
        abs_t = (t + self._epoch_offset) if self._epoch_offset is not None else None
        pos = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        q = msg.pose.pose.orientation
        self.state.update_vehicle(pos, (q.x, q.y, q.z, q.w), abs_t, t - self._t0)

    def _on_imu(self, msg: Imu):
        self._latest_imu_abs_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w = msg.angular_velocity
        a = msg.linear_acceleration
        self.state.update_imu((w.x, w.y, w.z), (a.x, a.y, a.z))

    def reset_for_new_run(self):
        """A new sequence's trail_node restarts its own relative clock from near-zero,
        so the frozen epoch offset from the previous run no longer applies -- must be
        recomputed against the new run's first /axis message."""
        self._t0 = None
        self._epoch_offset = None


class RosBridge:
    """Owns the rclpy node + spin thread. Call start() once, stop() on shutdown."""

    def __init__(self, path_topic='/path_recent', cloud_topic='/local_map_aged'):
        self.state = State()
        self._path_topic = path_topic
        self._cloud_topic = cloud_topic
        self._thread = None
        self._node = None

    def start(self):
        rclpy.init(args=None)
        self._node = _BridgeNode(self.state, self._path_topic, self._cloud_topic)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        rclpy.spin(self._node)

    def reset_for_new_run(self):
        self.state.reset()
        if self._node is not None:
            self._node.reset_for_new_run()

    def stop(self):
        if self._node is not None:
            self._node.destroy_node()
        rclpy.shutdown()
