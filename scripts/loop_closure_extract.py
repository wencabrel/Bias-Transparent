#!/usr/bin/env python3
"""Phase A of loop_closure.py's loop-closure correction: extract start/end LiDAR point
cloud windows from the rosbag, in the ROS2/rosbag2_py environment (must run inside the
chiyunnn/trail-odom:latest container -- numpy+rosbag2_py there, no open3d/scipy).

Writes <results_dir>/trail_output/loop_closure_{start,end}.npy (world-frame-relative
point clouds) and loop_closure_meta.npz (reference poses) for loop_closure.py (Phase B,
runs on the host with open3d/scipy) to pick up.

See scripts/loop_closure.py's module docstring for the full method and why it's split
this way.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path("/root/project")


def load_tum(path):
    ts, xyz, quat = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            ts.append(float(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            quat.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
    return np.array(ts), np.array(xyz), np.array(quat)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def pose_at(ts, xyz, quat, t_query):
    idx = np.searchsorted(ts, t_query)
    idx = np.clip(idx, 0, len(ts) - 1)
    return xyz[idx], quat[idx]


def load_calib(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    calib = cfg["Configor"]["DataStream"]["CalibParam"]
    q = calib["SO3_LtoB"]
    so3_ltob = quat_to_R([q["qx"], q["qy"], q["qz"], q["qw"]])
    pos_linb = np.array([calib["POS_LinB"]["r0c0"], calib["POS_LinB"]["r1c0"], calib["POS_LinB"]["r2c0"]])
    return so3_ltob, pos_linb


def extract_window_cloud(bagdir, t0_wall, t1_wall, ts_traj, xyz_traj, quat_traj,
                          so3_ltob, pos_linb, ref_xyz, ref_quat, max_points=200000):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2

    R_ref_inv = quat_to_R(ref_quat).T
    p_ref = ref_xyz

    storage_options = rosbag2_py.StorageOptions(uri=str(bagdir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic = "/gt_box/livox/lidar"

    pts_accum = []
    total = 0
    while reader.has_next():
        (t, data, bag_ts) = reader.read_next()
        if t != topic:
            continue
        t_s = bag_ts * 1e-9
        if t_s < t0_wall or t_s > t1_wall:
            continue
        msg = deserialize_message(data, PointCloud2)
        pts = np.array([[p[0], p[1], p[2]] for p in
                         pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
        if len(pts) == 0:
            continue
        pts_body = (so3_ltob @ pts.T).T + pos_linb
        xyz_w, quat_w = pose_at(ts_traj, xyz_traj, quat_traj, t_s)
        R_w = quat_to_R(quat_w)
        pts_world = (R_w @ pts_body.T).T + xyz_w
        pts_ref = (R_ref_inv @ (pts_world - p_ref).T).T
        pts_accum.append(pts_ref)
        total += len(pts_ref)
        if total > max_points:
            break
    if not pts_accum:
        return np.zeros((0, 3))
    return np.concatenate(pts_accum, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--rep", required=True, type=int)
    ap.add_argument("--window-s", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = REPO_ROOT / "results" / args.variant / args.sequence / f"rep{args.rep}" / "trail_output"
    poses_path = out_dir / "poses.txt"
    bag_dir = REPO_ROOT / "data" / "Boxi_ros2_sqlite" / args.sequence
    config_path = REPO_ROOT / "configs" / "base" / "boxi.yaml"

    ts, xyz, quat = load_tum(poses_path)
    print(f"loaded {len(ts)} poses from {poses_path}")
    so3_ltob, pos_linb = load_calib(config_path)

    t0, t1 = ts[0], ts[-1]
    ref_xyz0, ref_quat0 = xyz[0], quat[0]
    ref_xyz1, ref_quat1 = xyz[-1], quat[-1]

    print("extracting start-window point cloud...")
    cloud_start = extract_window_cloud(bag_dir, t0, t0 + args.window_s, ts, xyz, quat,
                                        so3_ltob, pos_linb, ref_xyz0, ref_quat0)
    print(f"  {len(cloud_start)} points")
    print("extracting end-window point cloud...")
    cloud_end = extract_window_cloud(bag_dir, t1 - args.window_s, t1, ts, xyz, quat,
                                      so3_ltob, pos_linb, ref_xyz1, ref_quat1)
    print(f"  {len(cloud_end)} points")

    if len(cloud_start) < 100 or len(cloud_end) < 100:
        print("ERROR: not enough points extracted for ICP", file=sys.stderr)
        sys.exit(1)

    np.save(out_dir / "loop_closure_start.npy", cloud_start)
    np.save(out_dir / "loop_closure_end.npy", cloud_end)
    np.savez(out_dir / "loop_closure_meta.npz",
             ref_xyz0=ref_xyz0, ref_quat0=ref_quat0, ref_xyz1=ref_xyz1, ref_quat1=ref_quat1)
    print(f"wrote loop_closure_{{start,end}}.npy and loop_closure_meta.npz to {out_dir}")


if __name__ == "__main__":
    main()
