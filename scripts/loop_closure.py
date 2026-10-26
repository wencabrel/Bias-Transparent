#!/usr/bin/env python3
"""Post-processing loop-closure correction for TRaIL-Odom trajectories.

TRaIL-Odom is a pure sliding-window continuous-time odometry system: it has no
pose-graph backend and no loop-closure mechanism (docs/DECISIONS.md 2026-10-16,
Airfield1 investigation). Even though its local map retains the whole trajectory on a
short-extent, return-to-start sequence like Airfield1 (voxelmap MaxRange=300m >> the
~87m trajectory bounding-box diagonal, so scan-to-map ICP already re-associates with
the start-area points when the robot returns there), that re-association only updates
the CURRENT window's spline knots -- there is no mechanism to propagate the correction
backward through the whole already-marginalized trajectory.

This script adds that missing piece as an offline post-process:
1. Load the estimated trajectory (poses.txt, TUM format, world frame).
2. Extract a LiDAR point cloud for a short window at the start and at the end of the
   sequence directly from the rosbag, each accumulated into its own local frame using
   the (locally accurate, only globally drifted) estimated poses.
3. Run point-to-plane ICP (Open3D) between the two windows to get a geometrically
   grounded loop-closure measurement -- NOT reusing the drifted odometry estimate of
   the relative start/end pose, an independent geometric measurement of it.
4. Compute the single rigid SE(3) correction needed at the end keyframe to make it
   consistent with the ICP loop measurement, and ramp it (SE(3): slerp rotation,
   linear translation) from identity at the start to the full correction at the end,
   applied to every pose by its fractional index along the trajectory.

Method note (2026-10-16): a full multi-edge SE(3) pose-graph formulation (Gauss-Newton
over all keyframes jointly) was built and tested first, but FAILED its own synthetic
ground-truth validation -- it closed the loop gap correctly but made mean trajectory
error WORSE (by 2-9x) across multiple synthetic drift models, including a realistic
non-canceling world-frame accumulating bias on an irregular path. The bug was never
isolated (suspected sign/frame-convention error in the relative-pose residual or the
left-perturbation composition). Per the project's rule against deploying unverified
code, that implementation was discarded entirely. The single-correction SE(3) ramp
used here was instead derived analytically and verified on 10/10 random synthetic
trials (translation + rotation drift, irregular non-circular paths) to reduce both
mean and max position error by 3-10x with zero failures -- see docs/DECISIONS.md for
the validation data. It is intentionally simpler (no per-edge information weighting,
no joint optimization) specifically because that simplicity made it possible to verify
correctly in the time available; it corrects only a single accumulated rigid-body
drift, not scale or non-uniform error profiles.

Two-phase split (no single environment here has both rosbag2_py and open3d/scipy):
  Phase A -- scripts/loop_closure_extract.py, run INSIDE the chiyunnn/trail-odom:latest
  container (has rosbag2_py, numpy; no open3d/scipy): extracts the start/end LiDAR
  point-cloud windows from the bag and saves them as .npy files.
  Phase B -- this script, run on the HOST (ros_env python: has open3d, scipy; no
  rosbag2_py): loads those .npy files, runs ICP, computes and applies the correction.

Usage (after running loop_closure_extract.py for the same variant/sequence/rep):
  python3 scripts/loop_closure.py --variant V4 --sequence Airfield1 --rep 1

Writes <results_dir>/trail_output/poses_loopclosed.txt alongside the original poses.txt.
Does not touch the original poses.txt or the online pipeline in any way -- strictly
offline, opt-in, separate output file.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_tum(path):
    ts, xyz, quat = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            ts.append(float(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            quat.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])  # xyzw
    return np.array(ts), np.array(xyz), np.array(quat)


def write_tum(path, ts, xyz, quat):
    with open(path, "w") as f:
        for t, p, q in zip(ts, xyz, quat):
            f.write(f"{t:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def pose_at(ts, xyz, quat, t_query):
    idx = np.searchsorted(ts, t_query)
    idx = np.clip(idx, 0, len(ts) - 1)
    return xyz[idx], quat[idx]


def icp_relative_pose(source_pts, target_pts, init_guess_T, voxel_size=0.1):
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_pts)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts)
    src = src.voxel_down_sample(voxel_size)
    tgt = tgt.voxel_down_sample(voxel_size)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=voxel_size * 5,
        init=init_guess_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return result.transformation, result.fitness, result.inlier_rmse


def se3_from(xyz, quat):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = xyz
    return T


def relative_pose(Ta, Tb):
    Ta_inv = np.eye(4)
    Ta_inv[:3, :3] = Ta[:3, :3].T
    Ta_inv[:3, 3] = -Ta[:3, :3].T @ Ta[:3, 3]
    return Ta_inv @ Tb


def apply_se3_ramp(ts, xyz, quat, T_corr_full):
    """Ramp T_corr_full (left-multiplicative, applied fully at the END) from identity
    at the first pose to full at the last pose, by fractional INDEX along the dense
    trajectory (verified analytically + on 10/10 synthetic trials, see module
    docstring)."""
    n = len(ts)
    rvec_full = Rotation.from_matrix(T_corr_full[:3, :3]).as_rotvec()
    t_full = T_corr_full[:3, 3]
    xyz_out = np.zeros_like(xyz)
    quat_out = np.zeros_like(quat)
    for i in range(n):
        alpha = i / (n - 1)
        Ri = Rotation.from_rotvec(rvec_full * alpha).as_matrix()
        ti = alpha * t_full
        xyz_out[i] = Ri @ xyz[i] + ti
        R_i_world = Rotation.from_quat(quat[i]).as_matrix()
        quat_out[i] = Rotation.from_matrix(Ri @ R_i_world).as_quat()
    return xyz_out, quat_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--rep", required=True, type=int)
    args = ap.parse_args()

    out_dir = REPO_ROOT / "results" / args.variant / args.sequence / f"rep{args.rep}"
    tdir = out_dir / "trail_output"
    poses_path = tdir / "poses.txt"

    ts, xyz, quat = load_tum(poses_path)
    print(f"loaded {len(ts)} poses from {poses_path}")
    print(f"estimated start pos {xyz[0]}, end pos {xyz[-1]}, gap {np.linalg.norm(xyz[0]-xyz[-1]):.3f} m")

    start_npy = tdir / "loop_closure_start.npy"
    end_npy = tdir / "loop_closure_end.npy"
    meta_npz = tdir / "loop_closure_meta.npz"
    if not (start_npy.exists() and end_npy.exists() and meta_npz.exists()):
        print(f"ERROR: run scripts/loop_closure_extract.py first (inside the docker "
              f"container) to produce {start_npy.name}/{end_npy.name}/{meta_npz.name}",
              file=sys.stderr)
        sys.exit(1)

    cloud_start = np.load(start_npy)
    cloud_end = np.load(end_npy)
    meta = np.load(meta_npz)
    ref_xyz0, ref_quat0 = meta["ref_xyz0"], meta["ref_quat0"]
    ref_xyz1, ref_quat1 = meta["ref_xyz1"], meta["ref_quat1"]
    print(f"loaded {len(cloud_start)} start points, {len(cloud_end)} end points")

    T0 = se3_from(ref_xyz0, ref_quat0)
    T1 = se3_from(ref_xyz1, ref_quat1)
    T_end_to_start_init = relative_pose(T1, T0)

    print("running ICP between start/end windows...")
    T_icp, fitness, inlier_rmse = icp_relative_pose(cloud_end, cloud_start, T_end_to_start_init)
    print(f"  ICP fitness={fitness:.4f} inlier_rmse={inlier_rmse:.4f}")
    if fitness < 0.3:
        print("WARNING: low ICP fitness, loop closure measurement may be unreliable", file=sys.stderr)

    # T_icp = relative_pose(T0, T1) refined: start-frame^{-1} * end-frame, geometrically measured.
    T_loop_meas = T_icp
    T_end_target = T0 @ T_loop_meas
    T_corr_full = T_end_target @ np.linalg.inv(T1)

    print(f"end correction: rotvec={Rotation.from_matrix(T_corr_full[:3,:3]).as_rotvec()}, "
          f"trans={T_corr_full[:3,3]}")

    xyz_corr, quat_corr = apply_se3_ramp(ts, xyz, quat, T_corr_full)

    out_path = out_dir / "trail_output" / "poses_loopclosed.txt"
    write_tum(out_path, ts, xyz_corr, quat_corr)
    print(f"wrote {out_path}")
    print(f"corrected start-end gap: {np.linalg.norm(xyz_corr[0] - xyz_corr[-1]):.3f} m")


if __name__ == "__main__":
    main()
