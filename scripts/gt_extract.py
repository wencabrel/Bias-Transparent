#!/usr/bin/env python3
"""Phase 0 step 3 (guideline Part H): extract Boxi ground truth as TUM trajectories.

Reads prism position from the ROS1 `*_jetson_ap20_robot.bag` (topic
`/gt_box/ap20/prism_position`, geometry_msgs/PointStamped) and writes it out as a
position-only TUM trajectory (identity quaternion -- no orientation GT is available from a
single total-station prism).

Also reads the static `box_base -> prism` and `box_base -> cpt7_imu` transforms from
`tf_static` and reports the prism->IMU lever arm. It is NOT applied to the GT: doing so
correctly requires rotating the lever arm into the world frame by the robot's orientation at
each timestamp, and no independent orientation GT exists here (see
docs/OPEN_QUESTIONS.md). Applying it with the *estimated* trajectory's orientation instead
would leak estimator output into GT and bias the comparison, so it is left unapplied and
recorded as metadata only.

Usage:
    python3 scripts/gt_extract.py <sequence_dir> <jetson_ap20_robot.bag> <tf_static_dir> <out.tum>
"""
import sys, json
from pathlib import Path
from rosbags.highlevel import AnyReader


def read_prism_positions(bag_path):
    rows = []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == '/gt_box/ap20/prism_position']
        if not conns:
            raise RuntimeError(f"no /gt_box/ap20/prism_position topic in {bag_path}")
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            rows.append((t, msg.point.x, msg.point.y, msg.point.z))
    rows.sort(key=lambda r: r[0])
    return rows


def read_lever_arm(tf_static_dir):
    with AnyReader([Path(tf_static_dir)]) as reader:
        conns = [c for c in reader.connections if c.topic == '/tf_static']
        transforms = {}
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            for tr in msg.transforms:
                key = (tr.header.frame_id, tr.child_frame_id)
                t = tr.transform.translation
                r = tr.transform.rotation
                transforms[key] = dict(t=[t.x, t.y, t.z], q_xyzw=[r.x, r.y, r.z, r.w])
    prism = transforms.get(('box_base', 'prism'))
    body = transforms.get(('box_base', 'cpt7_imu'))
    if prism is None or body is None:
        raise RuntimeError("missing box_base->prism or box_base->cpt7_imu in tf_static")
    if body['t'] != [0.0, 0.0, 0.0] or body['q_xyzw'] != [0.0, 0.0, 0.0, 1.0]:
        raise RuntimeError("box_base->cpt7_imu is not identity; lever-arm assumption below is wrong")
    return dict(prism_in_body=prism['t'], prism_rotation_xyzw=prism['q_xyzw'])


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    _, _, bag_path, tf_static_dir, out_tum = sys.argv
    rows = read_prism_positions(bag_path)
    lever = read_lever_arm(tf_static_dir)

    out_tum = Path(out_tum)
    with open(out_tum, 'w') as f:
        for t, x, y, z in rows:
            f.write(f"{t:.9f} {x:.9f} {y:.9f} {z:.9f} 0 0 0 1\n")

    meta_path = out_tum.with_suffix(out_tum.suffix + '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(dict(
            source_bag=str(bag_path), n_poses=len(rows),
            t_start=rows[0][0], t_end=rows[-1][0],
            lever_arm_note="prism_in_body is box_base->prism translation/rotation; "
                            "NOT applied to GT below -- see module docstring",
            **lever,
        ), f, indent=2)
    print(f"wrote {len(rows)} GT poses to {out_tum}")
    print(f"wrote lever-arm metadata to {meta_path}")


if __name__ == '__main__':
    main()
