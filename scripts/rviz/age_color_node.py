#!/usr/bin/env python3
"""Republish /local_map (xyz-only PointCloud2, see
ws/src/TRaIL-Odom/src/core/trail.cpp:450-462 -- EigenToPointCloud2/FillPointCloud2XYZ
never writes an intensity field) as /local_map_aged, adding an `intensity` field equal to
each point's age in seconds since it was first observed.

Why: VoxelHashMap::RemovePointsFarFromLocation (src/voxelmap/VoxelHashMap.cpp:167) prunes
by distance only, so a map window wide enough to persist is also wide enough to include
points affected by many seconds of compounding estimator drift (measured: ~0.64m median
positional drift by 20m traveled, ~0.92m by 40m, see scripts/run_live_rviz.sh). Dropping
old points (the previous MaxRange=40m fix) trades away persistence to avoid showing that
drift. Coloring by age instead keeps every point AND makes the drift-affected ones visibly
distinct (rviz's Intensity/Rainbow transformer: young=blue, old=red) -- no information is
discarded, and the viewer can see both the full environment and which parts of it are
trustworthy at a glance. This is a better trade than the hard cutoff: strictly more
information, not just a differently-placed cutoff.

Tracks first-seen time per voxel (grid cell, not exact point) since VoxelHashMap
republishes a dense snapshot each time, not incremental diffs -- a point recurring in
consecutive messages must keep its original age, not reset to zero.

Viz-only: not part of the research pipeline, run_seq.sh is untouched.
"""
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

VOXEL = 0.5  # age-tracking grid cell size (m) -- independent of the map's own VoxelSize


class AgeColorNode(Node):
    def __init__(self):
        super().__init__('age_color_node')
        self.first_seen = {}  # (vx,vy,vz) -> first-seen stamp (float seconds)
        self.sub = self.create_subscription(PointCloud2, '/local_map', self.on_cloud, 10)
        self.pub = self.create_publisher(PointCloud2, '/local_map_aged', 10)

    def on_cloud(self, msg: PointCloud2):
        offs = {f.name: f.offset for f in msg.fields}
        if not {'x', 'y', 'z'} <= offs.keys():
            return
        step = msg.point_step
        data = msg.data
        n = msg.width * msg.height
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        out_points = bytearray()
        seen_this_msg = set()
        for i in range(n):
            base = i * step
            x, y, z = struct.unpack_from('<fff', data, base + offs['x'])
            key = (round(x / VOXEL), round(y / VOXEL), round(z / VOXEL))
            seen_this_msg.add(key)
            t0 = self.first_seen.get(key)
            if t0 is None:
                t0 = now
                self.first_seen[key] = t0
            age = max(0.0, now - t0)
            out_points += struct.pack('<ffff', x, y, z, age)

        # drop tracked voxels no longer present (point left the map, e.g. pruned by
        # VoxelHashMap) so the dict doesn't grow unbounded over a long run
        stale = set(self.first_seen.keys()) - seen_this_msg
        for k in stale:
            del self.first_seen[k]

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = n
        out.is_bigendian = False
        out.point_step = 16
        out.row_step = out.point_step * n
        out.is_dense = msg.is_dense
        out.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out.data = bytes(out_points)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = AgeColorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
