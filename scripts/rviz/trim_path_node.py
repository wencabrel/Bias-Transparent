#!/usr/bin/env python3
"""Republish /path (trail_node's full accumulated trajectory since start, see
ws/src/TRaIL-Odom/src/core/trail.cpp:376 -- it never trims) as /path_recent, keeping
only the trailing poses within `max_range` meters (Euclidean, from the latest pose) of
the current position. This matches VoxelHashMap::RemovePointsFarFromLocation's own
Euclidean-distance window (ws/src/TRaIL-Odom/src/voxelmap/VoxelHashMap.cpp:167), which
is what /local_map is bounded by. Without this, /path shows the whole run while
/local_map (deliberately shrunk for live-viz legibility, see run_live_rviz.sh) only
covers the last `max_range` meters -- so most of the path line would extend into empty
space with no point cloud nearby.

Viz-only: not part of the research pipeline, run_seq.sh is untouched.

Usage: ros2 run (via python3 directly) trim_path_node.py --ros-args -p max_range:=20.0
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import math


class TrimPathNode(Node):
    def __init__(self):
        super().__init__('trim_path_node')
        self.declare_parameter('max_range', 20.0)
        self.sub = self.create_subscription(Path, '/path', self.on_path, 10)
        self.pub = self.create_publisher(Path, '/path_recent', 10)

    def on_path(self, msg: Path):
        if not msg.poses:
            return
        max_range = self.get_parameter('max_range').value
        last = msg.poses[-1].pose.position
        kept = []
        for p in msg.poses:
            pos = p.pose.position
            d = math.sqrt((pos.x - last.x) ** 2 + (pos.y - last.y) ** 2 + (pos.z - last.z) ** 2)
            if d <= max_range:
                kept.append(p)
        out = Path()
        out.header = msg.header
        out.poses = kept
        self.pub.publish(out)


def main():
    rclpy.init()
    node = TrimPathNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
