#!/usr/bin/env python3
"""Phase 0: convert Boxi's ROS1 .bag files (IMU + LiDAR + radar) into one merged ROS2
sqlite3 bag per sequence, suitable for `ros2 bag play` inside chiyunnn/trail-odom:latest.

Why this exists: the pre-existing `data/Boxi_ros2/*` (symlinked from ~/Desktop/Trail/bags)
is `.mcap`-format, but the Docker image only has the sqlite3 default storage plugin
installed (no rosbag2_storage_mcap) -- `ros2 bag play` on an .mcap bag fails with
"Could not load/open plugin with storage id 'mcap'". Converting the original ROS1 bags with
the pure-Python `rosbags` library avoids needing network/apt access inside the container.

Also works around a `rosbags` library quirk: it writes `offered_qos_profiles: []` (a YAML
list) in metadata.yaml, but ROS 2 Humble's rosbag2 parser expects a string (serialized QoS
profile YAML) there and throws "bad conversion" on the list form. Patched to `''` after
conversion.

Usage: python3 scripts/convert_boxi_bags.py [sequence ...]   (default: all 6)
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEQUENCES = ['Airfield1', 'Airfield2', 'BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2']
SUFFIXES = ['_cpt7_raw_imu.bag', '_nuc_livox_ready.bag', '_nuc_radar.bag']


def convert_one(seq):
    src_dir = REPO_ROOT / 'data' / 'Boxi' / seq
    srcs = []
    for suffix in SUFFIXES:
        matches = list(src_dir.glob(f'*{suffix}'))
        if not matches:
            raise RuntimeError(f"{seq}: no bag matching *{suffix} in {src_dir}")
        srcs.append(str(matches[0]))

    out_dir = REPO_ROOT / 'data' / 'Boxi_ros2_sqlite' / seq
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    subprocess.run(
        [sys.executable, '-m', 'rosbags.convert.cli',
         '--src', *srcs, '--dst', str(out_dir),
         '--dst-storage', 'sqlite3', '--dst-typestore', 'copy'],
        check=True, env={'PYTHONPATH': ''},
    )

    meta = out_dir / 'metadata.yaml'
    text = meta.read_text()
    meta.write_text(text.replace('offered_qos_profiles: []', "offered_qos_profiles: ''"))
    print(f"{seq}: converted -> {out_dir}")


def main():
    seqs = sys.argv[1:] or SEQUENCES
    for seq in seqs:
        convert_one(seq)


if __name__ == '__main__':
    main()
