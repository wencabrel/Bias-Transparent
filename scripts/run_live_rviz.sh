#!/usr/bin/env bash
# Run TRaIL-Odom on one Boxi sequence with rviz2, forwarded via X11 to the host desktop.
# Uses scripts/rviz/trail_live.rviz, our own copy of upstream's
# ws/src/TRaIL-Odom/config/trail_rviz.rviz with the point cloud recolored by height
# (AxisColor/rainbow instead of flat navy) and a dark background -- cosmetic display
# settings only, same rviz2 binary and same Path/Odometry/PointCloud2 plugins upstream
# uses. Kept as our own file rather than editing the submodule (CLAUDE.md rule 4).
#
# This replaces the earlier custom web-dashboard approach (scripts/visualization/
# trail_live.html): that reimplemented rendering (point cloud, path, vehicle pose)
# by hand in Three.js, which is exactly the kind of thing upstream already solved --
# their own trail_boxi.launch.py just runs rviz2 with trail_rviz.rviz alongside
# trail_node. Running their actual rviz2 means zero custom rendering code and zero
# risk of reintroducing the alignment/transform bug the web version had.
#
# Requires an X11 session on the host (checked below) and the base
# chiyunnn/trail-odom:latest image, which already ships rviz2 -- no custom image
# needed.
#
# Usage: scripts/run_live_rviz.sh <sequence> [start_time]
#   sequence   : Airfield1 | Airfield2 | BikeTunnel1 | BikeTunnel2 | Park1 | Park2
#   start_time : seconds into the bag to start from (default 20, matching run_seq.sh)
set -eo pipefail

SEQUENCE="$1"
START_TIME="${2:-20}"
if [ -z "$SEQUENCE" ]; then
    echo "usage: $0 <sequence> [start_time]" >&2
    exit 1
fi

if [ -z "$DISPLAY" ]; then
    echo "no \$DISPLAY set -- this needs an X11 desktop session to show rviz2" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_DIR="$REPO_ROOT/data/Boxi_ros2_sqlite/$SEQUENCE"
if [ ! -d "$BAG_DIR" ]; then
    echo "missing $BAG_DIR -- run scripts/convert_boxi_bags.py $SEQUENCE first" >&2
    exit 1
fi

OUT_DIR="/tmp/trail_live_viz/$SEQUENCE"
mkdir -p "$OUT_DIR"
CFG_PATH="$OUT_DIR/config.yaml"
OUT_DIR_CONTAINER="/root/live_viz/$SEQUENCE"
# MaxRange is shrunk from the research default (300m) to MAX_RANGE in this viz-only
# throwaway config. VoxelHashMap::RemovePointsFarFromLocation
# (src/voxelmap/VoxelHashMap.cpp:167) only prunes by spatial distance from the current
# pose, not by age, so /local_map otherwise accumulates the whole run -- including scans
# affected by tens of seconds of compounding estimator drift, which visually smears
# structure a short-horizon view wouldn't show. Verified before this change: raw scans
# align to within a few cm using GT position + independently gyro-integrated orientation
# (not the live estimator), so calibration/sync is not the cause.
#
# /path (trail.cpp:376) never trims -- it republishes the full trajectory since start --
# so once /local_map is windowed to MAX_RANGE, most of /path would extend outside the
# visible cloud. trim_path_node.py republishes /path as /path_recent, kept within the
# same MAX_RANGE of the current pose, and trail_live.rviz's Path display points at
# /path_recent instead, so both displays share one window.
#
# MAX_RANGE=40 is a measured trade-off, not a guess: using V4's real estimated trajectory
# (results/V4/Park1/rep1/trail_output/poses.txt) against GT, the position error's
# *variation* within a trailing W-meter window (the quantity that actually causes map
# blur -- a constant offset doesn't blur anything, only its change does) grows
# ~0.23m/0.38m/0.64m/0.92m median at W=5/10/20/40m and saturates near ~1.1m by W=60m+.
# 40m keeps that blur in the same ballpark as the map's own 1.0m voxel quantization while
# roughly doubling persistence over the previous 20m. There is no window size that is
# both long-persisting and sharp -- this is V4's actual drift, not a tunable artifact.
#
# Superseded by age-coloring (scripts/rviz/age_color_node.py): instead of deleting points
# older than MAX_RANGE, every point is kept and colored by seconds-since-first-observed
# (rviz Intensity transformer, blue=fresh/trustworthy -> red=old/likely drifted). That is
# strictly more informative than a hard cutoff -- nothing is hidden, and the viewer can
# see both the full environment and which parts of it to trust. MAX_RANGE is now just a
# generous memory/performance bound, not the thing controlling legibility, so it can stay
# much larger than the old 40m trade-off.
#
# Written to /tmp, never touches configs/base/boxi.yaml or results/, and run_seq.sh (the
# research path) is untouched. Override with the MAX_RANGE env var for a one-off.
MAX_RANGE="${MAX_RANGE:-150.0}"
sed -e "s|OutputPath: /tmp/trail_results/Boxi|OutputPath: $OUT_DIR_CONTAINER|" \
    -e "s|MaxRange: 300.0|MaxRange: $MAX_RANGE|" \
    "$REPO_ROOT/configs/base/boxi.yaml" > "$CFG_PATH"

docker rm -f trail_odom_live_rviz >/dev/null 2>&1 || true

xhost +local:docker >/dev/null

echo "Starting $SEQUENCE with rviz2 (upstream's trail_rviz.rviz) -- a window should" \
     "open on this desktop in a few seconds. Ctrl-C here stops everything."

docker run --rm -i --ipc=host --net=host \
    --gpus all \
    --name trail_odom_live_rviz \
    --user root \
    --hostname trail-odom \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --entrypoint /bin/bash \
    --volume="$REPO_ROOT:/root/project" \
    --volume="/tmp/trail_live_viz:/root/live_viz" \
    chiyunnn/trail-odom:latest \
    -lc '
set -eo pipefail
source /opt/ros/humble/setup.bash
source /root/project/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ros2 run rmw_zenoh_cpp rmw_zenohd &
ZENOHD_PID=$!
sleep 3
trap "kill $ZENOHD_PID 2>/dev/null || true" EXIT

ros2 run rviz2 rviz2 -d /root/project/scripts/rviz/trail_live.rviz &
RVIZ_PID=$!
trap "kill $ZENOHD_PID $RVIZ_PID 2>/dev/null || true" EXIT

python3 /root/project/scripts/rviz/trim_path_node.py --ros-args -p max_range:='"$MAX_RANGE"' &
TRIM_PID=$!
trap "kill $ZENOHD_PID $RVIZ_PID $TRIM_PID 2>/dev/null || true" EXIT

python3 /root/project/scripts/rviz/age_color_node.py &
AGE_PID=$!
trap "kill $ZENOHD_PID $RVIZ_PID $TRIM_PID $AGE_PID 2>/dev/null || true" EXIT

ros2 launch /root/project/scripts/launch/trail_boxi_headless.launch.py \
    config_path:='"$OUT_DIR_CONTAINER"'/config.yaml \
    rosbag_path:=/root/project/data/Boxi_ros2_sqlite/'"$SEQUENCE"' \
    start_time:='"$START_TIME"' grace_period:=15

echo "Bag finished -- rviz stays open showing the final accumulated map. Close its window (or Ctrl-C here) when done."
kill "$TRIM_PID" "$AGE_PID" 2>/dev/null || true
wait "$RVIZ_PID" 2>/dev/null || true
'
