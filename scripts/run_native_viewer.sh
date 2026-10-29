#!/usr/bin/env bash
# Run TRaIL-Odom on one Boxi sequence with the native Open3D viewer
# (scripts/native_viewer/viewer_app.py), forwarded via X11 to the host desktop.
#
# Same backend pipeline as scripts/run_live_rviz.sh (trail_node + age_color_node.py +
# trim_path_node.py), different frontend: a purpose-built point cloud + trajectory +
# vehicle view with a real side control panel (follow-camera, point size, view presets,
# live stats), modeled on ~/Light-3D/lightning-lm's UI structure
# (src/ui/ui_cloud.*, ui_trajectory.*, ui_car.*, imgui_panel.*) but built on Open3D's GUI
# toolkit + rclpy instead of Pangolin + ImGui (not in our Docker image; would need a
# multi-hour from-source build to add).
#
# Requires an X11 session on the host (checked below) and the
# trail-odom-native-viewer:latest image (chiyunnn/trail-odom:latest + open3d, see
# scripts/native_viewer/Dockerfile).
#
# The sequence can now be picked (and re-picked, and re-run) from inside the app itself
# -- a dropdown + Run/Stop in the panel, backed by scripts/native_viewer/sequence_runner.py,
# which starts/stops trail_node + bag-play as a subprocess of viewer_app.py. This script
# only brings up the supporting pipeline (zenohd, trim_path_node.py, age_color_node.py)
# and the viewer itself; it no longer launches trail_node/bag-play directly (that moved
# into sequence_runner.py so it can be done repeatedly, once per sequence switch, not
# just once at startup).
#
# Usage: scripts/run_native_viewer.sh [initial_sequence] [start_time] [method]
#   initial_sequence : Airfield1 | Airfield2 | BikeTunnel1 | BikeTunnel2 | Park1 | Park2
#                       (default Park1 -- just the dropdown's starting selection, not a
#                       hard requirement; pick a different one from the UI any time)
#   start_time        : seconds into the bag to start from (default 20, matching run_seq.sh)
#   method            : V4 (default, unmodified upstream) | PROXOBSE | PROXOBSEG |
#                       PROXOBSEGWF -- same research flags as scripts/run_seq.sh's variants
#                       of those names; they are exported into the container, so every run
#                       started from the panel inherits them. PROXOBSEGWF (PROXOBSEG + window
#                       cap + floor-multipath gate) is the only one that completes BikeTunnel2.
set -eo pipefail

SEQUENCE="${1:-Park1}"
START_TIME="${2:-20}"
METHOD="${3:-V4}"
case "$METHOD" in
    V4) METHOD_ENV=() ;;
    PROXOBSE) METHOD_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5) ;;
    PROXOBSEG) METHOD_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1) ;;
    PROXOBSEGWF) METHOD_ENV=(TRAIL_RESEARCH_ABLATION_PROX_OBS_GATE=1 TRAIL_RESEARCH_ABLATION_RADAR_ELEV_SHRINK=0.5 TRAIL_RESEARCH_ABLATION_GAMMA_INFO=1 TRAIL_RESEARCH_ABLATION_MAX_WINDOW_S=1.0 TRAIL_RESEARCH_ABLATION_RADAR_MIN_BODY_Z=0.0) ;;
    *) echo "unknown method '$METHOD' (expected V4, PROXOBSE, PROXOBSEG or PROXOBSEGWF)" >&2; exit 1 ;;
esac
METHOD_DOCKER_ENV=(--env="TRAIL_VIEWER_METHOD=$METHOD")
for kv in "${METHOD_ENV[@]}"; do METHOD_DOCKER_ENV+=(--env="$kv"); done

if [ -z "$DISPLAY" ]; then
    echo "no \$DISPLAY set -- this needs an X11 desktop session to show the viewer" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_DIR="$REPO_ROOT/data/Boxi_ros2_sqlite/$SEQUENCE"
if [ ! -d "$BAG_DIR" ]; then
    echo "missing $BAG_DIR -- run scripts/convert_boxi_bags.py $SEQUENCE first" >&2
    exit 1
fi

MAX_RANGE="${MAX_RANGE:-150.0}"

docker rm -f trail_odom_native_viewer >/dev/null 2>&1 || true

xhost +local:docker >/dev/null

echo "Starting $SEQUENCE ($METHOD) with the native viewer -- a window should open on this" \
     "desktop in a few seconds. Ctrl-C here stops everything."

docker run --rm -i --ipc=host --net=host \
    --device nvidia.com/gpu=all \
    --env="XDG_RUNTIME_DIR=/tmp/xdg-runtime" \
    --env="MAX_RANGE=$MAX_RANGE" \
    "${METHOD_DOCKER_ENV[@]}" \
    --name trail_odom_native_viewer \
    --user root \
    --hostname "$(hostname)" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --entrypoint /bin/bash \
    --volume="$REPO_ROOT:/root/project" \
    --volume="/tmp/trail_live_viz:/root/live_viz" \
    trail-odom-native-viewer:latest \
    -lc '
set -eo pipefail
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
source /opt/ros/humble/setup.bash
source /root/project/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ros2 run rmw_zenoh_cpp rmw_zenohd &
ZENOHD_PID=$!
sleep 3
trap "kill $ZENOHD_PID 2>/dev/null || true" EXIT

python3 /root/project/scripts/rviz/trim_path_node.py --ros-args -p max_range:='"$MAX_RANGE"' &
TRIM_PID=$!
trap "kill $ZENOHD_PID $TRIM_PID 2>/dev/null || true" EXIT

python3 /root/project/scripts/rviz/age_color_node.py &
AGE_PID=$!
trap "kill $ZENOHD_PID $TRIM_PID $AGE_PID 2>/dev/null || true" EXIT

echo "Opening the viewer -- pick a sequence and click Run in the panel, or it will" \
     "auto-start '"$SEQUENCE"' for you. Close its window (or Ctrl-C here) when done."

cd /root/project/scripts/native_viewer
python3 viewer_app.py '"$SEQUENCE"' '"$START_TIME"'
'
