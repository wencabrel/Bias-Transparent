#!/usr/bin/env bash
# Build TRaIL-Odom's thirdparty deps + the `trail` package inside chiyunnn/trail-odom:latest.
# Runs non-interactively (no entrypoint shell); logs go to stdout.
#
# Mounts the WHOLE research repo (not just ws/src/TRaIL-Odom) at /root/project, preserving
# the on-host relative path depth: the submodule's .git is a gitlink pointing up into the
# superproject's .git/modules/..., so the container needs that superproject directory present
# at the same relative location or git operations inside the submodule fail.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$REPO_ROOT/data"

docker run --rm -i --ipc=host --net=host \
    --env="RMW_IMPLEMENTATION=rmw_zenoh_cpp" \
    --name trail_odom_build \
    --user root \
    --hostname trail-odom \
    --entrypoint /bin/bash \
    --volume="$REPO_ROOT:/root/project" \
    --volume="$DATASET_DIR:/root/data" \
    chiyunnn/trail-odom:latest \
    -lc '
set -eo pipefail
source /opt/ros/humble/setup.bash

git config --global --add safe.directory "*"

cd /root/project/ws/src/TRaIL-Odom
chmod +x build_thirdparty.sh
./build_thirdparty.sh

cd /root/project/ws
colcon build --packages-select trail
'
