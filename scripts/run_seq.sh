#!/usr/bin/env bash
# Run TRaIL-Odom on one Boxi sequence, inside chiyunnn/trail-odom:latest.
#
# Usage: scripts/run_seq.sh <variant> <sequence> <rep>
#   variant  : V4 (upstream defaults; the only variant implemented so far -- Phase 0/H0)
#   sequence : Airfield1 | Airfield2 | BikeTunnel1 | BikeTunnel2 | Park1 | Park2
#   rep      : repetition number, e.g. 1, 2, 3
#
# Writes results/<variant>/<sequence>/rep<rep>/{config.yaml,run.log,run.json,trail_output/}.
# Assumes scripts/convert_boxi_bags.py has already produced data/Boxi_ros2_sqlite/<sequence>.
set -eo pipefail

VARIANT="$1"; SEQUENCE="$2"; REP="$3"
if [ -z "$VARIANT" ] || [ -z "$SEQUENCE" ] || [ -z "$REP" ]; then
    echo "usage: $0 <variant> <sequence> <rep>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$VARIANT" in
    V4) BASE_CFG="$REPO_ROOT/configs/base/boxi.yaml" ;;
    *) echo "unknown variant '$VARIANT' (only V4 is implemented so far)" >&2; exit 1 ;;
esac

BAG_DIR="$REPO_ROOT/data/Boxi_ros2_sqlite/$SEQUENCE"
if [ ! -d "$BAG_DIR" ]; then
    echo "missing $BAG_DIR -- run scripts/convert_boxi_bags.py $SEQUENCE first" >&2
    exit 1
fi

OUT_DIR="$REPO_ROOT/results/$VARIANT/$SEQUENCE/rep$REP"
mkdir -p "$OUT_DIR"
CFG_PATH="$OUT_DIR/config.yaml"
OUT_DIR_CONTAINER="/root/project/results/$VARIANT/$SEQUENCE/rep$REP"
sed "s|OutputPath: /tmp/trail_results/Boxi|OutputPath: $OUT_DIR_CONTAINER|" "$BASE_CFG" > "$CFG_PATH"

CONTAINER_NAME="trail_odom_run_${VARIANT}_${SEQUENCE}_${REP}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

START_TS="$(date -Is)"
set +e
docker run --rm -i --ipc=host --net=host \
    --name "$CONTAINER_NAME" \
    --user root \
    --hostname trail-odom \
    --entrypoint /bin/bash \
    --volume="$REPO_ROOT:/root/project" \
    chiyunnn/trail-odom:latest \
    -lc '
set -eo pipefail
source /opt/ros/humble/setup.bash
source /root/project/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ros2 run rmw_zenoh_cpp rmw_zenohd &
ZENOHD_PID=$!
sleep 3

timeout 700 ros2 launch /root/project/scripts/launch/trail_boxi_headless.launch.py \
    config_path:='"$OUT_DIR_CONTAINER"'/config.yaml \
    rosbag_path:=/root/project/data/Boxi_ros2_sqlite/'"$SEQUENCE"' \
    start_time:=20 grace_period:=15
EXIT=$?
kill $ZENOHD_PID 2>/dev/null || true
exit $EXIT
' > "$OUT_DIR/run.log" 2>&1
RUN_EXIT=$?
set -e
END_TS="$(date -Is)"

RESEARCH_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SUBMODULE_COMMIT="$(git -C "$REPO_ROOT/ws/src/TRaIL-Odom" rev-parse HEAD)"
CONFIG_SHA256="$(sha256sum "$CFG_PATH" | cut -d' ' -f1)"

python3 -c "
import json, sys
json.dump(dict(
    variant='$VARIANT', sequence='$SEQUENCE', rep=$REP,
    research_commit='$RESEARCH_COMMIT', trail_odom_commit='$SUBMODULE_COMMIT',
    config_sha256='$CONFIG_SHA256',
    flags=dict(), command=sys.argv[1],
    start_time='$START_TS', end_time='$END_TS', host='$(hostname)',
    exit_code=$RUN_EXIT,
), open('$OUT_DIR/run.json', 'w'), indent=2)
" "$0 $*"

if [ "$RUN_EXIT" -ne 0 ]; then
    echo "run FAILED (exit $RUN_EXIT) -- see $OUT_DIR/run.log" >&2
    exit "$RUN_EXIT"
fi
echo "run OK -> $OUT_DIR"
