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

timeout 900 ros2 launch /root/project/scripts/launch/trail_boxi_headless.launch.py \
    config_path:='"$OUT_DIR_CONTAINER"'/config.yaml \
    rosbag_path:=/root/project/data/Boxi_ros2_sqlite/'"$SEQUENCE"' \
    start_time:=20 grace_period:=120
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

# Part I validation: exit 0 from `ros2 launch` does NOT mean trail_node reached Save() --
# it can be SIGKILLed after failing to respond to SIGINT/SIGTERM (seen on BikeTunnel2: stuck
# mid-Solve() in a near-degenerate corridor segment, falling behind real-time, never caught
# up within the shutdown grace period) and `ros2 launch` still reports success. Verify the
# actual output instead of trusting the exit code.
VALIDATION="$(python3 -c "
import json, sys
poses_path = '$OUT_DIR/trail_output/poses.txt'
errors = []
try:
    with open(poses_path) as f:
        lines = [l for l in f if l.strip()]
except OSError as e:
    print(json.dumps({'ok': False, 'errors': [f'cannot open poses.txt: {e}']}))
    sys.exit()

if len(lines) == 0:
    errors.append('poses.txt is empty')
else:
    prev_t = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 8:
            errors.append(f'line {i}: expected 8 fields, got {len(parts)}')
            break
        vals = [float(x) for x in parts]
        if any(v != v or abs(v) == float('inf') for v in vals):  # NaN or Inf
            errors.append(f'line {i}: NaN/Inf value')
            break
        if prev_t is not None and vals[0] <= prev_t:
            errors.append(f'line {i}: non-monotonic timestamp')
            break
        prev_t = vals[0]
print(json.dumps({'ok': len(errors) == 0, 'errors': errors, 'n_poses': len(lines)}))
")"

python3 -c "
import json, sys
validation = json.loads('''$VALIDATION''')
exit_code = $RUN_EXIT if validation['ok'] else 1
json.dump(dict(
    variant='$VARIANT', sequence='$SEQUENCE', rep=$REP,
    research_commit='$RESEARCH_COMMIT', trail_odom_commit='$SUBMODULE_COMMIT',
    config_sha256='$CONFIG_SHA256',
    flags=dict(), command=sys.argv[1],
    start_time='$START_TS', end_time='$END_TS', host='$(hostname)',
    exit_code=exit_code, launch_exit_code=$RUN_EXIT, validation=validation,
), open('$OUT_DIR/run.json', 'w'), indent=2)
" "$0 $*"

VALIDATION_OK="$(python3 -c "import json; print(json.loads('''$VALIDATION''')['ok'])")"
if [ "$RUN_EXIT" -ne 0 ]; then
    echo "run FAILED (launch exit $RUN_EXIT) -- see $OUT_DIR/run.log" >&2
    exit "$RUN_EXIT"
fi
if [ "$VALIDATION_OK" != "True" ]; then
    echo "run FAILED validation: $VALIDATION -- see $OUT_DIR/run.log" >&2
    exit 1
fi
echo "run OK -> $OUT_DIR"
