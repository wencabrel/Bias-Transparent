#!/usr/bin/env bash
# Run TRaIL-Odom on one Boxi sequence, inside chiyunnn/trail-odom:latest.
#
# Usage: scripts/run_seq.sh <variant> <sequence> <rep>
#   variant  : V4 (upstream defaults) | V0,V1,V2,V3,V5,V6,V7 (Phase 2 ablation, Part D.1/H)
#              | V0P (paper-equivalent V0: alpha/gamma off only, prox+zerovel left on --
#                see the V0P case below) | CAL,CALP,CALROB (Phase 4 calibration)
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
    V4|V0|V0P|V1|V2|V3|V5|V6|V7|CAL|CALP|CALROB) BASE_CFG="$REPO_ROOT/configs/base/boxi.yaml" ;;
    *) echo "unknown variant '$VARIANT' (expected one of V0,V0P,V1,V2,V3,V4,V5,V6,V7,CAL,CALP,CALROB)" >&2; exit 1 ;;
esac

# Phase 2 ablation mapping (Part D.1/H Phase 2). V4 (upstream) leaves every
# TRAIL_RESEARCH_ABLATION_* variable unset, so AblationConfig's defaults (all on,
# kappa_thresh=5, eta=2.2) apply -- i.e. V4 is exactly upstream behaviour, unchanged.
declare -a ABLATION_ENV=()
case "$VARIANT" in
    V4) ;;  # all defaults (upstream)
    V0) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0
                       TRAIL_RESEARCH_ABLATION_PROX=0 TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;
    V0P) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0) ;;
        # "paper-equivalent" V0: the TRaIL paper's Table III ablation toggles ONLY alpha/gamma
        # (Sec III-F/G); it never mentions RadarFactor_prox (an ICP-init Tikhonov stabilizer
        # gated on degenerate_dim>0, state_manager.cpp:1621) or the x10 zero-vel boost
        # (:1550) -- both are undocumented-in-the-paper mechanisms this project's original
        # V0 ("nothing") also disabled, over-crippling it relative to what the paper actually
        # tested as its fixed-weight baseline. See docs/DECISIONS.md 2026-10-10 entry.
    V1) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0
                       TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # prox only
    V2) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_GAMMA=0 TRAIL_RESEARCH_ABLATION_PROX=0
                       TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # alpha only
    V3) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_PROX=0
                       TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # gamma only
    V5) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_PROX=0) ;;  # V4 without prox
    V6) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_KAPPA_THRESH=30 TRAIL_RESEARCH_ABLATION_ETA=2.0) ;;  # paper settings
    V7) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;  # no x10 zero-vel boost
    CAL|CALROB) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0
                               TRAIL_RESEARCH_ABLATION_PROX=0 TRAIL_RESEARCH_ABLATION_ZEROVEL=0) ;;
    CALP) ABLATION_ENV=(TRAIL_RESEARCH_ABLATION_ALPHA=0 TRAIL_RESEARCH_ABLATION_GAMMA=0) ;;
        # CAL built on V0P instead of V0 -- see V0P's comment above.
esac

# Phase 4 calibration mapping (Part D.1/H Phase 4). CAL = V0 + online noise calibration
# (E.2); CALROB additionally fixes the robust Doppler loss (E.4, fact F10). CALP = V0P +
# online noise calibration (the paper-equivalent-baseline version of CAL).
declare -a CALIBRATION_ENV=()
case "$VARIANT" in
    CAL|CALP) CALIBRATION_ENV=(TRAIL_RESEARCH_CALIBRATION_CAL=1) ;;
    CALROB) CALIBRATION_ENV=(TRAIL_RESEARCH_CALIBRATION_CAL=1 TRAIL_RESEARCH_CALIBRATION_ROB=1) ;;
esac
for kv in "${CALIBRATION_ENV[@]}"; do
    ABLATION_ENV+=("$kv")
done

# Determinism fixes (applied uniformly to every variant, not a Phase 2/4 ablation axis):
# single-threaded rclcpp executor removes IMU/LiDAR/Radar callback reordering as a source
# of run-to-run variance. Opt-in via env var upstream default is multi-threaded; combined
# with tbb::parallel_deterministic_reduce (compiled in, no flag) in BuildLinearSystem.
ABLATION_ENV+=(TRAIL_RESEARCH_SINGLE_THREADED_EXECUTOR=1)

ABLATION_EXPORTS=""
FLAGS_JSON="{"
for kv in "${ABLATION_ENV[@]}"; do
    ABLATION_EXPORTS="${ABLATION_EXPORTS}export ${kv}
"
    key="${kv%%=*}"; val="${kv#*=}"
    FLAGS_JSON="${FLAGS_JSON}\"${key}\":\"${val}\","
done
FLAGS_JSON="${FLAGS_JSON}}"
FLAGS_JSON="$(echo "$FLAGS_JSON" | sed 's/,}/}/')"

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
'"$ABLATION_EXPORTS"'
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
    flags=json.loads('''$FLAGS_JSON'''), command=sys.argv[1],
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
