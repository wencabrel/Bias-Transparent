"""Launches/stops the trail_node + bag-play backend for a chosen sequence, from inside
the running viewer app -- lets the GUI's sequence dropdown actually start a run instead
of only displaying one started externally by run_native_viewer.sh.

Runs as a subprocess of viewer_app.py itself. Since viewer_app.py is started from
run_native_viewer.sh's bash block *after* `source /opt/ros/humble/setup.bash` and
`export RMW_IMPLEMENTATION=rmw_zenoh_cpp`, this process already has those in its own
os.environ, and subprocess.Popen inherits that by default -- no re-sourcing needed.

Mirrors run_native_viewer.sh's own config generation (OutputPath + MaxRange
substitution) rather than shelling out to that script, since it needs to do this
repeatedly (once per sequence switch) and track/stop the specific launch process it
started, not everything the shell script also manages (zenohd, trim_path_node,
age_color_node -- those are started once by the launcher script and stay up across
sequence switches here).
"""
import os
import signal
import subprocess

BOXI_SEQUENCES = ['Airfield1', 'Airfield2', 'BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2']
# Same 7 sequences and order as scripts/run_seq.sh's GaRLILEO case (line ~319).
GARLILEO_SEQUENCES = ['Downstair', 'CorriLoop', 'BiCorridor', 'SlopeStair', 'Tunnel', 'Overpass', 'Quad']
SEQUENCES = BOXI_SEQUENCES + GARLILEO_SEQUENCES

REPO_ROOT = '/root/project'  # matches run_native_viewer.sh's container mount
LIVE_VIZ_DIR = '/root/live_viz'  # mounted from host /tmp/trail_live_viz
MAX_RANGE = os.environ.get('MAX_RANGE', '150.0')

# Method -> research flags: the same sets as scripts/run_seq.sh's variants of those names
# (and run_native_viewer.sh's initial-method case); keep the three in sync. A run started
# from the panel gets exactly the selected method's flags.
_P = 'TRAIL_RESEARCH_ABLATION_'
_PROXOBSE = {_P + 'PROX_OBS_GATE': '1', _P + 'RADAR_ELEV_SHRINK': '0.5'}
METHODS = {
    'V4': {},                                        # unmodified upstream
    'PROXOBSE': dict(_PROXOBSE),
    'PROXOBSEG': {**_PROXOBSE, _P + 'GAMMA_INFO': '1'},
    # PROXOBSEG + window cap + floor-multipath gate; the only one that completes BikeTunnel2
    'PROXOBSEGWF': {**_PROXOBSE, _P + 'GAMMA_INFO': '1', _P + 'MAX_WINDOW_S': '1.0',
                    _P + 'RADAR_MIN_BODY_Z': '0.0'},
}


def method_env(method):
    """Environment for a trail_node launch with `method`'s flags: this process's own env
    minus every ablation flag (the launcher exported the *initial* method's flags into
    the container, which must not leak into a different method chosen later), plus the
    method's own."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_P)}
    env.update(METHODS[method])
    return env


def _write_config(sequence):
    out_dir_container = f'{LIVE_VIZ_DIR}/{sequence}'
    os.makedirs(out_dir_container, exist_ok=True)
    cfg_path = f'{out_dir_container}/config.yaml'
    # Dataset-specific base config + its own upstream OutputPath/MaxRange placeholders,
    # same pairing as scripts/run_seq.sh (BASE_CFG/CFG_OUT_PLACEHOLDER, line ~318-329).
    if sequence in GARLILEO_SEQUENCES:
        base_cfg = f'{REPO_ROOT}/ws/src/TRaIL-Odom/dataset/GaRLILEO/config.yaml'
        out_placeholder = 'OutputPath: /tmp/trail_results/GaRLILEO'
        maxrange_placeholder = 'MaxRange: 100.0'
    else:
        base_cfg = f'{REPO_ROOT}/configs/base/boxi.yaml'
        out_placeholder = 'OutputPath: /tmp/trail_results/Boxi'
        maxrange_placeholder = 'MaxRange: 300.0'
    with open(base_cfg) as f:
        text = f.read()
    text = text.replace(out_placeholder, f'OutputPath: {out_dir_container}')
    text = text.replace(maxrange_placeholder, f'MaxRange: {MAX_RANGE}')
    with open(cfg_path, 'w') as f:
        f.write(text)
    return cfg_path


class SequenceRunner:
    def __init__(self):
        self._proc = None
        self.running_sequence = None

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def finished_naturally(self):
        """True when the launch exited on its own (bag ended + grace period), as
        opposed to stop() having killed it -- the viewer uses this to know the run's
        poses.txt is complete and ready for the final-spline evaluation."""
        return self._proc is not None and self._proc.poll() is not None

    def start(self, sequence, start_time=20, method=None):
        """method: key of METHODS; None keeps this process's environment unchanged.
        GaRLILEO sequences always start at bag time 0 (scripts/run_seq.sh's BAG_START=0
        for that dataset, line ~324) -- start_time is ignored for them regardless of
        what the caller passed in, same as the headless runs behind results/explore_garlileo."""
        self.stop()
        cfg_path = _write_config(sequence)
        if sequence in GARLILEO_SEQUENCES:
            bag_path = f'{REPO_ROOT}/data/GaRLILEO/{sequence}'
            start_time = 0
        else:
            bag_path = f'{REPO_ROOT}/data/Boxi_ros2_sqlite/{sequence}'
        cmd = [
            'ros2', 'launch', f'{REPO_ROOT}/scripts/launch/trail_boxi_headless.launch.py',
            f'config_path:={cfg_path}', f'rosbag_path:={bag_path}',
            f'start_time:={start_time}', 'grace_period:=15',
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=method_env(method) if method else None,
            preexec_fn=os.setsid)  # own process group, so stop() can signal the whole tree
        self.running_sequence = sequence

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                self._proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._proc = None
        self.running_sequence = None
