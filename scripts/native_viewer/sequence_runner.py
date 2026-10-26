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

SEQUENCES = ['Airfield1', 'Airfield2', 'BikeTunnel1', 'BikeTunnel2', 'Park1', 'Park2']

REPO_ROOT = '/root/project'  # matches run_native_viewer.sh's container mount
LIVE_VIZ_DIR = '/root/live_viz'  # mounted from host /tmp/trail_live_viz
MAX_RANGE = os.environ.get('MAX_RANGE', '150.0')


def _write_config(sequence):
    out_dir_container = f'{LIVE_VIZ_DIR}/{sequence}'
    os.makedirs(out_dir_container, exist_ok=True)
    cfg_path = f'{out_dir_container}/config.yaml'
    with open(f'{REPO_ROOT}/configs/base/boxi.yaml') as f:
        text = f.read()
    text = text.replace('OutputPath: /tmp/trail_results/Boxi', f'OutputPath: {out_dir_container}')
    text = text.replace('MaxRange: 300.0', f'MaxRange: {MAX_RANGE}')
    with open(cfg_path, 'w') as f:
        f.write(text)
    return cfg_path


class SequenceRunner:
    def __init__(self):
        self._proc = None
        self.running_sequence = None

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def start(self, sequence, start_time=20):
        self.stop()
        cfg_path = _write_config(sequence)
        bag_path = f'{REPO_ROOT}/data/Boxi_ros2_sqlite/{sequence}'
        cmd = [
            'ros2', 'launch', f'{REPO_ROOT}/scripts/launch/trail_boxi_headless.launch.py',
            f'config_path:={cfg_path}', f'rosbag_path:={bag_path}',
            f'start_time:={start_time}', 'grace_period:=15',
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
