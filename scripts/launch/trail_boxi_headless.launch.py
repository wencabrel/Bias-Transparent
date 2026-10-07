"""Headless variant of upstream's launch/trail_boxi.launch.py (ws/src/TRaIL-Odom), with the
rviz2 node removed. Kept as our own file rather than editing the submodule (CLAUDE.md rule 4
/ Part J.4: upstream stays unmodified) since batch runs have no DISPLAY.

trail_node has no automatic end-of-data exit when run against a finite bag (see
docs/DECISIONS.md): its IncrementalOptimization loop only checks "no more data" once its
processed-time has caught all the way up to the last-received-message time, and once
bag_play exits there are no more messages to advance that time, so it idles indefinitely.
Interactive upstream usage relies on a human watching rviz and Ctrl-C'ing when playback ends
-- Ctrl-C's SIGINT is what rclcpp relies on to make `rclcpp::ok()` false, which is what
actually breaks trail_node's `while (rclcpp::ok())` loop in TRaIL::Run() and lets main.cpp
fall through to `trail->Save()` (see src/core/trail.cpp:82, src/main.cpp:58-97). For batch
runs we reproduce that externally: wait a grace period after bag_play exits (for the last few
seconds of buffered data to be processed), then emit a Shutdown event, which launch delivers
as SIGINT to trail_node.

    ros2 launch /path/to/trail_boxi_headless.launch.py \
        config_path:=... rosbag_path:=... start_time:=20 [grace_period:=15]
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import EmitEvent
from launch.actions import ExecuteProcess
from launch.actions import RegisterEventHandler
from launch.actions import TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    trail_share = get_package_share_directory('trail')

    default_cfg_path = os.path.join(trail_share, 'dataset', 'Boxi', 'config.yaml')
    default_bag_path = '/path/to/your/dataset/Seq'
    default_start_time = '20'
    default_grace_period = '15'

    cfg_arg = DeclareLaunchArgument('config_path', default_value=default_cfg_path)
    bag_arg = DeclareLaunchArgument('rosbag_path', default_value=default_bag_path)
    start_arg = DeclareLaunchArgument('start_time', default_value=default_start_time)
    grace_arg = DeclareLaunchArgument('grace_period', default_value=default_grace_period)

    cfg_path = LaunchConfiguration('config_path')
    bag_path = LaunchConfiguration('rosbag_path')
    start_time = LaunchConfiguration('start_time')
    grace_period = LaunchConfiguration('grace_period')

    trail_node = Node(
        package='trail',
        executable='trail_node',
        name='trail_node',
        output='screen',
        parameters=[{
            'config_path': cfg_path
        }]
    )

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path,
             '--clock',
             '--start-offset', start_time,
             '-r', '1.0',
             '--log-level', 'error'],
        output='screen'
    )

    shutdown_after_grace_period = RegisterEventHandler(
        OnProcessExit(
            target_action=bag_play,
            on_exit=[
                TimerAction(
                    period=grace_period,
                    actions=[EmitEvent(event=Shutdown(reason='bag playback finished'))],
                ),
            ],
        )
    )

    return LaunchDescription([
        cfg_arg, bag_arg, start_arg, grace_arg,
        bag_play,
        trail_node,
        shutdown_after_grace_period,
    ])
