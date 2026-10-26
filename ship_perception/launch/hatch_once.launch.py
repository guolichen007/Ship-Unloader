"""One capture and one review; no continuously running detection node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("topic", default_value="/ship_scan/map"),
            DeclareLaunchArgument("output_dir", default_value="output"),
            DeclareLaunchArgument(
                "berth_reference",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("ship_perception"),
                        "config",
                        "site",
                        "berth_reference.yaml",
                    ]
                ),
            ),
            ExecuteProcess(
                cmd=[
                    PathJoinSubstitution(
                        [
                            FindPackagePrefix("ship_perception"),
                            "lib",
                            "ship_perception",
                            "ship_hatch_once",
                        ]
                    ),
                    "--topic",
                    LaunchConfiguration("topic"),
                    "--output-dir",
                    LaunchConfiguration("output_dir"),
                    "--berth-reference",
                    LaunchConfiguration("berth_reference"),
                ],
                output="screen",
            ),
        ]
    )
