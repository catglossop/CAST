import launch 
from launch_ros.actions import Node, SetRemap
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import UnlessCondition, IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_teleop_joy = get_package_share_directory('teleop_twist_joy')
    teleop_launch_file = PathJoinSubstitution(
    [pkg_teleop_joy, 'launch', 'teleop-launch.py'])
    
    # Create 3 robot model and description
    teleop_joy = GroupAction(
        actions =[
            SetRemap(src="/cmd_vel", dst="/teleop_vel"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([teleop_launch_file]),
                launch_arguments={
                        'joy_config': 'xbox',
                    }.items())
        ]
    )

    return launch.LaunchDescription([
        Node(
            package='deployment',
            executable='teleop_utils_node',
            name='teleop_utils_node',
            output='screen'),
        teleop_joy, 
    ])