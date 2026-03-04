from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf = ParameterValue(Command([
        'xacro ',
        PathJoinSubstitution([FindPackageShare('face_aware_3dbbox'), 'urdf', 'hero.urdf.xacro'])
    ]),
    value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description=''),

        DeclareLaunchArgument(
            'camera_topic',
            default_value='/carla/hero/rgb_front/image',
            description='Camera image topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/carla/hero/rgb_front/camera_info',
            description='Camera info topic'
        ),
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/carla/hero/lidar',
            description='lidar topic'
        ),
        DeclareLaunchArgument(
            'odometry',
            default_value='/carla/hero/odometry',
            description='odometry topic'
        ),

        Node(
            package='face_aware_3dbbox',
            executable='face_aware_3dbbox',
            name='face_aware_3dbbox',
            output='screen',
            parameters=[{
                'camera_topic': LaunchConfiguration('camera_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'lidar_topic': LaunchConfiguration('lidar_topic'),
                'odometry': LaunchConfiguration('odometry'),
            }]
        ),

        # Node(
        #     package='robot_state_publisher',
        #     executable='robot_state_publisher',
        #     name='robot_state_publisher',
        #     output='screen',
        #     parameters=[{'use_sim_time': use_sim_time, 'robot_description': urdf}],
        #     ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/taewook/.rviz2/face_aware_3dbbox.rviz']
        ),
    ])
