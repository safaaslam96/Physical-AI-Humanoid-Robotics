---
sidebar_position: 4
title: "Chapter 4: Building ROS 2 Packages with Python"
---

# Chapter 4: Building ROS 2 Packages with Python

## Learning Objectives
- Create ROS 2 packages using Python
- Build nodes with Python using rclpy
- Implement launch files and parameter management
- Understand the complete ROS 2 development workflow

## Creating ROS 2 Packages

ROS 2 packages are the fundamental building blocks of ROS 2 applications. Each package contains source code, dependencies, and configuration files that define specific functionality.

### Package Structure
A typical ROS 2 Python package follows this structure:
```
my_robot_package/
├── CMakeLists.txt          # Build configuration (for mixed packages)
├── package.xml             # Package metadata
├── setup.py                # Python package configuration
├── setup.cfg               # Installation configuration
├── resource/               # Resource files
├── test/                   # Test files
└── my_robot_package/       # Python module
    ├── __init__.py
    ├── main.py
    └── nodes/
        ├── __init__.py
        ├── sensor_node.py
        └── controller_node.py
```

### Creating a Package with Colcon
The `ros2 pkg create` command creates a new package with the proper structure:

```bash
ros2 pkg create --build-type ament_python my_robot_package
```

This command generates the basic package structure with necessary files and directories.

### Package.xml Configuration
The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package for humanoid robotics</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Setup.py Configuration
The `setup.py` file defines the Python package configuration:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Example ROS 2 package for humanoid robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_node = my_robot_package.nodes.sensor_node:main',
            'controller_node = my_robot_package.nodes.controller_node:main',
        ],
    },
)
```

## Building Nodes with Python

### Basic Node Structure
Every ROS 2 Python node follows a standard structure:

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        self.get_logger().info('My Robot Node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Publisher Node Example
Here's a complete example of a sensor publisher node:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import random

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for periodic publishing
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Joint names for a humanoid robot
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]

        self.get_logger().info('Joint State Publisher initialized')

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Simulate joint positions (in radians)
        msg.position = [random.uniform(-1.5, 1.5) for _ in self.joint_names]
        # Simulate joint velocities
        msg.velocity = [random.uniform(-0.5, 0.5) for _ in self.joint_names]
        # Simulate joint efforts
        msg.effort = [random.uniform(-10.0, 10.0) for _ in self.joint_names]

        self.publisher.publish(msg)
        self.get_logger().info(f'Published joint states for {len(msg.name)} joints')

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node Example
Here's an example of a subscriber node that processes sensor data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')

        # Create subscriber for laser scan data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        # Create publisher for velocity commands
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('Obstacle Avoidance Node initialized')

    def laser_callback(self, msg):
        # Process laser scan to detect obstacles
        min_distance = min(msg.ranges)

        # Create velocity command
        cmd_vel = Twist()

        if min_distance < 1.0:  # Obstacle too close
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Turn to avoid
        else:
            cmd_vel.linear.x = 0.5  # Move forward
            cmd_vel.angular.z = 0.0

        self.velocity_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files

Launch files allow you to start multiple nodes with a single command, making system deployment more manageable.

### Python Launch Files
ROS 2 uses Python for launch file creation:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'use_sim_time': True},
                {'publish_frequency': 50.0}
            ],
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='obstacle_avoidance_node',
            name='obstacle_avoidance_node',
            parameters=[
                {'safety_distance': 1.0},
                {'max_linear_speed': 0.5}
            ],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', [os.path.join(get_package_share_directory('my_robot_package'), 'config', 'robot.rviz')]]
        )
    ])
```

### Launch File Parameters
Launch files can accept command-line arguments:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare the arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_name_arg,
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name=[robot_name, '_controller'],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

## Parameter Management

### Node Parameters
Parameters allow runtime configuration of nodes:

```python
import rclpy
from rclpy.node import Node

class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with default values
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('control_frequency', 50.0)

        # Get parameter values
        self.max_speed = self.get_parameter('max_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.control_frequency = self.get_parameter('control_frequency').value

        # Callback for parameter changes
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            f'Node initialized with max_speed: {self.max_speed}, '
            f'safety_distance: {self.safety_distance}'
        )

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'max_speed' and param.type_ == param.Type.DOUBLE:
                self.max_speed = param.value
                self.get_logger().info(f'Max speed updated to: {self.max_speed}')
            elif param.name == 'safety_distance' and param.type_ == param.Type.DOUBLE:
                self.safety_distance = param.value
                self.get_logger().info(f'Safety distance updated to: {self.safety_distance}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterizedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### YAML Parameter Files
Parameters can also be defined in YAML files:

```yaml
# config/robot_params.yaml
my_robot_package:
  ros__parameters:
    max_speed: 1.5
    safety_distance: 0.8
    control_frequency: 100.0
    joint_limits:
      min_position: -2.0
      max_position: 2.0
      max_velocity: 5.0
    pid_gains:
      kp: 1.0
      ki: 0.1
      kd: 0.05
```

Loading parameters in launch files:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[config],
            output='screen'
        )
    ])
```

## Practical Example: Complete Humanoid Robot Package

Let's create a complete example that demonstrates the integration of all concepts:

**Main package file (my_robot_package/__init__.py):**
```python
from .robot_controller import RobotController
from .sensor_processor import SensorProcessor
from .motion_planner import MotionPlanner

__all__ = ['RobotController', 'SensorProcessor', 'MotionPlanner']
```

**Robot Controller Node (my_robot_package/robot_controller.py):**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'desired_joint_states', self.joint_command_callback, 10)

        # Timers
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

        # Robot state
        self.current_joint_positions = {}
        self.desired_joint_positions = {}

        # Declare parameters
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('max_joint_velocity', 1.0)

        self.get_logger().info('Robot Controller initialized')

    def joint_command_callback(self, msg):
        """Handle joint position commands"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.desired_joint_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop"""
        # Update joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'

        # Simulate joint position updates
        for joint_name in self.desired_joint_positions:
            if joint_name not in self.current_joint_positions:
                self.current_joint_positions[joint_name] = 0.0

            # Simple proportional control
            error = self.desired_joint_positions[joint_name] - self.current_joint_positions[joint_name]
            velocity = min(abs(error), self.get_parameter('max_joint_velocity').value)

            if error > 0:
                self.current_joint_positions[joint_name] += velocity * 0.02
            else:
                self.current_joint_positions[joint_name] -= velocity * 0.02

            # Update message
            joint_msg.name.append(joint_name)
            joint_msg.position.append(self.current_joint_positions[joint_name])

        self.joint_pub.publish(joint_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Building and Running the Package

### Building the Package
```bash
# Navigate to workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select my_robot_package

# Source the workspace
source install/setup.bash
```

### Running Nodes
```bash
# Run a single node
ros2 run my_robot_package joint_state_publisher

# Run with launch file
ros2 launch my_robot_package robot.launch.py

# Run with parameters
ros2 launch my_robot_package robot.launch.py use_sim_time:=true robot_name:=humanoid_robot
```

## Best Practices for Python Package Development

### Code Organization
- Use meaningful package and node names
- Organize nodes in subdirectories
- Separate concerns with different modules
- Use proper Python packaging conventions

### Error Handling
- Implement proper exception handling
- Use logging for debugging
- Handle ROS communication failures gracefully
- Implement node cleanup procedures

### Performance Considerations
- Minimize message publishing frequency
- Use appropriate QoS settings
- Optimize computational loops
- Consider real-time requirements

## Knowledge Check

1. What is the standard structure of a ROS 2 Python package?
2. How do you declare and use parameters in a ROS 2 node?
3. What is the purpose of launch files and how do they improve system management?
4. How do you handle parameter changes at runtime?

## Summary

This chapter covered the complete process of building ROS 2 packages with Python, from package creation to deployment. We explored the standard package structure, node development patterns, launch file creation, and parameter management. The chapter also provided practical examples demonstrating the integration of these concepts in a humanoid robot application.

## Next Steps

In the next chapter, we'll dive deeper into ROS 2 communication patterns, exploring nodes, topics, and services in more detail, and learn how to bridge Python agents to ROS controllers using rclpy.