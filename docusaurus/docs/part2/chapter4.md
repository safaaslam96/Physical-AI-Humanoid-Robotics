---
title: "Chapter 4: Building ROS 2 Packages with Python"
sidebar_label: "Chapter 4: Building ROS 2 Packages"
---

# Chapter 4: Building ROS 2 Packages with Python

## Learning Objectives
- Create ROS 2 packages using Python
- Build nodes with Python and the rclpy library
- Implement launch files and parameter management
- Apply best practices for Python-based ROS 2 development

## Introduction

Python has become one of the most popular languages for robotics development due to its simplicity, extensive libraries, and strong AI/ML ecosystem. The rclpy library provides Python bindings for ROS 2, enabling rapid prototyping and development of robotic applications. This chapter focuses on building robust ROS 2 packages using Python, emphasizing practical implementation and best practices for Physical AI systems.

## Creating ROS 2 Packages with Python

### Package Structure and Organization

A well-structured ROS 2 Python package follows a specific organization:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration
├── package.xml            # Package metadata
├── setup.py               # Python package configuration
├── setup.cfg              # Installation configuration
├── my_robot_package/      # Main Python package
│   ├── __init__.py
│   ├── robot_controller.py
│   └── utils/
│       ├── __init__.py
│       └── helper_functions.py
├── launch/                # Launch files
│   └── robot_launch.py
├── config/                # Configuration files
│   └── robot_params.yaml
├── test/                  # Test files
│   └── test_robot_controller.py
└── scripts/               # Executable scripts (optional)
```

### Creating a New Package

To create a Python-based ROS 2 package:

```bash
# Create a new package with Python build type
ros2 pkg create --build-type ament_python my_robot_controller

# Or using the ament_cmake build type for mixed C++/Python packages
ros2 pkg create --build-type ament_cmake my_mixed_robot_package
```

### Package.xml Configuration

The `package.xml` file contains essential metadata:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_controller</name>
  <version>0.0.0</version>
  <description>A Python-based robot controller package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

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

The `setup.py` file configures the Python package:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='A Python-based robot controller package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = my_robot_controller.robot_controller:main',
            'sensor_processor = my_robot_controller.sensor_processor:main',
        ],
    },
)
```

## Building Nodes with Python

### Basic Node Structure

A well-structured ROS 2 Python node includes:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Import message types
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Declare parameters with default values
        self.declare_parameter('control_frequency', 50)
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        self.control_freq = self.get_parameter('control_frequency').value
        self.max_vel = self.get_parameter('max_velocity').value

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            self._get_qos_profile()
        )

        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            self._get_qos_profile()
        )

        # Create services
        self.reset_service = self.create_service(
            Empty,  # You would define this message type
            'reset_robot',
            self.reset_callback
        )

        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self.control_loop
        )

        # Initialize robot state
        self.current_state = None
        self.target_velocity = Twist()

        self.get_logger().info(f'Robot Controller initialized with {self.control_freq}Hz frequency')

    def _get_qos_profile(self):
        """Create a QoS profile for real-time communication"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        return qos_profile

    def joint_state_callback(self, msg):
        """Handle incoming joint state messages"""
        self.current_state = msg
        # Process joint states as needed
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def control_loop(self):
        """Main control loop executed at specified frequency"""
        if self.current_state is not None:
            # Implement control logic here
            self.cmd_vel_pub.publish(self.target_velocity)

    def reset_callback(self, request, response):
        """Handle reset service requests"""
        self.get_logger().info('Resetting robot state')
        # Reset logic here
        return response

    def set_target_velocity(self, linear_x, angular_z):
        """Set target velocity for the robot"""
        self.target_velocity.linear.x = linear_x
        self.target_velocity.angular.z = angular_z


def main(args=None):
    rclpy.init(args=args)

    robot_controller = RobotControllerNode()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down robot controller')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Node Features

#### Parameter Management

ROS 2 provides robust parameter management:

```python
class AdvancedRobotNode(Node):
    def __init__(self):
        super().__init__('advanced_robot')

        # Declare parameters with descriptions
        self.declare_parameter(
            'robot_name',
            'default_robot',
            descriptor=rclpy.node.ParameterDescriptor(
                description='Name of the robot'
            )
        )

        self.declare_parameter('safety_limits', [1.0, 2.0, 3.0])
        self.declare_parameter('control_mode', 'velocity')

        # Callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Handle parameter changes"""
        for param in params:
            if param.name == 'control_mode' and param.value not in ['velocity', 'position', 'effort']:
                return SetParametersResult(successful=False, reason='Invalid control mode')
        return SetParametersResult(successful=True)
```

#### Lifecycle Nodes

For more complex applications, lifecycle nodes provide better state management:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn


class LifecycleRobotController(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_robot_controller')

    def on_configure(self, state):
        """Called when transitioning to CONFIGURING state"""
        self.get_logger().info('Configuring robot controller')
        # Initialize resources but don't start active operations
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Called when transitioning to ACTIVATING state"""
        self.get_logger().info('Activating robot controller')
        # Start active operations
        return super().on_activate(state)

    def on_deactivate(self, state):
        """Called when transitioning to DEACTIVATING state"""
        self.get_logger().info('Deactivating robot controller')
        # Stop active operations but keep resources
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        """Called when transitioning to CLEANINGUP state"""
        self.get_logger().info('Cleaning up robot controller')
        # Clean up resources
        return TransitionCallbackReturn.SUCCESS
```

## Launch Files for Python Nodes

### Python-based Launch Files

ROS 2 launch files can be written in Python:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessStart
from lifecycle_msgs.msg import Transition

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    # Create robot controller node
    robot_controller = Node(
        package='my_robot_controller',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name}
        ],
        output='screen'
    )

    # Create sensor processor node
    sensor_processor = Node(
        package='my_robot_controller',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        robot_controller,
        sensor_processor
    ])
```

### Advanced Launch Configuration

Launch files can include complex configurations:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Include other launch files
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ])
    )

    # Launch nodes with specific conditions
    robot_controller = Node(
        package='my_robot_controller',
        executable='robot_controller',
        name='robot_controller',
        parameters=[PathJoinSubstitution([
            FindPackageShare('my_robot_controller'),
            'config',
            'robot_config.yaml'
        ])],
        respawn=True,  # Restart if the node dies
        respawn_delay=2.0
    )

    # Launch with delay
    delayed_node = TimerAction(
        period=5.0,  # Wait 5 seconds before launching
        actions=[Node(
            package='my_robot_controller',
            executable='post_processing_node',
            name='post_processor'
        )]
    )

    return LaunchDescription([
        simulation_launch,
        robot_controller,
        delayed_node
    ])
```

## Parameter Management

### YAML Configuration Files

Configuration files provide flexible parameter management:

```yaml
# config/robot_config.yaml
/**:
  ros__parameters:
    # Robot specifications
    robot_name: "my_advanced_robot"
    max_velocity: 1.0
    max_angular_velocity: 1.5
    safety_distance: 0.5

    # Control parameters
    control_frequency: 100
    position_tolerance: 0.01
    velocity_tolerance: 0.05

    # Sensor parameters
    sensor_update_rate: 50
    sensor_timeout: 0.1

    # AI/ML parameters
    prediction_horizon: 10
    confidence_threshold: 0.8
```

### Loading Parameters

Parameters can be loaded in multiple ways:

```python
class ParameterizedRobotNode(Node):
    def __init__(self):
        super().__init__('parameterized_robot')

        # Load parameters from YAML file
        self.load_parameters_from_file()

        # Or load specific parameters
        self.declare_parameter('robot_config_file', 'config/robot_config.yaml')

    def load_parameters_from_file(self):
        """Load parameters from YAML configuration file"""
        config_file = self.get_parameter('robot_config_file').value

        # In practice, you'd use a configuration management system
        # This is a simplified example
        try:
            import yaml
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                # Process configuration
                self.get_logger().info(f'Loaded configuration from {config_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to load config file: {e}')
```

## Best Practices for Python-based ROS 2 Development

### Code Organization

1. **Separate Concerns**: Keep message handling, business logic, and ROS interface separate
2. **Use Type Hints**: Improve code readability and IDE support
3. **Error Handling**: Implement proper exception handling for robust operation
4. **Logging**: Use appropriate log levels for debugging and monitoring

### Performance Considerations

1. **Efficient Message Processing**: Minimize data copying and processing overhead
2. **Memory Management**: Be mindful of memory usage in long-running systems
3. **Threading**: Use ROS 2's built-in threading model appropriately
4. **QoS Configuration**: Choose appropriate QoS settings for your application

### Testing and Validation

```python
import unittest
from unittest.mock import Mock, patch
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


class TestRobotController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = RobotControllerNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_node_initialization(self):
        """Test that the node initializes correctly"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'robot_controller')

    def test_parameter_setting(self):
        """Test parameter setting functionality"""
        self.node.set_target_velocity(1.0, 0.5)
        self.assertEqual(self.node.target_velocity.linear.x, 1.0)
        self.assertEqual(self.node.target_velocity.angular.z, 0.5)
```

## Integration with Physical AI Systems

### AI/ML Integration

Python's strong AI/ML ecosystem integrates seamlessly with ROS 2:

```python
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class AIPoweredRobotNode(Node):
    def __init__(self):
        super().__init__('ai_robot')

        # Initialize AI model
        self.model = tf.keras.models.load_model('path/to/model')
        self.bridge = CvBridge()

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish AI decisions
        self.ai_cmd_pub = self.create_publisher(Twist, '/ai_cmd_vel', 10)

    def image_callback(self, msg):
        """Process image with AI model"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess image for model
        input_tensor = tf.convert_to_tensor(cv_image)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        # Run inference
        prediction = self.model(input_tensor)

        # Convert prediction to robot command
        cmd_vel = self.convert_prediction_to_command(prediction)
        self.ai_cmd_pub.publish(cmd_vel)
```

## Knowledge Check

1. Explain the difference between ament_python and ament_cmake build types.
2. Describe the purpose of launch files and how they improve system management.
3. What are the advantages of using parameters in ROS 2 development?

## Hands-On Exercise

1. Create a complete ROS 2 package with Python nodes
2. Implement parameter management with YAML configuration
3. Create a launch file that starts multiple nodes with dependencies
4. Test the system with different parameter configurations

## Summary

Building ROS 2 packages with Python provides a powerful and flexible approach to developing Physical AI systems. The combination of Python's rich ecosystem with ROS 2's robust communication framework enables rapid development of sophisticated robotic applications. Proper package structure, parameter management, and launch configuration are essential for building maintainable and scalable systems.

## Next Steps

In the following chapter, we'll explore ROS 2 nodes, topics, and services in greater depth, including how to bridge Python agents to ROS controllers using rclpy.