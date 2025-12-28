---
title: "Weeks 3-5: ROS 2 Fundamentals"
sidebar_label: "Weeks 3-5: ROS 2 Fundamentals"
---



# Weeks 3-5: ROS 2 Fundamentals

## Module 1: The Robotic Nervous System (ROS 2)

### Focus: Middleware for robot control

### Learning Objectives
- Understand ROS 2 architecture and core concepts
- Implement ROS 2 nodes, topics, and services
- Build ROS 2 packages with Python
- Configure launch files and parameter management
- Bridge Python AI agents to ROS controllers using rclpy
- Create URDF (Unified Robot Description Format) for humanoids

## ROS 2 Architecture and Core Concepts

### What is ROS 2?

ROS 2 (Robot Operating System 2) is flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Differences from ROS 1
- **Real-time support**: Better real-time capabilities
- **Multi-robot systems**: Improved support for multiple robots
- **Security**: Built-in security features
- **Middleware**: Uses DDS (Data Distribution Service) for communication
- **Quality of Service (QoS)**: Configurable communication policies

### Core Architecture Components

1. **Nodes**: Processes that perform computation
2. **Topics**: Named buses over which nodes exchange messages
3. **Services**: Synchronous request/response communication
4. **Actions**: Asynchronous goal-oriented communication
5. **Parameters**: Configuration values accessible to nodes
6. **Launch files**: XML/YAML files to start multiple nodes at once

## Nodes, Topics, and Services

### Nodes

A node is a process that performs computation. Nodes are combined together into a graph and communicate with each other using topics, services, and actions.

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello World!')
```

### Topics

Topics provide asynchronous, many-to-many communication using a publish-subscribe model:

```python
# Publisher
publisher = self.create_publisher(String, 'topic_name', 10)

# Subscriber
subscriber = self.create_subscription(
    String,
    'topic_name',
    self.listener_callback,
    10
)
```

### Services

Services provide synchronous request/response communication:

```python
# Service Server
service = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

# Service Client
client = self.create_client(AddTwoInts, 'add_two_ints')
```

## Building ROS 2 Packages with Python

### Package Structure
```
ros2_package/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
└── ros2_package/
    ├── __init__.py
    ├── publisher_member_function.py
    └── subscriber_member_function.py
```

### Creating a Package

```bash
ros2 pkg create --build-type ament_python my_robot_package
```

### Package.xml Example
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package for robot control</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Launch Files and Parameter Management

### Launch Files

Launch files allow you to start multiple nodes with a single command:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='publisher_node',
            name='publisher',
            parameters=[
                {'param_name': 'param_value'}
            ]
        ),
        Node(
            package='my_robot_package',
            executable='subscriber_node',
            name='subscriber'
        )
    ])
```

### Parameter Management

Parameters can be managed through YAML files:

```yaml
my_node:
  ros__parameters:
    param1: value1
    param2: 42
    param3: true
```

## Bridging Python AI Agents to ROS Controllers

### Using rclpy

rclpy is the Python client library for ROS 2:

```python
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import JointState

class AIController(Node):
    def __init__(self):
        super().__init__('ai_controller')

        # Subscribers for sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers for control commands
        self.command_publisher = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # Timer for AI control loop
        self.timer = self.create_timer(0.1, self.ai_control_loop)

    def joint_state_callback(self, msg):
        # Process joint state data from robot
        self.current_positions = msg.position

    def ai_control_loop(self):
        # Implement AI control logic here
        # This could include neural networks, planning algorithms, etc.
        command_msg = JointState()
        # ... fill in command based on AI decision
        self.command_publisher.publish(command_msg)
```

## Understanding URDF for Humanoids

### Unified Robot Description Format (URDF)

URDF is an XML format for representing a robot model. For humanoid robots, it describes the kinematic and dynamic properties.

### Basic URDF Structure
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

### Humanoid-Specific Considerations
- **Bipedal structure**: Left/right leg and arm symmetry
- **Degrees of freedom**: Multiple joints per limb for dexterity
- **Balance**: Center of mass considerations
- **Actuator specifications**: Joint limits and effort constraints

## Weekly Breakdown: Weeks 3-5

### Week 3: ROS 2 Architecture and Core Concepts
- ROS 2 vs ROS 1 differences
- Nodes, topics, services fundamentals
- Quality of Service (QoS) policies
- Basic publisher/subscriber patterns

### Week 4: Advanced ROS 2 Concepts
- Actions for goal-oriented communication
- Parameters and configuration management
- Launch files and system composition
- Testing and debugging tools

### Week 5: AI Integration with ROS 2
- Bridging Python AI agents to ROS controllers
- Sensor data processing pipelines
- Control command generation
- URDF implementation for humanoid robots

## Hands-On Exercises

### Exercise 1: Basic Publisher/Subscriber
Create a simple publisher that publishes sensor data and a subscriber that processes it.

### Exercise 2: Service Implementation
Implement a service that performs a calculation based on robot state.

### Exercise 3: URDF Creation
Create a basic URDF for a humanoid robot with at least 20 joints.

## Summary

This module has covered the fundamentals of ROS 2, the middleware that serves as the "nervous system" for robotic control. You've learned about nodes, topics, services, and how to bridge AI agents to ROS controllers. Understanding ROS 2 is crucial for controlling humanoid robots and will form the foundation for the subsequent modules on simulation and AI integration.

