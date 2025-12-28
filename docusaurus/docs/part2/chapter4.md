---
title: "Weeks 3-5: Building ROS 2 Packages with Python"
sidebar_label: "Weeks 3-5: Building ROS 2 Packages with Python"
---



# Weeks 3-5: Building ROS 2 Packages with Python

## Module 2: The Robotic Nervous System (ROS 2)

### Focus: Building ROS 2 packages with Python

### Learning Objectives
- Create ROS 2 packages using Python
- Build nodes with Python
- Configure launch files and parameter management
- Understand package structure and dependencies
- Implement ROS 2 client libraries in Python

## Building ROS 2 Packages with Python

### Package Structure

A typical ROS 2 Python package follows this structure:

```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
└── my_robot_package/
    ├── __init__.py
    ├── publisher_member_function.py
    └── subscriber_member_function.py
```

### Creating a Package

```bash
# Create a new Python package
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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User Name',
    maintainer_email='user@example.com',
    description='Example ROS 2 package for robot control',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher_member_function = my_robot_package.publisher_member_function:main',
            'subscriber_member_function = my_robot_package.subscriber_member_function:main',
        ],
    },
)
```

## Building Nodes with Python

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
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
            executable='publisher_member_function',
            name='publisher',
            parameters=[
                {'param_name': 'param_value'}
            ]
        ),
        Node(
            package='my_robot_package',
            executable='subscriber_member_function',
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
    robot_name: "my_robot"
    max_velocity: 1.0
```

### Using Parameters in Nodes

```python
class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with default values
        self.declare_parameter('param1', 'default_value')
        self.declare_parameter('param2', 10)
        self.declare_parameter('param3', True)

        # Get parameter values
        self.param1_value = self.get_parameter('param1').value
        self.param2_value = self.get_parameter('param2').value
        self.param3_value = self.get_parameter('param3').value

        self.get_logger().info(f'Parameters: {self.param1_value}, {self.param2_value}, {self.param3_value}')
```

## ROS 2 Client Libraries in Python (rclpy)

### Understanding rclpy

rclpy is the Python client library for ROS 2 that provides:

- Node creation and management
- Topic publishing and subscribing
- Service clients and servers
- Action clients and servers
- Parameter management
- Time and timer utilities

### Advanced rclpy Features

#### Services in Python

```python
# Service server
from example_interfaces.srv import AddTwoInts

class AddTwoIntsService(Node):
    def __init__(self):
        super().__init__('add_two_ints_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

# Service client
class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

#### Actions in Python

```python
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        # Send the goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))
```

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

### Exercise 1: Create a ROS 2 Package
Create a complete ROS 2 package with publisher, subscriber, and launch files.

### Exercise 2: Parameter Management
Implement a node that uses parameters to control its behavior.

### Exercise 3: Service Implementation
Create a service server and client for robot control commands.

## Summary

This module has covered building ROS 2 packages with Python, including package structure, node creation, launch files, and parameter management. You've learned how to use rclpy effectively to create robust robot applications with proper configuration and communication patterns.

