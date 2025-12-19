---
sidebar_position: 3
title: "Chapter 3: ROS 2 Architecture and Core Concepts"
---

# Chapter 3: ROS 2 Architecture and Core Concepts

## Learning Objectives
- Understand the fundamental architecture of ROS 2
- Identify and explain the core communication patterns: nodes, topics, services, and actions
- Explore the ROS 2 ecosystem and its components
- Implement basic ROS 2 communication patterns

## Introduction to ROS 2 Architecture

Robot Operating System 2 (ROS 2) represents a significant evolution from its predecessor, designed specifically to address the challenges of modern robotics applications. Unlike ROS 1, which was built on a peer-to-peer network model, ROS 2 is built on DDS (Data Distribution Service), providing enhanced reliability, real-time capabilities, and support for distributed systems.

ROS 2's architecture is designed with the following principles:
- **Distributed**: Components can run on different machines
- **Decentralized**: No single point of failure
- **Real-time capable**: Support for time-critical applications
- **Multi-language support**: Python, C++, and other languages
- **Security**: Built-in security features for safe deployment

## Core Components of ROS 2

### Nodes
Nodes are the fundamental execution units in ROS 2. Each node represents a single process that performs specific computational tasks.

**Key characteristics of nodes:**
- Encapsulate specific functionality
- Communicate with other nodes through messages
- Can be written in different programming languages
- Managed by the ROS 2 runtime system

**Creating a basic ROS 2 node in Python:**
```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Publishers/Subscribers
Topics enable asynchronous communication between nodes using a publish/subscribe pattern.

**Key concepts:**
- **Publisher**: Sends messages to a topic
- **Subscriber**: Receives messages from a topic
- **Message**: Data structure exchanged between nodes
- **Topic**: Named channel for message exchange

**Example of publisher/subscriber pattern:**
```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Services
Services provide synchronous request/response communication between nodes.

**Key characteristics:**
- Request/response pattern
- Synchronous communication
- Request and response message types
- Service server and client model

**Example service implementation:**
```python
# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {request.a} + {request.b} = {response.sum}')
        return response
```

### Actions
Actions provide a goal-oriented communication pattern for long-running tasks.

**Key components:**
- **Goal**: Request to perform a long-running task
- **Feedback**: Periodic updates during task execution
- **Result**: Final outcome of the task

**Action implementation:**
```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Understanding the ROS 2 Ecosystem

### DDS Implementation
ROS 2 uses DDS as its underlying communication middleware:
- **Connext DDS**: RTI implementation (commercial)
- **Fast DDS**: eProsima implementation (open source)
- **Cyclone DDS**: Eclipse Foundation implementation (open source)

### Tools and Utilities
- **ros2 run**: Execute nodes
- **ros2 topic**: Inspect and interact with topics
- **ros2 service**: Interact with services
- **ros2 action**: Interact with actions
- **ros2 param**: Manage parameters
- **rqt**: GUI tools for visualization

### Package Management
ROS 2 uses colcon for building packages:
- **colcon build**: Build workspace
- **colcon test**: Run tests
- **ament_cmake**: CMake-based build system
- **ament_python**: Python package build system

## Communication Patterns in Depth

### Quality of Service (QoS) Settings
QoS policies allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Example QoS profile for reliable communication
qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)
```

**QoS parameters:**
- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep all vs. keep last N messages
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to determine if a publisher is alive

### Parameters
Parameters provide a way to configure nodes at runtime:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        self.declare_parameter('my_parameter', 'default_value')

    def get_parameter_value(self):
        param = self.get_parameter('my_parameter')
        return param.value
```

## Practical Implementation: Building a Simple ROS 2 System

Let's create a complete example that demonstrates the core concepts:

**Publisher Node (sensor_publisher.py):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(Float32, 'sensor_data', 10)
        self.timer = self.create_timer(1.0, self.publish_sensor_data)

    def publish_sensor_data(self):
        msg = Float32()
        msg.data = random.uniform(0.0, 100.0)  # Simulated sensor reading
        self.publisher.publish(msg)
        self.get_logger().info(f'Published sensor data: {msg.data}')
```

**Subscriber Node (data_processor.py):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')
        self.subscription = self.create_subscription(
            Float32,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        processed_value = msg.data * 2.0  # Simple processing
        self.get_logger().info(f'Processed data: {processed_value}')
```

## Integration with Physical AI Systems

ROS 2 serves as the nervous system for Physical AI systems by:

1. **Enabling distributed processing**: Different AI modules can run as separate nodes
2. **Providing real-time communication**: Critical for responsive physical systems
3. **Supporting sensor integration**: Multiple sensor streams can be processed concurrently
4. **Facilitating hardware abstraction**: Same algorithms work across different robot platforms

## Knowledge Check

1. What are the four core communication patterns in ROS 2 and their use cases?
2. How does the publish/subscribe pattern differ from the service pattern?
3. What is the role of Quality of Service (QoS) settings in ROS 2?
4. Why is DDS important as the underlying communication middleware?

## Summary

This chapter introduced the fundamental architecture of ROS 2, covering its core components: nodes, topics, services, and actions. We explored the ROS 2 ecosystem, including DDS implementations, tools, and package management. The chapter also demonstrated practical implementation of these concepts and explained how ROS 2 serves as the nervous system for Physical AI systems.

## Next Steps

In the next chapter, we'll dive into building ROS 2 packages with Python, exploring the complete development workflow from package creation to deployment.