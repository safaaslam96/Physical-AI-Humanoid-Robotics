---
title: "Chapter 3: ROS 2 Architecture and Core Concepts"
sidebar_label: "Chapter 3: ROS 2 Architecture"
---

# Chapter 3: ROS 2 Architecture and Core Concepts

## Learning Objectives
- Understand the fundamental architecture of ROS 2
- Master the concepts of nodes, topics, services, and actions
- Explore the ROS 2 ecosystem and its components
- Apply ROS 2 principles to robotic control systems

## Introduction

The Robot Operating System 2 (ROS 2) serves as the "nervous system" of modern robotic platforms, providing the middleware infrastructure that enables complex robotic systems to function as integrated wholes. Unlike its predecessor, ROS 2 addresses the challenges of real-world deployment with improved security, real-time capabilities, and production-readiness. This chapter establishes the foundational understanding of ROS 2 architecture necessary for building sophisticated robotic applications.

## Evolution from ROS 1 to ROS 2

### Motivation for ROS 2

ROS 2 was developed to address critical limitations of ROS 1:

- **Security**: ROS 1 lacked authentication and encryption
- **Real-time Support**: Inadequate for time-critical applications
- **Multi-robot Systems**: Difficult to coordinate multiple robots
- **Production Deployment**: Not designed for commercial applications
- **Quality of Service (QoS)**: No guarantees for message delivery

### Key Improvements in ROS 2

1. **DDS Integration**: Data Distribution Service (DDS) for robust communication
2. **Security Framework**: Authentication, encryption, and access control
3. **Real-time Capabilities**: Support for real-time systems and determinism
4. **Multi-platform Support**: Windows, macOS, Linux, and embedded systems
5. **Package Management**: Improved build system and dependency management

## ROS 2 Architecture Overview

### Client Library Architecture

ROS 2 uses a layered architecture:

```
Application Layer (User Code)
├── rclcpp (C++)
├── rclpy (Python)
├── rcl (C)
└── DDS Implementation
```

This architecture allows the same ROS 2 applications to run across different DDS implementations while maintaining consistent APIs.

### Communication Patterns

ROS 2 supports multiple communication patterns:

1. **Publish/Subscribe (Topics)**: One-to-many, asynchronous communication
2. **Request/Response (Services)**: Synchronous client-server communication
3. **Action-Based Communication**: Asynchronous request/response with feedback
4. **Parameters**: Configuration management across nodes

## Core Concepts

### Nodes

A node is the fundamental unit of computation in ROS 2:

**Definition**: A process that performs computation and communicates with other nodes.

**Key Characteristics**:
- Each node runs independently
- Nodes communicate through topics, services, and actions
- Multiple nodes can run on the same machine or distributed systems
- Nodes are managed by the ROS 2 execution environment

**Node Lifecycle**:
1. **Unconfigured**: Node exists but not yet configured
2. **Inactive**: Configured but not active
3. **Active**: Running and processing data
4. **Finalized**: Node is shutting down

**Example Node Structure**:
```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        # Initialize publishers, subscribers, services, etc.
```

### Topics and Message Passing

**Topics** enable asynchronous, one-to-many communication:

**Characteristics**:
- Data is published to named topics
- Multiple subscribers can receive the same data
- Publishers and subscribers are decoupled in time
- No direct connection between publishers and subscribers

**Message Types**:
- Defined using `.msg` files in the `msg` directory
- Generated into language-specific code during build
- Strongly typed and validated at compile time

**Quality of Service (QoS)**:
ROS 2 provides QoS profiles to control communication behavior:

- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep last N messages vs. keep all
- **Deadline**: Maximum time between publications
- **Liveliness**: How to detect if publisher is alive

### Services

**Services** provide synchronous request/response communication:

**Characteristics**:
- One-to-one communication pattern
- Request must be processed before response
- Blocking call until response received
- Suitable for state queries and configuration

**Service Types**:
- Defined using `.srv` files with request/response structure
- Generated into language-specific code during build
- Support for complex data structures

### Actions

**Actions** provide asynchronous request/response with feedback:

**Characteristics**:
- Long-running operations with status updates
- Feedback messages during execution
- Goal preemption capability
- Suitable for navigation, manipulation, etc.

**Action Components**:
- **Goal**: Request for action execution
- **Feedback**: Status updates during execution
- **Result**: Final outcome of action

## Understanding the ROS 2 Ecosystem

### Core Components

1. **ros2cli**: Command-line interface tools
   - `ros2 run`: Execute nodes
   - `ros2 topic`: Topic inspection and publishing
   - `ros2 service`: Service inspection and calling
   - `ros2 node`: Node inspection and management
   - `ros2 param`: Parameter management

2. **rcl**: Robot Client Library implementations
   - rclcpp for C++
   - rclpy for Python
   - rclc for microcontrollers
   - rcljava for Java

3. **DDS Implementations**:
   - Fast DDS (default in most distributions)
   - Cyclone DDS
   - RTI Connext DDS
   - Eclipse Zenoh

### Build System: colcon

ROS 2 uses colcon as its build system:

**Features**:
- Parallel building of packages
- Support for multiple package types
- Flexible workspace management
- Integration with CMake, ament, and other build systems

**Workspace Structure**:
```
workspace/
├── src/          # Source packages
├── build/        # Build artifacts
├── install/      # Installation directory
└── log/          # Build logs
```

### Package Management

ROS 2 packages contain the basic software modules:

**Package Structure**:
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
├── include/
├── launch/
├── config/
├── test/
└── scripts/
```

## Practical Implementation: Creating a ROS 2 Node

### Setting up a ROS 2 Package

```bash
# Create a new package
ros2 pkg create --build-type ament_python my_robot_controller
```

### Creating a Publisher Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')
        self.publisher = self.create_publisher(String, 'robot_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Robot status: operational {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    robot_status_publisher = RobotStatusPublisher()
    rclpy.spin(robot_status_publisher)
    robot_status_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotStatusSubscriber(Node):
    def __init__(self):
        super().__init__('robot_status_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    robot_status_subscriber = RobotStatusSubscriber()
    rclpy.spin(robot_status_subscriber)
    robot_status_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## ROS 2 in Physical AI Context

### Middleware for Physical AI Systems

ROS 2 serves as the communication backbone for Physical AI systems:

**Sensor Integration**:
- Real-time sensor data distribution
- Synchronization across multiple sensors
- Quality of service for time-critical data

**Control System Integration**:
- Distributed control architectures
- Real-time command execution
- Safety system integration

**AI System Integration**:
- Data flow between perception and planning
- Command execution in physical systems
- Learning and adaptation feedback loops

### Best Practices for Physical AI Applications

1. **Real-time Considerations**:
   - Use appropriate QoS settings for time-critical data
   - Minimize communication overhead
   - Consider dedicated communication channels for safety-critical data

2. **Safety Integration**:
   - Implement safety state monitoring
   - Use parameter servers for safety configuration
   - Design fail-safe communication patterns

3. **Performance Optimization**:
   - Efficient message serialization
   - Appropriate message frequency
   - Memory management for continuous operation

## Knowledge Check

1. Explain the difference between topics, services, and actions in ROS 2.
2. What is Quality of Service (QoS) and why is it important in ROS 2?
3. Describe the node lifecycle in ROS 2 and its significance.

## Hands-On Exercise

1. Create a simple ROS 2 package with a publisher and subscriber
2. Implement a service that provides robot status information
3. Configure appropriate QoS settings for different types of messages
4. Test the communication between nodes and observe the behavior

## Summary

ROS 2 architecture provides the robust middleware foundation necessary for complex Physical AI systems. Understanding nodes, topics, services, and actions is crucial for building distributed robotic applications that can operate effectively in physical environments. The architecture's focus on security, real-time capabilities, and production readiness makes it ideal for Physical AI applications.

## Next Steps

In the following chapter, we'll explore building ROS 2 packages with Python, diving deeper into practical implementation of the concepts introduced here.