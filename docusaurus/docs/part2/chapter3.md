---
title: "Weeks 3-5: ROS 2 Architecture and Core Concepts"
sidebar_label: "Weeks 3-5: ROS 2 Architecture and Core Concepts"
---



# Weeks 3-5: ROS 2 Architecture and Core Concepts

## Module 2: The Robotic Nervous System (ROS 2)

### Focus: Middleware for robot control

### Learning Objectives
- Understand ROS 2 architecture and core concepts
- Learn about nodes, topics, services, and actions
- Understand the ROS 2 ecosystem
- Explore Quality of Service (QoS) policies
- Learn about the differences between ROS 1 and ROS 2

## ROS 2 Architecture and Core Concepts

### What is ROS 2?

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

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

## Nodes, Topics, Services, and Actions

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

### Actions

Actions provide goal-oriented communication with feedback:

```python
# Action Server
action_server = ActionServer(
    self,
    Fibonacci,
    'fibonacci',
    self.execute_callback)

# Action Client
action_client = ActionClient(self, Fibonacci, 'fibonacci')
```

## Quality of Service (QoS) Policies

QoS policies allow you to configure communication behavior:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Configure QoS profile
qos_profile = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)
```

## The DDS Middleware

### Data Distribution Service (DDS)

DDS is the middleware that enables ROS 2's distributed communication:

- **Data-Centric**: Focuses on data rather than services
- **Discovery**: Automatic discovery of participants
- **Reliability**: Built-in reliability mechanisms
- **Performance**: Optimized for real-time systems

### DDS Implementation in ROS 2

ROS 2 uses DDS implementations like:
- Fast DDS (default)
- Cyclone DDS
- RTI Connext DDS

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

### Exercise 3: Action Implementation
Create an action server that performs a multi-step task with feedback.

## Summary

This module has covered the fundamentals of ROS 2 architecture and core concepts. You've learned about the middleware that serves as the "nervous system" for robotic control, including nodes, topics, services, and actions. Understanding ROS 2 is crucial for controlling humanoid robots and will form the foundation for the subsequent modules on simulation and AI integration.

