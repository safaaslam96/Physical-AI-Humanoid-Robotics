---
title: "Weeks 3-5: ROS 2 Nodes, Topics, and Services"
sidebar_label: "Weeks 3-5: ROS 2 Nodes, Topics, and Services"
---



# Weeks 3-5: ROS 2 Nodes, Topics, and Services

## Module 2: The Robotic Nervous System (ROS 2)

### Focus: Deep dive into ROS 2 communication patterns

### Learning Objectives
- Master ROS 2 communication patterns: nodes, topics, services, and actions
- Implement publishers and subscribers for data exchange
- Create and use services for request/response communication
- Understand the difference between topics and services
- Bridge Python AI agents to ROS controllers using rclpy
- Understand URDF (Unified Robot Description Format) for humanoids

## ROS 2 Communication Patterns

### Nodes, Topics, Services, and Actions Overview

ROS 2 provides several communication patterns:

- **Nodes**: Processes that perform computation
- **Topics**: Asynchronous publish/subscribe communication
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous goal-oriented communication with feedback

### Deep Dive into Topics

Topics provide asynchronous, many-to-many communication using a publish-subscribe model:

```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

# Subscriber example
class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

### Quality of Service (QoS) Settings for Topics

```python
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

# Configure QoS for different needs
# Reliable communication with message history
reliable_qos = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Best effort for real-time data (like sensor streams)
best_effort_qos = QoSProfile(
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Publisher with custom QoS
publisher = self.create_publisher(String, 'sensor_data', best_effort_qos)
```

### Services for Request/Response Communication

Services provide synchronous communication:

```python
# Service definition (in srv/AddTwoInts.srv)
# int64 a
# int64 b
# ---
# int64 sum

# Service server
from example_interfaces.srv import AddTwoInts

class AddTwoIntsService(Node):
    def __init__(self):
        super().__init__('add_two_ints_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request: {request.a} + {request.b} = {response.sum}')
        return response

# Service client
class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create and send request
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Send asynchronously
        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)

        return self.future.result()
```

### Actions for Complex Task Management

Actions provide goal-oriented communication with feedback:

```python
# Action definition (in action/Fibonacci.action)
# int32 order
# ---
# int32[] sequence
# ---
# int32[] partial_sequence

from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Initialize result
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        # Execute the action
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1]
            )

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

        # Complete the goal
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Goal succeeded with result: {result.sequence}')

        return result

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
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
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.partial_sequence}')
```

## Bridging Python AI Agents to ROS Controllers

### Using rclpy for AI Integration

```python
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tensorflow as tf  # Example AI framework

class AIBridgeNode(Node):
    def __init__(self):
        super().__init__('ai_bridge_node')

        # Publishers for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Subscribers for sensor data
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Initialize AI model
        self.ai_model = self.load_ai_model()

        # Robot state storage
        self.current_odom = None
        self.current_joints = None
        self.current_laser = None

        # Control timer
        self.control_timer = self.create_timer(0.1, self.ai_control_loop)  # 10 Hz control loop

    def load_ai_model(self):
        """Load the AI model for robot control"""
        # This could be a pre-trained model or a neural network
        # For example, a policy network for navigation or manipulation
        try:
            # Load a pre-trained model
            model = tf.keras.models.load_model('robot_policy_model.h5')
            self.get_logger().info('AI model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().warn(f'Could not load AI model: {e}')
            # Fallback to simple control logic
            return None

    def odom_callback(self, msg):
        """Store odometry data"""
        self.current_odom = msg

    def joint_state_callback(self, msg):
        """Store joint state data"""
        self.current_joints = msg

    def laser_callback(self, msg):
        """Store laser scan data"""
        self.current_laser = msg

    def ai_control_loop(self):
        """Main AI control loop"""
        # Check if we have all required sensor data
        if not all([self.current_odom, self.current_laser]):
            return  # Wait for complete sensor data

        # Prepare input for AI model
        sensor_input = self.prepare_sensor_input()

        # Get AI decision
        if self.ai_model:
            # Use AI model to determine action
            action = self.ai_model.predict(sensor_input)
            cmd_vel = self.convert_action_to_cmd_vel(action)
        else:
            # Fallback to simple reactive control
            cmd_vel = self.simple_reactive_control()

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def prepare_sensor_input(self):
        """Prepare sensor data for AI model"""
        # Extract relevant features from sensor data
        linear_vel = self.current_odom.twist.twist.linear.x
        angular_vel = self.current_odom.twist.twist.angular.z

        # Process laser scan (e.g., get distances to obstacles in different directions)
        laser_ranges = np.array(self.current_laser.ranges)
        laser_ranges = np.nan_to_num(laser_ranges, nan=np.inf)  # Handle NaN values

        # Create input vector for AI model
        # This is an example - actual implementation depends on your AI model
        input_vector = np.concatenate([
            [linear_vel, angular_vel],  # Current velocity
            laser_ranges[::10]  # Downsampled laser ranges (every 10th value)
        ])

        return input_vector.reshape(1, -1)  # Reshape for model prediction

    def convert_action_to_cmd_vel(self, action):
        """Convert AI action to Twist command"""
        cmd = Twist()
        # Interpret action output from AI model
        # This depends on how your AI model is trained
        cmd.linear.x = float(action[0][0])  # Example: first output is linear velocity
        cmd.angular.z = float(action[0][1])  # Example: second output is angular velocity
        return cmd

    def simple_reactive_control(self):
        """Fallback simple reactive control"""
        cmd = Twist()

        # Simple obstacle avoidance based on laser scan
        if self.current_laser:
            # Check for obstacles in front (e.g., 30-degree cone in front)
            front_ranges = self.current_laser.ranges[:15] + self.current_laser.ranges[-15:]  # Approximate front
            min_front_dist = min([r for r in front_ranges if r > 0 and r < float('inf')], default=float('inf'))

            if min_front_dist < 0.5:  # Obstacle too close
                cmd.angular.z = 0.5  # Turn away
            else:
                cmd.linear.x = 0.3  # Move forward

        return cmd
```

## Understanding URDF for Humanoids

### Unified Robot Description Format (URDF)

URDF is an XML format for representing a robot model:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link - the root of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Head link -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>
  </link>

  <!-- Joint connecting head to base -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.0025"/>
    </inertial>
  </link>

  <!-- Left arm joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.1 0.15 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
```

### Using URDF in ROS 2 Launch Files

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'robot_description',
            default_value=[FindPackageShare('my_robot_description'), '/urdf/my_robot.urdf'],
            description='Full path to robot description file to load'
        )
    ]

    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': Command(['xacro ', LaunchConfiguration('robot_description')])}
        ]
    )

    return LaunchDescription(declared_arguments + [robot_state_publisher_node])
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
- Bridging Python AI agents to ROS controllers using rclpy
- Sensor data processing pipelines
- Control command generation
- URDF implementation for humanoid robots

## Hands-On Exercises

### Exercise 1: Custom Message Publisher/Subscriber
Create a custom message type and implement a publisher and subscriber for robot sensor data.

### Exercise 2: Service Implementation
Implement a service that performs path planning based on start and goal positions.

### Exercise 3: AI Controller Integration
Create a node that uses a simple neural network to control robot movement based on sensor data.

### Exercise 4: URDF Robot Model
Create a URDF model for a simple humanoid robot with at least 10 joints.

## Summary

This module has covered the core communication patterns in ROS 2: topics, services, and actions. You've learned how to implement each pattern and how to bridge AI agents to ROS controllers using rclpy. Additionally, you've gained an understanding of URDF for representing robot models, which is essential for humanoid robotics. These communication patterns form the "nervous system" of your robot, enabling coordinated behavior between different components.

