---
sidebar_position: 5
title: "Chapter 5: ROS 2 Nodes, Topics, and Services"
---

# Chapter 5: ROS 2 Nodes, Topics, and Services

## Learning Objectives
- Master advanced ROS 2 communication patterns
- Bridge Python agents to ROS controllers using rclpy
- Understand URDF (Unified Robot Description Format) for humanoids
- Implement complex multi-node systems with proper communication

## Advanced ROS 2 Communication Patterns

### Deep Dive into Topics and Publishers/Subscribers

In Chapter 3, we introduced the basic publish/subscribe pattern. Now let's explore advanced patterns and best practices for robust communication.

#### Message Synchronization
When dealing with multiple sensor streams, synchronization becomes crucial:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Create subscribers for different sensor types
        self.image_sub = Subscriber(self, Image, 'camera/image_raw')
        self.laser_sub = Subscriber(self, LaserScan, 'scan')
        self.imu_sub = Subscriber(self, Imu, 'imu/data')

        # Synchronize messages based on timestamps
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.laser_sub, self.imu_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.cv_bridge = CvBridge()
        self.get_logger().info('Sensor Fusion Node initialized')

    def synchronized_callback(self, image_msg, laser_msg, imu_msg):
        """Process synchronized sensor data"""
        # Convert image to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # Process laser scan
        ranges = np.array(laser_msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        # Process IMU data
        orientation = imu_msg.orientation
        angular_velocity = imu_msg.angular_velocity
        linear_acceleration = imu_msg.linear_acceleration

        # Fusion logic here
        self.get_logger().info(f'Fused data: image shape {cv_image.shape}, {len(valid_ranges)} valid laser ranges')

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
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

#### Publisher Throttling and Rate Control
For high-frequency data, rate control is essential:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.time import Time
import time

class RateControlledPublisher(Node):
    def __init__(self):
        super().__init__('rate_controlled_publisher')

        # Publisher with QoS profile for high frequency
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        qos_profile = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.publisher = self.create_publisher(JointState, 'high_freq_joint_states', qos_profile)

        # Use rate to control publishing frequency
        self.rate = self.create_rate(100)  # 100 Hz

        # Initialize joint names
        self.joint_names = [f'joint_{i}' for i in range(12)]

        self.get_logger().info('Rate Controlled Publisher initialized')

    def publish_joint_states(self):
        """Publish joint states at controlled rate"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        msg.position = [0.1 * i for i in range(len(self.joint_names))]
        msg.velocity = [0.05 * i for i in range(len(self.joint_names))]
        msg.effort = [0.01 * i for i in range(len(self.joint_names))]

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RateControlledPublisher()
    try:
        while rclpy.ok():
            node.publish_joint_states()
            node.rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Services: Advanced Patterns and Use Cases

#### Asynchronous Services
For long-running operations, consider using asynchronous service handling:

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from example_interfaces.srv import Trigger
import threading
import time

class AsyncServiceNode(Node):
    def __init__(self):
        super().__init__('async_service_node')

        # Create service with custom callback group for threading
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.service = self.create_service(
            Trigger,
            'async_operation',
            self.async_service_callback,
            callback_group=self.service_callback_group
        )

        self.get_logger().info('Async Service Node initialized')

    def async_service_callback(self, request, response):
        """Handle service request asynchronously"""
        self.get_logger().info('Starting async operation...')

        # Simulate long-running operation in separate thread
        def long_running_task():
            time.sleep(3)  # Simulate work
            response.success = True
            response.message = 'Async operation completed successfully'
            self.get_logger().info('Async operation completed')

        thread = threading.Thread(target=long_running_task)
        thread.start()
        thread.join()  # Wait for completion in this example

        return response

def main(args=None):
    rclpy.init(args=args)

    # Use multi-threaded executor to handle concurrent requests
    executor = MultiThreadedExecutor()

    node = AsyncServiceNode()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Service Composition
Creating services that coordinate multiple operations:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
from std_msgs.msg import Bool
import asyncio

class ServiceCompositionNode(Node):
    def __init__(self):
        super().__init__('service_composition_node')

        # Create multiple services for different operations
        self.calibration_service = self.create_service(
            Trigger, 'calibrate_sensors', self.calibrate_sensors_callback)
        self.home_service = self.create_service(
            Trigger, 'home_joints', self.home_joints_callback)
        self.reset_service = self.create_service(
            Trigger, 'reset_system', self.reset_system_callback)

        # Publisher for system status
        self.status_pub = self.create_publisher(Bool, 'system_ready', 10)

        self.get_logger().info('Service Composition Node initialized')

    def calibrate_sensors_callback(self, request, response):
        """Calibrate all sensors"""
        self.get_logger().info('Calibrating sensors...')

        # Simulate calibration process
        # In real implementation, this would call individual sensor calibration services
        time.sleep(2)

        response.success = True
        response.message = 'Sensors calibrated successfully'
        return response

    def home_joints_callback(self, request, response):
        """Home all joints to zero position"""
        self.get_logger().info('Homing joints...')

        # Simulate joint homing process
        time.sleep(3)

        response.success = True
        response.message = 'Joints homed successfully'
        return response

    def reset_system_callback(self, request, response):
        """Reset entire system"""
        self.get_logger().info('Resetting system...')

        # Call other services as part of reset
        calibrate_result = self.calibrate_sensors_callback(request, Trigger.Response())
        home_result = self.home_joints_callback(request, Trigger.Response())

        if calibrate_result.success and home_result.success:
            response.success = True
            response.message = 'System reset successfully'

            # Publish system ready status
            status_msg = Bool()
            status_msg.data = True
            self.status_pub.publish(status_msg)
        else:
            response.success = False
            response.message = 'System reset failed'

        return response

def main(args=None):
    rclpy.init(args=args)
    node = ServiceCompositionNode()
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

## Bridging Python Agents to ROS Controllers

### Agent-Controller Architecture
Creating bridges between high-level AI agents and low-level ROS controllers:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
import numpy as np
import threading

class AgentControllerBridge(Node):
    def __init__(self):
        super().__init__('agent_controller_bridge')

        # Subscribers for sensor data
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publishers for commands
        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)

        # Agent command interface
        self.agent_cmd_sub = self.create_subscription(
            String, 'agent_commands', self.agent_command_callback, 10)

        # Internal state
        self.current_joint_states = JointState()
        self.current_imu_data = Imu()
        self.agent_command_queue = []

        # Lock for thread safety
        self.state_lock = threading.Lock()

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info('Agent-Controller Bridge initialized')

    def joint_state_callback(self, msg):
        """Update joint state from ROS"""
        with self.state_lock:
            self.current_joint_states = msg

    def imu_callback(self, msg):
        """Update IMU data from ROS"""
        with self.state_lock:
            self.current_imu_data = msg

    def agent_command_callback(self, msg):
        """Receive commands from AI agent"""
        command = msg.data
        self.get_logger().info(f'Received agent command: {command}')

        with self.state_lock:
            self.agent_command_queue.append(command)

    def control_loop(self):
        """Main control loop that processes agent commands"""
        with self.state_lock:
            if self.agent_command_queue:
                command = self.agent_command_queue.pop(0)
                self.execute_agent_command(command)

    def execute_agent_command(self, command):
        """Execute a command received from the AI agent"""
        if command == 'move_forward':
            self.move_forward()
        elif command == 'turn_left':
            self.turn_left()
        elif command == 'move_arm':
            self.move_arm()
        elif command == 'balance':
            self.balance_robot()
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def move_forward(self):
        """Move robot forward"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5  # m/s
        cmd_vel.angular.z = 0.0
        self.velocity_pub.publish(cmd_vel)

    def turn_left(self):
        """Turn robot left"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.5  # rad/s
        self.velocity_pub.publish(cmd_vel)

    def move_arm(self):
        """Move arm joints"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow']
        joint_cmd.position = [0.5, 0.3, -0.5, 0.3]  # rad
        joint_cmd.velocity = [0.1, 0.1, 0.1, 0.1]  # rad/s
        self.joint_cmd_pub.publish(joint_cmd)

    def balance_robot(self):
        """Implement balance control based on IMU data"""
        with self.state_lock:
            # Simple balance control based on IMU pitch and roll
            pitch = self.current_imu_data.orientation.y
            roll = self.current_imu_data.orientation.x

            # Adjust joint positions to counteract imbalance
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.name = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']

            # Simple proportional control
            hip_adjustment = -roll * 0.1  # Adjust based on roll
            ankle_adjustment = -pitch * 0.1  # Adjust based on pitch

            joint_cmd.position = [
                hip_adjustment, -hip_adjustment,  # Hip adjustments
                ankle_adjustment, -ankle_adjustment  # Ankle adjustments
            ]
            self.joint_cmd_pub.publish(joint_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = AgentControllerBridge()
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

### Advanced Agent Integration
Creating a more sophisticated bridge with state management:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import numpy as np
from collections import deque
import json

class AdvancedAgentBridge(Node):
    def __init__(self):
        super().__init__('advanced_agent_bridge')

        # Publishers and subscribers
        self.state_pub = self.create_publisher(Float32MultiArray, 'robot_state', 10)
        self.action_sub = self.create_publisher(Float32MultiArray, 'agent_actions', 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # State history for temporal context
        self.state_history = deque(maxlen=10)

        # Robot parameters
        self.joint_names = []
        self.joint_positions = {}
        self.joint_velocities = {}
        self.imu_orientation = [0.0, 0.0, 0.0, 1.0]
        self.imu_angular_velocity = [0.0, 0.0, 0.0]
        self.imu_linear_acceleration = [0.0, 0.0, 0.0]

        # Control parameters
        self.max_linear_speed = 1.0
        self.max_angular_speed = 1.0
        self.control_frequency = 50.0

        # Timer for state publishing
        self.state_timer = self.create_timer(1.0/self.control_frequency, self.publish_robot_state)

        self.get_logger().info('Advanced Agent Bridge initialized')

    def joint_callback(self, msg):
        """Update joint state"""
        self.joint_names = msg.name
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        self.imu_angular_velocity = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        self.imu_linear_acceleration = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]

    def publish_robot_state(self):
        """Publish current robot state to AI agent"""
        state_vector = self.get_state_vector()

        state_msg = Float32MultiArray()
        state_msg.data = state_vector

        self.state_pub.publish(state_msg)

        # Add to history
        self.state_history.append(state_vector)

    def get_state_vector(self):
        """Create state vector for AI agent"""
        state = []

        # Joint positions (normalized)
        for joint_name in ['left_hip', 'left_knee', 'left_ankle',
                          'right_hip', 'right_knee', 'right_ankle',
                          'left_shoulder', 'left_elbow', 'left_wrist',
                          'right_shoulder', 'right_elbow', 'right_wrist']:
            pos = self.joint_positions.get(joint_name, 0.0)
            # Normalize to [-1, 1] range
            normalized_pos = max(-1.0, min(1.0, pos / 3.14))
            state.append(normalized_pos)

        # Joint velocities (normalized)
        for joint_name in ['left_hip', 'left_knee', 'left_ankle',
                          'right_hip', 'right_knee', 'right_ankle',
                          'left_shoulder', 'left_elbow', 'left_wrist',
                          'right_shoulder', 'right_elbow', 'right_wrist']:
            vel = self.joint_velocities.get(joint_name, 0.0)
            # Normalize to [-1, 1] range
            normalized_vel = max(-1.0, min(1.0, vel / 10.0))
            state.append(normalized_vel)

        # IMU orientation (roll, pitch, yaw)
        # Convert quaternion to Euler angles
        quat = self.imu_orientation
        roll = np.arctan2(2*(quat[3]*quat[0] + quat[1]*quat[2]),
                         1 - 2*(quat[0]**2 + quat[1]**2))
        pitch = np.arcsin(2*(quat[3]*quat[1] - quat[2]*quat[0]))
        yaw = np.arctan2(2*(quat[3]*quat[2] + quat[0]*quat[1]),
                        1 - 2*(quat[1]**2 + quat[2]**2))

        state.extend([roll, pitch, yaw])

        # IMU angular velocity
        state.extend(self.imu_angular_velocity)

        # IMU linear acceleration
        state.extend(self.imu_linear_acceleration)

        return state

    def execute_action(self, action_vector):
        """Execute action from AI agent"""
        if len(action_vector) >= 4:  # At least linear and angular commands
            # Decode action vector
            linear_x = max(-self.max_linear_speed, min(self.max_linear_speed, action_vector[0]))
            angular_z = max(-self.max_angular_speed, min(self.max_angular_speed, action_vector[1]))

            # Create and publish velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = linear_x
            cmd_vel.angular.z = angular_z
            self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedAgentBridge()
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

## URDF for Humanoid Robots

### Understanding URDF Structure
URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS. For humanoid robots, URDF becomes more complex due to the articulated structure.

**Basic URDF Structure:**
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Complete Humanoid URDF Example
Here's a more complete humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
    </inertial>
  </link>

  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_shoulder_to_upper_arm" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.075 0 -0.15" rpy="1.57 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="2.36" effort="50" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="left_upper_arm_to_lower_arm" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0" effort="30" velocity="2"/>
  </joint>

  <!-- Right Arm (similar to left, mirrored) -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_right_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.2 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_shoulder_to_upper_arm" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.075 0 -0.15" rpy="1.57 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="2.36" effort="50" velocity="2"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="right_upper_arm_to_lower_arm" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0" effort="30" velocity="2"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.1 0.15 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.15 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_left_hip" type="revolute">
    <parent link="torso"/>
    <child link="left_hip"/>
    <origin xyz="-0.1 0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.008"/>
    </inertial>
  </link>

  <joint name="left_hip_to_upper_leg" type="revolute">
    <parent link="left_hip"/>
    <child link="left_upper_leg"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.006"/>
    </inertial>
  </link>

  <joint name="left_upper_leg_to_lower_leg" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_lower_leg_to_foot" type="fixed">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <!-- Right Leg (similar to left) -->
  <link name="right_hip">
    <visual>
      <geometry>
        <box size="0.1 0.15 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.15 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_right_hip" type="revolute">
    <parent link="torso"/>
    <child link="right_hip"/>
    <origin xyz="-0.1 -0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.008"/>
    </inertial>
  </link>

  <joint name="right_hip_to_upper_leg" type="revolute">
    <parent link="right_hip"/>
    <child link="right_upper_leg"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.006"/>
    </inertial>
  </link>

  <joint name="right_upper_leg_to_lower_leg" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0.785" effort="100" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_lower_leg_to_foot" type="fixed">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugin for simulation -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

### URDF with Transmission Definitions
For controlling the joints in simulation:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_transmissions">
  <!-- Include the main robot definition -->
  <xacro:include filename="simple_humanoid.urdf"/>

  <!-- Transmissions for ROS Control -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="torso_to_left_hip">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_to_upper_leg">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Add transmissions for all other joints similarly -->
  <!-- ... -->
</robot>
```

## Practical Integration Example

Let's create a complete example that demonstrates how to use URDF with ROS 2 nodes:

**URDF Publisher Node:**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np

class URDFRobotController(Node):
    def __init__(self):
        super().__init__('urdf_robot_controller')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Transform broadcaster for TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing joint states
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

        # Initialize joint names
        self.joint_names = [
            'torso_to_head',
            'torso_to_left_shoulder', 'left_shoulder_to_upper_arm', 'left_upper_arm_to_lower_arm',
            'torso_to_right_shoulder', 'right_shoulder_to_upper_arm', 'right_upper_arm_to_lower_arm',
            'torso_to_left_hip', 'left_hip_to_upper_leg', 'left_upper_leg_to_lower_leg',
            'torso_to_right_hip', 'right_hip_to_upper_leg', 'right_upper_leg_to_lower_leg'
        ]

        # Initialize joint positions (oscillating for demo)
        self.time_offset = 0.0

        self.get_logger().info('URDF Robot Controller initialized')

    def publish_joint_states(self):
        """Publish joint states for visualization"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names

        # Create oscillating joint positions for demonstration
        positions = []
        for i, joint_name in enumerate(self.joint_names):
            # Different oscillation patterns for different joint types
            if 'hip' in joint_name or 'knee' in joint_name:
                # Leg joints with walking pattern
                pos = 0.5 * math.sin(self.time_offset + i * 0.5)
            elif 'shoulder' in joint_name or 'elbow' in joint_name:
                # Arm joints with reaching pattern
                pos = 0.3 * math.sin(self.time_offset * 1.5 + i * 0.3)
            elif 'head' in joint_name:
                # Head with nodding pattern
                pos = 0.2 * math.sin(self.time_offset * 2.0)
            else:
                pos = 0.0

            positions.append(pos)

        msg.position = positions
        self.joint_pub.publish(msg)

        # Broadcast transforms
        self.broadcast_transforms()

        self.time_offset += 0.05

    def broadcast_transforms(self):
        """Broadcast static and dynamic transforms"""
        # In a real implementation, you would compute transforms from forward kinematics
        # For this example, we'll broadcast a simple base transform

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = URDFRobotController()
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

## Knowledge Check

1. How do you synchronize multiple sensor streams in ROS 2?
2. What are the differences between services and topics, and when should each be used?
3. How does the agent-controller bridge pattern facilitate AI integration in robotics?
4. What are the key components of a URDF file for a humanoid robot?

## Summary

This chapter explored advanced ROS 2 communication patterns, including topic synchronization, service composition, and advanced publisher/subscriber patterns. We also covered the integration of Python agents with ROS controllers, demonstrating how to bridge high-level AI systems with low-level robot control. Finally, we examined URDF for humanoid robots, showing how to describe complex articulated structures in XML format for simulation and visualization.

## Next Steps

In the next module, we'll explore the Digital Twin environment with Gazebo simulation, learning how to set up physics simulation environments and integrate them with our ROS 2 systems.