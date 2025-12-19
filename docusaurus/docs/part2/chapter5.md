---
title: "Chapter 5: ROS 2 Nodes, Topics, and Services Deep Dive"
sidebar_label: "Chapter 5: ROS 2 Communication Deep Dive"
---

# Chapter 5: ROS 2 Nodes, Topics, and Services Deep Dive

## Learning Objectives
- Master advanced ROS 2 communication patterns
- Implement the bridge between Python agents and ROS controllers using rclpy
- Understand URDF (Unified Robot Description Format) for humanoid robots
- Apply advanced communication techniques for Physical AI systems

## Introduction

This chapter delves deep into the advanced communication mechanisms of ROS 2, focusing on practical implementation of nodes, topics, and services in the context of Physical AI systems. We'll explore how to effectively bridge Python-based AI agents with ROS-based robot controllers, and examine the critical role of URDF in describing humanoid robot systems. Understanding these advanced communication patterns is essential for building sophisticated Physical AI applications.

## Advanced Node Implementation

### Node Composition and Management

In complex Physical AI systems, nodes often need to be composed and managed as part of larger systems:

```python
#!/usr/bin/env python3
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import threading


class ComposedRobotNode(Node):
    def __init__(self):
        super().__init__('composed_robot_node')

        # Create multiple callback groups for different processing threads
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.ai_group = MutuallyExclusiveCallbackGroup()

        # Publishers with different QoS profiles
        self.sensor_pub = self.create_publisher(
            SensorData,
            'robot/sensor_data',
            qos_profile=self._get_sensor_qos()
        )

        self.control_pub = self.create_publisher(
            ControlCommand,
            'robot/control_cmd',
            qos_profile=self._get_control_qos()
        )

        # Subscribers with different callback groups
        self.ai_cmd_sub = self.create_subscription(
            AICommand,
            'ai/commands',
            self.ai_command_callback,
            qos_profile=self._get_ai_qos(),
            callback_group=self.ai_group
        )

        self.user_cmd_sub = self.create_subscription(
            UserCommand,
            'user/commands',
            self.user_command_callback,
            qos_profile=self._get_user_qos(),
            callback_group=self.control_group
        )

        # Create timers with different callback groups
        self.sensor_timer = self.create_timer(
            0.01,  # 100Hz for sensor processing
            self.sensor_processing,
            callback_group=self.sensor_group
        )

        self.control_timer = self.create_timer(
            0.02,  # 50Hz for control
            self.control_loop,
            callback_group=self.control_group
        )

        self.ai_timer = self.create_timer(
            0.1,   # 10Hz for AI processing
            self.ai_processing,
            callback_group=self.ai_group
        )

    def _get_sensor_qos(self):
        """QoS for high-frequency sensor data"""
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

    def _get_control_qos(self):
        """QoS for critical control commands"""
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

    def _get_ai_qos(self):
        """QoS for AI-generated commands"""
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

    def sensor_processing(self):
        """High-frequency sensor processing"""
        # Process sensor data
        sensor_data = self.acquire_sensor_data()
        self.sensor_pub.publish(sensor_data)

    def control_loop(self):
        """Control loop with safety checks"""
        # Implement control logic with safety constraints
        control_cmd = self.compute_control_command()

        # Safety check before publishing
        if self.is_safe_to_execute(control_cmd):
            self.control_pub.publish(control_cmd)

    def ai_processing(self):
        """AI processing logic"""
        # Process AI decisions
        pass

    def ai_command_callback(self, msg):
        """Handle AI-generated commands"""
        self.get_logger().info(f'AI Command received: {msg.command}')
        # Process AI command with safety validation

    def user_command_callback(self, msg):
        """Handle user commands"""
        self.get_logger().info(f'User Command received: {msg.command}')
        # Process user command with priority handling
```

### Node Lifecycle Management

Advanced node management includes lifecycle considerations:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor
import rclpy


class LifecycleRobotController(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_robot_controller')

        # Initialize components that will be managed through lifecycle
        self.publisher = None
        self.subscriber = None
        self.service = None
        self.action_server = None
        self.timer = None

    def on_configure(self, state):
        """Configure the node - create publishers, subscribers, etc."""
        self.get_logger().info('Configuring robot controller')

        # Create communication interfaces
        self.publisher = self.create_publisher(RobotState, 'robot/state', 10)
        self.subscriber = self.create_subscription(
            RobotCommand, 'robot/command', self.command_callback, 10
        )
        self.service = self.create_service(GetRobotInfo, 'get_robot_info', self.info_service)

        # Create action server for complex tasks
        self.action_server = ActionServer(
            self,
            RobotNavigation,
            'navigate_to_pose',
            self.execute_navigation,
            goal_callback=self.navigation_goal_callback,
            cancel_callback=self.navigation_cancel_callback
        )

        # Create timers
        self.timer = self.create_timer(0.1, self.state_publisher)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Activate the node - start active operations"""
        self.get_logger().info('Activating robot controller')

        # Enable active operations
        self.publisher.enable()

        return super().on_activate(state)

    def on_deactivate(self, state):
        """Deactivate the node - pause active operations"""
        self.get_logger().info('Deactivating robot controller')

        # Disable active operations
        self.publisher.disable()

        return super().on_deactivate(state)

    def on_cleanup(self, state):
        """Clean up resources"""
        self.get_logger().info('Cleaning up robot controller')

        # Destroy communication interfaces
        self.destroy_publisher(self.publisher)
        self.destroy_subscription(self.subscriber)
        self.destroy_service(self.service)
        self.destroy_timer(self.timer)
        self.action_server.destroy()

        self.publisher = None
        self.subscriber = None
        self.service = None
        self.timer = None
        self.action_server = None

        return TransitionCallbackReturn.SUCCESS

    def command_callback(self, msg):
        """Handle robot commands"""
        if self.get_current_state().id() == LifecycleState.ACTIVE.id():
            # Process command only when active
            self.execute_command(msg)

    def state_publisher(self):
        """Publish robot state"""
        if self.get_current_state().id() == LifecycleState.ACTIVE.id():
            state_msg = RobotState()
            # Populate state message
            self.publisher.publish(state_msg)
```

## Deep Dive into Topics and Message Passing

### Advanced Topic Patterns

#### Fan-in Pattern
Multiple nodes publishing to a single topic:

```python
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Multiple sensor subscribers
        self.imu_sub = self.create_subscription(
            Imu, 'sensors/imu', self.imu_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, 'sensors/lidar', self.lidar_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, 'sensors/camera', self.camera_callback, 10
        )

        # Single fused output
        self.fused_pub = self.create_publisher(
            SensorFusionOutput, 'sensors/fused_data', 10
        )

        # Data fusion timer
        self.fusion_timer = self.create_timer(0.05, self.fuse_sensor_data)

        # Storage for sensor data
        self.imu_data = None
        self.lidar_data = None
        self.camera_data = None

    def imu_callback(self, msg):
        """Store IMU data"""
        self.imu_data = msg
        self.get_logger().debug('Received IMU data')

    def lidar_callback(self, msg):
        """Store LIDAR data"""
        self.lidar_data = msg
        self.get_logger().debug('Received LIDAR data')

    def camera_callback(self, msg):
        """Store camera data"""
        self.camera_data = msg
        self.get_logger().debug('Received camera data')

    def fuse_sensor_data(self):
        """Fuse sensor data into unified representation"""
        if all([self.imu_data, self.lidar_data, self.camera_data]):
            fused_msg = SensorFusionOutput()
            # Implement sensor fusion algorithm
            fused_msg.timestamp = self.get_clock().now().to_msg()
            fused_msg.imu_data = self.imu_data
            fused_msg.lidar_data = self.lidar_data
            fused_msg.camera_data = self.camera_data

            # Apply fusion algorithm
            fused_msg.fused_state = self.apply_sensor_fusion(
                self.imu_data, self.lidar_data, self.camera_data
            )

            self.fused_pub.publish(fused_msg)
```

#### Fan-out Pattern
Single publisher to multiple subscribers:

```python
class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Single publisher for robot state
        self.state_pub = self.create_publisher(
            RobotState, 'robot/state', 10
        )

        # Multiple subscribers will receive the same data
        self.state_timer = self.create_timer(0.01, self.publish_robot_state)

    def publish_robot_state(self):
        """Publish robot state to multiple subscribers"""
        state_msg = RobotState()
        # Populate with current robot state
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.joint_states = self.get_joint_states()
        state_msg.odometry = self.get_odometry()
        state_msg.imu = self.get_imu_data()

        self.state_pub.publish(state_msg)
```

### Quality of Service (QoS) Advanced Configuration

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, Lifespan


class QoSDemonstrationNode(Node):
    def __init__(self):
        super().__init__('qos_demo')

        # Different QoS profiles for different use cases

        # Real-time sensor data (high frequency, no need to keep old data)
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0.01)  # Must arrive within 10ms
        )

        # Safety-critical commands (must be delivered, keep for recovery)
        self.safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
            lifespan=Lifespan(seconds=30.0)  # Keep for 30 seconds
        )

        # Configuration parameters (infrequent, must be persistent)
        self.config_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers with different QoS
        self.sensor_pub = self.create_publisher(SensorData, 'sensors/data', self.sensor_qos)
        self.safety_pub = self.create_publisher(SafetyCommand, 'safety/commands', self.safety_qos)
        self.config_pub = self.create_publisher(ConfigData, 'config/parameters', self.config_qos)
```

## Services and Advanced Communication

### Advanced Service Implementation

```python
from rclpy.service import Service
from rclpy.callback_groups import ReentrantCallbackGroup


class AdvancedRobotServices(Node):
    def __init__(self):
        super().__init__('robot_services')

        # Use reentrant callback group for services that might call other services
        self.service_group = ReentrantCallbackGroup()

        # Multiple services with different purposes
        self.move_service = self.create_service(
            MoveRobot,
            'robot/move',
            self.move_robot_callback,
            callback_group=self.service_group
        )

        self.get_state_service = self.create_service(
            GetRobotState,
            'robot/get_state',
            self.get_state_callback,
            callback_group=self.service_group
        )

        self.execute_action_service = self.create_service(
            ExecuteAction,
            'robot/execute_action',
            self.execute_action_callback,
            callback_group=self.service_group
        )

    def move_robot_callback(self, request, response):
        """Handle robot movement requests"""
        self.get_logger().info(f'Moving robot to position: {request.target_position}')

        try:
            # Validate request
            if not self.is_valid_target(request.target_position):
                response.success = False
                response.message = 'Invalid target position'
                return response

            # Execute movement
            success = self.execute_movement(request.target_position)

            response.success = success
            response.message = 'Movement completed' if success else 'Movement failed'

        except Exception as e:
            self.get_logger().error(f'Error in move_robot: {e}')
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_state_callback(self, request, response):
        """Provide current robot state"""
        response.state = self.get_current_robot_state()
        response.timestamp = self.get_clock().now().to_msg()
        return response

    def execute_action_callback(self, request, response):
        """Execute complex robot action"""
        self.get_logger().info(f'Executing action: {request.action_name}')

        # Complex action execution
        result = self.execute_complex_action(request)

        response.success = result.success
        response.message = result.message
        response.execution_time = result.execution_time

        return response
```

## Bridging Python Agents to ROS Controllers

### AI Agent Integration Pattern

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge


class AIAgentBridge(Node):
    def __init__(self):
        super().__init__('ai_agent_bridge')

        # ROS interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # AI agent interfaces
        self.ai_model = self.load_ai_model()
        self.cv_bridge = CvBridge()

        # Data storage
        self.latest_laser_data = None
        self.latest_image_data = None
        self.ai_command_queue = []

        # AI processing timer
        self.ai_timer = self.create_timer(0.1, self.ai_processing_loop)

        # Safety parameters
        self.safety_distance = 0.5  # meters
        self.max_velocity = 1.0     # m/s

    def load_ai_model(self):
        """Load the AI model for navigation"""
        try:
            # Load your trained model
            model = tf.keras.models.load_model('path/to/navigation_model')
            self.get_logger().info('AI model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load AI model: {e}')
            return None

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.latest_laser_data = np.array(msg.ranges)
        # Filter out invalid ranges (inf, nan)
        self.latest_laser_data = np.where(
            (self.latest_laser_data == float('inf')) |
            (self.latest_laser_data == float('nan')),
            msg.range_max,
            self.latest_laser_data
        )

    def image_callback(self, msg):
        """Process camera image data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize for model input
            input_image = cv2.resize(cv_image, (224, 224))
            # Normalize
            input_image = input_image.astype(np.float32) / 255.0
            self.latest_image_data = input_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def ai_processing_loop(self):
        """Main AI processing loop"""
        if self.ai_model is None:
            return

        if self.latest_laser_data is not None and self.latest_image_data is not None:
            # Prepare input for AI model
            model_input = self.prepare_model_input(
                self.latest_laser_data,
                self.latest_image_data
            )

            # Get AI prediction
            ai_output = self.ai_model.predict(np.expand_dims(model_input, axis=0))

            # Convert AI output to ROS command
            cmd_vel = self.convert_ai_output_to_cmd_vel(ai_output)

            # Apply safety checks
            safe_cmd_vel = self.apply_safety_constraints(cmd_vel)

            # Publish command
            self.cmd_vel_pub.publish(safe_cmd_vel)

    def prepare_model_input(self, laser_data, image_data):
        """Prepare sensor data for AI model input"""
        # Normalize laser data
        normalized_laser = laser_data / np.max(laser_data) if np.max(laser_data) > 0 else laser_data

        # Combine sensor data (this is a simplified example)
        # In practice, you might use more sophisticated fusion techniques
        return {
            'laser': normalized_laser,
            'image': image_data
        }

    def convert_ai_output_to_cmd_vel(self, ai_output):
        """Convert AI model output to Twist message"""
        cmd_vel = Twist()

        # Example: AI outputs [linear_velocity, angular_velocity]
        cmd_vel.linear.x = float(ai_output[0][0])
        cmd_vel.angular.z = float(ai_output[0][1])

        return cmd_vel

    def apply_safety_constraints(self, cmd_vel):
        """Apply safety constraints to commands"""
        # Limit velocities
        cmd_vel.linear.x = max(-self.max_velocity, min(self.max_velocity, cmd_vel.linear.x))
        cmd_vel.angular.z = max(-1.0, min(1.0, cmd_vel.angular.z))

        # Safety check based on laser data
        if self.latest_laser_data is not None:
            min_distance = np.min(self.latest_laser_data)
            if min_distance < self.safety_distance:
                # Emergency stop if obstacle is too close
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.get_logger().warn('Safety stop: obstacle detected')

        return cmd_vel


def main(args=None):
    rclpy.init(args=args)

    ai_agent_bridge = AIAgentBridge()

    try:
        rclpy.spin(ai_agent_bridge)
    except KeyboardInterrupt:
        ai_agent_bridge.get_logger().info('Shutting down AI agent bridge')
    finally:
        ai_agent_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## URDF for Humanoid Robots

### Understanding URDF Structure

URDF (Unified Robot Description Format) is XML-based and describes robot kinematics:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints connecting links -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <origin xyz="0 0.1 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.1 0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

</robot>
```

### URDF with Transmission and Gazebo Integration

```xml
<?xml version="1.0"?>
<robot name="humanoid_with_transmissions" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include other URDF files -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.urdf.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/transmissions.urdf.xacro"/>

  <!-- Robot base -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo-specific tags -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <!-- Joint with transmission for ROS control -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Transmission for ros_control -->
  <transmission name="trans_joint1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="motor1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

## Integration with Physical AI Systems

### Multi-Agent Communication Pattern

```python
class MultiAgentRobotSystem(Node):
    def __init__(self):
        super().__init__('multi_agent_system')

        # AI agent communication
        self.ai_command_pub = self.create_publisher(AICommand, 'ai/commands', 10)
        self.ai_feedback_sub = self.create_subscription(AIFeedback, 'ai/feedback', self.ai_feedback_callback, 10)

        # Robot controller communication
        self.robot_command_pub = self.create_publisher(RobotCommand, 'robot/commands', 10)
        self.robot_state_sub = self.create_subscription(RobotState, 'robot/state', self.robot_state_callback, 10)

        # Planning and execution
        self.planning_pub = self.create_publisher(Plan, 'planning/goal', 10)
        self.execution_sub = self.create_subscription(ExecutionStatus, 'execution/status', self.execution_callback, 10)

        # Main processing timer
        self.main_timer = self.create_timer(0.05, self.main_processing_loop)

        # Internal state
        self.current_robot_state = None
        self.current_plan = None
        self.ai_goals = []

    def main_processing_loop(self):
        """Main processing loop coordinating AI and robot control"""
        if self.current_robot_state is None:
            return

        # Process AI goals and generate robot commands
        if self.ai_goals:
            # Plan based on AI goals
            plan = self.generate_plan_from_goals(self.ai_goals, self.current_robot_state)

            # Execute plan
            cmd = self.plan_to_robot_command(plan)
            self.robot_command_pub.publish(cmd)

    def ai_feedback_callback(self, msg):
        """Handle feedback from AI system"""
        self.get_logger().info(f'AI feedback: {msg.status}')
        # Process AI feedback and update internal state

    def robot_state_callback(self, msg):
        """Update current robot state"""
        self.current_robot_state = msg

    def execution_callback(self, msg):
        """Handle execution status updates"""
        if msg.status == ExecutionStatus.COMPLETED:
            # Remove completed goal
            if self.ai_goals:
                self.ai_goals.pop(0)
```

## Knowledge Check

1. Explain the difference between fan-in and fan-out communication patterns in ROS 2.
2. Describe how Quality of Service (QoS) profiles affect communication reliability.
3. What are the key components of a URDF file for humanoid robots?

## Hands-On Exercise

1. Create a ROS 2 node that implements the AI agent bridge pattern
2. Design a URDF file for a simple humanoid robot with at least 6 degrees of freedom
3. Implement a service that provides robot state information to multiple clients
4. Test the system with different QoS configurations to observe behavior differences

## Summary

This chapter has explored advanced ROS 2 communication patterns, focusing on the integration between Python-based AI agents and ROS-based robot controllers. The deep dive into topics, services, and URDF demonstrates how to build sophisticated Physical AI systems that can effectively bridge the gap between AI decision-making and physical robot control. Understanding these advanced patterns is crucial for developing robust and efficient Physical AI applications.

## Next Steps

In the following chapters, we'll explore simulation environments, NVIDIA Isaac integration, and the implementation of advanced Physical AI systems for humanoid robotics.