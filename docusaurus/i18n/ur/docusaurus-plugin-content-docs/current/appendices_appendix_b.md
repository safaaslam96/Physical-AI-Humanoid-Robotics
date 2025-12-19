---
title: "اپنڈکس بی: سافٹ ویئر کی تنصیب اور کنفیگریشن"
sidebar_label: "اپنڈکس بی: سافٹ ویئر کی تنصیب"
---

# اپنڈکس بی: سافٹ ویئر کی تنصیب اور کنفیگریشن

## سیکھنے کے اہداف

- ہیومنوائڈ روبوٹکس کے لیے سافٹ ویئر اسٹیک کو سمجھنا
- ROS 2 اور متعلقہ لائبریریز کی تنصیب کرنا
- سافٹ ویئر کنفیگریشن اور کیلیبریشن کا عمل سیکھنا
- ڈیبگنگ اور ٹیسٹنگ کے ٹولز کا استعمال کرنا

## سافٹ ویئر کا جامع اسٹیک

### سافٹ ویئر کی ترتیب

ہیومنوائڈ روبوٹکس کے لیے سافٹ ویئر کا اسٹیک مختلف لیئروں میں منظم ہے:

```yaml
Software Stack Layers:
  application_layer:
    description: High-level applications and behaviors
    components:
      - humanoid_behavior_engine
      - task_planning_system
      - human_robot_interaction
      - cognitive_reasoning
  middleware_layer:
    description: Communication and coordination
    components:
      - ros2_framework
      - real_time_communication
      - data_distribution_service
  control_layer:
    description: Motion and control algorithms
    components:
      - whole_body_controller
      - trajectory_planner
      - balance_control
      - inverse_kinematics
  hardware_abstraction:
    description: Hardware interface layer
    components:
      - hardware_drivers
      - sensor_interfaces
      - actuator_controllers
      - communication_protocols
  operating_system:
    description: Base system layer
    components:
      - real_time_linux
      - device_drivers
      - power_management
      - security_modules
```

### ROS 2 کی تنصیب

#### ROS 2 Iron Irwini کی تنصیب

ROS 2 Iron Irwini کی تنصیب کے لیے درج ذیل اقدامات کریں:

```bash
# Ubuntu 22.04 کے لیے ROS 2 Iron تنصیب کے اقدامات
sudo apt update && sudo apt upgrade -y

# ROS 2 GPG کی اور ریپوزٹری شامل کریں
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update

sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-iron-desktop-full
sudo apt install ros-dev-tools
```

#### ROS 2 کنفیگریشن

```bash
# ROS 2 کنفیگریشن
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
source ~/.bashrc

# ROS 2 workspace تیار کریں
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws
colcon build --symlink-install

echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ضروری لائبریریز اور ڈیپنڈنسیز

#### کمپیوٹر وژن لائبریریز

```bash
# OpenCV کی تنصیب
sudo apt install libopencv-dev python3-opencv

# Intel RealSense SDK
sudo apt install librealsense2-dev librealsense2-utils

# YOLO اور deep learning لائبریریز
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install ultralytics opencv-python
```

#### مشین لرننگ لائبریریز

```bash
# Python ML لائبریریز
pip3 install numpy scipy pandas matplotlib
pip3 install scikit-learn tensorflow keras
pip3 install transformers torch torchvision
pip3 install stable-baselines3 gymnasium
```

#### کنٹرول اور میتھمیٹیکل لائبریریز

```bash
# NumPy، SciPy، SymPy
pip3 install numpy scipy sympy

# Control Systems Library
pip3 install control slycot

# Symbolic Mathematics
pip3 install sympy

# Optimization Libraries
pip3 install cvxpy scipy.optimize
```

## ہیومنوائڈ سافٹ ویئر کمپوننٹس

### ROS 2 پیکجز

#### اصل ہیومنوائڈ پیکجز

```yaml
Humanoid ROS Packages:
  humanoid_description:
    purpose: URDF model of the humanoid robot
    files:
      - urdf/humanoid.urdf.xacro
      - meshes/base_link.STL
      - launch/description.launch.py
      - config/joint_limits.yaml
    dependencies: xacro joint_state_publisher robot_state_publisher
  humanoid_control:
    purpose: Whole-body control system
    files:
      - src/controller_manager.cpp
      - config/controllers.yaml
      - launch/control.launch.py
      - include/humanoid_control/humanoid_controller.hpp
    dependencies: controller_manager joint_trajectory_controller
  humanoid_perception:
    purpose: Vision and sensor processing
    files:
      - src/object_detector.cpp
      - src/face_tracker.cpp
      - src/environment_mapper.cpp
      - launch/perception.launch.py
    dependencies: opencv_contrib_libs realsense2_camera
  humanoid_navigation:
    purpose: Path planning and navigation
    files:
      - src/path_planner.cpp
      - src/local_planner.cpp
      - config/nav2_params.yaml
      - launch/navigation.launch.py
    dependencies: nav2_bringup slam_toolbox
  humanoid_manipulation:
    purpose: Grasping and manipulation
    files:
      - src/grasp_planner.cpp
      - src/ik_solver.cpp
      - config/manipulation.yaml
      - launch/manipulation.launch.py
    dependencies: moveit2 moveit_msgs geometric_shapes
```

### کنٹرول سسٹم کی ترتیب

#### ہول بڈی کنٹرول (Whole Body Control)

```cpp
// Whole body control system
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainiksolver.hpp>

class WholeBodyController {
public:
    WholeBodyController();
    ~WholeBodyController();

    bool initialize();
    bool computeControl(const std::vector<double>& joint_states,
                       const std::vector<double>& desired_positions,
                       std::vector<double>& control_outputs);

    // Balance control
    void updateBalanceControl(const Eigen::Vector3d& com_position,
                             const Eigen::Vector3d& com_velocity,
                             const Eigen::Vector3d& zmp_reference);

    // Trajectory generation
    void generateTrajectory(const Eigen::VectorXd& start_pos,
                           const Eigen::VectorXd& end_pos,
                           double duration);

    // Inverse kinematics
    bool solveInverseKinematics(const std::vector<Eigen::Affine3d>& desired_poses,
                               std::vector<double>& joint_angles);

private:
    // Robot model
    KDL::Chain left_leg_chain_;
    KDL::Chain right_leg_chain_;
    KDL::Chain left_arm_chain_;
    KDL::Chain right_arm_chain_;

    // Controllers
    std::vector<std::unique_ptr<PIDController>> joint_controllers_;
    std::unique_ptr<BalanceController> balance_controller_;
    std::unique_ptr<IKSolver> ik_solver_;

    // Trajectory generators
    std::vector<std::unique_ptr<TrajectoryGenerator>> trajectory_generators_;

    // Control parameters
    double control_frequency_;
    double dt_; // time step
};

WholeBodyController::WholeBodyController() : control_frequency_(1000.0), dt_(0.001) {
    // Initialize joint controllers
    for(int i = 0; i < NUM_JOINTS; i++) {
        joint_controllers_.push_back(std::make_unique<PIDController>());
    }

    // Initialize balance controller
    balance_controller_ = std::make_unique<BalanceController>();

    // Initialize IK solver
    ik_solver_ = std::make_unique<IKSolver>();
}

bool WholeBodyController::computeControl(const std::vector<double>& joint_states,
                                        const std::vector<double>& desired_positions,
                                        std::vector<double>& control_outputs) {
    if(joint_states.size() != desired_positions.size() ||
       joint_states.size() != NUM_JOINTS) {
        return false;
    }

    control_outputs.resize(NUM_JOINTS);

    // Compute control for each joint
    for(size_t i = 0; i < joint_controllers_.size(); i++) {
        double error = desired_positions[i] - joint_states[i];
        control_outputs[i] = joint_controllers_[i]->compute(error, dt_);
    }

    return true;
}

void WholeBodyController::updateBalanceControl(const Eigen::Vector3d& com_position,
                                              const Eigen::Vector3d& com_velocity,
                                              const Eigen::Vector3d& zmp_reference) {
    // Implement ZMP-based balance control
    balance_controller_->update(com_position, com_velocity, zmp_reference);

    // Adjust desired joint positions based on balance requirements
    balance_controller_->adjustJointPositions();
}
```

### ROS 2 لانچ فائلز

#### ہیومنوائڈ کنٹرول لانچ فائل

```python
# launch/humanoid_control.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package name
    pkg_name = 'humanoid_control'

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_rviz = LaunchConfiguration('launch_rviz')

    # Paths
    config_dir = os.path.join(get_package_share_directory(pkg_name), 'config')
    rviz_config_path = os.path.join(config_dir, 'humanoid_control.rviz')

    # Controller manager node
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            os.path.join(config_dir, 'controllers.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'publish_frequency': 50.0}
        ],
        output='screen'
    )

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Whole body controller
    whole_body_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['whole_body_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz2 node (conditionally launched)
    rviz_node = Node(
        condition=IfCondition(launch_rviz),
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Event handler for RViz
    rviz_event_handler = RegisterEventHandler(
        condition=IfCondition(PythonExpression(['"', launch_rviz, '" == "true"'])),
        event_handler=rviz_node
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz if true'
        ),
        controller_manager_node,
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        whole_body_controller_spawner,
        rviz_event_handler
    ])
```

### کنفیگریشن فائلز

#### کنٹرولرز کی کنفیگریشن

```yaml
# config/controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz
    use_sim_time: false

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    whole_body_controller:
      type: humanoid_control/HumanoidController

    left_leg_controller:
      type: position_controllers/JointGroupPositionController

    right_leg_controller:
      type: position_controllers/JointGroupPositionController

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

    right_arm_controller:
      type: position_controllers/JointGroupPositionController

    head_controller:
      type: position_controllers/JointGroupPositionController

whole_body_controller:
  ros__parameters:
    joints:
      - left_hip_yaw
      - left_hip_roll
      - left_hip_pitch
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll
      - right_hip_yaw
      - right_hip_roll
      - right_hip_pitch
      - right_knee
      - right_ankle_pitch
      - right_ankle_roll
      - torso_yaw
      - torso_pitch
      - torso_roll
      - left_shoulder_pitch
      - left_shoulder_roll
      - left_shoulder_yaw
      - left_elbow
      - left_wrist_yaw
      - left_wrist_pitch
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
      - right_wrist_yaw
      - right_wrist_pitch
      - head_yaw
      - head_pitch

    gains:
      kp: 100.0
      ki: 0.0
      kd: 10.0

    control_mode: position_velocity
    publish_rate: 1000.0

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_yaw
      - left_hip_roll
      - left_hip_pitch
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll
    interface_name: position

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_yaw
      - right_hip_roll
      - right_hip_pitch
      - right_knee
      - right_ankle_pitch
      - right_ankle_roll
    interface_name: position

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch
      - left_shoulder_roll
      - left_shoulder_yaw
      - left_elbow
      - left_wrist_yaw
      - left_wrist_pitch
    interface_name: position

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
      - right_wrist_yaw
      - right_wrist_pitch
    interface_name: position

head_controller:
  ros__parameters:
    joints:
      - head_yaw
      - head_pitch
    interface_name: position
```

## ڈیبگنگ اور ٹیسٹنگ کے ٹولز

### ROS 2 ڈیبگنگ ٹولز

#### ros2cli ٹولز

```bash
# Nodes کی فہرست حاصل کریں
ros2 node list

# Topics کی فہرست حاصل کریں
ros2 topic list

# Services کی فہرست حاصل کریں
ros2 service list

# Parameters کی فہرست حاصل کریں
ros2 param list <node_name>

# Topic کا محتوا دیکھیں
ros2 topic echo /joint_states

# Service کال کریں
ros2 service call /set_parameters rcl_interfaces/srv/SetParameters "{parameters: [{name: 'kp', value: {double_value: 100.0}}]}"

# Action کال کریں
ros2 action send_goal /follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{trajectory: {...}}"
```

#### rviz2 کنفیگریشن

```yaml
# RViz2 کنفیگریشن فائل
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
        - /InteractiveMarkers1
      Splitter Ratio: 0.5
    Tree Height: 787
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Mass Properties:
        Inertia: false
        Mass: false
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/SetGoal
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1028
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd0000000100000000000002a2000003a2fc0200000008fb000000100044006900730070006c006100790073010000003d000003a2000000c900fffffffb0000000800480065006c00700000000342000000bb0000006e00fffffffb0000000a0056006900650077007300000003b0000000b0000000a400fffffffb0000000c0054006f006f006c00730100000000ffffffff0000000000000000fb0000000a00500061006e0065006c00730100000000000003a20000000100000000fb0000000a00500061006e0065006c00730100000000000003a20000000100000000fb0000000a00530065006c0065006300740069006f006e010000025a000000b20000000000000000fb0000000a00560069006500770073030000004e00000080000000e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d00650100000000000004500000000000000000000004ba000003a200000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1853
  X: 67
  Y: 27
```

### ٹیسٹنگ فریمورک

#### یونٹ ٹیسٹنگ

```cpp
// Unit testing for humanoid controller
#include <gtest/gtest.h>
#include <humanoid_control/humanoid_controller.hpp>

class HumanoidControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        controller_ = std::make_unique<HumanoidController>();
        controller_->initialize();
    }

    void TearDown() override {
        controller_.reset();
    }

    std::unique_ptr<HumanoidController> controller_;
};

TEST_F(HumanoidControllerTest, Initialization) {
    ASSERT_TRUE(controller_ != nullptr);
    ASSERT_TRUE(controller_->isInitialized());
}

TEST_F(HumanoidControllerTest, JointControl) {
    std::vector<double> joint_states(29, 0.0);
    std::vector<double> desired_positions(29, 0.1);
    std::vector<double> control_outputs;

    bool success = controller_->computeControl(joint_states, desired_positions, control_outputs);

    EXPECT_TRUE(success);
    EXPECT_EQ(control_outputs.size(), 29);
}

TEST_F(HumanoidControllerTest, BalanceControl) {
    Eigen::Vector3d com_pos(0.0, 0.0, 0.8);
    Eigen::Vector3d com_vel(0.0, 0.0, 0.0);
    Eigen::Vector3d zmp_ref(0.0, 0.0, 0.0);

    controller_->updateBalanceControl(com_pos, com_vel, zmp_ref);

    // Add assertions based on expected behavior
    EXPECT_TRUE(true); // Placeholder - actual assertions would be more specific
}

TEST_F(HumanoidControllerTest, TrajectoryGeneration) {
    Eigen::VectorXd start_pos = Eigen::VectorXd::Zero(29);
    Eigen::VectorXd end_pos = Eigen::VectorXd::Ones(29) * 0.5;
    double duration = 1.0;

    controller_->generateTrajectory(start_pos, end_pos, duration);

    // Verify trajectory was generated
    EXPECT_TRUE(true); // Placeholder - actual assertions would be more specific
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

#### انٹیگریشن ٹیسٹنگ

```python
#!/usr/bin/env python3
# Integration testing script for humanoid robot

import unittest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

class HumanoidIntegrationTest(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('humanoid_integration_test')

        # Create subscribers to monitor robot state
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Create action clients for testing
        self.trajectory_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        self.joint_states = None
        self.received_states = False

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def joint_state_callback(self, msg):
        self.joint_states = msg
        self.received_states = True

    def test_joint_state_subscription(self):
        """Test that joint states are being published correctly"""
        timeout = 5.0  # seconds
        start_time = self.node.get_clock().now()

        while not self.received_states:
            current_time = self.node.get_clock().now()
            if (current_time - start_time).nanoseconds > timeout * 1e9:
                self.fail("Joint states not received within timeout")

            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertIsNotNone(self.joint_states)
        self.assertGreater(len(self.joint_states.name), 0)
        self.assertEqual(len(self.joint_states.name), len(self.joint_states.position))

    def test_trajectory_execution(self):
        """Test that trajectory execution works"""
        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.fail("Trajectory server not available")

        # Create a simple trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll'
        ]

        # Add trajectory points
        for i in range(5):
            point = JointTrajectoryPoint()
            point.positions = [i * 0.1] * 6  # 6 joints
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
            point.time_from_start = Duration(sec=i+1, nanosec=0)
            goal_msg.trajectory.points.append(point)

        # Send goal
        future = self.trajectory_client.send_goal_async(goal_msg)

        # Wait for result (with timeout)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)

        if future.result() is None:
            self.fail("Trajectory execution failed or timed out")

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.fail("Trajectory goal was not accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=10.0)

        if result_future.result() is None:
            self.fail("Trajectory result not received")

        result = result_future.result().result
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

## کیلیبریشن اور ٹیوننگ

### سینسر کیلیبریشن

#### کیمرہ کیلیبریشن

```bash
# Intel RealSense کیمرہ کیلیبریشن
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.0245 \
  --ros-args -r image:=/camera/color/image_raw \
  -p /camera/matrix:=/camera/color/camera_info
```

#### IMU کیلیبریشن

```cpp
// IMU calibration algorithm
class IMUCalibrator {
public:
    IMUCalibrator();
    ~IMUCalibrator();

    void addSample(const Eigen::Vector3d& accel_reading,
                   const Eigen::Vector3d& gyro_reading);

    bool calibrate(Eigen::Vector3d& bias_offset,
                   Eigen::Matrix3d& scale_matrix);

    bool isReadyForCalibration() const;

private:
    std::vector<Eigen::Vector3d> accel_samples_;
    std::vector<Eigen::Vector3d> gyro_samples_;
    size_t min_samples_needed_;

    bool performAccelCalibration(Eigen::Vector3d& bias);
    bool performGyroCalibration(Eigen::Vector3d& bias);
    double calculateGravityMagnitude(const std::vector<Eigen::Vector3d>& samples);
};

bool IMUCalibrator::calibrate(Eigen::Vector3d& bias_offset,
                              Eigen::Matrix3d& scale_matrix) {
    if (!isReadyForCalibration()) {
        return false;
    }

    // Calibrate accelerometer bias (should be at rest)
    Eigen::Vector3d accel_bias;
    if (!performAccelCalibration(accel_bias)) {
        return false;
    }

    // Calibrate gyroscope bias (should be at rest)
    Eigen::Vector3d gyro_bias;
    if (!performGyroCalibration(gyro_bias)) {
        return false;
    }

    bias_offset << accel_bias.x(), accel_bias.y(), accel_bias.z(),
                  gyro_bias.x(), gyro_bias.y(), gyro_bias.z();

    // Identity scale matrix initially
    scale_matrix.setIdentity();

    return true;
}
```

### کنٹرول ٹیوننگ

#### PID گین ٹیوننگ

```python
#!/usr/bin/env python3
# PID tuning script for humanoid joints

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class PIDTuner(Node):
    def __init__(self):
        super().__init__('pid_tuner')

        # Joint to tune
        self.target_joint = 'left_knee'
        self.current_setpoint = 0.0

        # Subscribers and publishers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.command_pub = self.create_publisher(
            Float64MultiArray, '/position_commands', 10
        )

        # Data storage for analysis
        self.time_data = []
        self.position_data = []
        self.error_data = []
        self.start_time = self.get_clock().now()

        # Initial PID parameters (Ziegler-Nichols method)
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.1

        # Integral and derivative terms
        self.prev_error = 0.0
        self.integral = 0.0

    def joint_state_callback(self, msg):
        try:
            idx = msg.name.index(self.target_joint)
            current_pos = msg.position[idx]
            current_time = self.get_clock().now()

            # Calculate error
            error = self.current_setpoint - current_pos

            # Store data for analysis
            elapsed = (current_time - self.start_time).nanoseconds / 1e9
            self.time_data.append(elapsed)
            self.position_data.append(current_pos)
            self.error_data.append(error)

            # PID control law
            self.integral += error * 0.001  # assuming 1ms dt
            derivative = (error - self.prev_error) / 0.001

            output = (self.kp * error +
                     self.ki * self.integral +
                     self.kd * derivative)

            # Publish command
            cmd_msg = Float64MultiArray()
            cmd_msg.data = [float(output)]
            self.command_pub.publish(cmd_msg)

            self.prev_error = error

        except ValueError:
            # Joint not found
            pass

    def tune_pid(self, setpoint, duration=10.0):
        """Tune PID parameters using optimization"""
        self.current_setpoint = setpoint

        # Collect initial data with current parameters
        self.get_logger().info(f"Tuning PID for joint {self.target_joint} to {setpoint} rad")

        # Wait for data collection
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Optimize PID parameters
        result = minimize(self.objective_function,
                         x0=[self.kp, self.ki, self.kd],
                         method='Nelder-Mead')

        if result.success:
            self.kp, self.ki, self.kd = result.x
            self.get_logger().info(f"Optimized PID: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}")
        else:
            self.get_logger().warn("PID optimization failed")

    def objective_function(self, params):
        """Objective function for PID optimization"""
        kp, ki, kd = params

        # Simulate response with these parameters
        # This is a simplified simulation - in practice, you'd run real experiments
        error_sum = sum(abs(e) for e in self.error_data)
        overshoot_penalty = self.calculate_overshoot_penalty()

        # Minimize error while penalizing instability
        objective = error_sum + 10 * overshoot_penalty
        return objective

    def calculate_overshoot_penalty(self):
        """Calculate penalty for overshoot"""
        if not self.position_data:
            return 0.0

        # Simplified overshoot calculation
        max_deviation = max(abs(p - self.current_setpoint) for p in self.position_data)
        return max(0, max_deviation - 0.1)  # 0.1 rad tolerance

def main():
    rclpy.init()
    tuner = PIDTuner()

    # Tune for a specific position
    tuner.tune_pid(setpoint=0.5, duration=10.0)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(tuner.time_data, tuner.position_data, label='Actual Position')
    plt.axhline(y=tuner.current_setpoint, color='r', linestyle='--', label='Setpoint')
    plt.title(f'Joint Response: {tuner.target_joint}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tuner.time_data, tuner.error_data, label='Error', color='orange')
    plt.title('Tracking Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## سیفٹی اور فیل سیف میکانزمز

### ایمرجنسی سٹاپ سسٹم

```cpp
// Emergency stop system for humanoid robot
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>

class EmergencyStopSystem {
public:
    EmergencyStopSystem();
    ~EmergencyStopSystem();

    void initialize();
    void checkSafetyConditions();
    void triggerEmergencyStop();
    void clearEmergencyStop();

    bool isEmergencyActive() const;
    void addSafetyCondition(std::function<bool()> condition);

private:
    std::atomic<bool> emergency_active_;
    std::atomic<bool> system_running_;
    std::mutex safety_mutex_;

    std::vector<std::function<bool()>> safety_conditions_;
    std::thread monitoring_thread_;

    void monitoringLoop();
    void handleEmergency();
    void publishEmergencyStatus();
};

EmergencyStopSystem::EmergencyStopSystem()
    : emergency_active_(false), system_running_(true) {

    // Add default safety conditions
    safety_conditions_.push_back([this]() { return checkJointLimits(); });
    safety_conditions_.push_back([this]() { return checkTorsoOrientation(); });
    safety_conditions_.push_back([this]() { return checkBatteryLevel(); });
    safety_conditions_.push_back([this]() { return checkCommunicationStatus(); });
}

void EmergencyStopSystem::monitoringLoop() {
    while (system_running_) {
        if (!emergency_active_) {
            // Check all safety conditions
            for (const auto& condition : safety_conditions_) {
                if (!condition()) {
                    triggerEmergencyStop();
                    break;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 100 Hz check
    }
}

bool EmergencyStopSystem::checkJointLimits() {
    // Check if any joint is beyond safe limits
    // Implementation depends on actual joint state monitoring
    return true; // Placeholder - implement actual check
}

bool EmergencyStopSystem::checkTorsoOrientation() {
    // Check if torso is tilted beyond safe angle (e.g., >45 degrees)
    // Implementation depends on IMU data
    return true; // Placeholder - implement actual check
}

bool EmergencyStopSystem::checkBatteryLevel() {
    // Check if battery level is critically low
    // Implementation depends on battery monitoring
    return true; // Placeholder - implement actual check
}

bool EmergencyStopSystem::checkCommunicationStatus() {
    // Check if critical communication is lost
    // Implementation depends on heartbeat monitoring
    return true; // Placeholder - implement actual check
}

void EmergencyStopSystem::triggerEmergencyStop() {
    if (!emergency_active_.exchange(true)) {
        // First time emergency is triggered
        handleEmergency();
        publishEmergencyStatus();
    }
}

void EmergencyStopSystem::handleEmergency() {
    // Stop all joint movements
    // Disable all actuators safely
    // Log emergency event
    // Optionally sound alarm
}
```

## ڈیپلومنٹ اور ڈیبگنگ کے بہترین طریقے

### ڈیبگنگ کے لیے ROS 2 لانچ فائل

```python
# launch/debug_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_debug = LaunchConfiguration('enable_debug')
    enable_profiling = LaunchConfiguration('enable_profiling')

    # Include main control launch
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('humanoid_control'),
                        'launch', 'humanoid_control.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Debug nodes
    debug_nodes = []

    # Performance monitor node
    perf_monitor_node = Node(
        condition=IfCondition(enable_profiling),
        package='humanoid_utils',
        executable='performance_monitor',
        parameters=[
            {'control_frequency': 1000.0},
            {'publish_period': 1.0},
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    debug_nodes.append(perf_monitor_node)

    # Diagnostic aggregator
    diag_aggregator_node = Node(
        condition=IfCondition(enable_debug),
        package='diagnostic_aggregator',
        executable='aggregator_node',
        parameters=[os.path.join(get_package_share_directory('humanoid_control'),
                                'config', 'diagnostics.yaml')],
        output='screen'
    )
    debug_nodes.append(diag_aggregator_node)

    # Logger node
    logger_node = Node(
        condition=IfCondition(enable_debug),
        package='humanoid_utils',
        executable='data_logger',
        parameters=[
            {'log_topics': ['/joint_states', '/tf', '/imu/data']},
            {'log_frequency': 100.0},
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    debug_nodes.append(logger_node)

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'enable_debug',
            default_value='false',
            description='Enable debug nodes'
        ),
        DeclareLaunchArgument(
            'enable_profiling',
            default_value='false',
            description='Enable performance profiling'
        ),
        control_launch,
    ] + debug_nodes)
```

## کمیونٹی کے وسائل اور مدد

### ROS 2 کے لیے دستیاب وسائل

#### آفیشل ڈاکومنٹیشن

- [ROS 2 Documentation](https://docs.ros.org/en/rolling/)
- [MoveIt 2 Documentation](https://moveit.picknik.ai/main/index.html)
- [Nav2 Documentation](https://navigation.ros.org/)
- [Gazebo Harmonic Documentation](https://gazebosim.org/docs/harmonic)

#### کمیونٹی فورمز

- [ROS Answers](https://answers.ros.org/questions/)
- [ROS Discourse](https://discourse.ros.org/)
- [Robotics Stack Exchange](https://robotics.stackexchange.com/)

#### ٹیوٹوریلز اور کورسز

- ROS 2 Tutorials
- MoveIt 2 Tutorials
- Navigation 2 Tutorials
- Gazebo Tutorials

## زیادہ تر پوچھے گئے سوالات (FAQ)

### سافٹ ویئر کے بارے میں سوالات

**Q: کیا میں ROS 1 کے بجائے ROS 2 استعمال کر سکتا ہوں؟**
A: ROS 2 کی تجویز کی جاتی ہے کیونکہ یہ بہتر کارکردگی، محفوظ تھریڈنگ، اور بہتر سسٹم انضمام فراہم کرتا ہے۔

**Q: کیا میں اپنی موجودہ ROS 2 تنصیب کو اپ ڈیٹ کر سکتا ہوں؟**
A: ہاں، لیکن اس سے پہلے اپنے کسٹم کوڈ اور کنفیگریشنز کا بیک اپ لیں۔

**Q: کیا میں کسٹم کنٹرولر تیار کر سکتا ہوں؟**
A: ہاں، ROS 2 Control فریمورک کسٹم کنٹرولرز کی تعمیر کی اجازت دیتا ہے۔

**Q: کیا میں سافٹ ویئر کو ریموٹ کنٹرول کر سکتا ہوں؟**
A: ہاں، ROS 2 DDS کے ذریعے ریموٹ کنٹرول کی اجازت دیتا ہے۔

## خلاصہ

یہ اپنڈکس ہیومنوائڈ روبوٹکس کے لیے مکمل سافٹ ویئر تنصیب اور کنفیگریشن کے طریقے فراہم کرتا ہے، بشمول ROS 2، کنٹرول سسٹم، ڈیبگنگ ٹولز، اور سیفٹی میکانزمز۔ کامیاب ایکٹویشن کے لیے احتیاط سے کنفیگریشن کی سفارش کی جاتی ہے اور تمام سیفٹی اقدامات کو نافذ کرنا چاہیے۔

## اگلے اقدامات

اپنڈکس سی میں، ہم ہیومنوائڈ روبوٹکس کے لیے جامع تشخیصی اور جائزہ گاہ کے ڈھانچے کا جائزہ لیں گے، جو نظام کی کارکردگی کو یقینی بناتا ہے۔