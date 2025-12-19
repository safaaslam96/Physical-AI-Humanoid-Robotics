---
sidebar_position: 6
title: "Chapter 6: Gazebo Simulation Environment Setup"
---

# Chapter 6: Gazebo Simulation Environment Setup

## Learning Objectives
- Set up Gazebo simulation environment for humanoid robotics
- Understand physics simulation principles and their importance
- Configure realistic physics properties for humanoid robots
- Integrate Gazebo with ROS 2 for robot simulation and control

## Introduction to Gazebo Simulation

Gazebo is a powerful 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. For humanoid robotics, Gazebo serves as a "digital twin" that allows for safe testing, algorithm development, and validation before deploying on real hardware.

### Why Gazebo for Humanoid Robotics?

Gazebo is particularly well-suited for humanoid robotics because:

- **Realistic Physics**: Accurate simulation of gravity, friction, collisions, and dynamics
- **Sensor Simulation**: Realistic simulation of cameras, LIDAR, IMUs, and force/torque sensors
- **Complex Environments**: Creation of realistic indoor and outdoor environments
- **Hardware Integration**: Seamless integration with ROS 2 through Gazebo ROS packages
- **Safety**: Risk-free testing of complex behaviors without hardware damage
- **Cost-Effective**: No need for expensive physical prototypes

### Gazebo Architecture

Gazebo follows a client-server architecture:

- **Gazebo Server**: Runs the physics simulation and handles all simulation logic
- **Gazebo Client**: Provides the graphical user interface for visualization
- **Plugins**: Extend functionality through custom code (physics, sensors, controllers)
- **ROS 2 Interface**: Bridges simulation to ROS 2 topics and services

## Installing and Setting Up Gazebo

### System Requirements
Before installing Gazebo, ensure your system meets the requirements:

- **Operating System**: Ubuntu 22.04 LTS (recommended) or similar Linux distribution
- **Graphics**: GPU with OpenGL 2.1+ support (for GUI)
- **RAM**: 8GB+ recommended for complex simulations
- **Storage**: 10GB+ for Gazebo models and environments

### Installation Process

For ROS 2 Humble Hawksbill (Ubuntu 22.04):

```bash
# Install Gazebo Garden (recommended for ROS 2 Humble)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Install additional dependencies
sudo apt install gazebo
```

### Verification
Test the installation:

```bash
# Launch Gazebo GUI
gazebo

# Launch Gazebo server only (no GUI)
gzserver

# Launch Gazebo client only (connects to running server)
gzclient
```

## Physics Simulation Principles

### Understanding Physics Engines

Gazebo supports multiple physics engines, each with different characteristics:

#### ODE (Open Dynamics Engine)
- Default physics engine for Gazebo
- Good balance of performance and accuracy
- Suitable for most humanoid robotics applications
- Supports complex contact dynamics

#### Bullet Physics
- High-performance physics engine
- Good for real-time applications
- Excellent collision detection
- Good for humanoid balance simulation

#### DART (Dynamic Animation and Robotics Toolkit)
- Advanced physics simulation
- Better handling of complex contacts
- Excellent for humanoid locomotion
- More accurate than ODE for complex interactions

### Physics Configuration in SDF

The Simulation Description Format (SDF) allows configuration of physics properties:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine configuration -->
    <physics type="ode" name="default_physics">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>

      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- World content -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Key Physics Parameters

#### Time Step Configuration
```xml
<max_step_size>0.001</max_step_size>  <!-- Simulation time step (seconds) -->
<real_time_factor>1.0</real_time_factor>  <!-- Simulation speed multiplier -->
<real_time_update_rate>1000.0</real_time_update_rate>  <!-- Hz -->
```

**Best practices for humanoid robots:**
- Use smaller time steps (0.001s or smaller) for stability
- Balance accuracy with performance requirements
- Consider real-time factor for training vs. testing scenarios

#### Gravity and Environmental Forces
```xml
<gravity>0 0 -9.8</gravity>  <!-- Standard Earth gravity -->
```

For humanoid balance testing, you might want to adjust gravity:
```xml
<gravity>0 0 -1.62</gravity>  <!-- Moon gravity for stability testing -->
```

#### Contact Properties
```xml
<contact_max_correcting_vel>100.0</contact_max_correcting_vel>  <!-- Max contact velocity -->
<contact_surface_layer>0.001</contact_surface_layer>  <!-- Surface layer thickness -->
```

## Creating Simulation Environments

### Basic World File Structure

A Gazebo world file is an SDF file that defines the simulation environment:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Physics configuration -->
    <physics type="ode" name="default_physics">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Models in the environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom environment objects -->
    <model name="table">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="table_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>50.0</mass>
          <inertia>
            <ixx>5.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>5.0</iyy>
            <iyz>0.0</iyz>
            <izz>5.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Lighting -->
    <light name="room_light" type="point">
      <pose>0 0 3 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.5</constant>
        <linear>0.1</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
    </light>
  </world>
</sdf>
```

### Creating Custom Models

To create a custom humanoid robot model for Gazebo, extend your URDF with Gazebo-specific tags:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include main URDF content -->
  <xacro:include filename="humanoid.urdf"/>

  <!-- Gazebo-specific plugins and configurations -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- Ground truth pose publisher -->
  <gazebo>
    <plugin name="p3d_base_controller" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>ground_truth/state</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>map</frameName>
    </plugin>
  </gazebo>

  <!-- IMU sensor plugin -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.17</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.17</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.17</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- Camera sensor plugin -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_optical_frame</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>100.0</max_depth>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LIDAR sensor plugin -->
  <gazebo reference="lidar_link">
    <sensor name="laser" type="ray">
      <always_on>true</always_on>
      <visualize>true</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
        <frame_name>lidar_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## ROS 2 Integration with Gazebo

### Launching Gazebo with ROS 2

Create a launch file to start Gazebo with your robot:

```python
# launch/humanoid_gazebo.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='humanoid_lab.sdf',
        description='Choose one of the world files from `/my_robot_gazebo/worlds`'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                LaunchConfiguration('world')
            ])
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': Command(['xacro ', PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'humanoid.urdf.xacro'
            ])])
        }]
    )

    return LaunchDescription([
        world_arg,
        robot_state_publisher,
        gazebo,
        spawn_entity
    ])
```

### Robot Control in Simulation

To control your robot in Gazebo, you need to set up ROS 2 controllers. Create a controller configuration file:

```yaml
# config/humanoid_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_controller:
      type: joint_state_controller/JointStateController

    left_leg_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - left_hip
        - left_knee
        - left_ankle

    right_leg_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - right_hip
        - right_knee
        - right_ankle

    left_arm_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - left_shoulder
        - left_elbow

    right_arm_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - right_shoulder
        - right_elbow

    head_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - head_pan
        - head_tilt
```

### Launching Controllers

Create a launch file to start the controllers:

```python
# launch/humanoid_control.launch.py
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([
                FindPackageShare("my_robot_description"),
                "urdf",
                "humanoid.urdf.xacro",
            ]),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # Spawn controllers
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_controller", "-c", "/controller_manager"],
    )

    left_leg_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_leg_controller", "-c", "/controller_manager"],
    )

    right_leg_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_leg_controller", "-c", "/controller_manager"],
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller", "-c", "/controller_manager"],
    )

    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller", "-c", "/controller_manager"],
    )

    head_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["head_controller", "-c", "/controller_manager"],
    )

    # Return launch description
    return LaunchDescription([
        robot_state_publisher,
        joint_state_broadcaster_spawner,
        left_leg_controller_spawner,
        right_leg_controller_spawner,
        left_arm_controller_spawner,
        right_arm_controller_spawner,
        head_controller_spawner,
    ])
```

## Advanced Simulation Techniques

### Dynamic Environment Objects

Create dynamic objects that interact with your robot:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="movable_box">
    <pose>3 0 0.5 0 0 0</pose>
    <link name="box_link">
      <collision name="collision">
        <geometry>
          <box>
            <size>0.2 0.2 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.2 0.2 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.4 0.2 1</ambient>
          <diffuse>0.8 0.4 0.2 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.0083</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.0083</iyy>
          <iyz>0.0</iyz>
          <izz>0.0083</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Allow the object to be grasped by the robot -->
    <plugin name="gazebo_ros_grasp_fix" filename="libgazebo_ros_grasp_fix.so">
      <arm>
        <arm_name>left_arm</arm_name>
        <collision_name>box::box_link::collision</collision_name>
      </arm>
    </plugin>
  </model>
</sdf>
```

### Terrain and Complex Environments

For humanoid locomotion training, create challenging terrains:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="challenging_terrain">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane with texture -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grass</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <kinematic>0</kinematic>
        <gravity>1</gravity>
      </link>
    </model>

    <!-- Obstacles for navigation -->
    <model name="obstacle_1">
      <pose>5 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Stairs for locomotion training -->
    <model name="stairs">
      <pose>8 0 0 0 0 0</pose>
      <link name="base_link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>file://meshes/stairs.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://meshes/stairs.dae</uri>
            </mesh>
          </geometry>
        </visual>
        <inertial>
          <mass>100.0</mass>
          <inertia>
            <ixx>100.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>100.0</iyy>
            <iyz>0.0</iyz>
            <izz>100.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Simulation Workflows

### Training vs. Testing Environments

Different simulation configurations are needed for training versus testing:

#### Training Configuration
```xml
<!-- Training environment: Simplified physics, faster simulation -->
<physics type="ode" name="training_physics">
  <max_step_size>0.01</max_step_size>  <!-- Larger steps for speed -->
  <real_time_factor>5.0</real_time_factor>  <!-- 5x faster than real-time -->
  <real_time_update_rate>100.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>  <!-- Fewer iterations for speed -->
      <sor>1.3</sor>
    </solver>
  </ode>
</physics>
```

#### Testing Configuration
```xml
<!-- Testing environment: Accurate physics, realistic simulation -->
<physics type="ode" name="testing_physics">
  <max_step_size>0.001</max_step_size>  <!-- Smaller steps for accuracy -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time simulation -->
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- More iterations for accuracy -->
      <sor>1.0</sor>
    </solver>
  </ode>
</physics>
```

### Multi-Robot Simulation

Simulate multiple robots in the same environment:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="multi_robot_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Robot 1 -->
    <include>
      <name>humanoid_robot_1</name>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Robot 2 -->
    <include>
      <name>humanoid_robot_2</name>
      <uri>model://humanoid_robot</uri>
      <pose>2 0 1 0 0 0</pose>
    </include>

    <!-- Communication between robots -->
    <gazebo>
      <plugin name="multi_robot_communication" filename="libmulti_robot_comm.so">
        <robot_namespace>/humanoid_robot_1</robot_namespace>
        <robot_namespace>/humanoid_robot_2</robot_namespace>
        <communication_range>10.0</communication_range>
      </plugin>
    </gazebo>
  </world>
</sdf>
```

## Performance Optimization

### Simulation Performance Tips

1. **Adjust Time Step**: Balance accuracy vs. performance
2. **Limit Physics Iterations**: Fewer iterations = faster simulation
3. **Use Simplified Models**: Lower polygon count for visual meshes
4. **Optimize Collision Geometry**: Use simpler shapes for collision detection
5. **Reduce Update Rates**: Lower sensor update rates where possible

### Resource Management

Monitor simulation performance:

```bash
# Monitor Gazebo server performance
htop -p $(pgrep gzserver)

# Monitor physics update rate
gz stats

# Check for bottlenecks
ros2 topic hz /clock  # Should match real_time_update_rate
```

## Troubleshooting Common Issues

### Physics Instability
If your humanoid robot is unstable in simulation:

```xml
<!-- Increase constraint solver iterations -->
<ode>
  <solver>
    <iters>200</iters>  <!-- Increase from default -->
    <sor>1.0</sor>      <!-- Lower for stability -->
  </solver>
  <constraints>
    <contact_surface_layer>0.005</contact_surface_layer>  <!-- Increase slightly -->
    <contact_max_correcting_vel>10.0</contact_max_correcting_vel>
  </constraints>
</ode>
```

### Joint Limits and Control
Ensure proper joint limits in URDF:

```xml
<joint name="left_knee" type="revolute">
  <parent link="left_upper_leg"/>
  <child link="left_lower_leg"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.36" upper="0.785" effort="100" velocity="2"/>  <!-- Proper limits -->
  <dynamics damping="1.0" friction="0.1"/>  <!-- Add damping for stability -->
</joint>
```

## Knowledge Check

1. What are the key physics parameters that affect humanoid robot simulation stability?
2. How do you configure different physics settings for training vs. testing environments?
3. What are the essential Gazebo plugins needed for humanoid robot simulation?
4. How do you integrate Gazebo with ROS 2 controllers?

## Summary

This chapter covered the setup and configuration of Gazebo simulation environment for humanoid robotics. We explored physics simulation principles, world creation, ROS 2 integration, and advanced simulation techniques. The chapter also provided practical examples for creating realistic environments and optimizing simulation performance for humanoid robot applications.

## Next Steps

In the next chapter, we'll explore URDF and SDF robot description formats in more detail, learning how to create complex robot models and configure physics properties for simulation.