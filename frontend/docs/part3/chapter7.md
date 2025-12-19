---
sidebar_position: 7
title: "Chapter 7: URDF and SDF Robot Description Formats"
---

# Chapter 7: URDF and SDF Robot Description Formats

## Learning Objectives
- Understand URDF (Unified Robot Description Format) and SDF (Simulation Description Format)
- Create complex robot descriptions for humanoid robots
- Configure physics simulation and sensor simulation
- Implement proper robot modeling techniques for simulation

## Introduction to URDF and SDF

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are XML-based formats used to describe robots and simulation environments in ROS and Gazebo respectively. These formats are fundamental to robotics development, enabling the creation of digital twins for simulation, visualization, and control.

### URDF vs SDF: Key Differences

| Aspect | URDF | SDF |
|--------|------|-----|
| **Purpose** | Robot description | Complete simulation environment |
| **Scope** | Single robot model | World, models, physics, plugins |
| **Primary Use** | ROS ecosystem | Gazebo simulation |
| **Extensibility** | Through Xacro macros | Native XML structure |
| **Physics** | Basic inertial properties | Comprehensive physics engine |
| **Sensors** | Basic definitions | Advanced sensor simulation |

## URDF Fundamentals

### Basic URDF Structure

A URDF file describes a robot's kinematic and dynamic properties using a tree structure of links connected by joints:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link - root of the kinematic tree -->
  <link name="base_link">
    <!-- Visual properties for display -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>

    <!-- Collision properties for physics simulation -->
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>

    <!-- Inertial properties for dynamics -->
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Additional links and joints would follow -->
</robot>
```

### Link Properties

#### Visual Properties
Visual elements define how the robot appears in simulation and visualization tools:

```xml
<link name="visual_example">
  <visual name="main_visual">
    <!-- Position and orientation relative to link frame -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>

    <!-- Geometry types -->
    <geometry>
      <!-- Box: width, depth, height -->
      <box size="0.1 0.2 0.3"/>
      <!-- Cylinder: radius, length -->
      <!-- <cylinder radius="0.1" length="0.2"/> -->
      <!-- Sphere: radius -->
      <!-- <sphere radius="0.1"/> -->
      <!-- Mesh: external 3D model -->
      <!-- <mesh filename="package://my_robot/meshes/link.stl"/> -->
    </geometry>

    <!-- Material properties -->
    <material name="red">
      <color rgba="1 0 0 1"/>
      <!-- Or reference texture -->
      <!-- <texture filename="package://my_robot/materials/textures/red.png"/> -->
    </material>
  </visual>
</link>
```

#### Collision Properties
Collision elements define how the robot interacts with the environment in physics simulation:

```xml
<link name="collision_example">
  <collision name="main_collision">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Often simplified compared to visual geometry for performance -->
      <cylinder radius="0.1" length="0.2"/>
    </geometry>
  </collision>
</link>
```

#### Inertial Properties
Inertial properties are crucial for physics simulation and control:

```xml
<link name="inertial_example">
  <inertial>
    <!-- Center of mass location -->
    <origin xyz="0.01 0 0.02" rpy="0 0 0"/>
    <!-- Mass in kg -->
    <mass value="2.5"/>
    <!-- Inertia matrix (in link frame) -->
    <inertia
      ixx="0.01" ixy="0.0" ixz="0.001"
      iyy="0.02" iyz="0.0"
      izz="0.015"/>
  </inertial>
</link>
```

### Joint Properties

Joints connect links and define the degrees of freedom:

```xml
<!-- Revolute joint (rotational) -->
<joint name="shoulder_joint" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.2 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Rotation axis -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>

<!-- Continuous joint (unlimited rotation) -->
<joint name="continuous_joint" type="continuous">
  <parent link="base"/>
  <child link="rotating_part"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.5"/>
</joint>

<!-- Prismatic joint (linear motion) -->
<joint name="prismatic_joint" type="prismatic">
  <parent link="base"/>
  <child link="slider"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="200" velocity="1"/>
</joint>

<!-- Fixed joint (no motion) -->
<joint name="fixed_joint" type="fixed">
  <parent link="link1"/>
  <child link="link2"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>
```

## Advanced URDF Techniques

### Using Xacro for Complex Models

Xacro (XML Macros) allows parameterization and modularization of URDF files:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid">

  <!-- Parameters -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="base_width" value="0.3"/>
  <xacro:property name="base_length" value="0.5"/>
  <xacro:property name="base_height" value="0.2"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia
          ixx="${0.25 * 0.5 * wheel_radius * wheel_radius}"
          ixy="0" ixz="0"
          iyy="${0.5 * 0.5 * (3 * wheel_radius * wheel_radius + wheel_width * wheel_width) / 12}"
          iyz="0"
          izz="${0.25 * 0.5 * wheel_radius * wheel_radius}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia
        ixx="${10.0 * (base_length * base_length + base_height * base_height) / 12}"
        ixy="0" ixz="0"
        iyy="${10.0 * (base_width * base_width + base_height * base_height) / 12}"
        iyz="0"
        izz="${10.0 * (base_width * base_width + base_length * base_length) / 12}"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.15 0.2 -0.1" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.15 -0.2 -0.1" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.15 0.2 -0.1" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.15 -0.2 -0.1" rpy="0 0 0"/>

</robot>
```

### Complex Humanoid URDF Structure

Here's a more complete humanoid robot URDF using Xacro:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Include other xacro files -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/transmissions.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/gazebo.xacro"/>

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_mass" value="10.0"/>
  <xacro:property name="head_mass" value="2.0"/>
  <xacro:property name="arm_mass" value="1.5"/>
  <xacro:property name="leg_mass" value="3.0"/>

  <!-- Torso macro -->
  <xacro:macro name="torso" params="name parent xyz rpy">
    <joint name="${name}_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
    </joint>

    <link name="${name}_link">
      <visual>
        <geometry>
          <box size="0.3 0.2 0.5"/>
        </geometry>
        <material name="white"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.3 0.2 0.5"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${torso_mass}"/>
        <inertia
          ixx="${torso_mass * (0.2 * 0.2 + 0.5 * 0.5) / 12}"
          ixy="0" ixz="0"
          iyy="${torso_mass * (0.3 * 0.3 + 0.5 * 0.5) / 12}"
          iyz="0"
          izz="${torso_mass * (0.3 * 0.3 + 0.2 * 0.2) / 12}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Limb macro -->
  <xacro:macro name="limb" params="name parent xyz rpy joint_type joint_axis lower upper max_effort max_velocity">
    <!-- Upper part -->
    <joint name="${name}_joint" type="${joint_type}">
      <parent link="${parent}"/>
      <child link="${name}_upper"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="${joint_axis}"/>
      <limit lower="${lower}" upper="${upper}" effort="${max_effort}" velocity="${max_velocity}"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${name}_upper">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.4"/>
        </geometry>
        <material name="gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.4"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${arm_mass}"/>
        <inertia
          ixx="${arm_mass * (3 * 0.05 * 0.05 + 0.4 * 0.4) / 12}"
          ixy="0" ixz="0"
          iyy="${arm_mass * (3 * 0.05 * 0.05 + 0.4 * 0.4) / 12}"
          iyz="0"
          izz="${arm_mass * 0.05 * 0.05 * 0.5}"/>
      </inertial>
    </link>

    <!-- Lower part -->
    <joint name="${name}_lower_joint" type="${joint_type}">
      <parent link="${name}_upper"/>
      <child link="${name}_lower"/>
      <origin xyz="0 0 -0.4" rpy="0 0 0"/>
      <axis xyz="${joint_axis}"/>
      <limit lower="${lower}" upper="${upper}" effort="${max_effort}" velocity="${max_velocity}"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${name}_lower">
      <visual>
        <geometry>
          <cylinder radius="0.04" length="0.4"/>
        </geometry>
        <material name="gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.04" length="0.4"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${arm_mass * 0.8}"/>
        <inertia
          ixx="${arm_mass * 0.8 * (3 * 0.04 * 0.04 + 0.4 * 0.4) / 12}"
          ixy="0" ixz="0"
          iyy="${arm_mass * 0.8 * (3 * 0.04 * 0.04 + 0.4 * 0.4) / 12}"
          iyz="0"
          izz="${arm_mass * 0.8 * 0.04 * 0.04 * 0.5}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Torso -->
  <xacro:torso name="torso" parent="base_link" xyz="0 0 0.3" rpy="0 0 0"/>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="10" velocity="2"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${head_mass}"/>
      <inertia
        ixx="${0.4 * head_mass * 0.1 * 0.1}"
        ixy="0" ixz="0"
        iyy="${0.4 * head_mass * 0.1 * 0.1}"
        iyz="0"
        izz="${0.4 * head_mass * 0.1 * 0.1}"/>
    </inertial>
  </link>

  <!-- Arms -->
  <xacro:limb name="left_arm" parent="torso_link"
              xyz="0.2 0.1 0.1" rpy="0 0 0"
              joint_type="revolute" joint_axis="0 1 0"
              lower="${-M_PI/2}" upper="${M_PI/2}"
              max_effort="50" max_velocity="2"/>

  <xacro:limb name="right_arm" parent="torso_link"
              xyz="0.2 -0.1 0.1" rpy="0 0 0"
              joint_type="revolute" joint_axis="0 1 0"
              lower="${-M_PI/2}" upper="${M_PI/2}"
              max_effort="50" max_velocity="2"/>

  <!-- Legs -->
  <xacro:limb name="left_leg" parent="torso_link"
              xyz="-0.1 0.1 -0.25" rpy="0 0 0"
              joint_type="revolute" joint_axis="0 1 0"
              lower="${-M_PI/2}" upper="${M_PI/4}"
              max_effort="100" max_velocity="1"/>

  <xacro:limb name="right_leg" parent="torso_link"
              xyz="-0.1 -0.1 -0.25" rpy="0 0 0"
              joint_type="revolute" joint_axis="0 1 0"
              lower="${-M_PI/2}" upper="${M_PI/4}"
              max_effort="100" max_velocity="1"/>

  <!-- Feet -->
  <joint name="left_foot_joint" type="fixed">
    <parent link="left_leg_lower"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
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

  <joint name="right_foot_joint" type="fixed">
    <parent link="right_leg_lower"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
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

</robot>
```

### Materials and Colors

Define materials in a separate file for reusability:

```xml
<!-- materials.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

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
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>

  <material name="orange">
    <color rgba="1.0 0.4235 0.0392 1.0"/>
  </material>

  <material name="brown">
    <color rgba="0.8706 0.8118 0.7647 1.0"/>
  </material>

  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>

  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <material name="skin">
    <color rgba="0.8 0.6 0.4 1.0"/>
  </material>

</robot>
```

## SDF Fundamentals

### Basic SDF Structure

SDF is more comprehensive than URDF and can describe entire simulation environments:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <!-- World definition -->
  <world name="my_world">
    <!-- Physics engine configuration -->
    <physics type="ode" name="default_physics">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Models in the world -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Light sources -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
  </world>
</sdf>
```

### SDF Model Definition

A complete SDF model definition includes more simulation-specific features:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="advanced_robot">
    <!-- Model pose -->
    <pose>0 0 0.5 0 0 0</pose>

    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>

      <!-- Inertial properties -->
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.4</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.4</iyy>
          <iyz>0.0</iyz>
          <izz>0.4</izz>
        </inertia>
      </inertial>

      <!-- Visual elements -->
      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.5</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>0.8 0.2 0.2 1</diffuse>
          <specular>0.8 0.8 0.8 1</specular>
        </material>
      </visual>

      <!-- Collision elements -->
      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.5</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <soft_cfm>0</soft_cfm>
              <soft_erp>0.2</soft_erp>
              <kp>1e+10</kp>
              <kd>1</kd>
              <max_vel>100.0</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>

      <!-- Sensors -->
      <sensor name="imu_sensor" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <pose>0 0 0 0 0 0</pose>
        <imu>
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
              </noise>
            </x>
          </linear_acceleration>
        </imu>
      </sensor>
    </link>

    <!-- Joints -->
    <joint name="arm_joint" type="revolute">
      <parent>base_link</parent>
      <child>arm_link</child>
      <pose>0.3 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>2</velocity>
        </limit>
        <dynamics>
          <damping>1.0</damping>
          <friction>0.1</friction>
        </dynamics>
      </axis>
    </joint>

    <link name="arm_link">
      <pose>0.3 0 0 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <visual name="arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

## Physics Simulation Configuration

### Inertial Calculations

Proper inertial properties are crucial for stable simulation. Here are formulas for common shapes:

```xml
<!-- Box: width(w), depth(d), height(h), mass(m) -->
<inertia
  ixx="${m * (d*d + h*h) / 12}"
  ixy="0" ixz="0"
  iyy="${m * (w*w + h*h) / 12}"
  iyz="0"
  izz="${m * (w*w + d*d) / 12}"/>

<!-- Cylinder: radius(r), length(l), mass(m) -->
<inertia
  ixx="${m * (3*r*r + l*l) / 12}"
  ixy="0" ixz="0"
  iyy="${m * (3*r*r + l*l) / 12}"
  iyz="0"
  izz="${m * r*r / 2}"/>

<!-- Sphere: radius(r), mass(m) -->
<inertia
  ixx="${0.4 * m * r*r}"
  ixy="0" ixz="0"
  iyy="${0.4 * m * r*r}"
  iyz="0"
  izz="${0.4 * m * r*r}"/>
```

### Friction and Contact Properties

Proper friction settings are essential for humanoid robot stability:

```xml
<collision name="foot_collision">
  <geometry>
    <box size="0.2 0.1 0.05"/>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>    <!-- Static friction coefficient -->
        <mu2>1.0</mu2>  <!-- Secondary friction coefficient -->
        <fdir1>0 0 0</fdir1>  <!-- Friction direction -->
      </ode>
      <!-- Bullet physics alternative -->
      <!--
      <bullet>
        <friction>1.0</friction>
        <friction2>1.0</friction2>
      </bullet>
      -->
    </friction>
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>     <!-- Constraint Force Mixing -->
        <soft_erp>0.2</soft_erp>     <!-- Error Reduction Parameter -->
        <kp>1e+6</kp>               <!-- Spring stiffness -->
        <kd>1.0</kd>                <!-- Damping coefficient -->
        <max_vel>100.0</max_vel>     <!-- Maximum contact correction velocity -->
        <min_depth>0.001</min_depth> <!-- Minimum contact depth -->
      </ode>
    </contact>
  </surface>
</collision>
```

## Sensor Simulation

### Camera Sensors

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30.0</update_rate>
  <visualize>true</visualize>
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
    <min_depth>0.1</min_depth>
    <max_depth>100.0</max_depth>
  </plugin>
</sensor>
```

### IMU Sensors

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <visualize>false</visualize>
  <topic>__default_topic__</topic>
  <pose>0 0 0 0 0 0</pose>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### LIDAR Sensors

```xml
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
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </ray>
  <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
    <topic_name>scan</topic_name>
    <frame_name>laser_frame</frame_name>
  </plugin>
</sensor>
```

## Validation and Debugging

### URDF Validation

Validate your URDF files using ROS tools:

```bash
# Check URDF syntax
check_urdf my_robot.urdf

# Parse Xacro to URDF
xacro my_robot.urdf.xacro > my_robot.urdf

# Visualize in RViz
ros2 run rviz2 rviz2
# Add RobotModel display and set robot description to your URDF topic
```

### Common Issues and Solutions

#### Joint Limit Issues
```xml
<!-- Problem: Joint limits too restrictive -->
<limit lower="-0.1" upper="0.1" effort="100" velocity="1"/>

<!-- Solution: Proper joint limits based on mechanical constraints -->
<limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="2"/>
```

#### Inertial Issues
```xml
<!-- Problem: Zero or negative inertial values -->
<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>

<!-- Solution: Proper positive definite inertia matrix -->
<inertia ixx="0.1" ixy="0.001" ixz="0.001" iyy="0.1" iyz="0.001" izz="0.1"/>
```

#### Mass Issues
```xml
<!-- Problem: Zero or negative mass -->
<mass value="0"/>

<!-- Solution: Proper positive mass -->
<mass value="1.0"/>
```

## Integration with ROS 2

### Robot State Publisher

To publish robot state from URDF:

```python
# Launch file for robot state publisher
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindPackageShare("my_robot_description"), "urdf", "robot.urdf.xacro"]),
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

    return LaunchDescription([robot_state_publisher])
```

### Joint State Publisher

For interactive joint control:

```python
# Joint state publisher GUI
joint_state_publisher_gui = Node(
    package='joint_state_publisher_gui',
    executable='joint_state_publisher_gui',
    name='joint_state_publisher_gui',
    parameters=[{'use_gui': True}]
)
```

## Best Practices

### URDF Best Practices

1. **Use Xacro**: Modularize complex robots using Xacro macros
2. **Proper Inertials**: Calculate or measure actual inertial properties
3. **Realistic Joint Limits**: Set limits based on mechanical constraints
4. **Consistent Units**: Use SI units throughout (meters, kilograms, seconds)
5. **Kinematic Loops**: Use fixed joints for closed kinematic chains
6. **Frame Names**: Use descriptive, consistent frame names

### SDF Best Practices

1. **Physics Tuning**: Adjust contact properties for stable simulation
2. **Sensor Noise**: Include realistic noise models for sensors
3. **Performance**: Use simplified collision geometries when possible
4. **World Design**: Create representative environments for testing
5. **Plugin Configuration**: Properly configure Gazebo plugins for ROS integration

## Knowledge Check

1. What are the key differences between URDF and SDF formats?
2. How do you calculate proper inertial properties for robot links?
3. What are the essential elements for a humanoid robot URDF?
4. How do you configure realistic sensor simulation in SDF?

## Summary

This chapter covered the fundamentals of URDF and SDF robot description formats, including their structure, elements, and proper configuration for humanoid robots. We explored advanced techniques using Xacro, physics simulation configuration, sensor integration, and best practices for creating realistic robot models for simulation and control.

## Next Steps

In the next chapter, we'll explore Unity visualization and sensor simulation, learning how to create high-fidelity visualizations and advanced sensor models for humanoid robotics applications.