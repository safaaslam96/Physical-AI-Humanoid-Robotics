---
title: "Chapter 7: URDF and SDF Robot Description Formats"
sidebar_label: "Chapter 7: URDF and SDF Formats"
---



# Chapter 7: URDF and SDF Robot Description Formats

## Learning Objectives
- Understand URDF (Unified Robot Description Format) and SDF (Simulation Description Format)
- Create detailed robot models for humanoid robots using XML-based descriptions
- Implement physics and sensor simulation in robot models
- Validate robot models for both simulation and real-world deployment

## Introduction

Robot Description Formats (URDF and SDF) serve as the "DNA" of robotic systems, encoding all physical and functional characteristics of robots in structured XML files. These formats are fundamental to both simulation and real-world robot deployment, providing the necessary information for physics engines, controllers, and visualization tools. This chapter explores the creation and validation of robot models specifically for humanoid robotics applications.

## Understanding URDF (Unified Robot Description Format)

### What is URDF?

URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. It defines:
- **Physical structure**: Links, joints, and their relationships
- **Visual properties**: Meshes, colors, and shapes for visualization
- **Collision properties**: Collision geometries for physics simulation
- **Inertial properties**: Mass, center of mass, and inertia tensors
- **Sensor configurations**: Mounting positions and parameters

### URDF Structure for Humanoid Robots

Humanoid robots require special considerations in URDF:
- **Bipedal structure**: Two legs with feet for locomotion
- **Arms with hands**: For manipulation tasks
- **Torso and head**: For balance and perception
- **Complex joint configurations**: Multiple degrees of freedom

### Basic URDF Elements

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## Creating URDF Models for Humanoid Robots

### Link Definitions

Links represent rigid bodies in the robot. For humanoid robots:
- **base_link**: The main body or pelvis
- **torso**: Upper body segment
- **head**: Contains cameras and sensors
- **arms**: Upper arm, lower arm, and hand segments
- **legs**: Thigh, shin, and foot segments

Example link with complete properties:
```xml
<link name="upper_arm">
  <visual>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://robot_description/meshes/upper_arm.stl"/>
    </geometry>
    <material name="light_grey">
      <color rgba="0.7 0.7 0.7 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.3"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
  </inertial>
</link>
```

### Joint Definitions

Joints connect links and define their motion. Humanoid robots use various joint types:
- **Revolute**: Single-axis rotation (most common for humanoid joints)
- **Continuous**: Unlimited rotation (for wheels or continuous rotation joints)
- **Prismatic**: Linear motion
- **Fixed**: No motion (for sensors or attachments)

Example joint definition:
```xml
<joint name="shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.05 0.2 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Complete Humanoid Robot URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Pelvis link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.3 0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.15 0.2 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.25"/>
    </inertial>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>
</robot>
```

## Understanding SDF (Simulation Description Format)

### What is SDF?

SDF (Simulation Description Format) is the native format for Gazebo simulation. While URDF is primarily used for ROS, SDF is specifically designed for physics simulation and provides:
- **Advanced physics properties**: Complex collision models, surface properties
- **Sensor integration**: Detailed sensor specifications and noise models
- **Plugin support**: Custom simulation plugins
- **Environment description**: World properties and object placement

### SDF vs URDF Comparison

| Feature | URDF | SDF |
|---------|------|-----|
| Primary Use | ROS robot description | Gazebo simulation |
| Physics | Basic inertial properties | Advanced physics simulation |
| Sensors | Basic mounting | Detailed sensor models |
| Plugins | Limited | Extensive plugin support |
| Integration | ROS ecosystem | Gazebo ecosystem |

### SDF Structure for Robot Models

SDF can also describe robot models with more simulation-specific features:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0.5 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.2</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.3</iyy>
          <iyz>0</iyz>
          <izz>0.4</izz>
        </inertia>
      </inertial>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.2 0.3 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.2 0.3 0.1</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Joints -->
    <joint name="torso_joint" type="fixed">
      <parent>base_link</parent>
      <child>torso</child>
    </joint>

    <!-- Sensors -->
    <link name="head">
      <sensor name="camera" type="camera">
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>
```

## Physics Simulation and Sensor Integration

### Collision Models

For humanoid robots, collision models must balance accuracy and performance:
- **Simple shapes**: Boxes, cylinders, spheres for basic collision detection
- **Complex meshes**: Detailed models for accurate interaction
- **Multiple collision elements**: Different shapes for different parts of a link

### Inertial Properties

Accurate inertial properties are crucial for humanoid robot simulation:
- **Mass distribution**: Affects balance and locomotion
- **Center of mass**: Critical for stability calculations
- **Inertia tensors**: Determine how the robot responds to forces

### Sensor Simulation

Humanoid robots typically include multiple sensor types:
- **IMUs**: For balance and orientation
- **Cameras**: For vision-based perception
- **LIDAR**: For environment mapping
- **Force/Torque sensors**: For manipulation and contact detection

## Creating Robot Models: Best Practices

### Model Validation

Before using robot models, validate them using:
```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Visualize in RViz
ros2 run rviz2 rviz2

# Use Gazebo for physics validation
gazebo --verbose /path/to/model.sdf
```

### Performance Considerations

- **Mesh complexity**: Use simplified meshes for real-time simulation
- **Collision geometry**: Balance accuracy with computational efficiency
- **Inertial properties**: Ensure realistic values for stable simulation

### Reusability and Modularity

- **Macros**: Use xacro macros for parameterized robot models
- **Include files**: Modularize common components
- **Parameterization**: Make models adaptable for different configurations

## Xacro: XML Macros for URDF

Xacro (XML Macros) extends URDF with programming-like features:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="leg_length" value="0.5" />

  <!-- Macro for creating limbs -->
  <xacro:macro name="leg" params="side">
    <link name="${side}_thigh">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="${leg_length}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="${leg_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:leg side="left"/>
  <xacro:leg side="right"/>
</robot>
```

## Hands-On Exercise: Creating a Simple Humanoid Model

### Exercise Objectives
- Create a basic humanoid robot URDF model
- Validate the model using URDF tools
- Load the model in RViz for visualization
- Understand the relationship between model and simulation

### Step-by-Step Instructions

1. **Create a new URDF file** for a simple humanoid robot
2. **Define basic links**: pelvis, torso, head, arms, legs
3. **Add joints** to connect the links
4. **Validate the URDF** using check_urdf tool
5. **Visualize in RViz** to verify the structure

### Expected Outcomes
- Valid URDF file that passes validation
- Correct visualization in RViz
- Understanding of link-joint relationships

## Knowledge Check

1. What is the difference between URDF and SDF formats?
2. Explain the purpose of inertial properties in robot models.
3. Why are collision models important in simulation?
4. How does xacro enhance URDF capabilities?

## Summary

This chapter covered the fundamental robot description formats (URDF and SDF) essential for humanoid robotics. Understanding these formats is crucial for both simulation and real-world robot deployment. Proper robot modeling enables accurate physics simulation, effective visualization, and successful real-world transfer of control algorithms.

## Next Steps

In Chapter 8, we'll explore Unity visualization and sensor simulation, expanding our digital twin capabilities to include high-fidelity rendering and advanced sensor modeling.

