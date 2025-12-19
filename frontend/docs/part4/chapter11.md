---
sidebar_position: 11
title: "Chapter 11: Nav2 and Path Planning for Humanoid Robots"
---

# Chapter 11: Nav2 and Path Planning for Humanoid Robots

## Learning Objectives
- Understand Nav2 navigation stack for humanoid robots
- Implement path planning algorithms for bipedal locomotion
- Design navigation systems for legged robots
- Apply reinforcement learning for robot control

## Introduction to Nav2 for Humanoid Robots

Nav2 (Navigation 2) is ROS 2's state-of-the-art navigation stack, designed for mobile robots. For humanoid robots, Nav2 requires specialized adaptations to handle the unique challenges of bipedal locomotion, balance constraints, and dynamic movement patterns.

### Nav2 Architecture Overview

Nav2 consists of several key components:

- **Navigation Server**: Main orchestrator of navigation tasks
- **Planners**: Global and local path planning
- **Controllers**: Trajectory execution and control
- **Behavior Trees**: Task orchestration and decision making
- **Sensors**: Perception and localization integration
- **Transforms**: Coordinate frame management

### Challenges with Humanoid Navigation

Unlike wheeled robots, humanoid robots face unique navigation challenges:

- **Dynamic Balance**: Maintaining stability during movement
- **Step Planning**: Careful foot placement on terrain
- **Bipedal Kinematics**: Complex movement patterns
- **Energy Efficiency**: Minimizing power consumption
- **Terrain Adaptation**: Handling various surface types
- **Obstacle Avoidance**: Navigating with human-like behavior

## Nav2 Configuration for Humanoid Robots

### Basic Nav2 Setup

```python
# humanoid_nav2_config.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')

    # Create the launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'))

    ld.add_action(
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically start components'))

    ld.add_action(
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(
                get_package_share_directory('humanoid_nav2_bringup'),
                'config',
                'humanoid_nav2_params.yaml'),
            description='Full path to the ROS2 parameters file to use for all launched nodes'))

    # Create the navigation server
    navigation_server_node = Node(
        package='nav2_navigation_server',
        executable='navigation_server',
        name='navigation_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'),
                   ('/tf_static', 'tf_static')])

    # Lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                   {'autostart': autostart},
                   {'node_names': ['navigation_server',
                                 'bt_navigator',
                                 'controller_server',
                                 'planner_server',
                                 'recoveries_server',
                                 'waypoint_follower']}])

    # Add nodes to launch description
    ld.add_action(navigation_server_node)
    ld.add_action(lifecycle_manager)

    return ld
```

### Humanoid-Specific Parameters

```yaml
# humanoid_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: "map"
    robot_base_frame: "base_footprint"
    odom_topic: "odom"
    default_bt_xml_filename: "humanoid_navigator_bt.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.2
      wz_std: 0.3
      vx_max: 0.5
      vx_min: -0.2
      vy_max: 0.3
      wz_max: 0.3
      sim_period: 0.05
      trajectory_visualization: true
      # Humanoid-specific parameters
      step_size: 0.3  # Typical humanoid step size
      max_step_height: 0.1  # Maximum step height
      balance_threshold: 0.1  # Balance maintenance threshold

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_footprint"
      use_sim_time: False
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Humanoid radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: "map"
      robot_base_frame: "base_footprint"
      use_sim_time: False
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

### Humanoid Behavior Tree

```xml
<!-- humanoid_navigator_bt.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <RecoveryNode number_of_retries="6" name="NavigateRecovery">
            <PipelineSequence name="NavigateWithReplanning">
                <RateController hz="1.0">
                    <ComputePathToPose goal="current_goal" path="path"/>
                </RateController>
                <RecoveryNode number_of_retries="1" name="FollowPathRecovery">
                    <FollowPath path="path" velocity="default_velocity"/>
                    <ReactiveFallback name="FollowPathWithRecoveryFallback">
                        <GoalReached goal="current_goal" path="path"/>
                        <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                    </ReactiveFallback>
                </RecoveryNode>
            </PipelineSequence>
            <ReactiveFallback name="RecoveryFallback">
                <GoalUpdated goal="current_goal"/>
                <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
            </ReactiveFallback>
        </RecoveryNode>
    </BehaviorTree>
</root>
```

## Bipedal Path Planning Algorithms

### Humanoid-Specific Path Planning

```python
# humanoid_path_planner.py
import numpy as np
from scipy.spatial import KDTree
import math
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time

class HumanoidPathPlanner:
    def __init__(self):
        self.step_size = 0.3  # Typical humanoid step size (meters)
        self.max_step_height = 0.1  # Maximum step height (meters)
        self.balance_margin = 0.1  # Balance maintenance margin
        self.support_polygon = self.calculate_support_polygon()

    def calculate_support_polygon(self):
        """Calculate the support polygon for bipedal balance"""
        # For a humanoid, the support polygon is between the feet
        # This is a simplified model - in practice, this would be more complex
        return [
            Point(x=-0.1, y=-0.1, z=0.0),  # Left foot
            Point(x=-0.1, y=0.1, z=0.0),   # Right foot
        ]

    def plan_bipedal_path(self, start_pose, goal_pose, costmap):
        """Plan a path suitable for bipedal locomotion"""
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_pose.position.x, start_pose.position.y, costmap)
        goal_grid = self.world_to_grid(goal_pose.position.x, goal_pose.position.y, costmap)

        # Use A* with humanoid-specific constraints
        path = self.bipedal_astar(start_grid, goal_grid, costmap)

        # Smooth the path for natural human-like movement
        smoothed_path = self.smooth_path(path)

        # Convert to ROS Path message
        ros_path = self.create_ros_path(smoothed_path, costmap)

        return ros_path

    def bipedal_astar(self, start, goal, costmap):
        """A* algorithm with bipedal constraints"""
        import heapq

        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # Priority queue: (cost, current, path)
        frontier = [(0, start, [start])]
        explored = set()

        while frontier:
            cost, current, path = heapq.heappop(frontier)

            if current == goal:
                return path

            if current in explored:
                continue

            explored.add(current)

            # Get valid neighbors (with bipedal constraints)
            for neighbor in self.get_valid_neighbors(current, costmap):
                if neighbor not in explored:
                    new_cost = cost + 1  # Simple cost model
                    new_path = path + [neighbor]
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor, new_path))

        return []  # No path found

    def get_valid_neighbors(self, pos, costmap):
        """Get valid neighbors considering bipedal constraints"""
        neighbors = []
        grid_size = len(costmap.data) // costmap.info.height

        # Consider 8-connectivity for more natural movement
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                new_x = pos[0] + dx
                new_y = pos[1] + dy

                # Check bounds
                if (0 <= new_x < costmap.info.width and
                    0 <= new_y < costmap.info.height):

                    # Check if cell is free (cost < 50 to avoid unknown areas)
                    grid_index = new_y * costmap.info.width + new_x
                    if grid_index < len(costmap.data) and costmap.data[grid_index] < 50:
                        neighbors.append((new_x, new_y))

        return neighbors

    def smooth_path(self, path):
        """Smooth the path for natural human-like movement"""
        if len(path) < 3:
            return path

        smoothed = [path[0]]

        i = 0
        while i < len(path) - 1:
            j = i + 1

            # Try to skip intermediate points while maintaining safety
            while j < len(path) - 1 and self.is_line_clear(path[i], path[j+1]):
                j += 1

            smoothed.append(path[j])
            i = j

        return smoothed

    def is_line_clear(self, start, end):
        """Check if line between two points is clear of obstacles"""
        # Simplified implementation - in practice, this would check costmap
        return True

    def create_ros_path(self, grid_path, costmap):
        """Convert grid path to ROS Path message"""
        path_msg = Path()
        path_msg.header.stamp = Time(sec=0, nanosec=0)  # Will be filled by caller
        path_msg.header.frame_id = "map"

        for grid_x, grid_y in grid_path:
            world_x, world_y = self.grid_to_world(grid_x, grid_y, costmap)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0  # Assuming 2D navigation

            # Set orientation to face the next point
            if grid_path.index((grid_x, grid_y)) < len(grid_path) - 1:
                next_x, next_y = grid_path[grid_path.index((grid_x, grid_y)) + 1]
                next_world_x, next_world_y = self.grid_to_world(next_x, next_y, costmap)

                angle = math.atan2(next_world_y - world_y, next_world_x - world_x)
                pose.pose.orientation.z = math.sin(angle / 2)
                pose.pose.orientation.w = math.cos(angle / 2)

            path_msg.poses.append(pose)

        return path_msg

    def world_to_grid(self, x, y, costmap):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - costmap.info.origin.position.x) / costmap.info.resolution)
        grid_y = int((y - costmap.info.origin.position.y) / costmap.info.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y, costmap):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * costmap.info.resolution + costmap.info.origin.position.x
        world_y = grid_y * costmap.info.resolution + costmap.info.origin.position.y
        return (world_x, world_y)
```

### Step Planning for Humanoid Locomotion

```python
# step_planner.py
import numpy as np
from scipy.spatial import distance
from geometry_msgs.msg import Point

class StepPlanner:
    def __init__(self):
        self.step_length = 0.3  # Typical step length for humanoid
        self.step_width = 0.2   # Typical step width
        self.max_step_height = 0.1  # Maximum step height
        self.balance_margin = 0.05  # Balance margin

    def plan_steps(self, path, robot_pose):
        """Plan individual steps along the path"""
        steps = []

        if len(path.poses) < 2:
            return steps

        current_pose = robot_pose
        current_foot = "left"  # Start with left foot

        for i in range(len(path.poses) - 1):
            start_pose = path.poses[i].pose.position
            end_pose = path.poses[i + 1].pose.position

            # Calculate required step parameters
            step_vector = np.array([end_pose.x - start_pose.x,
                                  end_pose.y - start_pose.y])
            step_distance = np.linalg.norm(step_vector)

            if step_distance > 0:
                step_direction = step_vector / step_distance

                # Plan steps between start and end
                num_steps = max(1, int(step_distance / self.step_length))

                for j in range(num_steps):
                    step_fraction = (j + 1) / num_steps
                    step_position = start_pose + step_fraction * step_direction

                    # Alternate feet
                    foot_side = "left" if current_foot == "right" else "right"

                    # Calculate foot placement with slight offset for balance
                    foot_offset = self.calculate_foot_offset(foot_side, step_direction)
                    final_position = Point()
                    final_position.x = step_position[0] + foot_offset[0]
                    final_position.y = step_position[1] + foot_offset[1]
                    final_position.z = 0.0  # Ground level

                    step = {
                        'position': final_position,
                        'foot': foot_side,
                        'step_number': len(steps) + 1
                    }

                    steps.append(step)
                    current_foot = foot_side

        return steps

    def calculate_foot_offset(self, foot_side, step_direction):
        """Calculate foot offset for balance"""
        # Calculate perpendicular vector for foot offset
        perp_direction = np.array([-step_direction[1], step_direction[0]])

        # Offset based on foot side
        offset_magnitude = self.step_width / 2
        if foot_side == "left":
            offset = perp_direction * offset_magnitude
        else:  # right foot
            offset = -perp_direction * offset_magnitude

        return offset

    def validate_step_sequence(self, steps, terrain_map):
        """Validate step sequence on terrain"""
        valid_steps = []

        for i, step in enumerate(steps):
            # Check if step location is valid
            if self.is_step_valid(step, terrain_map):
                # Check if transition from previous step is valid
                if i > 0:
                    prev_step = steps[i-1]
                    if self.is_transition_valid(prev_step, step, terrain_map):
                        valid_steps.append(step)
                else:
                    valid_steps.append(step)
            else:
                # Try alternative step placement
                alternative_step = self.find_alternative_step(step, terrain_map)
                if alternative_step and self.is_step_valid(alternative_step, terrain_map):
                    valid_steps.append(alternative_step)

        return valid_steps

    def is_step_valid(self, step, terrain_map):
        """Check if a single step is valid"""
        # Check if position is on walkable terrain
        x, y = int(step['position'].x), int(step['position'].y)

        if not (0 <= x < terrain_map.width and 0 <= y < terrain_map.height):
            return False

        # Check terrain cost (simplified)
        terrain_cost = terrain_map.get_cost(x, y)
        if terrain_cost > 50:  # Too costly to step here
            return False

        # Check step height (simplified)
        height_diff = abs(step['position'].z - terrain_map.get_height(x, y))
        if height_diff > self.max_step_height:
            return False

        return True

    def is_transition_valid(self, prev_step, current_step, terrain_map):
        """Check if transition between steps is valid"""
        # Check if step is within reach
        pos1 = np.array([prev_step['position'].x, prev_step['position'].y])
        pos2 = np.array([current_step['position'].x, current_step['position'].y])

        distance = np.linalg.norm(pos2 - pos1)
        if distance > self.step_length * 1.5:  # Allow some flexibility
            return False

        return True

    def find_alternative_step(self, original_step, terrain_map):
        """Find alternative step placement if original is invalid"""
        # Try small adjustments around original position
        adjustments = [
            (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1),  # Direct neighbors
            (0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (-0.1, -0.1)  # Diagonal neighbors
        ]

        for dx, dy in adjustments:
            alt_step = {
                'position': Point(
                    x=original_step['position'].x + dx,
                    y=original_step['position'].y + dy,
                    z=original_step['position'].z
                ),
                'foot': original_step['foot'],
                'step_number': original_step['step_number']
            }

            if self.is_step_valid(alt_step, terrain_map):
                return alt_step

        return None
```

## Navigation Systems for Humanoid Robots

### Humanoid Navigation Server

```python
# humanoid_navigation_server.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import threading

class HumanoidNavigationServer(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_server')

        # Action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_navigate_to_pose,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'navigation_status', 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Path planner and controller
        self.path_planner = HumanoidPathPlanner()
        self.step_planner = StepPlanner()
        self.controller = HumanoidController()

        # Navigation state
        self.current_goal = None
        self.navigation_active = False
        self.navigation_thread = None

        self.get_logger().info('Humanoid Navigation Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject navigation goal"""
        self.get_logger().info('Received navigation goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject navigation cancel request"""
        self.get_logger().info('Received navigation cancel request')
        return CancelResponse.ACCEPT

    async def execute_navigate_to_pose(self, goal_handle):
        """Execute navigation to pose goal"""
        self.get_logger().info('Executing navigation to pose')

        goal = goal_handle.request.pose
        feedback = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        # Set current goal
        self.current_goal = goal
        self.navigation_active = True

        try:
            # Plan path
            self.get_logger().info('Planning path...')
            path = self.path_planner.plan_bipedal_path(
                self.get_robot_pose(), goal, self.get_costmap())

            if not path.poses:
                self.get_logger().error('Failed to plan path')
                result.result = result.FAILURE
                goal_handle.succeed()
                return result

            # Plan steps
            self.get_logger().info('Planning steps...')
            steps = self.step_planner.plan_steps(path, self.get_robot_pose())

            if not steps:
                self.get_logger().error('Failed to plan steps')
                result.result = result.FAILURE
                goal_handle.succeed()
                return result

            # Execute navigation
            self.get_logger().info('Executing navigation...')
            success = self.controller.execute_navigation_steps(steps)

            if success:
                result.result = result.SUCCEEDED
                self.get_logger().info('Navigation completed successfully')
            else:
                result.result = result.FAILURE
                self.get_logger().error('Navigation failed')

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            result.result = result.FAILURE

        goal_handle.succeed()
        self.navigation_active = False
        return result

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))

            pose = PoseStamped()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation

            return pose.pose
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return None

    def get_costmap(self):
        """Get current costmap (simplified)"""
        # In practice, this would get the actual costmap from Nav2
        from nav_msgs.msg import OccupancyGrid
        costmap = OccupancyGrid()
        costmap.header.frame_id = 'map'
        costmap.info.resolution = 0.05
        costmap.info.width = 100
        costmap.info.height = 100
        costmap.info.origin.position.x = -2.5
        costmap.info.origin.position.y = -2.5
        costmap.data = [0] * (costmap.info.width * costmap.info.height)  # Empty costmap
        return costmap

class HumanoidController:
    def __init__(self):
        self.balance_controller = BalanceController()
        self.step_controller = StepController()
        self.speed = 0.1  # m/s
        self.step_duration = 1.0  # seconds per step

    def execute_navigation_steps(self, steps):
        """Execute planned steps for navigation"""
        for step in steps:
            if not self.execute_single_step(step):
                return False  # Navigation failed

            # Check balance after each step
            if not self.balance_controller.is_balanced():
                self.get_logger().error('Robot lost balance during navigation')
                return False

        return True

    def execute_single_step(self, step):
        """Execute a single step"""
        try:
            # Move to step position using appropriate humanoid locomotion
            success = self.step_controller.move_to_step(step)

            if success:
                # Wait for step completion
                import time
                time.sleep(self.step_duration)

            return success
        except Exception as e:
            self.get_logger().error(f'Step execution failed: {e}')
            return False

class BalanceController:
    def __init__(self):
        self.balance_threshold = 0.1  # Balance threshold in meters
        self.com_height = 0.8  # Center of mass height

    def is_balanced(self):
        """Check if robot is currently balanced"""
        # This would interface with actual balance sensors
        # For simulation, return True
        return True

    def adjust_balance(self, target_com_position):
        """Adjust robot balance to target COM position"""
        # Implementation would use joint control to adjust balance
        pass

class StepController:
    def __init__(self):
        self.step_height = 0.05  # Step height for lifting foot
        self.step_duration = 1.0  # Duration for each step

    def move_to_step(self, step):
        """Move robot foot to specified step position"""
        # This would control the actual humanoid joints
        # For simulation, return True
        return True

def main(args=None):
    rclpy.init(args=args)

    node = HumanoidNavigationServer()

    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor(num_threads=4)
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

## Reinforcement Learning for Robot Control

### Deep Reinforcement Learning for Navigation

```python
# humanoid_rl_navigation.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from gym import spaces

class HumanoidNavigationEnv(gym.Env):
    """Custom environment for humanoid navigation with RL"""

    def __init__(self):
        super(HumanoidNavigationEnv, self).__init__()

        # Define action and observation space
        # Actions: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5]),  # min linear, min angular
            high=np.array([0.5, 0.5]),   # max linear, max angular
            dtype=np.float32
        )

        # Observations: [robot_x, robot_y, robot_yaw, goal_x, goal_y, obstacle_distances...]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 20),  # 20 observation values
            high=np.array([np.inf] * 20),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        # Random start and goal positions
        self.robot_pos = np.random.uniform(-5, 5, 2)
        self.goal_pos = np.random.uniform(-5, 5, 2)

        # Random obstacles
        self.obstacles = np.random.uniform(-6, 6, (5, 2))

        # Robot orientation
        self.robot_yaw = np.random.uniform(-np.pi, np.pi)

        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        # Apply action to robot
        linear_vel, angular_vel = action

        # Update robot position (simplified kinematics)
        dt = 0.1  # Time step
        self.robot_yaw += angular_vel * dt
        self.robot_pos[0] += linear_vel * np.cos(self.robot_yaw) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_yaw) * dt

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        done = self._is_done()

        # Get next observation
        observation = self._get_observation()

        # Info dictionary
        info = {}

        return observation, reward, done, info

    def _get_observation(self):
        """Get current observation from environment"""
        # Robot position and orientation
        obs = [self.robot_pos[0], self.robot_pos[1], self.robot_yaw]

        # Goal position
        obs.extend([self.goal_pos[0], self.goal_pos[1]])

        # Distance to obstacles (simplified)
        for obstacle in self.obstacles:
            dist = np.linalg.norm(self.robot_pos - obstacle)
            obs.append(min(dist, 10.0))  # Cap distance at 10m

        # Pad to fixed size
        while len(obs) < 20:
            obs.append(0.0)

        return np.array(obs[:20], dtype=np.float32)

    def _calculate_reward(self):
        """Calculate reward for current state"""
        # Distance to goal
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        # Reward based on getting closer to goal
        reward = -dist_to_goal * 0.1

        # Bonus for reaching goal
        if dist_to_goal < 0.5:
            reward += 100

        # Penalty for collisions
        for obstacle in self.obstacles:
            dist_to_obstacle = np.linalg.norm(self.robot_pos - obstacle)
            if dist_to_obstacle < 0.3:  # Collision threshold
                reward -= 50

        return reward

    def _is_done(self):
        """Check if episode is done"""
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        # Episode done if goal reached or too far from goal
        return dist_to_goal < 0.5 or dist_to_goal > 20

class DQN(nn.Module):
    """Deep Q-Network for humanoid navigation"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    """DQN Agent for humanoid navigation"""

    def __init__(self, state_size, action_size, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_humanoid_navigation_agent():
    """Train the humanoid navigation agent"""
    env = HumanoidNavigationEnv()
    state_size = env.observation_space.shape[0]
    action_size = 2  # linear and angular velocity

    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    scores = deque(maxlen=100)

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        for time_step in range(500):  # Max steps per episode
            action = agent.act(state)

            # Convert discrete action to continuous (simplified)
            continuous_action = np.array([action/10, action/20])  # Map to meaningful velocities

            next_state, reward, done, _ = env.step(continuous_action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)

        # Update target network periodically
        if e % 100 == 0:
            agent.update_target_network()

        print(f"Episode {e}, Score: {total_reward:.2f}, Average Score: {np.mean(scores):.2f}, Epsilon: {agent.epsilon:.2f}")

    return agent

# Example usage
if __name__ == "__main__":
    trained_agent = train_humanoid_navigation_agent()
```

### Humanoid Locomotion Control with RL

```python
# humanoid_locomotion_rl.py
import torch
import torch.nn as nn
import numpy as np
import math

class HumanoidLocomotionEnv:
    """Environment for humanoid locomotion learning"""

    def __init__(self):
        # Robot parameters
        self.torso_height = 0.8
        self.step_length = 0.3
        self.max_joint_angles = np.array([0.5, 1.0, 0.5])  # Hip, knee, ankle limits
        self.balance_threshold = 0.1  # Balance maintenance threshold

        # State initialization
        self.reset()

    def reset(self):
        """Reset to initial standing position"""
        self.torso_pos = np.array([0.0, 0.0, self.torso_height])
        self.torso_vel = np.array([0.0, 0.0, 0.0])
        self.torso_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
        self.left_foot_pos = np.array([-0.1, 0.1, 0.0])
        self.right_foot_pos = np.array([-0.1, -0.1, 0.0])
        self.time = 0.0

        return self.get_state()

    def get_state(self):
        """Get current state representation"""
        state = np.concatenate([
            self.torso_pos,           # 3: torso position
            self.torso_vel,           # 3: torso velocity
            self.torso_orientation,   # 4: torso orientation
            self.left_foot_pos[:2],   # 2: left foot x,y (z is 0)
            self.right_foot_pos[:2],  # 2: right foot x,y (z is 0)
            [self.time]               # 1: time
        ])
        return state

    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Action contains desired joint angles or torques
        # Simplified physics simulation
        self.apply_action(action)

        # Update physics
        self.update_physics()

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination
        done = self.is_terminal()

        self.time += 0.05  # 20Hz simulation

        return self.get_state(), reward, done, {}

    def apply_action(self, action):
        """Apply control action to robot"""
        # In a real implementation, this would set joint torques or positions
        # For simulation, we'll just use the action to influence movement
        pass

    def update_physics(self):
        """Update robot physics"""
        # Simplified physics update
        # In reality, this would involve complex dynamics simulation
        pass

    def calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0.0

        # Reward forward movement
        if self.torso_pos[0] > 0:  # Moving forward
            reward += self.torso_pos[0] * 0.1

        # Penalty for falling
        if abs(self.torso_pos[2] - self.torso_height) > 0.2:  # Fallen
            reward -= 100

        # Reward balance maintenance
        if abs(self.torso_pos[1]) < 0.1:  # Staying centered laterally
            reward += 0.1

        # Penalty for excessive joint angles
        # This would check actual joint angles in a real implementation

        return reward

    def is_terminal(self):
        """Check if episode should terminate"""
        # Fallen or moved too far
        return (abs(self.torso_pos[2] - self.torso_height) > 0.3 or
                abs(self.torso_pos[0]) > 10 or
                self.time > 100)  # Max time

class PPOAgent:
    """Proximal Policy Optimization agent for humanoid locomotion"""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor = self.build_actor(state_dim, action_dim)
        self.critic = self.build_critic(state_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lam = 0.95

    def build_actor(self, state_dim, action_dim):
        """Build actor network (policy)"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions between -1 and 1
        )

    def build_critic(self, state_dim):
        """Build critic network (value function)"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_mean = self.actor(state_tensor)
            action = torch.normal(action_mean, 0.1)  # Add noise for exploration
            action = torch.clamp(action, -1, 1)

        return action.squeeze().numpy()

    def compute_returns(self, rewards, values, dones):
        """Compute discounted returns for training"""
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i].item()
            else:
                next_value = values[i + 1].item()

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i].item()
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i].item())

        return torch.FloatTensor(returns)

def train_locomotion_agent():
    """Train humanoid locomotion agent"""
    env = HumanoidLocomotionEnv()
    agent = PPOAgent(state_dim=env.get_state().shape[0], action_dim=12)  # 12 joint controls

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while steps < 1000:  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {steps}")
```

## Knowledge Check

1. What are the key differences between Nav2 for wheeled robots and humanoid robots?
2. How do you modify path planning algorithms for bipedal locomotion constraints?
3. What are the essential components of a humanoid navigation system?
4. How can reinforcement learning be applied to humanoid robot control?

## Summary

This chapter covered Nav2 and path planning specifically for humanoid robots, addressing the unique challenges of bipedal locomotion. We explored humanoid-specific Nav2 configuration, step planning algorithms, navigation systems design, and reinforcement learning approaches for robot control. The chapter provided practical implementations for creating navigation systems that account for the balance, kinematic, and dynamic constraints of humanoid robots.

## Next Steps

In the next chapter, we'll examine sim-to-real transfer techniques, exploring the principles of transferring learned behaviors from simulation to real-world humanoid robots, including domain randomization and best practices for successful deployment.