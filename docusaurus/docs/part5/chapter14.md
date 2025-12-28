---
title: "Chapter 14: Bipedal Locomotion and Balance Control"
sidebar_label: "Chapter 14: Bipedal Locomotion"
---



# Chapter 14: Bipedal Locomotion and Balance Control

## Learning Objectives
- Understand the principles of bipedal locomotion and human walking patterns
- Master balance control mechanisms for humanoid robots
- Implement ZMP (Zero Moment Point) theory and control strategies
- Apply reinforcement learning techniques for gait optimization

## Introduction

Bipedal locomotion represents one of the most challenging problems in robotics, requiring the integration of complex control systems, dynamic modeling, and real-time adaptation. Unlike wheeled or tracked robots, humanoid robots must maintain balance while transferring their center of mass over alternating support points, mimicking the intricate balance control mechanisms found in human walking. This chapter explores the biomechanics of human walking, the physics of bipedal balance, and advanced control strategies for stable humanoid locomotion.

## Biomechanics of Human Walking

### Understanding Human Gait Cycle

Human walking follows a predictable gait cycle consisting of two main phases:

1. **Stance Phase (60%)**: Foot is in contact with ground
2. **Swing Phase (40%)**: Foot is off ground, moving forward

```python
# Gait cycle analysis
import numpy as np
import matplotlib.pyplot as plt

class HumanGaitAnalyzer:
    def __init__(self):
        self.cycle_duration = 1.0  # seconds
        self.stance_ratio = 0.6    # 60% stance, 40% swing
        self.stride_length = 0.7   # meters

    def analyze_gait_phase(self, time):
        """Determine gait phase at given time"""
        normalized_time = (time % self.cycle_duration) / self.cycle_duration

        if normalized_time < self.stance_ratio:
            # Stance phase
            stance_time = normalized_time / self.stance_ratio
            return 'stance', stance_time
        else:
            # Swing phase
            swing_time = (normalized_time - self.stance_ratio) / (1 - self.stance_ratio)
            return 'swing', swing_time

    def human_foot_trajectory(self, phase, phase_time, leg='left'):
        """Calculate human-like foot trajectory during gait cycle"""
        if phase == 'stance':
            # Foot is on ground, minimal movement
            x = phase_time * self.stride_length  # Move forward with body
            y = 0 if leg == 'left' else -0.2    # Lateral position
            z = 0  # Ground contact
        else:  # swing
            # Smooth trajectory lifting foot and moving forward
            x = self.stance_ratio + phase_time * (1 - self.stance_ratio) * self.stride_length
            y = 0 if leg == 'left' else -0.2

            # Vertical lift profile (parabolic)
            lift_factor = 4 * phase_time * (1 - phase_time)  # 0->1->0
            z = 0.05 * lift_factor  # 5cm maximum lift

        return np.array([x, y, z])

    def com_horizontal_movement(self, time):
        """Model horizontal CoM movement during walking"""
        # Smooth forward progression with slight lateral sway
        normalized_time = (time % self.cycle_duration) / self.cycle_duration

        # Forward progression
        x = (time // self.cycle_duration) * self.stride_length + normalized_time * self.stride_length

        # Lateral sway following double-support pattern
        # CoM shifts toward stance leg
        y = 0.05 * np.sin(2 * np.pi * time / self.cycle_duration)

        # Vertical oscillation (natural in human walking)
        z = 0.8 + 0.01 * np.cos(4 * np.pi * time / self.cycle_duration)

        return np.array([x, y, z])

    def step_timing_analysis(self, walking_speed):
        """Analyze step timing based on walking speed"""
        # Adjust gait cycle based on speed
        cycle_time = max(0.6, 1.0 / walking_speed)  # Min 0.6s for stability

        # Adjust stride length based on speed
        adjusted_stride = min(self.stride_length * walking_speed, 0.9)  # Max 0.9m

        return cycle_time, adjusted_stride
```

### Key Biomechanical Principles

Human walking involves several key biomechanical principles:

1. **Center of Mass Movement**: CoM moves in a figure-8 pattern
2. **Weight Transfer**: Smooth transition between support legs
3. **Ankle Strategy**: Ankle adjustments for balance
4. **Hip Strategy**: Hip movements for stability
5. **Arm Swing**: Counterbalance for leg movements

## Zero Moment Point (ZMP) Theory

### Understanding ZMP Fundamentals

The Zero Moment Point (ZMP) is a critical concept in bipedal robotics that represents the point on the ground where the net moment of the ground reaction forces equals zero.

```python
# ZMP calculation and analysis
class ZMPAnalyzer:
    def __init__(self, robot_height=0.8):
        self.robot_height = robot_height
        self.gravity = 9.81

    def calculate_zmp_simple(self, com_position, com_acceleration, foot_position):
        """Calculate ZMP using simplified inverted pendulum model"""
        # ZMP_x = CoM_x - h/g * CoM_x_ddot
        # ZMP_y = CoM_y - h/g * CoM_y_ddot
        # where h = CoM height, g = gravity, CoM_ddot = acceleration

        zmp_x = com_position[0] - (self.robot_height / self.gravity) * com_acceleration[0]
        zmp_y = com_position[1] - (self.robot_height / self.gravity) * com_acceleration[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def calculate_zmp_full(self, com_position, com_velocity, com_acceleration,
                          angular_momentum, external_forces):
        """Calculate ZMP using full dynamic equations"""
        # More complex ZMP calculation considering angular momentum
        # ZMP = (Σ(mi * (ri × ai) + hi_dot) × g) / (Σmi * g)
        # where mi = mass of link i, ri = position, ai = acceleration, hi = angular momentum

        # Simplified implementation
        zmp_x = com_position[0] - (self.robot_height / self.gravity) * com_acceleration[0]
        zmp_y = com_position[1] - (self.robot_height / self.gravity) * com_acceleration[1]

        # Add angular momentum correction
        h_dot = np.array([0, 0, 0])  # Rate of change of angular momentum
        if np.any(angular_momentum):
            # Simplified correction term
            correction = np.cross(angular_momentum, [0, 0, 1])[:2] / (self.total_mass * self.gravity)
            zmp_x += correction[0]
            zmp_y += correction[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def calculate_support_polygon(self, left_foot, right_foot, foot_size=[0.2, 0.1]):
        """Calculate support polygon based on foot positions"""
        # Create polygon vertices based on foot positions and sizes
        if left_foot is not None and right_foot is not None:
            # Double support - convex hull of both feet
            left_vertices = self.foot_polygon(left_foot, foot_size)
            right_vertices = self.foot_polygon(right_foot, foot_size)

            all_vertices = np.vstack([left_vertices, right_vertices])
            return self.convex_hull(all_vertices)
        elif left_foot is not None:
            # Left foot support
            return self.foot_polygon(left_foot, foot_size)
        elif right_foot is not None:
            # Right foot support
            return self.foot_polygon(right_foot, foot_size)
        else:
            # No support (airborne)
            return np.array([])

    def foot_polygon(self, foot_center, foot_size):
        """Create polygon representing foot contact area"""
        half_length, half_width = foot_size[0] / 2, foot_size[1] / 2
        dx, dy = foot_center[0], foot_center[1]

        return np.array([
            [dx - half_length, dy - half_width],
            [dx + half_length, dy - half_width],
            [dx + half_length, dy + half_width],
            [dx - half_length, dy + half_width]
        ])

    def is_zmp_stable(self, zmp_position, support_polygon):
        """Check if ZMP is within support polygon"""
        if len(support_polygon) == 0:
            return False  # No support polygon means unstable

        return self.point_in_polygon(zmp_position[:2], support_polygon)

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def convex_hull(self, points):
        """Calculate convex hull of points (simplified implementation)"""
        # This would use Graham scan or other convex hull algorithm
        # For simplicity, return the points as-is
        return points
```

### ZMP-Based Control Strategies

```python
# ZMP-based balance controllers
class ZMPController:
    def __init__(self, sampling_time=0.005):
        self.dt = sampling_time
        self.zmp_reference = np.zeros(2)
        self.zmp_actual = np.zeros(2)
        self.com_state = {'position': np.zeros(3), 'velocity': np.zeros(3), 'acceleration': np.zeros(3)}

        # Controller gains
        self.kp = 10.0  # Proportional gain
        self.kd = 2.0   # Derivative gain
        self.ki = 0.5   # Integral gain

        # Integral and derivative terms
        self.zmp_error_integral = np.zeros(2)
        self.prev_zmp_error = np.zeros(2)

    def compute_balance_control(self, measured_zmp, desired_zmp, support_polygon):
        """Compute balance control output based on ZMP error"""
        # Calculate ZMP error
        zmp_error = desired_zmp[:2] - measured_zmp[:2]

        # Update integral term with anti-windup
        self.zmp_error_integral += zmp_error * self.dt
        max_integral = 5.0  # Limit integral windup
        self.zmp_error_integral = np.clip(self.zmp_error_integral, -max_integral, max_integral)

        # Calculate derivative term
        zmp_error_derivative = (zmp_error - self.prev_zmp_error) / self.dt
        self.prev_zmp_error = zmp_error

        # Compute control output
        proportional_term = self.kp * zmp_error
        integral_term = self.ki * self.zmp_error_integral
        derivative_term = self.kd * zmp_error_derivative

        control_output = proportional_term + integral_term + derivative_term

        # Apply saturation limits
        max_control = 1.0
        control_output = np.clip(control_output, -max_control, max_control)

        return control_output

    def generate_zmp_trajectory(self, step_time, start_zmp, end_zmp):
        """Generate smooth ZMP trajectory between two points"""
        # Use quintic polynomial for smooth transition
        t = np.linspace(0, step_time, int(step_time / self.dt))

        # Quintic polynomial coefficients for smooth trajectory
        # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # With boundary conditions: s(0)=0, s(T)=1, s'(0)=0, s'(T)=0, s''(0)=0, s''(T)=0
        a0, a1, a2, a3, a4, a5 = 0, 0, 0, 10/(step_time**3), -15/(step_time**4), 6/(step_time**5)

        s = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)

        # Generate ZMP trajectory
        zmp_trajectory = np.zeros((len(t), 2))
        for i in range(len(t)):
            zmp_trajectory[i] = start_zmp + s[i] * (end_zmp - start_zmp)

        return zmp_trajectory

    def linear_inverted_pendulum_control(self, com_position, com_velocity, desired_zmp):
        """Control using Linear Inverted Pendulum Model"""
        # LIPM: x_ddot = ω²(x - x_zmp)
        # where ω² = g / h (h = CoM height)

        com_height = com_position[2]  # z-component
        omega_squared = self.gravity / com_height

        # Calculate required CoM acceleration
        com_acceleration = omega_squared * (com_position[:2] - desired_zmp)

        return com_acceleration
```

## Balance Control Mechanisms

### Ankle Strategy Control

The ankle strategy is the primary mechanism for maintaining balance during quiet standing:

```python
# Ankle strategy balance control
class AnkleStrategyController:
    def __init__(self, robot_height=0.8, ankle_stiffness=1000, ankle_damping=100):
        self.robot_height = robot_height
        self.ankle_stiffness = ankle_stiffness
        self.ankle_damping = ankle_damping
        self.gravity = 9.81

    def ankle_balance_control(self, measured_com_position, measured_com_velocity,
                             desired_com_position, foot_position):
        """Implement ankle strategy for balance control"""
        # Calculate CoM deviation from upright position
        com_deviation = measured_com_position[:2] - desired_com_position[:2]

        # Calculate required ankle torques based on inverted pendulum model
        # Tau = K * theta + D * theta_dot
        # where theta is the tilt angle from vertical

        # Calculate tilt angles
        tilt_x = np.arctan2(
            measured_com_position[0] - foot_position[0],
            self.robot_height
        )
        tilt_y = np.arctan2(
            measured_com_position[1] - foot_position[1],
            self.robot_height
        )

        # Calculate angular velocities
        angular_vel_x = measured_com_velocity[0] / self.robot_height
        angular_vel_y = measured_com_velocity[1] / self.robot_height

        # Calculate required ankle torques
        torque_x = -self.ankle_stiffness * tilt_x - self.ankle_damping * angular_vel_x
        torque_y = -self.ankle_stiffness * tilt_y - self.ankle_damping * angular_vel_y

        return np.array([torque_x, torque_y])

    def ankle_impedance_control(self, desired_position, desired_velocity,
                               actual_position, actual_velocity, stiffness, damping):
        """Impedance control for ankle joints"""
        # F = K(x_d - x) + D(v_d - v)
        position_error = desired_position - actual_position
        velocity_error = desired_velocity - actual_velocity

        force = stiffness * position_error + damping * velocity_error

        return force
```

### Hip Strategy Control

For larger perturbations, the hip strategy becomes necessary:

```python
# Hip strategy balance control
class HipStrategyController:
    def __init__(self, hip_stiffness=2000, hip_damping=200):
        self.hip_stiffness = hip_stiffness
        self.hip_damping = hip_damping
        self.upper_body_controller = None

    def hip_balance_control(self, com_position, com_velocity,
                           pelvis_orientation, pelvis_angular_velocity):
        """Implement hip strategy for balance control"""
        # Calculate required hip torques to move CoM back to stable region

        # Determine if ankle strategy is insufficient
        com_deviation = np.linalg.norm(com_position[:2])
        if com_deviation > 0.05:  # If CoM is too far from upright
            # Engage hip strategy
            return self.engaged_hip_control(
                com_position, com_velocity,
                pelvis_orientation, pelvis_angular_velocity
            )
        else:
            # Use minimal hip adjustment
            return self.minimal_hip_adjustment(
                com_position, pelvis_orientation
            )

    def engaged_hip_control(self, com_position, com_velocity,
                           pelvis_orientation, pelvis_angular_velocity):
        """Full hip strategy engagement"""
        # Calculate required CoM movement to return to stable zone
        desired_com_offset = -0.3 * com_position[:2]  # Move CoM toward support

        # Calculate required pelvis orientation change
        pelvis_roll = 0.1 * desired_com_offset[1]  # Roll toward stable side
        pelvis_pitch = -0.1 * desired_com_offset[0]  # Pitch forward/back

        # Calculate required hip torques
        current_roll = pelvis_orientation[0]
        current_pitch = pelvis_orientation[1]

        roll_error = pelvis_roll - current_roll
        pitch_error = pelvis_pitch - current_pitch

        roll_torque = self.hip_stiffness * roll_error - self.hip_damping * pelvis_angular_velocity[0]
        pitch_torque = self.hip_stiffness * pitch_error - self.hip_damping * pelvis_angular_velocity[1]

        # Return torques for left and right hips
        return np.array([roll_torque, pitch_torque, roll_torque, pitch_torque])  # [L_roll, L_pitch, R_roll, R_pitch]

    def minimal_hip_adjustment(self, com_position, pelvis_orientation):
        """Minimal hip adjustments to complement ankle strategy"""
        # Small hip adjustments to fine-tune balance
        hip_adjustment = -0.1 * com_position[:2]  # Small corrective movement
        return np.array([0.1 * hip_adjustment[1], 0, 0.1 * hip_adjustment[1], 0])
```

### Capture Point Control

The capture point indicates where the CoM must step to come to rest:

```python
# Capture point calculation and control
class CapturePointController:
    def __init__(self, robot_height=0.8):
        self.robot_height = robot_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.robot_height)

    def calculate_capture_point(self, com_position, com_velocity):
        """Calculate capture point where robot should step to stop"""
        # Capture Point = CoM + CoM_velocity / ω
        # where ω = √(g/h)

        capture_point = com_position[:2] + com_velocity[:2] / self.omega

        return capture_point

    def capture_point_control(self, current_capture_point, target_capture_point,
                             foot_position, foot_placement_strategy='desired'):
        """Control based on capture point positioning"""
        # Calculate required foot placement to achieve target capture point

        if foot_placement_strategy == 'desired':
            # Place foot at desired capture point
            desired_foot_position = target_capture_point
        elif foot_placement_strategy == 'compromise':
            # Place foot at compromise between current and target
            desired_foot_position = 0.7 * target_capture_point + 0.3 * foot_position[:2]
        else:
            # Default to target
            desired_foot_position = target_capture_point

        # Calculate step parameters
        step_vector = desired_foot_position - foot_position[:2]
        step_distance = np.linalg.norm(step_vector)

        if step_distance > 0:
            step_direction = step_vector / step_distance
        else:
            step_direction = np.array([0, 0])

        return {
            'step_direction': step_direction,
            'step_distance': step_distance,
            'desired_foot_position': desired_foot_position
        }

    def predictive_capture_point(self, com_position, com_velocity, time_horizon):
        """Predict capture point at future time"""
        # Account for expected CoM motion over time horizon
        predicted_com_pos = com_position[:2] + com_velocity[:2] * time_horizon
        predicted_com_vel = com_velocity[:2]  # Assume constant velocity for short term

        predicted_capture_point = predicted_com_pos + predicted_com_vel / self.omega

        return predicted_capture_point
```

## Walking Pattern Generation

### Footstep Planning

```python
# Footstep planning for bipedal walking
class FootstepPlanner:
    def __init__(self, step_length=0.6, step_width=0.2, step_height=0.05):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.nominal_width = step_width

    def plan_straight_walk(self, num_steps, start_position, start_orientation):
        """Plan footsteps for straight walking"""
        footsteps = []

        current_pos = np.array(start_position)
        current_orient = start_orientation  # Angle in radians

        for i in range(num_steps):
            # Calculate foot position based on step number
            # Odd steps: right foot, even steps: left foot
            if i % 2 == 0:  # Left foot
                foot_offset = np.array([-self.step_width/2, 0])
            else:  # Right foot
                foot_offset = np.array([self.step_width/2, 0])

            # Rotate offset by current orientation
            cos_o, sin_o = np.cos(current_orient), np.sin(current_orient)
            rotation_matrix = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
            rotated_offset = rotation_matrix @ foot_offset

            # Calculate foot position
            foot_pos = current_pos + np.array([self.step_length * (i+1), 0])
            foot_pos[:2] += rotated_offset

            footsteps.append({
                'position': foot_pos,
                'orientation': current_orient,
                'step_number': i + 1,
                'support_leg': 'right' if i % 2 == 0 else 'left'
            })

        return footsteps

    def plan_turning_walk(self, num_steps, turn_angle, start_position, start_orientation):
        """Plan footsteps for turning walk"""
        footsteps = []

        current_pos = np.array(start_position)
        current_orient = start_orientation
        turn_per_step = turn_angle / num_steps

        for i in range(num_steps):
            # Calculate turning offset
            if i % 2 == 0:  # Left foot
                foot_offset = np.array([-self.step_width/2, 0])
            else:  # Right foot
                foot_offset = np.array([self.step_width/2, 0])

            # Apply turning offset (creates turning arc)
            turn_radius = self.step_width / (2 * np.sin(turn_per_step/2)) if turn_per_step != 0 else float('inf')

            # Calculate foot position accounting for turning
            step_arc = self.step_length if turn_radius == float('inf') else self.step_length
            step_angle = current_orient + turn_per_step * i

            # Calculate position with turning
            dx = step_arc * np.cos(step_angle)
            dy = step_arc * np.sin(step_angle)

            foot_pos = current_pos + np.array([dx, dy, 0])

            # Add lateral offset
            cos_o, sin_o = np.cos(current_orient + i * turn_per_step), np.sin(current_orient + i * turn_per_step)
            rotation_matrix = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
            rotated_offset = rotation_matrix @ foot_offset
            foot_pos[:2] += rotated_offset

            footsteps.append({
                'position': foot_pos,
                'orientation': current_orient + (i + 1) * turn_per_step,
                'step_number': i + 1,
                'support_leg': 'right' if i % 2 == 0 else 'left'
            })

        return footsteps

    def plan_terrain_adaptive_steps(self, terrain_map, start_pos, goal_pos):
        """Plan footsteps adapting to terrain conditions"""
        # This would implement path planning considering terrain constraints
        # such as slopes, obstacles, and surface stability

        # For simplicity, implement basic obstacle avoidance
        footsteps = []

        # Calculate straight-line path
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction)
        num_steps = int(distance / self.step_length) + 1

        for i in range(num_steps):
            t = i / num_steps
            nominal_pos = start_pos + t * direction

            # Check terrain for this position
            if self.terrain_is_safe(nominal_pos, terrain_map):
                foot_pos = nominal_pos
            else:
                # Find alternative safe position
                foot_pos = self.find_safe_alternative(nominal_pos, terrain_map)

            # Alternate feet
            if i % 2 == 0:
                foot_pos[1] += (-1) ** (i // 2) * self.step_width / 2
            else:
                foot_pos[1] += (-1) ** (i // 2) * self.step_width / 2

            footsteps.append({
                'position': foot_pos,
                'orientation': np.arctan2(direction[1], direction[0]),
                'step_number': i + 1,
                'support_leg': 'right' if i % 2 == 0 else 'left'
            })

        return footsteps

    def terrain_is_safe(self, position, terrain_map):
        """Check if terrain at position is safe for stepping"""
        # Check for obstacles, drop-offs, unstable surfaces
        return True  # Simplified

    def find_safe_alternative(self, nominal_pos, terrain_map):
        """Find safe alternative position near nominal"""
        # Search in a spiral pattern around nominal position
        return nominal_pos  # Simplified
```

### Trajectory Generation

```python
# Walking trajectory generation
class WalkingTrajectoryGenerator:
    def __init__(self, step_duration=1.0, double_support_ratio=0.1):
        self.step_duration = step_duration
        self.ds_ratio = double_support_ratio  # 10% double support
        self.ss_ratio = 1 - double_support_ratio  # 90% single support

        # CoM trajectory parameters
        self.com_height = 0.8
        self.com_lateral_sway = 0.05
        self.com_forward_progression = 0.6

    def generate_com_trajectory(self, left_footsteps, right_footsteps):
        """Generate CoM trajectory following footsteps"""
        # Create timeline
        total_time = len(left_footsteps) * self.step_duration
        dt = 0.01  # 100Hz trajectory
        time_array = np.arange(0, total_time, dt)

        com_trajectory = np.zeros((len(time_array), 3))  # [x, y, z]

        for i, t in enumerate(time_array):
            # Determine which phase we're in
            step_num = int(t / self.step_duration)
            if step_num >= len(left_footsteps):
                step_num = len(left_footsteps) - 1

            phase_time = (t % self.step_duration) / self.step_duration

            # Calculate CoM position based on walking phase
            com_pos = self.calculate_com_position_at_time(
                step_num, phase_time, left_footsteps, right_footsteps
            )

            com_trajectory[i] = com_pos

        return com_trajectory

    def calculate_com_position_at_time(self, step_num, phase_time, left_footsteps, right_footsteps):
        """Calculate CoM position at specific time in gait cycle"""
        # Forward progression
        forward_pos = step_num * self.com_forward_progression + phase_time * self.com_forward_progression

        # Lateral swaying (shifts weight between feet)
        if step_num % 2 == 0:  # Left foot support phase
            # CoM shifts over left foot
            lateral_pos = -self.com_lateral_sway * np.cos(np.pi * phase_time)
        else:  # Right foot support phase
            # CoM shifts over right foot
            lateral_pos = self.com_lateral_sway * np.cos(np.pi * phase_time)

        # Vertical oscillation (natural in walking)
        vertical_oscillation = 0.02 * np.cos(2 * np.pi * phase_time)
        height = self.com_height + vertical_oscillation

        return np.array([forward_pos, lateral_pos, height])

    def generate_foot_trajectory(self, touchdown_pos, liftoff_pos, phase_time, support_phase=True):
        """Generate smooth foot trajectory between two positions"""
        # Different trajectories for swing vs stance phases
        if support_phase:
            # Foot on ground - minimal movement
            return np.array([
                liftoff_pos[0] + phase_time * (touchdown_pos[0] - liftoff_pos[0]),
                liftoff_pos[1] + phase_time * (touchdown_pos[1] - liftoff_pos[1]),
                0  # Ground contact
            ])
        else:
            # Swing phase - lifting and moving foot
            # Horizontal movement
            x = liftoff_pos[0] + phase_time * (touchdown_pos[0] - liftoff_pos[0])
            y = liftoff_pos[1] + phase_time * (touchdown_pos[1] - liftoff_pos[1])

            # Vertical lift (parabolic trajectory)
            lift_profile = 4 * phase_time * (1 - phase_time)  # 0->1->0
            z = 0.05 * lift_profile  # Maximum 5cm lift

            return np.array([x, y, z])

    def generate_joint_trajectories(self, com_trajectory, footsteps):
        """Generate joint angle trajectories from CoM and foot trajectories"""
        # This would use inverse kinematics to calculate joint angles
        # that achieve the desired CoM and foot positions

        # Simplified approach: calculate for key poses
        joint_trajectories = []

        for i, com_pos in enumerate(com_trajectory):
            # Calculate required joint angles using inverse kinematics
            # This would call the humanoid IK system
            joint_angles = self.calculate_inverse_kinematics(com_pos, footsteps)
            joint_trajectories.append(joint_angles)

        return joint_trajectories

    def calculate_inverse_kinematics(self, com_position, foot_positions):
        """Calculate joint angles for given CoM and foot positions"""
        # This would implement complex IK for the entire humanoid
        # For now, return placeholder
        return np.zeros(28)  # Assuming 28 DOF humanoid
```

## Reinforcement Learning for Gait Optimization

### Gait Learning Framework

```python
# Reinforcement learning for gait optimization
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class GaitLearningEnvironment:
    def __init__(self, robot_simulator):
        self.robot = robot_simulator
        self.action_space = 10  # Joint angle adjustments
        self.observation_space = 20  # Sensor readings and state
        self.max_episode_length = 1000

    def reset(self):
        """Reset environment to initial state"""
        self.robot.reset()
        self.timestep = 0
        self.cumulative_reward = 0

        # Return initial observation
        return self.get_observation()

    def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        # Apply action (joint angle adjustments)
        self.robot.apply_action(action)

        # Simulate one timestep
        self.robot.simulate_timestep()

        # Get new observation
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward()
        self.cumulative_reward += reward

        # Check termination conditions
        done = self.check_termination()
        info = {'cumulative_reward': self.cumulative_reward}

        self.timestep += 1

        return observation, reward, done, info

    def get_observation(self):
        """Get current observation from robot sensors"""
        # Combine various sensor readings
        sensor_data = self.robot.get_sensor_data()
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        com_state = self.robot.get_com_state()
        foot_contacts = self.robot.get_foot_contacts()

        # Concatenate all observations
        observation = np.concatenate([
            sensor_data.flatten(),
            joint_positions,
            joint_velocities,
            com_state,
            foot_contacts
        ])

        return observation[:self.observation_space]  # Truncate to expected size

    def calculate_reward(self):
        """Calculate reward based on walking performance"""
        reward = 0

        # Forward velocity reward
        forward_vel = self.robot.get_forward_velocity()
        reward += 0.1 * forward_vel

        # Balance reward (CoM over support polygon)
        com_pos = self.robot.get_com_position()
        zmp_pos = self.robot.get_zmp_position()
        com_zmp_dist = np.linalg.norm(com_pos[:2] - zmp_pos[:2])
        reward += -0.5 * com_zmp_dist  # Penalty for CoM-ZMP distance

        # Energy efficiency reward
        joint_torques = self.robot.get_joint_torques()
        energy_penalty = -0.01 * np.sum(np.abs(joint_torques))
        reward += energy_penalty

        # Upright posture reward
        pitch_angle = self.robot.get_pitch_angle()
        reward += -0.2 * abs(pitch_angle)  # Penalty for tilting

        # Smooth motion reward
        joint_acc = self.robot.get_joint_accelerations()
        smoothness_penalty = -0.001 * np.sum(np.square(joint_acc))
        reward += smoothness_penalty

        return reward

    def check_termination(self):
        """Check if episode should terminate"""
        # Terminate if fallen
        pitch_angle = self.robot.get_pitch_angle()
        if abs(pitch_angle) > np.pi / 3:  # Fallen (60 degrees)
            return True

        # Terminate if CoM too far from support
        com_pos = self.robot.get_com_position()
        zmp_pos = self.robot.get_zmp_position()
        if np.linalg.norm(com_pos[:2] - zmp_pos[:2]) > 0.3:  # Too far
            return True

        # Terminate if exceeded max steps
        if self.timestep >= self.max_episode_length:
            return True

        return False

class GaitPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(GaitPolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Actor network outputs mean and std for Gaussian policy
        self.actor_mean = nn.Linear(output_dim, output_dim)
        self.actor_std = nn.Linear(output_dim, output_dim)

    def forward(self, state):
        features = self.network(state)
        mean = torch.tanh(self.actor_mean(features))  # Actions between -1 and 1
        log_std = self.actor_std(features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Std between exp(-20) and exp(2)
        std = torch.exp(log_std)

        return mean, std

class GaitLearningAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network
        self.policy_net = GaitPolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update parameter

    def select_action(self, state, evaluate=False):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)

        if evaluate:
            return mean.cpu().numpy()[0]
        else:
            # Sample from Gaussian distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        """Train the policy network"""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples yet

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # Calculate targets
        with torch.no_grad():
            next_means, next_stds = self.policy_net(next_states)
            next_dist = torch.distributions.Normal(next_means, next_stds)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions).sum(dim=1, keepdim=True)

            next_q_values = rewards + (self.gamma * (1 - dones.float()) * next_log_probs)

        # Calculate current Q-values
        means, stds = self.policy_net(states)
        dist = torch.distributions.Normal(means, stds)
        sampled_actions = dist.rsample()
        log_probs = dist.log_prob(sampled_actions).sum(dim=1, keepdim=True)

        # Calculate loss
        q_values = dist.log_prob(actions).sum(dim=1, keepdim=True)
        actor_loss = (log_probs - next_q_values).mean()

        # Update policy
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
```

## Practical Implementation and Control

### Integrated Balance Controller

```python
# Integrated balance and locomotion controller
class IntegratedBalanceController:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.zmp_controller = ZMPController()
        self.ankle_controller = AnkleStrategyController()
        self.hip_controller = HipStrategyController()
        self.capture_point_controller = CapturePointController()
        self.footstep_planner = FootstepPlanner()

        # State estimation
        self.state_estimator = StateEstimator()

        # Walking parameters
        self.walking_speed = 0.6  # m/s
        self.step_length = 0.6    # m
        self.step_width = 0.2     # m
        self.step_height = 0.05   # m

    def update_balance_control(self, dt):
        """Main control loop for balance and locomotion"""
        # Estimate current state
        current_state = self.state_estimator.estimate_state()

        # Calculate ZMP
        measured_zmp = self.calculate_current_zmp(current_state)

        # Determine if we need balance control or walking control
        if self.is_quasi_static(current_state):
            # Balance control mode
            balance_control = self.balance_control_mode(current_state, measured_zmp)
            walking_control = np.zeros(28)  # No walking commands
        else:
            # Walking mode
            balance_control = self.balance_during_walking(current_state, measured_zmp)
            walking_control = self.walking_control_mode(current_state)

        # Combine controls
        total_control = balance_control + walking_control

        # Apply control to robot
        self.robot.apply_control(total_control)

        return total_control

    def balance_control_mode(self, state, measured_zmp):
        """Pure balance control (standing, recovery)"""
        # Calculate desired ZMP (usually under feet)
        support_polygon = self.calculate_support_polygon(state)
        desired_zmp = self.find_stable_zmp_location(measured_zmp, support_polygon)

        # Generate balance control commands
        balance_torques = self.zmp_controller.compute_balance_control(
            measured_zmp, desired_zmp, support_polygon
        )

        # Add ankle strategy control
        ankle_torques = self.ankle_controller.ankle_balance_control(
            state['com_position'], state['com_velocity'],
            state['desired_com'], state['left_foot_pos']
        )

        # Add hip strategy if needed
        if self.should_engage_hip_strategy(state):
            hip_torques = self.hip_controller.hip_balance_control(
                state['com_position'], state['com_velocity'],
                state['pelvis_orientation'], state['pelvis_angular_vel']
            )
        else:
            hip_torques = np.zeros(4)  # [L_roll, L_pitch, R_roll, R_pitch]

        # Combine all balance controls
        total_torques = np.zeros(28)  # Assuming 28 DOF humanoid
        # Map balance torques to appropriate joints
        # This mapping would be specific to the robot model

        return total_torques

    def walking_control_mode(self, state):
        """Walking control with balance maintenance"""
        # Plan footsteps
        footsteps = self.footstep_planner.plan_straight_walk(
            10,  # Plan next 10 steps
            state['robot_position'],
            state['robot_orientation']
        )

        # Generate walking trajectory
        trajectory_gen = WalkingTrajectoryGenerator()
        com_trajectory = trajectory_gen.generate_com_trajectory(
            footsteps['left'], footsteps['right']
        )

        # Calculate desired ZMP from trajectory
        desired_zmp = self.calculate_trajectory_zmp(com_trajectory)

        # Generate walking control commands
        walking_commands = self.generate_walking_commands(
            state, footsteps, com_trajectory
        )

        return walking_commands

    def balance_during_walking(self, state, measured_zmp):
        """Balance control during walking motion"""
        # Calculate desired ZMP following the walking pattern
        desired_zmp = self.calculate_walking_zmp_reference(state)

        # Apply ZMP control for balance during walking
        balance_control = self.zmp_controller.compute_balance_control(
            measured_zmp, desired_zmp, self.calculate_support_polygon(state)
        )

        # Apply ankle and hip strategies as needed
        ankle_control = self.ankle_controller.ankle_balance_control(
            state['com_position'], state['com_velocity'],
            state['desired_com'], state['support_foot_pos']
        )

        return balance_control + ankle_control

    def calculate_current_zmp(self, state):
        """Calculate current ZMP from force sensors"""
        # Get ground reaction forces from foot sensors
        left_foot_forces = state['left_foot_forces']  # [Fx, Fy, Fz, Mx, My, Mz]
        right_foot_forces = state['right_foot_forces']

        # Calculate ZMP from force measurements
        # ZMP = -[M] / Fz where [M] = [Mx, My] and Fz is normal force

        # Left foot ZMP
        if left_foot_forces[2] > 10:  # If sufficient normal force
            left_zmp_x = -left_foot_forces[4] / left_foot_forces[2]  # -My / Fz
            left_zmp_y = left_foot_forces[3] / left_foot_forces[2]   # Mx / Fz
        else:
            left_zmp_x, left_zmp_y = 0, 0

        # Right foot ZMP
        if right_foot_forces[2] > 10:
            right_zmp_x = -right_foot_forces[4] / right_foot_forces[2]
            right_zmp_y = right_foot_forces[3] / right_foot_forces[2]
        else:
            right_zmp_x, right_zmp_y = 0, 0

        # Weighted average based on support forces
        total_fz = max(left_foot_forces[2], 0) + max(right_foot_forces[2], 0)
        if total_fz > 0:
            weighted_zmp_x = (left_foot_forces[2] * left_zmp_x + right_foot_forces[2] * right_zmp_x) / total_fz
            weighted_zmp_y = (left_foot_forces[2] * left_zmp_y + right_foot_forces[2] * right_zmp_y) / total_fz
        else:
            weighted_zmp_x, weighted_zmp_y = 0, 0

        return np.array([weighted_zmp_x, weighted_zmp_y, 0.0])

    def is_quasi_static(self, state):
        """Determine if robot is in quasi-static condition"""
        # Quasi-static: low velocities and accelerations
        com_velocity = np.linalg.norm(state['com_velocity'])
        com_acceleration = np.linalg.norm(state['com_acceleration'])

        return com_velocity < 0.1 and com_acceleration < 0.5

    def should_engage_hip_strategy(self, state):
        """Determine if hip strategy should be engaged"""
        com_deviation = np.linalg.norm(state['com_position'][:2] - state['desired_com'][:2])
        return com_deviation > 0.08  # Engage if CoM deviates by more than 8cm

    def find_stable_zmp_location(self, current_zmp, support_polygon):
        """Find nearest stable ZMP location within support polygon"""
        if len(support_polygon) == 0:
            return current_zmp  # No support, return current

        # If current ZMP is in support, return it
        if self.zmp_controller.is_zmp_stable(current_zmp, support_polygon):
            return current_zmp

        # Find closest point in polygon to current ZMP
        closest_point = self.find_closest_point_in_polygon(
            current_zmp[:2], support_polygon
        )

        return np.array([closest_point[0], closest_point[1], 0.0])

    def find_closest_point_in_polygon(self, point, polygon):
        """Find closest point on polygon to given point"""
        # This would implement the closest point algorithm
        # For now, return centroid as approximation
        centroid = np.mean(polygon, axis=0)
        return centroid
```

## Safety and Robustness Considerations

### Fall Prevention and Recovery

```python
# Fall prevention and recovery systems
class FallPreventionSystem:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.balance_margin = 0.1  # 10cm safety margin
        self.fall_threshold = 1.0  # Fall if CoM-ZMP > 1.0m
        self.recovery_enabled = True

    def monitor_stability(self, state):
        """Monitor robot stability and trigger countermeasures"""
        # Calculate stability metrics
        com_zmp_distance = np.linalg.norm(
            state['com_position'][:2] - state['zmp_position'][:2]
        )

        pitch_angle = state['pitch_angle']
        roll_angle = state['roll_angle']

        # Check if falling
        is_falling = (
            com_zmp_distance > self.fall_threshold or
            abs(pitch_angle) > np.pi / 3 or  # 60 degrees
            abs(roll_angle) > np.pi / 3
        )

        if is_falling:
            return self.initiate_fall_protection(state)

        # Check if approaching fall conditions
        is_unstable = (
            com_zmp_distance > self.fall_threshold * 0.8 or
            abs(pitch_angle) > np.pi / 6 or  # 30 degrees
            abs(roll_angle) > np.pi / 6
        )

        if is_unstable:
            return self.apply_aggressive_balance_control(state)

        return "stable"

    def initiate_fall_protection(self, state):
        """Initiate fall protection measures"""
        if not self.recovery_enabled:
            return "falling"

        # Emergency measures
        # 1. Reduce stiffness to minimize impact damage
        self.reduce_joint_stiffness()

        # 2. Prepare for protective landing
        self.prepare_protective_posture(state)

        # 3. Attempt last-resort recovery
        recovery_attempt = self.last_resort_recovery(state)

        if recovery_attempt['success']:
            return "recovered"
        else:
            return "falling_with_protection"

    def apply_aggressive_balance_control(self, state):
        """Apply aggressive balance control to prevent fall"""
        # Increase control gains
        self.increase_control_gains()

        # Attempt rapid recovery steps
        recovery_step = self.plan_recovery_step(state)

        if recovery_step:
            # Execute recovery step
            self.execute_recovery_step(recovery_step)
            return "recovering"
        else:
            # No recovery step possible, apply other measures
            self.apply_upper_body_control(state)
            return "critical_balance"

    def last_resort_recovery(self, state):
        """Attempt last-resort recovery maneuvers"""
        # 1. Arm swing for angular momentum
        self.activate_arm_swing_recovery(state)

        # 2. Rapid stepping if possible
        if self.can_take_emergency_step(state):
            emergency_step = self.plan_emergency_step(state)
            self.execute_emergency_step(emergency_step)
            return {'success': True, 'type': 'step_recovery'}

        # 3. Hip strategy with maximum effort
        hip_correction = self.maximum_hip_correction(state)
        self.apply_hip_correction(hip_correction)

        # Check if recovery successful
        new_com_zmp_dist = self.estimate_new_com_zmp_distance(state, hip_correction)

        return {'success': new_com_zmp_dist < self.fall_threshold, 'type': 'hip_recovery'}

    def reduce_joint_stiffness(self):
        """Reduce joint stiffness to minimize impact damage"""
        # Set low stiffness for impact absorption
        low_stiffness = 50  # N*m/rad
        self.robot.set_joint_stiffness(low_stiffness)

    def increase_control_gains(self):
        """Increase control gains for more aggressive response"""
        # Increase PID gains for faster response
        self.robot.increase_pid_gains(factor=2.0)

    def can_take_emergency_step(self, state):
        """Check if emergency step is possible"""
        # Check if swing leg is in position to step
        # Check if there's time before impact
        # Check if ground is accessible

        swing_leg_ready = state['swing_foot_clearance'] > 0.02  # 2cm clearance
        time_to_impact = self.estimate_time_to_impact(state)

        return swing_leg_ready and time_to_impact > 0.1  # Need 100ms

    def estimate_time_to_impact(self, state):
        """Estimate time until fall impact"""
        # Simplified estimation based on current CoM state
        com_vel = np.linalg.norm(state['com_velocity'])
        com_acc = np.linalg.norm(state['com_acceleration'])

        # Estimate time using kinematic equations
        # This is a simplified approximation
        if com_acc > 0:
            time_estimate = com_vel / com_acc
        else:
            time_estimate = 1.0  # Default estimate

        return min(time_estimate, 1.0)  # Cap at 1 second
```

## Hands-On Exercise: Implementing Balance Control

### Exercise Objectives
- Implement a basic ZMP-based balance controller
- Create a simple walking pattern generator
- Test balance control with simulated perturbations
- Analyze stability margins and recovery capabilities

### Step-by-Step Instructions

1. **Set up simulation environment** with humanoid robot model
2. **Implement ZMP calculation** and support polygon detection
3. **Create balance controller** using PID approach
4. **Develop simple walking pattern** generation
5. **Test with perturbations** (pushes, uneven terrain)
6. **Analyze stability metrics** and adjust parameters
7. **Implement recovery strategies** for near-fall conditions

### Expected Outcomes
- Working balance control system
- Stable walking gait
- Understanding of ZMP concepts
- Experience with humanoid control challenges

## Knowledge Check

1. What is the Zero Moment Point (ZMP) and why is it important for bipedal balance?
2. Explain the difference between ankle and hip balance strategies.
3. How does the Capture Point concept help with balance recovery?
4. What are the key phases of the human gait cycle?

## Summary

This chapter covered the complex topic of bipedal locomotion and balance control, which is fundamental to humanoid robotics. We explored the biomechanics of human walking, ZMP theory, balance control mechanisms, and practical implementation strategies. The integration of multiple control strategies—ankle, hip, and stepping—is essential for stable humanoid locomotion. Understanding these principles is crucial for developing robust, human-like walking capabilities in robotic systems.

## Next Steps

In Chapter 15, we'll explore manipulation and grasping techniques for humanoid robots, building upon the locomotion and balance foundations to enable dexterous interaction with the environment.

