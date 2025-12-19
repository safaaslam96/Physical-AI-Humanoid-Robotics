---
sidebar_position: 14
title: "Chapter 14: Bipedal Locomotion and Balance Control"
---

# Chapter 14: Bipedal Locomotion and Balance Control

## Learning Objectives
- Understand the principles of bipedal locomotion in humanoid robots
- Implement balance control mechanisms for stable walking
- Master stability control algorithms for humanoid robots
- Design walking patterns and gait generation systems

## Introduction to Bipedal Locomotion

Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control algorithms to maintain balance while walking. Unlike wheeled robots, bipedal robots must constantly manage their center of mass and adjust their posture to avoid falling.

### Fundamentals of Human Walking

Human walking involves a complex interplay of biomechanics, neural control, and environmental adaptation. For humanoid robots, we must replicate these principles through mathematical models and control systems.

### Key Concepts in Bipedal Locomotion

1. **Center of Mass (CoM)**: The point where the robot's mass is concentrated
2. **Zero Moment Point (ZMP)**: The point where the net moment of the ground reaction force is zero
3. **Support Polygon**: The area defined by the feet in contact with the ground
4. **Stance Phase**: When a foot is in contact with the ground
5. **Swing Phase**: When a foot is moving through the air
6. **Double Support Phase**: When both feet are in contact with the ground

## Walking Pattern Generation

### ZMP-Based Walking Pattern Generation

```python
# zmp_walking.py
import numpy as np
import matplotlib.pyplot as plt
from math import sinh, cosh, sqrt

class ZMPWalkingPatternGenerator:
    def __init__(self, robot_height=0.8, step_length=0.3, step_width=0.2, step_height=0.05):
        self.robot_height = robot_height  # Height of CoM from ground
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.omega = sqrt(9.81 / robot_height)  # Natural frequency
        self.g = 9.81  # Gravity

    def generate_footsteps(self, num_steps, start_pos=(0, 0), start_yaw=0):
        """Generate footstep positions for walking"""
        footsteps = []

        current_x, current_y = start_pos
        current_yaw = start_yaw

        for i in range(num_steps):
            # Determine which foot to step with (alternating)
            foot = 'left' if i % 2 == 0 else 'right'

            # Calculate step position
            if foot == 'left':
                step_x = current_x + self.step_length * cos(current_yaw)
                step_y = current_y + self.step_width * sin(current_yaw)
                step_yaw = current_yaw
            else:
                step_x = current_x + self.step_length * cos(current_yaw)
                step_y = current_y - self.step_width * sin(current_yaw)
                step_yaw = current_yaw

            footsteps.append({
                'step': i + 1,
                'foot': foot,
                'position': (step_x, step_y),
                'yaw': step_yaw
            })

            # Update current position for next step
            current_x = step_x
            current_y = step_y

        return footsteps

    def calculate_zmp_trajectory(self, footsteps, dt=0.01, double_support_ratio=0.1):
        """Calculate ZMP trajectory for the footsteps"""
        # Calculate total walking time
        total_time = len(footsteps) * 1.0  # 1 second per step

        # Time vector
        t = np.arange(0, total_time, dt)
        zmp_x = np.zeros_like(t)
        zmp_y = np.zeros_like(t)

        # Generate ZMP trajectory following the footsteps
        for i, step in enumerate(footsteps):
            step_start = i
            step_end = i + 1

            # Support phase: ZMP follows the supporting foot
            support_start_idx = int(step_start / dt)
            support_end_idx = int((step_start + 1 - double_support_ratio) / dt)

            if support_end_idx >= len(t):
                support_end_idx = len(t) - 1

            # Set ZMP to foot position during support
            foot_pos = step['position']
            zmp_x[support_start_idx:support_end_idx] = foot_pos[0]
            zmp_y[support_start_idx:support_end_idx] = foot_pos[1]

            # Double support phase transition
            if i < len(footsteps) - 1:  # Not the last step
                next_foot_pos = footsteps[i + 1]['position']
                transition_start_idx = support_end_idx
                transition_end_idx = int((step_start + 1) / dt)

                if transition_end_idx >= len(t):
                    transition_end_idx = len(t) - 1

                # Linear interpolation between feet
                transition_duration = transition_end_idx - transition_start_idx
                if transition_duration > 0:
                    for j in range(transition_duration):
                        alpha = j / transition_duration
                        zmp_x[transition_start_idx + j] = (1 - alpha) * foot_pos[0] + alpha * next_foot_pos[0]
                        zmp_y[transition_start_idx + j] = (1 - alpha) * foot_pos[1] + alpha * next_foot_pos[1]

        return t, zmp_x, zmp_y

    def generate_com_trajectory(self, zmp_trajectory, dt=0.01):
        """Generate CoM trajectory from ZMP using inverted pendulum model"""
        t, zmp_x, zmp_y = zmp_trajectory

        # Initialize CoM trajectory
        com_x = np.zeros_like(zmp_x)
        com_y = np.zeros_like(zmp_y)

        # Initial conditions (start at CoM position above first foot)
        if len(zmp_x) > 0:
            com_x[0] = zmp_x[0]  # Start at ZMP
            com_y[0] = zmp_y[0]

        # Velocity and acceleration
        com_dx = np.zeros_like(zmp_x)
        com_dy = np.zeros_like(zmp_y)

        # Integrate inverted pendulum equation: (CoM - ZMP) = (h/g) * CoM_ddot
        # Rearranged: CoM_ddot = (g/h) * (CoM - ZMP)

        for i in range(1, len(t)):
            # Calculate acceleration based on ZMP-COM relationship
            com_ddot_x = (self.g / self.robot_height) * (com_x[i-1] - zmp_x[i-1])
            com_ddot_y = (self.g / self.robot_height) * (com_y[i-1] - zmp_y[i-1])

            # Integrate to get velocity
            com_dx[i] = com_dx[i-1] + com_ddot_x * dt
            com_dy[i] = com_dy[i-1] + com_ddot_y * dt

            # Integrate to get position
            com_x[i] = com_x[i-1] + com_dx[i-1] * dt
            com_y[i] = com_y[i-1] + com_dy[i-1] * dt

        return t, com_x, com_y, com_dx, com_dy

    def generate_foot_trajectory(self, footsteps, dt=0.01):
        """Generate smooth foot trajectories for walking"""
        # For each footstep, create a trajectory from previous position to new position
        # Include lift and swing phases

        total_time = len(footsteps) * 1.0  # 1 second per step
        t = np.arange(0, total_time, dt)

        left_foot_x = np.zeros_like(t)
        left_foot_y = np.zeros_like(t)
        left_foot_z = np.zeros_like(t)
        right_foot_x = np.zeros_like(t)
        right_foot_y = np.zeros_like(t)
        right_foot_z = np.zeros_like(t)

        # Initialize feet positions
        left_support = True  # Start with left foot as support
        left_pos = [0, self.step_width/2, 0]
        right_pos = [0, -self.step_width/2, 0]

        for i, step in enumerate(footsteps):
            step_start_idx = int(i / dt)
            step_duration_idx = int(1.0 / dt)

            if step_start_idx >= len(t):
                break

            end_idx = min(step_start_idx + step_duration_idx, len(t))

            # Determine which foot is moving
            moving_foot = step['foot']
            target_pos = step['position']

            # Generate trajectory for moving foot
            for j in range(step_start_idx, end_idx):
                time_in_step = (j - step_start_idx) * dt

                # Swing phase: parabolic trajectory for smooth motion
                if moving_foot == 'left':
                    # Calculate swing trajectory
                    swing_progress = min(time_in_step * 2, 1.0)  # 0 to 1 during swing
                    if swing_progress < 1.0:
                        # Lift phase (first half of swing)
                        lift_progress = min(swing_progress * 2, 1.0)
                        left_foot_z[j] = self.step_height * 4 * lift_progress * (1 - lift_progress)  # Parabolic lift
                        left_foot_x[j] = left_pos[0] + (target_pos[0] - left_pos[0]) * swing_progress
                        left_foot_y[j] = left_pos[1] + (target_pos[1] - left_pos[1]) * swing_progress
                    else:
                        left_foot_x[j] = target_pos[0]
                        left_foot_y[j] = target_pos[1]
                        left_foot_z[j] = 0  # On ground
                else:  # right foot
                    swing_progress = min(time_in_step * 2, 1.0)
                    if swing_progress < 1.0:
                        lift_progress = min(swing_progress * 2, 1.0)
                        right_foot_z[j] = self.step_height * 4 * lift_progress * (1 - lift_progress)
                        right_foot_x[j] = right_pos[0] + (target_pos[0] - right_pos[0]) * swing_progress
                        right_foot_y[j] = right_pos[1] + (target_pos[1] - right_pos[1]) * swing_progress
                    else:
                        right_foot_x[j] = target_pos[0]
                        right_foot_y[j] = target_pos[1]
                        right_foot_z[j] = 0  # On ground

            # Update the position of the moved foot
            if moving_foot == 'left':
                left_pos = [target_pos[0], target_pos[1], 0]
            else:
                right_pos = [target_pos[0], target_pos[1], 0]

        return t, (left_foot_x, left_foot_y, left_foot_z), (right_foot_x, right_foot_y, right_foot_z)

# Example usage
def example_zmp_walking():
    generator = ZMPWalkingPatternGenerator()

    # Generate footsteps
    footsteps = generator.generate_footsteps(5)
    print("Generated footsteps:")
    for step in footsteps:
        print(f"Step {step['step']}: {step['foot']} foot at {step['position']}")

    # Generate ZMP trajectory
    zmp_trajectory = generator.calculate_zmp_trajectory(footsteps)

    # Generate CoM trajectory
    com_trajectory = generator.generate_com_trajectory(zmp_trajectory)

    # Generate foot trajectories
    foot_trajectories = generator.generate_foot_trajectory(footsteps)

    # Plot results
    t, com_x, com_y, _, _ = com_trajectory
    t_zmp, zmp_x, zmp_y = zmp_trajectory

    plt.figure(figsize=(12, 8))

    # Plot CoM and ZMP trajectories
    plt.subplot(2, 2, 1)
    plt.plot(com_x, com_y, 'b-', label='CoM Trajectory', linewidth=2)
    plt.plot(zmp_x, zmp_y, 'r--', label='ZMP Trajectory', linewidth=2)
    plt.scatter([step['position'][0] for step in footsteps],
                [step['position'][1] for step in footsteps],
                c='g', s=100, label='Footsteps', zorder=5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM and ZMP Trajectories')
    plt.legend()
    plt.grid(True)

    # Plot X trajectories over time
    plt.subplot(2, 2, 2)
    plt.plot(t, com_x, label='CoM X', linewidth=2)
    plt.plot(t_zmp, zmp_x, label='ZMP X', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('X Trajectory Over Time')
    plt.legend()
    plt.grid(True)

    # Plot Y trajectories over time
    plt.subplot(2, 2, 3)
    plt.plot(t, com_y, label='CoM Y', linewidth=2)
    plt.plot(t_zmp, zmp_y, label='ZMP Y', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Y Trajectory Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example_zmp_walking()
```

### Footstep Planning and Path Following

```python
# footstep_planning.py
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, max_turn=0.3):
        self.step_length = step_length
        self.step_width = step_width
        self.max_turn = max_turn  # Maximum turning angle per step

    def plan_path_footsteps(self, path, start_pos, start_yaw):
        """Plan footsteps along a given path"""
        footsteps = []
        current_pos = np.array(start_pos)
        current_yaw = start_yaw

        # Start with current foot positions
        left_foot = current_pos + np.array([0, self.step_width/2])
        right_foot = current_pos + np.array([0, -self.step_width/2])

        # Determine which foot to step with first
        use_left = True  # Start with left foot

        path_idx = 0
        while path_idx < len(path) - 1:
            # Calculate direction to next path point
            target = np.array(path[path_idx])
            direction = target - current_pos
            distance = np.linalg.norm(direction)

            if distance < 0.1:  # Close enough to path point
                path_idx += 1
                continue

            # Normalize direction
            direction = direction / distance

            # Calculate desired step position
            step_x = current_pos[0] + self.step_length * direction[0]
            step_y = current_pos[1] + self.step_length * direction[1]

            # Add some turning capability
            target_yaw = np.arctan2(direction[1], direction[0])
            yaw_diff = target_yaw - current_yaw

            # Limit turning per step
            if abs(yaw_diff) > self.max_turn:
                yaw_diff = np.sign(yaw_diff) * self.max_turn

            current_yaw += yaw_diff

            # Calculate foot position with turning
            if use_left:
                foot_pos = np.array([step_x, step_y + self.step_width/2])
                foot = 'left'
            else:
                foot_pos = np.array([step_x, step_y - self.step_width/2])
                foot = 'right'

            # Check if step is feasible (not too far from supporting foot)
            support_foot = right_foot if use_left else left_foot
            step_distance_from_support = np.linalg.norm(foot_pos - support_foot)

            if step_distance_from_support <= 0.5:  # Max step distance
                footsteps.append({
                    'step': len(footsteps) + 1,
                    'foot': foot,
                    'position': foot_pos.copy(),
                    'yaw': current_yaw
                })

                # Update supporting foot position
                if use_left:
                    left_foot = foot_pos.copy()
                else:
                    right_foot = foot_pos.copy()

                # Move robot position forward
                current_pos = foot_pos.copy()

                # Switch feet
                use_left = not use_left

            else:
                # If step is too large, try a smaller step
                adjusted_pos = support_foot + 0.4 * direction  # Reduce step size
                footsteps.append({
                    'step': len(footsteps) + 1,
                    'foot': 'left' if use_left else 'right',
                    'position': adjusted_pos,
                    'yaw': current_yaw
                })

                if use_left:
                    left_foot = adjusted_pos.copy()
                else:
                    right_foot = adjusted_pos.copy()

                current_pos = adjusted_pos.copy()
                use_left = not use_left

            path_idx += 1

        return footsteps

    def optimize_footsteps(self, footsteps):
        """Optimize footsteps to improve stability and efficiency"""
        if len(footsteps) < 2:
            return footsteps

        optimized = [footsteps[0]]  # Keep first step

        for i in range(1, len(footsteps)):
            current = footsteps[i]
            previous = optimized[-1]

            # Calculate ideal position based on gait pattern
            dx = current['position'][0] - previous['position'][0]
            dy = current['position'][1] - previous['position'][1]

            # Ensure alternating feet
            if current['foot'] == previous['foot']:
                # This shouldn't happen in proper planning, but fix if needed
                current['foot'] = 'right' if current['foot'] == 'left' else 'left'

            # Apply stability constraints
            distance_from_prev = np.sqrt(dx**2 + dy**2)
            if distance_from_prev > 0.4:  # Too far, adjust
                direction = np.array([dx, dy]) / distance_from_prev
                adjusted_pos = np.array(previous['position']) + 0.3 * direction
                current['position'] = adjusted_pos

            optimized.append(current)

        return optimized

    def check_stability(self, footsteps):
        """Check if the footstep sequence is stable"""
        if len(footsteps) < 2:
            return True

        stable = True
        reasons = []

        for i in range(1, len(footsteps)):
            current = footsteps[i]
            previous = footsteps[i-1]

            # Check step size
            step_distance = np.linalg.norm(
                np.array(current['position']) - np.array(previous['position'])
            )

            if step_distance > 0.4:  # Too large a step
                stable = False
                reasons.append(f"Step {i}: Distance {step_distance:.2f}m too large")

            # Check foot placement relative to body
            # For now, just check that feet alternate appropriately
            if current['foot'] == previous['foot']:
                stable = False
                reasons.append(f"Step {i}: Same foot as previous step")

        return stable, reasons

# Example usage
def example_footstep_planning():
    planner = FootstepPlanner()

    # Define a simple path (x, y coordinates)
    path = [
        [0, 0],
        [1, 0],
        [2, 0.5],
        [3, 1],
        [4, 1.5],
        [5, 1.5]
    ]

    # Plan footsteps
    footsteps = planner.plan_path_footsteps(path, start_pos=[0, 0], start_yaw=0)

    # Optimize footsteps
    optimized_footsteps = planner.optimize_footsteps(footsteps)

    # Check stability
    stable, reasons = planner.check_stability(optimized_footsteps)

    print("Footstep planning results:")
    print(f"Stable: {stable}")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")

    print(f"\nPlanned footsteps ({len(optimized_footsteps)} total):")
    for step in optimized_footsteps:
        print(f"  Step {step['step']}: {step['foot']} foot at ({step['position'][0]:.2f}, {step['position'][1]:.2f})")

if __name__ == "__main__":
    example_footstep_planning()
```

## Balance Control Mechanisms

### Center of Mass Control

```python
# balance_control.py
import numpy as np
from math import sin, cos, atan2

class BalanceController:
    def __init__(self, robot_height=0.8, control_frequency=100):
        self.robot_height = robot_height
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # PID gains for balance control
        self.com_x_pid = self.PIDController(kp=800, ki=10, kd=200)
        self.com_y_pid = self.PIDController(kp=800, ki=10, kd=200)
        self.com_z_pid = self.PIDController(kp=500, ki=5, kd=100)  # For height control

        # State variables
        self.current_com = np.array([0.0, 0.0, robot_height])
        self.desired_com = np.array([0.0, 0.0, robot_height])
        self.com_velocity = np.array([0.0, 0.0, 0.0])

        # ZMP-based balance control
        self.zmp_error_integrator = np.array([0.0, 0.0])
        self.zmp_error_derivative = np.array([0.0, 0.0])
        self.previous_zmp_error = np.array([0.0, 0.0])

    class PIDController:
        def __init__(self, kp, ki, kd):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.integral = 0
            self.previous_error = 0

        def compute(self, error, dt):
            self.integral += error * dt
            derivative = (error - self.previous_error) / dt if dt > 0 else 0

            output = self.kp * error + self.ki * self.integral + self.kd * derivative

            self.previous_error = error
            return output

    def update_balance(self, current_com, current_zmp, desired_zmp, dt=None):
        """Update balance control based on CoM and ZMP errors"""
        if dt is None:
            dt = self.dt

        # Calculate errors
        com_error = self.desired_com - current_com
        zmp_error = desired_zmp - current_zmp

        # Update ZMP error integrator and derivative
        self.zmp_error_integrator += zmp_error * dt
        if dt > 0:
            self.zmp_error_derivative = (zmp_error - self.previous_zmp_error) / dt
        self.previous_zmp_error = zmp_error.copy()

        # Compute balance corrections using PID control
        balance_correction = np.zeros(3)

        # X-axis balance (using ZMP error to correct CoM)
        balance_correction[0] = self.com_x_pid.compute(zmp_error[0], dt)

        # Y-axis balance (using ZMP error to correct CoM)
        balance_correction[1] = self.com_y_pid.compute(zmp_error[1], dt)

        # Z-axis (height) control
        height_error = self.desired_com[2] - current_com[2]
        balance_correction[2] = self.com_z_pid.compute(height_error, dt)

        # Apply limits to prevent excessive corrections
        max_correction = 0.05  # 5cm max correction per step
        balance_correction = np.clip(balance_correction, -max_correction, max_correction)

        # Update desired CoM position based on corrections
        self.desired_com += balance_correction * dt * 10  # Scale appropriately

        return balance_correction

    def compute_ankle_torque_balance(self, com_error, zmp_error):
        """Compute ankle torques for balance based on CoM and ZMP errors"""
        # Map CoM and ZMP errors to ankle torques
        # This is a simplified model - in practice, this would involve more complex kinematics

        ankle_torques = {}

        # Left ankle corrections
        ankle_torques['left_ankle_pitch'] = -100 * com_error[1] - 50 * zmp_error[1]  # Y-direction balance
        ankle_torques['left_ankle_roll'] = -100 * com_error[0] - 50 * zmp_error[0]   # X-direction balance

        # Right ankle corrections
        ankle_torques['right_ankle_pitch'] = -100 * com_error[1] - 50 * zmp_error[1]  # Y-direction balance
        ankle_torques['right_ankle_roll'] = -100 * com_error[0] - 50 * zmp_error[0]   # X-direction balance

        return ankle_torques

    def compute_hip_balance(self, com_error, zmp_error):
        """Compute hip adjustments for balance"""
        hip_adjustments = {}

        # Hip pitch for forward/backward balance
        hip_adjustments['left_hip_pitch'] = -50 * com_error[0] - 25 * zmp_error[0]
        hip_adjustments['right_hip_pitch'] = -50 * com_error[0] - 25 * zmp_error[0]

        # Hip roll for lateral balance
        hip_adjustments['left_hip_roll'] = -40 * com_error[1] - 20 * zmp_error[1]
        hip_adjustments['right_hip_roll'] = 40 * com_error[1] + 20 * zmp_error[1]  # Opposite for right hip

        return hip_adjustments

    def compute_balance_strategy(self, com_pos, zmp_pos, support_polygon):
        """Determine appropriate balance strategy based on current state"""
        # Calculate distance from CoM to support polygon
        com_xy = com_pos[:2]

        # Simple check if CoM is within support polygon (simplified for rectangular support)
        if len(support_polygon) >= 2:
            min_x = min(p[0] for p in support_polygon)
            max_x = max(p[0] for p in support_polygon)
            min_y = min(p[1] for p in support_polygon)
            max_y = max(p[1] for p in support_polygon)

            com_in_support = (min_x <= com_xy[0] <= max_x) and (min_y <= com_xy[1] <= max_y)
        else:
            com_in_support = False

        # Determine balance strategy
        if not com_in_support:
            # Emergency balance strategy - large corrections needed
            strategy = 'emergency_balance'
            margin = min(
                abs(com_xy[0] - min_x), abs(max_x - com_xy[0]),
                abs(com_xy[1] - min_y), abs(max_y - com_xy[1])
            )
        else:
            # Normal balance - small adjustments
            strategy = 'normal_balance'
            margin = 0.1  # Default safety margin

        return {
            'strategy': strategy,
            'margin': margin,
            'com_in_support': com_in_support
        }

    def adjust_step_timing(self, balance_state):
        """Adjust step timing based on balance state"""
        if balance_state['strategy'] == 'emergency_balance':
            # Speed up step timing to recover balance faster
            return 0.8  # 80% of normal step time
        else:
            # Normal step timing
            return 1.0

class InvertedPendulumBalancer:
    """Advanced balance controller using inverted pendulum model"""
    def __init__(self, com_height=0.8, control_frequency=200):
        self.com_height = com_height
        self.omega = np.sqrt(9.81 / com_height)  # Natural frequency
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # State: [x, x_dot, y, y_dot] where x,y are CoM positions
        self.state = np.zeros(4)

        # Desired state
        self.desired_state = np.zeros(4)

        # Control gains (LQR-based, simplified)
        self.K = np.array([1.0, 0.1, 1.0, 0.1])  # Simplified gains

    def update(self, measured_com_pos, measured_com_vel, desired_com_pos, dt=None):
        """Update inverted pendulum balance controller"""
        if dt is None:
            dt = self.dt

        # Update state vector [x, x_dot, y, y_dot]
        self.state[0] = measured_com_pos[0]  # x position
        self.state[1] = measured_com_vel[0]  # x velocity
        self.state[2] = measured_com_pos[1]  # y position
        self.state[3] = measured_com_vel[1]  # y velocity

        # Update desired state
        self.desired_state[0] = desired_com_pos[0]
        self.desired_state[2] = desired_com_pos[1]

        # Calculate error
        error = self.desired_state - self.state

        # Simple control law (in practice, this would be more sophisticated)
        zmp_correction = self.K * error

        # Convert to ZMP command
        zmp_cmd = np.array([
            self.desired_state[0] - zmp_correction[0] / (self.omega**2),
            self.desired_state[2] - zmp_correction[2] / (self.omega**2)
        ])

        return zmp_cmd

# Example usage
def example_balance_control():
    balance_controller = BalanceController()
    pendulum_balancer = InvertedPendulumBalancer()

    # Simulate some CoM and ZMP measurements
    current_com = np.array([0.02, -0.01, 0.79])  # Slightly off balance
    current_zmp = np.array([0.01, -0.005])  # ZMP slightly off
    desired_zmp = np.array([0.0, 0.0])  # Want ZMP at origin

    # Update balance control
    balance_correction = balance_controller.update_balance(
        current_com, current_zmp, desired_zmp
    )

    print(f"Balance correction: {balance_correction}")

    # Compute ankle torques
    com_error = np.array([0.02, -0.01, 0])  # X, Y, Z errors
    ankle_torques = balance_controller.compute_ankle_torque_balance(com_error[:2],
                                                                 desired_zmp - current_zmp)

    print(f"Ankle torques: {ankle_torques}")

    # Test inverted pendulum balancer
    desired_com = np.array([0.0, 0.0, 0.8])
    com_vel = np.array([0.01, -0.005, 0.0])  # Small velocities

    zmp_cmd = pendulum_balancer.update(current_com, com_vel, desired_com)
    print(f"ZMP command from inverted pendulum: {zmp_cmd}")

if __name__ == "__main__":
    example_balance_control()
```

## Stability Control Algorithms

### Capture Point and Divergent Component of Motion

```python
# stability_algorithms.py
import numpy as np
from math import sqrt, exp

class CapturePointController:
    """Controller based on Capture Point theory for humanoid balance"""
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)

    def compute_capture_point(self, com_pos, com_vel):
        """Compute capture point from current CoM state"""
        # Capture point = CoM position + CoM velocity / omega
        capture_point = com_pos + com_vel / self.omega
        return capture_point

    def is_stable(self, com_pos, com_vel, support_polygon):
        """Check if current state is stable based on capture point"""
        capture_point = self.compute_capture_point(com_pos[:2], com_vel[:2])

        # Check if capture point is within support polygon
        # Simplified for rectangular support
        if len(support_polygon) >= 2:
            min_x = min(p[0] for p in support_polygon)
            max_x = max(p[0] for p in support_polygon)
            min_y = min(p[1] for p in support_polygon)
            max_y = max(p[1] for p in support_polygon)

            return (min_x <= capture_point[0] <= max_x) and (min_y <= capture_point[1] <= max_y)

        return False

    def compute_stabilizing_step(self, com_pos, com_vel, support_polygon):
        """Compute where to step to achieve stability"""
        capture_point = self.compute_capture_point(com_pos[:2], com_vel[:2])

        # For stability, step should be placed at or beyond the capture point
        # Add a safety margin
        safety_margin = 0.05  # 5cm safety margin

        # Calculate step location
        step_location = capture_point + safety_margin * (capture_point - com_pos[:2]) / \
                       np.linalg.norm(capture_point - com_pos[:2] + 1e-6)

        # Ensure step is within reasonable bounds
        step_location[0] = np.clip(step_location[0], -0.3, 0.3)  # Limit step size
        step_location[1] = np.clip(step_location[1], -0.2, 0.2)

        return step_location

class DCMController:
    """Controller based on Divergent Component of Motion (DCM)"""
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)

    def compute_dcm(self, com_pos, com_vel):
        """Compute Divergent Component of Motion"""
        # DCM = CoM position + CoM velocity / omega
        dcm = com_pos + com_vel / self.omega
        return dcm

    def compute_desired_dcm(self, target_pos, time_to_switch):
        """Compute desired DCM to reach target in given time"""
        # For exponential convergence: DCM_desired = target + (current_DCM - target) * exp(-omega * t)
        current_dcm = self.compute_dcm(self.current_com_pos, self.current_com_vel)
        desired_dcm = target_pos + (current_dcm - target_pos) * exp(-self.omega * time_to_switch)
        return desired_dcm

    def compute_balance_feedback(self, current_dcm, desired_dcm, dt):
        """Compute feedback control based on DCM error"""
        dcm_error = desired_dcm - current_dcm

        # Simple proportional control
        feedback_gain = 10.0  # Adjust based on robot properties
        com_adjustment = feedback_gain * dcm_error * dt

        return com_adjustment

class WalkingStabilizer:
    """Comprehensive walking stabilizer using multiple techniques"""
    def __init__(self, com_height=0.8):
        self.com_height = com_height
        self.capture_point_controller = CapturePointController(com_height)
        self.dcm_controller = DCMController(com_height)

        # Walking state
        self.current_support_foot = 'left'
        self.swing_foot = 'right'
        self.step_time = 1.0  # Time per step
        self.time_in_step = 0.0

        # Balance margins
        self.stability_margin = 0.05  # 5cm margin
        self.max_com_velocity = 0.5   # 0.5 m/s max CoM velocity

    def update_stabilization(self, com_pos, com_vel, support_polygon, dt):
        """Update stabilization strategy"""
        self.time_in_step += dt

        # Check stability using multiple methods
        cp_stable = self.capture_point_controller.is_stable(com_pos, com_vel, support_polygon)
        dcm = self.dcm_controller.compute_dcm(com_pos[:2], com_vel[:2])

        # Determine if we need to take a step to maintain balance
        need_step = not cp_stable

        # Calculate step location if needed
        step_location = None
        if need_step:
            step_location = self.capture_point_controller.compute_stabilizing_step(
                com_pos, com_vel, support_polygon
            )

        # Calculate CoM adjustments
        com_adjustment = self.calculate_com_adjustment(com_pos, com_vel, dcm, support_polygon)

        # Calculate timing adjustments
        timing_factor = self.calculate_timing_adjustment(com_pos, com_vel, cp_stable)

        return {
            'need_step': need_step,
            'step_location': step_location,
            'com_adjustment': com_adjustment,
            'timing_factor': timing_factor,
            'capture_point': self.capture_point_controller.compute_capture_point(com_pos[:2], com_vel[:2]),
            'dcm': dcm,
            'stable': cp_stable
        }

    def calculate_com_adjustment(self, com_pos, com_vel, dcm, support_polygon):
        """Calculate CoM adjustments for stability"""
        adjustments = np.zeros(3)

        # DCM-based adjustment
        # Want to move DCM toward center of support polygon
        if len(support_polygon) >= 2:
            support_center = np.mean(support_polygon, axis=0)
            dcm_error = support_center - dcm

            # Apply feedback to move CoM appropriately
            adjustments[0] = 0.5 * dcm_error[0]  # X adjustment
            adjustments[1] = 0.5 * dcm_error[1]  # Y adjustment

        # Velocity damping
        max_velocity = self.max_com_velocity
        if abs(com_vel[0]) > max_velocity:
            adjustments[0] -= 0.1 * (abs(com_vel[0]) - max_velocity) * np.sign(com_vel[0])
        if abs(com_vel[1]) > max_velocity:
            adjustments[1] -= 0.1 * (abs(com_vel[1]) - max_velocity) * np.sign(com_vel[1])

        # Limit adjustments
        adjustments = np.clip(adjustments, -0.02, 0.02)  # ±2cm adjustments

        return adjustments

    def calculate_timing_adjustment(self, com_pos, com_vel, stable):
        """Calculate step timing adjustments based on stability"""
        if stable:
            # If stable, maintain normal timing
            return 1.0
        else:
            # If unstable, speed up to recover balance faster
            com_speed = np.linalg.norm(com_vel[:2])
            if com_speed > 0.3:  # If moving fast
                return 0.8  # 80% of normal time
            else:
                return 0.9  # 90% of normal time

# Example usage
def example_stability_algorithms():
    stabilizer = WalkingStabilizer()

    # Simulate robot state
    com_pos = np.array([0.03, -0.02, 0.79])  # CoM slightly off
    com_vel = np.array([0.05, -0.03, 0.0])   # CoM moving
    support_polygon = np.array([[-0.05, 0.1], [-0.05, -0.1]])  # Simple rectangular support

    # Update stabilization
    dt = 0.01  # 100Hz control
    stability_info = stabilizer.update_stabilization(com_pos, com_vel, support_polygon, dt)

    print("Stability Analysis:")
    print(f"  Stable: {stability_info['stable']}")
    print(f"  Need step: {stability_info['need_step']}")
    print(f"  Capture Point: {stability_info['capture_point']}")
    print(f"  DCM: {stability_info['dcm']}")
    print(f"  CoM adjustment: {stability_info['com_adjustment']}")
    print(f"  Timing factor: {stability_info['timing_factor']}")

    if stability_info['step_location'] is not None:
        print(f"  Recommended step location: {stability_info['step_location']}")

if __name__ == "__main__":
    example_stability_algorithms()
```

## Walking Pattern Implementation

### Pattern Generators and Interpolation

```python
# walking_patterns.py
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class WalkingPatternGenerator:
    """Generate complete walking patterns with smooth interpolation"""
    def __init__(self, com_height=0.8, step_time=1.0, dsp_ratio=0.1):
        self.com_height = com_height
        self.step_time = step_time
        self.dsp_ratio = dsp_ratio  # Double support phase ratio
        self.ssp_ratio = 1.0 - dsp_ratio  # Single support phase ratio

        # Walking parameters
        self.step_length = 0.3  # m
        self.step_width = 0.2   # m
        self.step_height = 0.05 # m
        self.com_height_variation = 0.02  # m (up and down motion)

    def generate_com_trajectory(self, steps, start_pos=(0, 0, 0.8)):
        """Generate CoM trajectory for multiple steps"""
        # Total time
        total_time = len(steps) * self.step_time

        # Time vector
        t = np.linspace(0, total_time, int(total_time * 200))  # 200Hz sampling

        # Initialize trajectory arrays
        com_x = np.zeros_like(t)
        com_y = np.zeros_like(t)
        com_z = np.zeros_like(t)

        # Start position
        current_x, current_y, current_z = start_pos

        # Generate trajectory for each step
        for i, step in enumerate(steps):
            step_start_time = i * self.step_time
            step_end_time = (i + 1) * self.step_time

            # Find time indices for this step
            step_indices = np.where((t >= step_start_time) & (t < step_end_time))[0]

            if len(step_indices) == 0:
                continue

            step_t = t[step_indices]
            step_duration = step_end_time - step_start_time

            # X trajectory - follows walking direction
            if i == 0:
                # Start at initial position
                com_x[step_indices] = current_x + (step['position'][0] - current_x) * \
                    (step_t - step_start_time) / step_duration
            else:
                # Smooth transition between steps
                start_x = com_x[step_indices[0] - 1] if step_indices[0] > 0 else current_x
                com_x[step_indices] = start_x + (step['position'][0] - start_x) * \
                    (step_t - step_start_time) / step_duration

            # Y trajectory - maintains balance between feet
            support_foot_y = step['position'][1]
            if step['foot'] == 'left':
                # Left foot support, CoM shifts slightly right
                target_y = support_foot_y - 0.02
            else:
                # Right foot support, CoM shifts slightly left
                target_y = support_foot_y + 0.02

            # Smooth transition to target Y
            if step_indices[0] > 0:
                start_y = com_y[step_indices[0] - 1]
            else:
                start_y = start_pos[1]

            com_y[step_indices] = start_y + (target_y - start_y) * \
                (step_t - step_start_time) / step_duration

            # Z trajectory - slight up/down motion for natural walking
            # Oscillate at walking frequency
            omega = 2 * np.pi / self.step_time
            z_variation = self.com_height_variation * np.sin(omega * (step_t - step_start_time))
            com_z[step_indices] = self.com_height + z_variation

        return t, com_x, com_y, com_z

    def generate_foot_trajectory(self, steps, foot_type='left'):
        """Generate smooth foot trajectory for a specific foot"""
        total_time = len(steps) * self.step_time
        t = np.linspace(0, total_time, int(total_time * 200))

        foot_x = np.zeros_like(t)
        foot_y = np.zeros_like(t)
        foot_z = np.zeros_like(t)

        # Find steps where this foot is moving (not supporting)
        moving_steps = []
        for i, step in enumerate(steps):
            if step['foot'] == foot_type:
                moving_steps.append((i, step))

        # Set initial position to first supporting position
        for i, step in enumerate(steps):
            if step['foot'] != foot_type:  # Supporting foot
                initial_pos = step['position']
                break
        else:
            initial_pos = (0, 0)  # Default if no supporting step found

        # Initialize with supporting position
        foot_x.fill(initial_pos[0])
        foot_y.fill(initial_pos[1])
        foot_z.fill(0)  # On ground when supporting

        # Generate trajectories for each time the foot moves
        for step_idx, (move_step_idx, move_step) in enumerate(moving_steps):
            # When this foot moves, it goes from previous support position to new position
            start_time = move_step_idx * self.step_time
            end_time = (move_step_idx + 1) * self.step_time

            # Find indices for this step
            step_indices = np.where((t >= start_time) & (t < end_time))[0]

            if len(step_indices) == 0:
                continue

            step_t = t[step_indices]

            # Find the previous support position (position of other foot)
            if move_step_idx > 0:
                prev_support = steps[move_step_idx - 1]
                if prev_support['foot'] != foot_type:
                    prev_pos = prev_support['position']
                else:
                    # This shouldn't happen in proper alternating gait
                    prev_pos = initial_pos
            else:
                prev_pos = initial_pos

            # Calculate swing trajectory
            # Use parabolic trajectory for foot lift
            for j, time_idx in enumerate(step_indices):
                local_t = (t[time_idx] - start_time) / self.step_time  # 0 to 1

                if local_t < self.dsp_ratio:
                    # Double support phase - foot still on ground
                    foot_x[time_idx] = prev_pos[0]
                    foot_y[time_idx] = prev_pos[1]
                    foot_z[time_idx] = 0
                elif local_t < self.dsp_ratio + self.ssp_ratio:
                    # Single support phase - foot swings
                    swing_t = (local_t - self.dsp_ratio) / self.ssp_ratio  # 0 to 1 in swing phase

                    # Parabolic trajectory in X and Y
                    foot_x[time_idx] = prev_pos[0] + (move_step['position'][0] - prev_pos[0]) * swing_t
                    foot_y[time_idx] = prev_pos[1] + (move_step['position'][1] - prev_pos[1]) * swing_t

                    # Parabolic lift in Z (swing phase)
                    lift_shape = 4 * swing_t * (1 - swing_t)  # Parabola from 0 to 1
                    foot_z[time_idx] = self.step_height * lift_shape
                else:
                    # Second double support phase - foot lands
                    foot_x[time_idx] = move_step['position'][0]
                    foot_y[time_idx] = move_step['position'][1]
                    foot_z[time_idx] = 0  # On ground

        return t, foot_x, foot_y, foot_z

    def generate_joints_from_cartesian(self, com_trajectory, left_foot_trajectory, right_foot_trajectory):
        """Generate joint angles from Cartesian trajectories using inverse kinematics"""
        # This would typically use inverse kinematics solvers
        # For simplicity, we'll return placeholder joint trajectories
        t, com_x, com_y, com_z = com_trajectory
        _, lf_x, lf_y, lf_z = left_foot_trajectory
        _, rf_x, rf_y, rf_z = right_foot_trajectory

        # Placeholder: return time vector and some joint data
        # In practice, this would solve inverse kinematics for each time step
        joint_trajectories = {
            'time': t,
            'left_hip_pitch': np.zeros_like(t),
            'left_knee': np.zeros_like(t),
            'left_ankle_pitch': np.zeros_like(t),
            'right_hip_pitch': np.zeros_like(t),
            'right_knee': np.zeros_like(t),
            'right_ankle_pitch': np.zeros_like(t),
        }

        # Generate simple joint patterns based on foot trajectories
        for i in range(len(t)):
            # Simple inverse kinematics approximation
            # Left leg
            leg_length = 0.75  # Approximate leg length
            left_foot_height = lf_z[i]
            if left_foot_height < 0.01:  # Foot on ground
                # Calculate approximate joint angles
                hip_pitch = np.arcsin(min(0.3, max(-0.3, (0.1 - com_z[i] + 0.1) / leg_length)))
                knee_angle = -2 * hip_pitch  # Simplified
                ankle_angle = -hip_pitch
            else:  # Foot in air
                hip_pitch = 0.1
                knee_angle = -0.2
                ankle_angle = 0.05

            joint_trajectories['left_hip_pitch'][i] = hip_pitch
            joint_trajectories['left_knee'][i] = knee_angle
            joint_trajectories['left_ankle_pitch'][i] = ankle_angle

            # Right leg (similar calculation)
            if rf_z[i] < 0.01:  # Foot on ground
                hip_pitch = np.arcsin(min(0.3, max(-0.3, (0.1 - com_z[i] + 0.1) / leg_length)))
                knee_angle = -2 * hip_pitch
                ankle_angle = -hip_pitch
            else:
                hip_pitch = 0.1
                knee_angle = -0.2
                ankle_angle = 0.05

            joint_trajectories['right_hip_pitch'][i] = hip_pitch
            joint_trajectories['right_knee'][i] = knee_angle
            joint_trajectories['right_ankle_pitch'][i] = ankle_angle

        return joint_trajectories

class SmoothTrajectoryGenerator:
    """Generate smooth trajectories using splines"""
    def __init__(self):
        pass

    def generate_spline_trajectory(self, waypoints, duration, frequency=200):
        """Generate smooth trajectory using cubic splines"""
        if len(waypoints) < 2:
            return np.array([0]), np.array([waypoints[0]]) if waypoints else np.array([0])

        # Create time vector
        t_total = duration
        t = np.linspace(0, t_total, int(t_total * frequency))

        # Separate waypoints by dimension
        if isinstance(waypoints[0], (list, tuple, np.ndarray)):
            n_dims = len(waypoints[0])
            waypoint_times = np.linspace(0, t_total, len(waypoints))

            trajectory = np.zeros((len(t), n_dims))

            for dim in range(n_dims):
                waypoint_values = [wp[dim] for wp in waypoints]
                spline = CubicSpline(waypoint_times, waypoint_values)
                trajectory[:, dim] = spline(t)
        else:
            # Single dimension
            waypoint_times = np.linspace(0, t_total, len(waypoints))
            spline = CubicSpline(waypoint_times, waypoints)
            trajectory = spline(t)

        return t, trajectory

# Example usage
def example_walking_patterns():
    # Create pattern generator
    pattern_gen = WalkingPatternGenerator()

    # Create some example footsteps
    footsteps = [
        {'step': 1, 'foot': 'left', 'position': (0.3, 0.1), 'yaw': 0},
        {'step': 2, 'foot': 'right', 'position': (0.6, -0.1), 'yaw': 0},
        {'step': 3, 'foot': 'left', 'position': (0.9, 0.1), 'yaw': 0},
        {'step': 4, 'foot': 'right', 'position': (1.2, -0.1), 'yaw': 0},
    ]

    # Generate CoM trajectory
    com_trajectory = pattern_gen.generate_com_trajectory(footsteps)

    # Generate foot trajectories
    left_foot_trajectory = pattern_gen.generate_foot_trajectory(footsteps, 'left')
    right_foot_trajectory = pattern_gen.generate_foot_trajectory(footsteps, 'right')

    # Generate joint trajectories
    joint_trajectories = pattern_gen.generate_joints_from_cartesian(
        com_trajectory, left_foot_trajectory, right_foot_trajectory
    )

    # Plot results
    t, com_x, com_y, com_z = com_trajectory

    plt.figure(figsize=(15, 10))

    # Plot CoM trajectory
    plt.subplot(2, 3, 1)
    plt.plot(com_x, com_y, 'b-', linewidth=2, label='CoM Path')
    for step in footsteps:
        color = 'red' if step['foot'] == 'left' else 'green'
        plt.scatter([step['position'][0]], [step['position'][1]],
                   c=color, s=100, label=f"{step['foot']} foot", zorder=5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM and Footstep Trajectory')
    plt.legend()
    plt.grid(True)

    # Plot CoM height over time
    plt.subplot(2, 3, 2)
    plt.plot(t, com_z, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('CoM Height Over Time')
    plt.grid(True)

    # Plot foot trajectories
    t_f, lf_x, lf_y, lf_z = left_foot_trajectory
    t_f, rf_x, rf_y, rf_z = right_foot_trajectory

    plt.subplot(2, 3, 3)
    plt.plot(t_f, lf_z, label='Left Foot Z', linewidth=2)
    plt.plot(t_f, rf_z, label='Right Foot Z', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Foot Z Trajectories')
    plt.legend()
    plt.grid(True)

    # Plot joint trajectories
    plt.subplot(2, 3, 4)
    plt.plot(joint_trajectories['time'], joint_trajectories['left_hip_pitch'],
             label='Left Hip Pitch', linewidth=2)
    plt.plot(joint_trajectories['time'], joint_trajectories['right_hip_pitch'],
             label='Right Hip Pitch', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Hip Pitch Trajectories')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(joint_trajectories['time'], joint_trajectories['left_knee'],
             label='Left Knee', linewidth=2)
    plt.plot(joint_trajectories['time'], joint_trajectories['right_knee'],
             label='Right Knee', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Knee Angle Trajectories')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Generated trajectories for {len(footsteps)} steps")
    print(f"Total simulation time: {len(joint_trajectories['time']) / 200:.1f} seconds")
    print(f"Trajectory points: {len(joint_trajectories['time'])}")

if __name__ == "__main__":
    example_walking_patterns()
```

## Knowledge Check

1. What is the Zero Moment Point (ZMP) and why is it crucial for bipedal walking?
2. How do Capture Point and Divergent Component of Motion (DCM) help in balance control?
3. What are the main phases of a walking cycle and how do they affect balance?
4. How does the inverted pendulum model apply to humanoid balance control?

## Summary

This chapter covered the fundamental concepts of bipedal locomotion and balance control for humanoid robots. We explored ZMP-based walking pattern generation, footstep planning, balance control mechanisms, and stability algorithms including Capture Point and DCM theory. The chapter provided practical implementations for generating stable walking patterns and maintaining balance during locomotion.

## Next Steps

In the next chapter, we'll explore manipulation and grasping techniques for humanoid robots, covering dexterous manipulation strategies, grasp planning, and human-robot interaction design.