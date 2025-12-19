---
sidebar_position: 13
title: "Chapter 13: Humanoid Robot Kinematics and Dynamics"
---

# Chapter 13: Humanoid Robot Kinematics and Dynamics

## Learning Objectives
- Understand forward and inverse kinematics for humanoid robots
- Master the mathematical foundations of humanoid dynamics
- Implement kinematic solvers for complex humanoid movements
- Design control systems based on dynamic models

## Introduction to Humanoid Kinematics

Humanoid robot kinematics is the study of motion without considering the forces that cause the motion. It involves understanding the geometric relationships between the robot's links and joints, which is crucial for controlling the robot's movements and interactions with the environment.

### Humanoid Robot Structure

A typical humanoid robot consists of:
- **Trunk/Torso**: Central body connecting upper and lower body
- **Head**: Contains sensors (cameras, microphones, etc.)
- **Arms**: Two manipulator arms with hands
- **Legs**: Two legs with feet for locomotion and balance

The kinematic structure is typically a tree topology with multiple branches, making it more complex than serial manipulators.

### Denavit-Hartenberg (DH) Parameters for Humanoid Robots

```python
# dh_parameters.py
import numpy as np
from math import sin, cos, sqrt

class HumanoidDHParameters:
    """DH parameters for a humanoid robot structure"""

    def __init__(self):
        # Define DH parameters for left arm
        self.left_arm_dh = [
            # [a, alpha, d, theta_offset]
            [0, np.pi/2, 0.1, 0],      # Shoulder joint 1 (yaw)
            [0, -np.pi/2, 0, 0],       # Shoulder joint 2 (pitch)
            [0.15, 0, 0, -np.pi/2],    # Shoulder joint 3 (roll)
            [0, np.pi/2, 0, 0],        # Elbow joint (pitch)
            [0.15, 0, 0, 0],           # Wrist joint 1 (pitch)
            [0, 0, 0.05, 0]            # Wrist joint 2 (yaw)
        ]

        # Define DH parameters for right arm
        self.right_arm_dh = [
            [0, np.pi/2, 0.1, 0],      # Shoulder joint 1 (yaw)
            [0, -np.pi/2, 0, 0],       # Shoulder joint 2 (pitch)
            [0.15, 0, 0, np.pi/2],     # Shoulder joint 3 (roll) - opposite to left
            [0, np.pi/2, 0, 0],        # Elbow joint (pitch)
            [0.15, 0, 0, 0],           # Wrist joint 1 (pitch)
            [0, 0, 0.05, 0]            # Wrist joint 2 (yaw)
        ]

        # Define DH parameters for left leg
        self.left_leg_dh = [
            [0, -np.pi/2, 0.05, 0],    # Hip joint 1 (yaw)
            [0, np.pi/2, 0, 0],        # Hip joint 2 (roll)
            [0, -np.pi/2, 0.2, 0],     # Hip joint 3 (pitch)
            [0, np.pi/2, 0, 0],        # Knee joint (pitch)
            [0, 0, 0.2, 0],            # Ankle joint 1 (pitch)
            [0, 0, 0.05, 0]            # Ankle joint 2 (roll)
        ]

        # Define DH parameters for right leg
        self.right_leg_dh = [
            [0, -np.pi/2, 0.05, 0],    # Hip joint 1 (yaw)
            [0, np.pi/2, 0, 0],        # Hip joint 2 (roll) - opposite to left
            [0, -np.pi/2, 0.2, 0],     # Hip joint 3 (pitch)
            [0, np.pi/2, 0, 0],        # Knee joint (pitch)
            [0, 0, 0.2, 0],            # Ankle joint 1 (pitch)
            [0, 0, 0.05, 0]            # Ankle joint 2 (roll)
        ]

    def dh_transform(self, a, alpha, d, theta):
        """Calculate DH transformation matrix"""
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles, dh_params):
        """Calculate forward kinematics using DH parameters"""
        T = np.eye(4)  # Identity transformation

        for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = np.dot(T, T_i)

        return T

    def get_end_effector_pose(self, joint_angles, dh_params):
        """Get end effector position and orientation"""
        T = self.forward_kinematics(joint_angles, dh_params)

        position = T[:3, 3]
        orientation = T[:3, :3]

        return position, orientation

# Example usage
def example_dh_usage():
    dh_params = HumanoidDHParameters()

    # Example joint angles for left arm (radians)
    left_arm_angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Calculate forward kinematics
    pos, orient = dh_params.get_end_effector_pose(left_arm_angles, dh_params.left_arm_dh)

    print(f"Left arm end effector position: {pos}")
    print(f"Left arm end effector orientation:\n{orient}")

if __name__ == "__main__":
    example_dh_usage()
```

## Forward Kinematics

Forward kinematics calculates the end-effector position and orientation given the joint angles. For humanoid robots, this involves complex multi-chain kinematics.

```python
# forward_kinematics.py
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HumanoidForwardKinematics:
    def __init__(self):
        # Robot dimensions (example values in meters)
        self.upper_arm_length = 0.3
        self.lower_arm_length = 0.25
        self.upper_leg_length = 0.4
        self.lower_leg_length = 0.35
        self.torso_height = 0.6
        self.head_height = 0.2

    def rotation_matrix_x(self, angle):
        """Rotation matrix around X axis"""
        return np.array([
            [1, 0, 0, 0],
            [0, cos(angle), -sin(angle), 0],
            [0, sin(angle), cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_y(self, angle):
        """Rotation matrix around Y axis"""
        return np.array([
            [cos(angle), 0, sin(angle), 0],
            [0, 1, 0, 0],
            [-sin(angle), 0, cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_z(self, angle):
        """Rotation matrix around Z axis"""
        return np.array([
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def translation_matrix(self, x, y, z):
        """Translation matrix"""
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    def left_arm_kinematics(self, joint_angles):
        """Calculate left arm forward kinematics"""
        # Joint angles: [shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, wrist_pitch, wrist_yaw]
        if len(joint_angles) < 6:
            raise ValueError("Need 6 joint angles for arm")

        # Start from shoulder (relative to torso)
        T = self.translation_matrix(0.15, 0.1, self.torso_height)  # Shoulder position

        # Shoulder yaw
        T = T @ self.rotation_matrix_z(joint_angles[0])
        # Shoulder pitch
        T = T @ self.rotation_matrix_y(joint_angles[1])
        # Shoulder roll
        T = T @ self.rotation_matrix_x(joint_angles[2])

        # Upper arm
        T = T @ self.translation_matrix(0, 0, -self.upper_arm_length)

        # Elbow pitch
        T = T @ self.rotation_matrix_y(joint_angles[3])

        # Lower arm
        T = T @ self.translation_matrix(0, 0, -self.lower_arm_length)

        # Wrist pitch
        T = T @ self.rotation_matrix_y(joint_angles[4])
        # Wrist yaw
        T = T @ self.rotation_matrix_z(joint_angles[5])

        return T

    def right_arm_kinematics(self, joint_angles):
        """Calculate right arm forward kinematics"""
        # Joint angles: [shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, wrist_pitch, wrist_yaw]
        if len(joint_angles) < 6:
            raise ValueError("Need 6 joint angles for arm")

        # Start from shoulder (relative to torso)
        T = self.translation_matrix(0.15, -0.1, self.torso_height)  # Shoulder position

        # Shoulder yaw
        T = T @ self.rotation_matrix_z(joint_angles[0])
        # Shoulder pitch
        T = T @ self.rotation_matrix_y(joint_angles[1])
        # Shoulder roll (opposite direction to left arm)
        T = T @ self.rotation_matrix_x(-joint_angles[2])

        # Upper arm
        T = T @ self.translation_matrix(0, 0, -self.upper_arm_length)

        # Elbow pitch
        T = T @ self.rotation_matrix_y(joint_angles[3])

        # Lower arm
        T = T @ self.translation_matrix(0, 0, -self.lower_arm_length)

        # Wrist pitch
        T = T @ self.rotation_matrix_y(joint_angles[4])
        # Wrist yaw
        T = T @ self.rotation_matrix_z(joint_angles[5])

        return T

    def left_leg_kinematics(self, joint_angles):
        """Calculate left leg forward kinematics"""
        # Joint angles: [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]
        if len(joint_angles) < 6:
            raise ValueError("Need 6 joint angles for leg")

        # Start from hip (relative to torso)
        T = self.translation_matrix(-0.05, 0.08, 0)  # Hip position

        # Hip yaw
        T = T @ self.rotation_matrix_z(joint_angles[0])
        # Hip roll
        T = T @ self.rotation_matrix_x(joint_angles[1])
        # Hip pitch
        T = T @ self.rotation_matrix_y(joint_angles[2])

        # Upper leg
        T = T @ self.translation_matrix(0, 0, -self.upper_leg_length)

        # Knee pitch
        T = T @ self.rotation_matrix_y(joint_angles[3])

        # Lower leg
        T = T @ self.translation_matrix(0, 0, -self.lower_leg_length)

        # Ankle pitch
        T = T @ self.rotation_matrix_y(joint_angles[4])
        # Ankle roll
        T = T @ self.rotation_matrix_x(joint_angles[5])

        return T

    def right_leg_kinematics(self, joint_angles):
        """Calculate right leg forward kinematics"""
        # Joint angles: [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]
        if len(joint_angles) < 6:
            raise ValueError("Need 6 joint angles for leg")

        # Start from hip (relative to torso)
        T = self.translation_matrix(-0.05, -0.08, 0)  # Hip position

        # Hip yaw
        T = T @ self.rotation_matrix_z(joint_angles[0])
        # Hip roll (opposite to left leg)
        T = T @ self.rotation_matrix_x(-joint_angles[1])
        # Hip pitch
        T = T @ self.rotation_matrix_y(joint_angles[2])

        # Upper leg
        T = T @ self.translation_matrix(0, 0, -self.upper_leg_length)

        # Knee pitch
        T = T @ self.rotation_matrix_y(joint_angles[3])

        # Lower leg
        T = T @ self.translation_matrix(0, 0, -self.lower_leg_length)

        # Ankle pitch
        T = T @ self.rotation_matrix_y(joint_angles[4])
        # Ankle roll (opposite to left leg)
        T = T @ self.rotation_matrix_x(-joint_angles[5])

        return T

    def full_body_kinematics(self, joint_angles_dict):
        """Calculate full body forward kinematics"""
        # Joint angles dictionary contains angles for all limbs
        result = {}

        # Torso is at origin
        result['torso'] = np.eye(4)
        result['head'] = self.translation_matrix(0, 0, self.torso_height + self.head_height)

        # Calculate each limb
        if 'left_arm' in joint_angles_dict:
            result['left_arm'] = self.left_arm_kinematics(joint_angles_dict['left_arm'])
        if 'right_arm' in joint_angles_dict:
            result['right_arm'] = self.right_arm_kinematics(joint_angles_dict['right_arm'])
        if 'left_leg' in joint_angles_dict:
            result['left_leg'] = self.left_leg_kinematics(joint_angles_dict['left_leg'])
        if 'right_leg' in joint_angles_dict:
            result['right_leg'] = self.right_leg_kinematics(joint_angles_dict['right_leg'])

        return result

    def visualize_pose(self, joint_angles_dict):
        """Visualize the current pose of the humanoid robot"""
        transforms = self.full_body_kinematics(joint_angles_dict)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot torso
        self.plot_limb(ax, [0, 0, 0], [0, 0, self.torso_height], 'b-', label='Torso')

        # Plot head
        if 'head' in transforms:
            head_pos = transforms['head'][:3, 3]
            torso_pos = [0, 0, self.torso_height]
            self.plot_limb(ax, torso_pos, head_pos, 'b-', label='Head')

        # Plot arms
        if 'left_arm' in transforms:
            left_arm_pos = transforms['left_arm'][:3, 3]
            shoulder_pos = [0.15, 0.1, self.torso_height]
            self.plot_limb(ax, shoulder_pos, left_arm_pos, 'r-', label='Left Arm')
        if 'right_arm' in transforms:
            right_arm_pos = transforms['right_arm'][:3, 3]
            shoulder_pos = [0.15, -0.1, self.torso_height]
            self.plot_limb(ax, shoulder_pos, right_arm_pos, 'g-', label='Right Arm')

        # Plot legs
        if 'left_leg' in transforms:
            left_foot_pos = transforms['left_leg'][:3, 3]
            hip_pos = [-0.05, 0.08, 0]
            self.plot_limb(ax, hip_pos, left_foot_pos, 'c-', label='Left Leg')
        if 'right_leg' in transforms:
            right_foot_pos = transforms['right_leg'][:3, 3]
            hip_pos = [-0.05, -0.08, 0]
            self.plot_limb(ax, hip_pos, right_foot_pos, 'm-', label='Right Leg')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Humanoid Robot Pose')
        ax.legend()
        ax.grid(True)

        plt.show()

    def plot_limb(self, ax, start, end, style, label=None):
        """Helper function to plot a limb as a line"""
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], style, label=label)

# Example usage
def example_forward_kinematics():
    fk = HumanoidForwardKinematics()

    # Define joint angles for a pose
    joint_angles = {
        'left_arm': [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
        'right_arm': [-0.1, 0.2, 0.3, 0.4, 0.5, -0.1],
        'left_leg': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'right_leg': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    # Calculate transforms
    transforms = fk.full_body_kinematics(joint_angles)

    # Print end effector positions
    for limb, transform in transforms.items():
        if limb != 'torso' and limb != 'head':
            pos = transform[:3, 3]
            print(f"{limb} end effector position: {pos}")

    # Visualize the pose (uncomment to see plot)
    # fk.visualize_pose(joint_angles)

if __name__ == "__main__":
    example_forward_kinematics()
```

## Inverse Kinematics

Inverse kinematics calculates the joint angles needed to achieve a desired end-effector position and orientation. This is more complex than forward kinematics and often has multiple solutions.

```python
# inverse_kinematics.py
import numpy as np
from scipy.optimize import minimize
from math import sin, cos, acos, atan2, sqrt

class HumanoidInverseKinematics:
    def __init__(self):
        # Robot dimensions
        self.upper_arm_length = 0.3
        self.lower_arm_length = 0.25
        self.upper_leg_length = 0.4
        self.lower_leg_length = 0.35
        self.torso_height = 0.6

    def jacobian(self, joint_angles, forward_func):
        """Calculate Jacobian matrix using numerical differentiation"""
        n = len(joint_angles)
        end_effector_pos = forward_func(joint_angles)[:3, 3]

        J = np.zeros((3, n))
        eps = 1e-6

        for i in range(n):
            # Positive perturbation
            angles_plus = joint_angles.copy()
            angles_plus[i] += eps
            pos_plus = forward_func(angles_plus)[:3, 3]

            # Calculate partial derivative
            J[:, i] = (pos_plus - end_effector_pos) / eps

        return J

    def left_arm_ik(self, target_pos, target_orient=None, current_angles=None):
        """Inverse kinematics for left arm using numerical optimization"""
        if current_angles is None:
            current_angles = [0.0] * 6  # Default joint angles

        def objective_function(angles):
            # Calculate current end effector position
            fk = HumanoidForwardKinematics()
            current_transform = fk.left_arm_kinematics(angles)
            current_pos = current_transform[:3, 3]

            # Calculate distance to target
            error = np.linalg.norm(current_pos - target_pos)

            # Add joint limit penalties
            joint_limit_penalty = 0
            for i, angle in enumerate(angles):
                # Example joint limits (should be customized per robot)
                if angle < -2.0 or angle > 2.0:  # Example limits
                    joint_limit_penalty += 100 * (abs(angle) - 2.0)**2

            return error + joint_limit_penalty

        # Optimize to find joint angles
        result = minimize(objective_function, current_angles, method='BFGS')

        if result.success:
            return result.x
        else:
            print("IK optimization failed")
            return current_angles

    def right_arm_ik(self, target_pos, target_orient=None, current_angles=None):
        """Inverse kinematics for right arm"""
        if current_angles is None:
            current_angles = [0.0] * 6

        def objective_function(angles):
            fk = HumanoidForwardKinematics()
            current_transform = fk.right_arm_kinematics(angles)
            current_pos = current_transform[:3, 3]

            error = np.linalg.norm(current_pos - target_pos)

            # Add joint limit penalties
            joint_limit_penalty = 0
            for i, angle in enumerate(angles):
                if angle < -2.0 or angle > 2.0:
                    joint_limit_penalty += 100 * (abs(angle) - 2.0)**2

            return error + joint_limit_penalty

        result = minimize(objective_function, current_angles, method='BFGS')

        if result.success:
            return result.x
        else:
            print("IK optimization failed")
            return current_angles

    def left_arm_analytical_ik(self, target_pos, shoulder_pos):
        """Analytical IK for simple 2-DOF arm (shoulder + elbow)"""
        # Simplified 2D case for demonstration
        # target_pos: [x, y, z] in world coordinates
        # shoulder_pos: [x, y, z] shoulder position

        # Calculate relative position
        rel_x = target_pos[0] - shoulder_pos[0]
        rel_y = target_pos[1] - shoulder_pos[1]
        rel_z = target_pos[2] - shoulder_pos[2]

        # Project to 2D plane (simplified)
        dist_2d = sqrt(rel_x**2 + rel_z**2)  # Working in XZ plane

        # Check if target is reachable
        total_length = self.upper_arm_length + self.lower_arm_length
        if dist_2d > total_length:
            print("Target out of reach")
            return None

        # Calculate elbow angle using law of cosines
        cos_elbow = (self.upper_arm_length**2 + self.lower_arm_length**2 - dist_2d**2) / \
                    (2 * self.upper_arm_length * self.lower_arm_length)
        cos_elbow = np.clip(cos_elbow, -1, 1)  # Clamp to valid range
        elbow_angle = acos(cos_elbow)

        # Calculate shoulder angles
        shoulder_angle_1 = atan2(rel_z, rel_x)  # Shoulder yaw

        # For the 2D case in XZ plane
        cos_shoulder = (self.upper_arm_length**2 + dist_2d**2 - self.lower_arm_length**2) / \
                      (2 * self.upper_arm_length * dist_2d)
        cos_shoulder = np.clip(cos_shoulder, -1, 1)
        shoulder_angle_2 = atan2(sqrt(1 - cos_shoulder**2), cos_shoulder)

        # Adjust for actual arm configuration
        shoulder_angle_2 = shoulder_angle_2  # This is simplified

        # Return joint angles (others set to 0 for simplicity)
        joint_angles = np.array([shoulder_angle_1, shoulder_angle_2, 0, elbow_angle, 0, 0])

        return joint_angles

    def jacobian_ik(self, target_pos, current_angles, forward_func, max_iterations=100, tolerance=1e-4):
        """Jacobian-based inverse kinematics"""
        current_angles = np.array(current_angles)

        for i in range(max_iterations):
            # Calculate current position
            current_transform = forward_func(current_angles)
            current_pos = current_transform[:3, 3]

            # Calculate error
            error = target_pos - current_pos

            if np.linalg.norm(error) < tolerance:
                print(f"Converged after {i+1} iterations")
                return current_angles

            # Calculate Jacobian
            J = self.jacobian(current_angles, forward_func)

            # Calculate joint angle updates using pseudo-inverse
            J_pinv = np.linalg.pinv(J)
            delta_theta = J_pinv @ error

            # Update joint angles
            current_angles = current_angles + delta_theta * 0.1  # Step size

        print(f"Did not converge after {max_iterations} iterations")
        return current_angles

    def resolve_ik_ambiguities(self, joint_angles, preferred_angles, weights=None):
        """Resolve IK solution ambiguities by preferring certain configurations"""
        if weights is None:
            weights = np.ones_like(joint_angles)

        # Calculate differences from preferred angles
        differences = joint_angles - preferred_angles

        # Apply weights to penalize deviations from preferred configuration
        weighted_differences = differences * weights

        return joint_angles - weighted_differences * 0.1  # Small adjustment

# Example usage
def example_inverse_kinematics():
    ik = HumanoidInverseKinematics()
    fk = HumanoidForwardKinematics()

    # Define a target position for the left hand
    target_pos = np.array([0.5, 0.2, 0.4])  # x, y, z in meters

    # Calculate IK
    initial_angles = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
    joint_angles = ik.left_arm_ik(target_pos, current_angles=initial_angles)

    if joint_angles is not None:
        print(f"Calculated joint angles: {joint_angles}")

        # Verify with forward kinematics
        final_transform = fk.left_arm_kinematics(joint_angles)
        final_pos = final_transform[:3, 3]
        print(f"Final position: {final_pos}")
        print(f"Target position: {target_pos}")
        print(f"Position error: {np.linalg.norm(final_pos - target_pos)}")

    # Example of analytical IK for simplified case
    shoulder_pos = np.array([0.15, 0.1, 0.6])  # Shoulder position
    simplified_target = np.array([0.4, 0.1, 0.4])
    analytical_solution = ik.left_arm_analytical_ik(simplified_target, shoulder_pos)

    if analytical_solution is not None:
        print(f"Analytical IK solution: {analytical_solution}")

if __name__ == "__main__":
    example_inverse_kinematics()
```

## Humanoid Dynamics

Humanoid dynamics involves understanding the forces and torques required to produce the desired motion. This is crucial for balance control and stable locomotion.

```python
# humanoid_dynamics.py
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

class HumanoidDynamics:
    def __init__(self):
        # Robot parameters (example values)
        self.link_masses = {
            'torso': 10.0,
            'head': 2.0,
            'upper_arm': 1.5,
            'lower_arm': 1.0,
            'hand': 0.5,
            'upper_leg': 3.0,
            'lower_leg': 2.5,
            'foot': 1.0
        }

        # Moments of inertia (simplified as spheres)
        self.link_inertias = {
            'torso': np.diag([1.0, 1.0, 1.0]),
            'head': np.diag([0.1, 0.1, 0.1]),
            'upper_arm': np.diag([0.1, 0.1, 0.1]),
            'lower_arm': np.diag([0.05, 0.05, 0.05]),
            'hand': np.diag([0.02, 0.02, 0.02]),
            'upper_leg': np.diag([0.3, 0.3, 0.3]),
            'lower_leg': np.diag([0.2, 0.2, 0.2]),
            'foot': np.diag([0.05, 0.05, 0.05])
        }

        # Link lengths (meters)
        self.link_lengths = {
            'upper_arm': 0.3,
            'lower_arm': 0.25,
            'upper_leg': 0.4,
            'lower_leg': 0.35,
            'torso': 0.6,
            'head': 0.2
        }

        # Gravity
        self.g = 9.81  # m/s^2

    def compute_lagrangian_dynamics(self, joint_angles, joint_velocities, joint_accelerations):
        """
        Compute inverse dynamics using Lagrangian formulation
        This calculates required joint torques for given motion
        """
        # This is a simplified example - full implementation would be very complex
        n = len(joint_angles)  # Number of joints

        # Mass matrix M(q)
        M = self.compute_mass_matrix(joint_angles)

        # Coriolis and centrifugal terms C(q, q_dot)
        C = self.compute_coriolis_matrix(joint_angles, joint_velocities)

        # Gravity terms g(q)
        G = self.compute_gravity_terms(joint_angles)

        # Required joint torques: τ = M(q)*q_ddot + C(q, q_dot)*q_dot + G(q)
        torques = M @ joint_accelerations + C @ joint_velocities + G

        return torques

    def compute_mass_matrix(self, joint_angles):
        """Compute the mass matrix M(q)"""
        # Simplified mass matrix calculation
        # In reality, this would involve complex recursive calculations
        n = len(joint_angles)
        M = np.zeros((n, n))

        # Diagonal elements (simplified)
        for i in range(n):
            # Approximate moment of inertia contribution
            M[i, i] = 1.0  # This would be calculated based on link properties

        return M

    def compute_coriolis_matrix(self, joint_angles, joint_velocities):
        """Compute Coriolis and centrifugal matrix C(q, q_dot)"""
        n = len(joint_angles)
        C = np.zeros((n, n))

        # Simplified Coriolis terms
        # In reality, this would involve complex velocity-dependent terms
        for i in range(n):
            for j in range(n):
                # Simplified velocity coupling
                C[i, j] = 0.1 * joint_velocities[j] if i != j else 0

        return C

    def compute_gravity_terms(self, joint_angles):
        """Compute gravity terms g(q)"""
        n = len(joint_angles)
        G = np.zeros(n)

        # Simplified gravity effects
        for i in range(n):
            # Gravity effect based on joint configuration
            G[i] = 0.5 * sin(joint_angles[i])  # Simplified model

        return G

    def euler_lagrange_equation(self, joint_angles, joint_velocities, torques):
        """
        Solve forward dynamics: M(q)*q_ddot = τ - C(q, q_dot)*q_dot - G(q)
        This calculates accelerations for given torques
        """
        M = self.compute_mass_matrix(joint_angles)
        C = self.compute_coriolis_matrix(joint_angles, joint_velocities)
        G = self.compute_gravity_terms(joint_angles)

        # Solve for accelerations: q_ddot = M^(-1) * (τ - C*q_dot - G)
        accelerations = np.linalg.solve(M, torques - C @ joint_velocities - G)

        return accelerations

    def compute_center_of_mass(self, joint_angles):
        """Compute center of mass position"""
        # Simplified CoM calculation
        total_mass = sum(self.link_masses.values())

        # This would involve complex kinematic calculations in reality
        # For now, return a simplified estimate
        com_x = 0.0
        com_y = 0.0
        com_z = self.link_lengths['torso'] / 2  # Approximate CoM height

        return np.array([com_x, com_y, com_z])

    def compute_zero_moment_point(self, joint_angles, joint_velocities, joint_accelerations):
        """Compute Zero Moment Point (ZMP) for balance analysis"""
        # ZMP = [x, y] where net moment about this point is zero
        # ZMP_x = (g * (CoM_x - ZMP_x)) / (g + CoM_z_ddot) + CoM_x_ddot / (g + CoM_z_ddot)
        # ZMP_y = (g * (CoM_y - ZMP_y)) / (g + CoM_z_ddot) + CoM_y_ddot / (g + CoM_z_ddot)

        # This would require full dynamic model and CoM calculations
        # Simplified return
        return np.array([0.0, 0.0])

    def compute_balance_metrics(self, joint_angles, joint_velocities, joint_accelerations):
        """Compute various balance-related metrics"""
        # Center of Mass
        com_pos = self.compute_center_of_mass(joint_angles)

        # Zero Moment Point (simplified)
        zmp_pos = self.compute_zero_moment_point(joint_angles, joint_velocities, joint_accelerations)

        # Support polygon (simplified - just the feet positions)
        support_polygon = self.calculate_support_polygon(joint_angles)

        # Calculate distance from CoM to support polygon
        com_in_support = self.is_com_in_support_polygon(com_pos[:2], support_polygon)

        # Calculate margin of stability
        stability_margin = self.calculate_stability_margin(com_pos[:2], support_polygon)

        return {
            'center_of_mass': com_pos,
            'zero_moment_point': zmp_pos,
            'support_polygon': support_polygon,
            'com_in_support': com_in_support,
            'stability_margin': stability_margin
        }

    def calculate_support_polygon(self, joint_angles):
        """Calculate support polygon based on foot positions"""
        # This would use forward kinematics to get foot positions
        # Simplified return: approximate foot positions
        left_foot = np.array([-0.1, 0.1])  # Simplified
        right_foot = np.array([-0.1, -0.1])  # Simplified

        return np.array([left_foot, right_foot])

    def is_com_in_support_polygon(self, com_xy, support_polygon):
        """Check if CoM is within support polygon"""
        # Simplified for rectangular support
        if len(support_polygon) >= 2:
            min_x = min(point[0] for point in support_polygon)
            max_x = max(point[0] for point in support_polygon)
            min_y = min(point[1] for point in support_polygon)
            max_y = max(point[1] for point in support_polygon)

            return (min_x <= com_xy[0] <= max_x) and (min_y <= com_xy[1] <= max_y)

        return False

    def calculate_stability_margin(self, com_xy, support_polygon):
        """Calculate minimum distance from CoM to support polygon edge"""
        if len(support_polygon) < 2:
            return 0.0

        # Simplified calculation for rectangular support
        min_x = min(point[0] for point in support_polygon)
        max_x = max(point[0] for point in support_polygon)
        min_y = min(point[1] for point in support_polygon)
        max_y = max(point[1] for point in support_polygon)

        # Calculate distances to each edge
        dx1 = abs(com_xy[0] - min_x)
        dx2 = abs(max_x - com_xy[0])
        dy1 = abs(com_xy[1] - min_y)
        dy2 = abs(max_y - com_xy[1])

        return min(dx1, dx2, dy1, dy2)

    def simulate_dynamics_step(self, joint_angles, joint_velocities, joint_torques, dt):
        """Simulate one time step of robot dynamics"""
        # Calculate accelerations
        joint_accelerations = self.euler_lagrange_equation(
            joint_angles, joint_velocities, joint_torques
        )

        # Update velocities and positions using simple integration
        new_joint_velocities = joint_velocities + joint_accelerations * dt
        new_joint_angles = joint_angles + new_joint_velocities * dt

        return new_joint_angles, new_joint_velocities, joint_accelerations

# Example usage
def example_dynamics():
    dynamics = HumanoidDynamics()

    # Example joint state
    joint_angles = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0] * 4)  # 24 joints (simplified)
    joint_velocities = np.array([0.0] * len(joint_angles))
    joint_accelerations = np.array([0.0] * len(joint_angles))

    # Calculate required torques for some motion
    torques = dynamics.compute_lagrangian_dynamics(joint_angles, joint_velocities, joint_accelerations)

    print(f"Required joint torques: {torques}")

    # Calculate balance metrics
    balance_metrics = dynamics.compute_balance_metrics(joint_angles, joint_velocities, joint_accelerations)

    print(f"Center of Mass: {balance_metrics['center_of_mass']}")
    print(f"ZMP: {balance_metrics['zero_moment_point']}")
    print(f"Stability Margin: {balance_metrics['stability_margin']}")
    print(f"CoM in support: {balance_metrics['com_in_support']}")

if __name__ == "__main__":
    example_dynamics()
```

## Control Systems for Humanoid Robots

### PID Controllers for Joint Control

```python
# control_systems.py
import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    """PID controller for joint control"""
    def __init__(self, kp, ki, kd, output_limits=(-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.reset()

    def reset(self):
        """Reset the PID controller"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None

    def compute(self, setpoint, measurement, dt=None):
        """Compute control output"""
        current_time = time.time()
        if dt is None:
            if self.previous_time is not None:
                dt = current_time - self.previous_time
            else:
                dt = 0.01  # Default time step
        self.previous_time = current_time

        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Store values for next iteration
        self.previous_error = error

        return output

class JointController:
    """Controller for a single joint with safety limits"""
    def __init__(self, joint_name, kp=100, ki=1, kd=10):
        self.joint_name = joint_name
        self.pid = PIDController(kp, ki, kd)
        self.position_limit = (-np.pi, np.pi)  # Default limits
        self.velocity_limit = 5.0  # rad/s
        self.torque_limit = 100.0  # Nm

    def set_position_limits(self, min_pos, max_pos):
        """Set position limits for the joint"""
        self.position_limit = (min_pos, max_pos)

    def set_velocity_limit(self, max_vel):
        """Set velocity limit for the joint"""
        self.velocity_limit = max_vel

    def set_torque_limit(self, max_torque):
        """Set torque limit for the joint"""
        self.torque_limit = max_torque

    def compute_torque(self, desired_pos, current_pos, current_vel, dt=0.001):
        """Compute required torque to reach desired position"""
        # Check position limits
        if not (self.position_limit[0] <= desired_pos <= self.position_limit[1]):
            print(f"Warning: Desired position {desired_pos} outside limits {self.position_limit} for {self.joint_name}")

        # Compute control effort
        torque = self.pid.compute(desired_pos, current_pos, dt)

        # Apply torque limits
        torque = np.clip(torque, -self.torque_limit, self.torque_limit)

        return torque

class HumanoidController:
    """Controller for the entire humanoid robot"""
    def __init__(self):
        # Create controllers for each joint
        self.joint_controllers = {}
        self.create_joint_controllers()

    def create_joint_controllers(self):
        """Create PID controllers for all joints"""
        # Define joint names
        joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_yaw', 'left_shoulder_pitch', 'left_shoulder_roll',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_yaw', 'right_shoulder_pitch', 'right_shoulder_roll',
            'right_elbow', 'right_wrist_pitch', 'right_wrist_yaw'
        ]

        # Create controllers with appropriate parameters
        for joint_name in joint_names:
            if 'hip' in joint_name or 'knee' in joint_name or 'ankle' in joint_name:
                # Legs typically need higher torque
                controller = JointController(joint_name, kp=200, ki=2, kd=20)
            else:
                # Arms can use lower torque
                controller = JointController(joint_name, kp=100, ki=1, kd=10)

            self.joint_controllers[joint_name] = controller

    def compute_joint_torques(self, desired_positions, current_positions, current_velocities, dt=0.001):
        """Compute torques for all joints"""
        torques = {}

        for joint_name in self.joint_controllers:
            if joint_name in desired_positions and joint_name in current_positions and joint_name in current_velocities:
                torque = self.joint_controllers[joint_name].compute_torque(
                    desired_positions[joint_name],
                    current_positions[joint_name],
                    current_velocities[joint_name],
                    dt
                )
                torques[joint_name] = torque

        return torques

    def balance_control(self, com_error, zmp_error):
        """Balance control to maintain stability"""
        # Simple balance control based on CoM and ZMP errors
        # This would involve more complex algorithms in practice
        balance_torques = {}

        # Adjust ankle joints based on balance errors
        for joint_name in self.joint_controllers:
            if 'ankle' in joint_name:
                # Apply balance correction
                balance_torque = -10 * com_error - 5 * zmp_error  # Simplified
                balance_torques[joint_name] = balance_torque

        return balance_torques

# Example usage
def example_control_system():
    controller = HumanoidController()

    # Example desired and current positions
    desired_positions = {name: 0.0 for name in controller.joint_controllers}
    current_positions = {name: 0.01 for name in controller.joint_controllers}  # Small error
    current_velocities = {name: 0.0 for name in controller.joint_controllers}

    # Compute required torques
    torques = controller.compute_joint_torques(desired_positions, current_positions, current_velocities)

    print("Computed joint torques:")
    for joint_name, torque in list(torques.items())[:6]:  # Show first 6
        print(f"  {joint_name}: {torque:.3f} Nm")

    # Example balance control
    balance_torques = controller.balance_control(com_error=0.01, zmp_error=0.005)
    print("\nBalance control torques:")
    for joint_name, torque in balance_torques.items():
        print(f"  {joint_name}: {torque:.3f} Nm")

if __name__ == "__main__":
    import time
    example_control_system()
```

## Mathematical Foundations

### Rigid Body Dynamics

```python
# mathematical_foundations.py
import numpy as np
from math import sin, cos, sqrt

def skew_symmetric_matrix(vector):
    """Create skew-symmetric matrix from a 3D vector"""
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def rotation_matrix_from_axis_angle(axis, angle):
    """Create rotation matrix from axis-angle representation"""
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    K = skew_symmetric_matrix(axis)
    I = np.eye(3)

    # Rodrigues' rotation formula
    R = I + sin(angle) * K + (1 - cos(angle)) * K @ K
    return R

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion"""
    trace = np.trace(R)

    if trace > 0:
        s = sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    return np.array([qw, qx, qy, qz])

def transform_point(point, transform_matrix):
    """Transform a 3D point using a 4x4 transformation matrix"""
    # Convert to homogeneous coordinates
    homogeneous_point = np.append(point, 1)
    # Apply transformation
    transformed_point = transform_matrix @ homogeneous_point
    # Convert back to 3D
    return transformed_point[:3]

def compose_transforms(T1, T2):
    """Compose two transformation matrices"""
    return T1 @ T2

def inverse_transform(transform_matrix):
    """Calculate inverse of a transformation matrix"""
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]

    R_inv = R.T
    t_inv = -R.T @ t

    inv_transform = np.eye(4)
    inv_transform[:3, :3] = R_inv
    inv_transform[:3, 3] = t_inv

    return inv_transform

# Example usage
def example_mathematical_foundations():
    print("Mathematical Foundations Examples:")

    # Skew-symmetric matrix
    v = np.array([1, 2, 3])
    S = skew_symmetric_matrix(v)
    print(f"Skew-symmetric matrix of {v}:")
    print(S)

    # Axis-angle rotation
    axis = np.array([0, 0, 1])  # Z-axis
    angle = np.pi / 4  # 45 degrees
    R = rotation_matrix_from_axis_angle(axis, angle)
    print(f"\nRotation matrix for {angle:.2f} rad around {axis}:")
    print(R)

    # Quaternion operations
    q = np.array([1, 0, 0, 0])  # Identity quaternion
    R_from_q = quaternion_to_rotation_matrix(q)
    print(f"\nRotation matrix from identity quaternion:")
    print(R_from_q)

    # Point transformation
    point = np.array([1, 0, 0])
    T = np.eye(4)
    T[:3, 3] = [2, 0, 0]  # Translation by 2 units in X
    transformed_point = transform_point(point, T)
    print(f"\nTransforming point {point} by translation matrix:")
    print(f"Result: {transformed_point}")

if __name__ == "__main__":
    example_mathematical_foundations()
```

## Knowledge Check

1. What are the main differences between forward and inverse kinematics for humanoid robots?
2. How do you calculate the center of mass for a humanoid robot?
3. What is the Zero Moment Point (ZMP) and why is it important for balance?
4. How do joint limits affect inverse kinematics solutions?

## Summary

This chapter covered the fundamental kinematics and dynamics of humanoid robots. We explored forward and inverse kinematics using both analytical and numerical methods, examined the dynamics of multi-link systems, and discussed control systems for joint actuation. The chapter provided mathematical foundations including rotation matrices, quaternions, and transformation mathematics essential for humanoid robot control.

## Next Steps

In the next chapter, we'll explore bipedal locomotion and balance control, diving into the principles of walking algorithms, stability control, and the dynamics of bipedal movement for humanoid robots.