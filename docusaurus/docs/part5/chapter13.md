---
title: "Chapter 13: Humanoid Robot Kinematics and Dynamics"
sidebar_label: "Chapter 13: Kinematics and Dynamics"
---



# Chapter 13: Humanoid Robot Kinematics and Dynamics

## Learning Objectives
- Understand the mathematical foundations of humanoid robot kinematics
- Master forward and inverse kinematics for multi-link humanoid systems
- Analyze dynamic modeling for bipedal movement and balance
- Implement control systems based on kinematic and dynamic principles

## Introduction

Humanoid robot kinematics and dynamics form the mathematical foundation for understanding and controlling the complex movements of human-like robotic systems. Unlike simpler robots, humanoid robots must navigate the challenges of bipedal locomotion, multi-degree-of-freedom manipulation, and balance maintenance. This chapter explores the mathematical principles that govern humanoid robot motion, from basic kinematic relationships to complex dynamic models required for stable bipedal movement.

## Mathematical Foundations for Humanoid Robotics

### Coordinate Systems and Transformations

Humanoid robots require multiple coordinate systems to describe their complex structure:

```python
# Coordinate system transformations for humanoid robots
import numpy as np
from scipy.spatial.transform import Rotation as R

class CoordinateSystem:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation = np.eye(3)  # 3x3 rotation matrix
        self.transform_matrix = np.eye(4)  # 4x4 homogeneous transformation

    def set_pose(self, position, orientation):
        """Set position and orientation (rotation matrix)"""
        self.position = np.array(position)
        self.orientation = np.array(orientation)

        # Update homogeneous transformation matrix
        self.transform_matrix[:3, :3] = self.orientation
        self.transform_matrix[:3, 3] = self.position

    def get_transform_to_parent(self):
        """Get transformation matrix to parent coordinate system"""
        return self.transform_matrix

    def transform_point(self, point):
        """Transform point from this coordinate system to parent"""
        point_homogeneous = np.append(point, 1)
        transformed = self.transform_matrix @ point_homogeneous
        return transformed[:3]

class HumanoidKinematicChain:
    def __init__(self):
        self.links = []
        self.joints = []
        self.coordinate_frames = {}

    def add_link(self, name, length, joint_type='revolute'):
        """Add a link to the kinematic chain"""
        link = {
            'name': name,
            'length': length,
            'joint_type': joint_type,
            'dof': 1 if joint_type == 'revolute' else 3
        }
        self.links.append(link)

    def create_dh_parameters(self):
        """Create Denavit-Hartenberg parameters for the robot"""
        # DH parameters: [theta, d, a, alpha]
        # theta: joint angle
        # d: link offset
        # a: link length
        # alpha: link twist
        dh_params = []

        for i, link in enumerate(self.links):
            # Default DH parameters for a simple serial chain
            dh_param = {
                'theta': 0,  # Joint angle (variable)
                'd': 0,      # Link offset (constant for revolute)
                'a': link['length'],  # Link length
                'alpha': 0   # Link twist
            }
            dh_params.append(dh_param)

        return dh_params
```

### Vector and Matrix Mathematics

Humanoid robotics heavily relies on linear algebra:

```python
# Mathematical utilities for humanoid robotics
class MathUtils:
    @staticmethod
    def skew_symmetric(vector):
        """Create skew-symmetric matrix from 3D vector"""
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    @staticmethod
    def rotation_matrix_from_axis_angle(axis, angle):
        """Create rotation matrix from axis-angle representation"""
        axis = axis / np.linalg.norm(axis)  # Normalize axis
        kx, ky, kz = axis
        c = np.cos(angle)
        s = np.sin(angle)
        v = 1 - c

        return np.array([
            [kx*kx*v + c, kx*ky*v - kz*s, kx*kz*v + ky*s],
            [kx*ky*v + kz*s, ky*ky*v + c, ky*kz*v - kx*s],
            [kx*kz*v - ky*s, ky*kz*v + kx*s, kz*kz*v + c]
        ])

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    @staticmethod
    def homogeneous_transform(rotation, translation):
        """Create 4x4 homogeneous transformation matrix"""
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
```

## Forward Kinematics

### Understanding Forward Kinematics

Forward kinematics calculates the end-effector position and orientation from given joint angles. For humanoid robots, this involves multiple kinematic chains (arms, legs, torso, head).

```python
# Forward kinematics implementation
class ForwardKinematics:
    def __init__(self, dh_parameters):
        self.dh_params = dh_parameters  # List of DH parameter dictionaries

    def dh_transform(self, theta, d, a, alpha):
        """Calculate Denavit-Hartenberg transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def calculate_forward_kinematics(self, joint_angles):
        """Calculate forward kinematics for the entire chain"""
        if len(joint_angles) != len(self.dh_params):
            raise ValueError("Number of joint angles must match DH parameters")

        # Start with identity transformation
        cumulative_transform = np.eye(4)
        transforms = [cumulative_transform.copy()]  # Store all transforms

        for i, angle in enumerate(joint_angles):
            # Get DH parameters for this joint
            dh = self.dh_params[i]
            theta = dh['theta'] + angle  # Add joint angle to offset
            d = dh['d']
            a = dh['a']
            alpha = dh['alpha']

            # Calculate individual joint transformation
            joint_transform = self.dh_transform(theta, d, a, alpha)

            # Accumulate transformation
            cumulative_transform = cumulative_transform @ joint_transform
            transforms.append(cumulative_transform.copy())

        # Extract end-effector position and orientation
        end_effector_pos = cumulative_transform[:3, 3]
        end_effector_rot = cumulative_transform[:3, :3]

        return {
            'position': end_effector_pos,
            'orientation': end_effector_rot,
            'all_transforms': transforms
        }

    def calculate_link_positions(self, joint_angles):
        """Calculate positions of all links in the chain"""
        result = self.calculate_forward_kinematics(joint_angles)
        transforms = result['all_transforms']

        link_positions = []
        for transform in transforms:
            pos = transform[:3, 3]
            link_positions.append(pos)

        return link_positions
```

### Humanoid-Specific Forward Kinematics

Humanoid robots have multiple kinematic chains that must be considered together:

```python
# Humanoid-specific forward kinematics
class HumanoidFK:
    def __init__(self):
        # Define kinematic chains for different body parts
        self.chains = {
            'left_arm': self.create_arm_dh_params('left'),
            'right_arm': self.create_arm_dh_params('right'),
            'left_leg': self.create_leg_dh_params('left'),
            'right_leg': self.create_leg_dh_params('right'),
            'torso': self.create_torso_dh_params()
        }

        # FK calculators for each chain
        self.fk_calculators = {}
        for chain_name, dh_params in self.chains.items():
            self.fk_calculators[chain_name] = ForwardKinematics(dh_params)

    def create_arm_dh_params(self, side):
        """Create DH parameters for humanoid arm"""
        # Simplified 7-DOF arm model
        sign = -1 if side == 'left' else 1

        dh_params = [
            {'theta': 0, 'd': 0.1, 'a': 0, 'alpha': -np.pi/2},  # Shoulder joint
            {'theta': 0, 'd': 0, 'a': 0.2, 'alpha': 0},         # Shoulder flexion
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': -np.pi/2},    # Shoulder rotation
            {'theta': 0, 'd': 0.3, 'a': 0, 'alpha': -np.pi/2},  # Elbow joint
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': np.pi/2},     # Elbow rotation
            {'theta': 0, 'd': 0.25, 'a': 0, 'alpha': -np.pi/2}, # Wrist joint
            {'theta': 0, 'd': 0.05, 'a': 0, 'alpha': 0}         # Wrist rotation
        ]
        return dh_params

    def create_leg_dh_params(self, side):
        """Create DH parameters for humanoid leg"""
        # Simplified 6-DOF leg model
        sign = -1 if side == 'left' else 1

        dh_params = [
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': -np.pi/2},    # Hip abduction
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': np.pi/2},     # Hip flexion
            {'theta': 0, 'd': -0.4, 'a': 0, 'alpha': 0},        # Hip rotation
            {'theta': 0, 'd': -0.4, 'a': 0, 'alpha': 0},        # Knee joint
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': 0},           # Ankle pitch
            {'theta': 0, 'd': -0.05, 'a': 0, 'alpha': 0}        # Ankle roll
        ]
        return dh_params

    def calculate_humanoid_pose(self, joint_angles):
        """Calculate full humanoid pose from joint angles"""
        poses = {}

        for chain_name, fk_calc in self.fk_calculators.items():
            # Extract relevant joint angles for this chain
            chain_angles = self.extract_chain_angles(chain_name, joint_angles)

            # Calculate forward kinematics
            pose = fk_calc.calculate_forward_kinematics(chain_angles)
            poses[chain_name] = pose

        return poses

    def extract_chain_angles(self, chain_name, all_angles):
        """Extract joint angles for specific chain"""
        # This would map from full joint angle vector to chain-specific angles
        # Implementation depends on joint ordering in the robot
        chain_map = {
            'left_arm': slice(0, 7),      # Joints 0-6
            'right_arm': slice(7, 14),    # Joints 7-13
            'left_leg': slice(14, 20),    # Joints 14-19
            'right_leg': slice(20, 26),   # Joints 20-25
            'torso': slice(26, 29)        # Joints 26-28
        }

        return all_angles[chain_map[chain_name]]
```

## Inverse Kinematics

### Understanding Inverse Kinematics

Inverse kinematics (IK) calculates the required joint angles to achieve a desired end-effector position and orientation. This is crucial for humanoid robots to perform tasks like reaching, walking, and manipulation.

```python
# Inverse kinematics implementation
class InverseKinematics:
    def __init__(self, dh_parameters):
        self.dh_params = dh_parameters
        self.forward_kin = ForwardKinematics(dh_parameters)

    def jacobian(self, joint_angles):
        """Calculate geometric Jacobian matrix"""
        # Calculate current end-effector position using FK
        fk_result = self.forward_kin.calculate_forward_kinematics(joint_angles)
        current_pos = fk_result['position']

        # Calculate Jacobian columns
        jacobian = np.zeros((6, len(joint_angles)))  # 6 DOF (position + orientation)

        # Get all link positions
        link_positions = self.forward_kin.calculate_link_positions(joint_angles)

        for i in range(len(joint_angles)):
            # Calculate axis of rotation for joint i
            # For revolute joints: axis is along z of joint frame
            transform_to_joint = self.forward_kin.calculate_forward_kinematics(
                joint_angles[:i+1]
            )['all_transforms'][-1] if i > 0 else np.eye(4)

            # Joint axis in world coordinates
            z_axis = transform_to_joint[:3, 2]  # z-axis of joint frame
            joint_pos = link_positions[i]

            # Position vector from joint to end-effector
            r = current_pos - joint_pos

            # Jacobian column for position
            jacobian[:3, i] = np.cross(z_axis, r)

            # Jacobian column for orientation
            jacobian[3:, i] = z_axis

        return jacobian

    def inverse_kinematics_analytical(self, target_pose, initial_angles):
        """Analytical IK solution (for simple chains)"""
        # This is a simplified example for a 2-DOF planar arm
        # Real humanoid IK requires numerical methods

        # Extract target position
        target_pos = target_pose[:3, 3]

        # For a 2-DOF arm: solve for joint angles analytically
        # This is just an example - humanoid robots require more complex solutions
        x, y = target_pos[0], target_pos[1]

        # Calculate distance from base to target
        distance = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        link_lengths = [self.dh_params[0]['a'], self.dh_params[1]['a']]
        max_reach = sum(link_lengths)

        if distance > max_reach:
            # Target is out of reach - return closest possible
            scale = max_reach / distance
            target_pos = np.array([x * scale, y * scale, target_pos[2]])
            distance = max_reach

        # Analytical solution for 2-DOF arm
        # (This is simplified and wouldn't work for complex humanoid chains)
        pass

    def inverse_kinematics_numerical(self, target_pose, initial_angles,
                                   max_iterations=100, tolerance=1e-6):
        """Numerical IK solution using Jacobian transpose/pseudo-inverse"""
        current_angles = np.array(initial_angles, dtype=float)

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            current_pose = self.forward_kin.calculate_forward_kinematics(
                current_angles
            )

            # Calculate error
            pos_error = target_pose[:3, 3] - current_pose['position']
            rot_error = self.rotation_error(
                target_pose[:3, :3],
                current_pose['orientation']
            )

            # Combine position and rotation errors
            error = np.concatenate([pos_error, rot_error])

            # Check convergence
            if np.linalg.norm(error) < tolerance:
                return current_angles, True  # Success

            # Calculate Jacobian
            jacobian = self.jacobian(current_angles)

            # Use damped least squares (Levenberg-Marquardt) for stability
            damping = 0.01
            j_pinv = np.linalg.pinv(jacobian + damping * np.eye(jacobian.shape[0]))

            # Calculate joint angle update
            delta_theta = j_pinv @ error

            # Update joint angles
            current_angles += delta_theta

        return current_angles, False  # Failed to converge

    def rotation_error(self, target_rot, current_rot):
        """Calculate rotation error between two rotation matrices"""
        # Use rotation vector representation
        target_r = R.from_matrix(target_rot)
        current_r = R.from_matrix(current_rot)

        # Calculate relative rotation
        relative_r = target_r * current_r.inv()

        # Get rotation vector
        rot_vec = relative_r.as_rotvec()

        return rot_vec
```

### Humanoid-Specific Inverse Kinematics

Humanoid robots require specialized IK approaches due to their complex structure:

```python
# Humanoid-specific inverse kinematics
class HumanoidIK:
    def __init__(self):
        self.chains = HumanoidFK()
        self.ik_solvers = {}

        # Initialize IK solvers for each chain
        for chain_name, dh_params in self.chains.chains.items():
            self.ik_solvers[chain_name] = InverseKinematics(dh_params)

    def solve_arm_ik(self, arm_side, target_pose, current_angles):
        """Solve inverse kinematics for specific arm"""
        chain_name = f"{arm_side}_arm"
        ik_solver = self.ik_solvers[chain_name]

        # Extract current angles for this chain
        chain_angles = self.chains.extract_chain_angles(chain_name, current_angles)

        # Solve IK
        solution, success = ik_solver.inverse_kinematics_numerical(
            target_pose, chain_angles
        )

        return solution, success

    def solve_leg_ik(self, leg_side, target_pose, current_angles):
        """Solve inverse kinematics for specific leg"""
        chain_name = f"{leg_side}_leg"
        ik_solver = self.ik_solvers[chain_name]

        # Extract current angles for this chain
        chain_angles = self.chains.extract_chain_angles(chain_name, current_angles)

        # Solve IK
        solution, success = ik_solver.inverse_kinematics_numerical(
            target_pose, chain_angles
        )

        return solution, success

    def whole_body_ik(self, targets, current_angles, weights=None):
        """Solve whole-body inverse kinematics with multiple targets"""
        if weights is None:
            weights = {'left_arm': 1.0, 'right_arm': 1.0, 'left_leg': 1.0, 'right_leg': 1.0}

        # This would implement a more complex optimization approach
        # considering all targets simultaneously
        # Common approaches: prioritized IK, optimization-based IK

        # Simplified approach: solve each chain separately with coordination
        result_angles = current_angles.copy()

        for chain_name, target_pose in targets.items():
            if chain_name in self.ik_solvers:
                ik_solver = self.ik_solvers[chain_name]

                # Extract chain-specific angles
                chain_slice = self.get_chain_slice(chain_name)
                chain_angles = current_angles[chain_slice]

                # Solve IK for this chain
                new_chain_angles, success = ik_solver.inverse_kinematics_numerical(
                    target_pose, chain_angles
                )

                if success:
                    result_angles[chain_slice] = new_chain_angles

        return result_angles

    def get_chain_slice(self, chain_name):
        """Get slice for extracting chain-specific angles"""
        chain_map = {
            'left_arm': slice(0, 7),
            'right_arm': slice(7, 14),
            'left_leg': slice(14, 20),
            'right_leg': slice(20, 26),
            'torso': slice(26, 29)
        }
        return chain_map[chain_name]

    def balance_aware_ik(self, targets, current_angles, com_target=None):
        """Solve IK while maintaining balance"""
        # This would integrate center of mass considerations
        # with IK solution to maintain stable balance

        # First solve basic IK
        ik_solution = self.whole_body_ik(targets, current_angles)

        # Then adjust for balance if needed
        if com_target is not None:
            # Calculate current CoM
            current_com = self.calculate_center_of_mass(ik_solution)

            # Adjust solution to move CoM toward target
            adjusted_solution = self.adjust_for_balance(
                ik_solution, current_com, com_target
            )

            return adjusted_solution

        return ik_solution

    def calculate_center_of_mass(self, joint_angles):
        """Calculate center of mass given joint configuration"""
        # This would use robot's mass properties and current configuration
        # to calculate CoM position
        pass

    def adjust_for_balance(self, current_config, current_com, target_com):
        """Adjust configuration to improve balance"""
        # This would use optimization to adjust joint angles
        # while maintaining task requirements and improving balance
        pass
```

## Dynamic Modeling

### Understanding Robot Dynamics

Robot dynamics describes the relationship between forces acting on a robot and the resulting motion. For humanoid robots, dynamics are crucial for stable locomotion and manipulation.

```python
# Rigid body dynamics for humanoid robots
class RigidBodyDynamics:
    def __init__(self, mass, inertia_tensor):
        self.mass = mass
        self.inertia_tensor = np.array(inertia_tensor)  # 3x3 matrix
        self.inertia_tensor_inv = np.linalg.inv(self.inertia_tensor)

    def rigid_body_equation(self, pose, twist, wrench):
        """Calculate acceleration from wrench (force/torque)"""
        # pose: [position, orientation]
        # twist: [linear_velocity, angular_velocity]
        # wrench: [force, torque]

        linear_force = wrench[:3]
        torque = wrench[3:]

        # Linear acceleration (F = ma)
        linear_accel = linear_force / self.mass

        # Angular acceleration (tau = I*alpha + omega x (I*omega))
        angular_velocity = twist[3:]
        inertia_omega = self.inertia_tensor @ angular_velocity
        angular_accel = (torque - np.cross(angular_velocity, inertia_omega)) @ self.inertia_tensor_inv

        acceleration = np.concatenate([linear_accel, angular_accel])

        return acceleration

class HumanoidDynamics:
    def __init__(self, robot_description):
        self.links = robot_description['links']
        self.joints = robot_description['joints']
        self.mass_properties = self.calculate_mass_properties()

    def calculate_mass_properties(self):
        """Calculate mass properties for each link"""
        mass_props = {}

        for link_name, link_data in self.links.items():
            mass_props[link_name] = {
                'mass': link_data.get('mass', 1.0),
                'com': np.array(link_data.get('center_of_mass', [0, 0, 0])),
                'inertia': np.array(link_data.get('inertia', np.eye(3)))
            }

        return mass_props

    def euler_lagrange_dynamics(self, joint_positions, joint_velocities, joint_torques):
        """Calculate joint accelerations using Euler-Lagrange formulation"""
        # M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau
        # where:
        # M(q) = mass matrix
        # C(q, q_dot) = Coriolis and centrifugal forces
        # g(q) = gravity forces
        # tau = joint torques

        M = self.mass_matrix(joint_positions)
        C = self.coriolis_matrix(joint_positions, joint_velocities)
        g = self.gravity_vector(joint_positions)

        # Solve: M * q_ddot = tau - C * q_dot - g
        q_ddot = np.linalg.solve(M, joint_torques - C @ joint_velocities - g)

        return q_ddot

    def mass_matrix(self, joint_positions):
        """Calculate mass matrix M(q)"""
        # This would use composite rigid body algorithm or other methods
        # to calculate the mass matrix
        n = len(joint_positions)
        M = np.zeros((n, n))

        # Simplified calculation - in practice this involves
        # complex recursive algorithms
        for i in range(n):
            for j in range(n):
                M[i, j] = self.calculate_inertial_coupling(i, j, joint_positions)

        return M

    def coriolis_matrix(self, joint_positions, joint_velocities):
        """Calculate Coriolis and centrifugal matrix C(q, q_dot)"""
        # This matrix accounts for velocity-dependent forces
        n = len(joint_positions)
        C = np.zeros((n, n))

        # Calculate Christoffel symbols and form C matrix
        # This is a complex calculation involving partial derivatives
        # of the mass matrix
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    christoffel = self.christoffel_symbol(i, j, k, joint_positions)
                    C[i, j] += christoffel * joint_velocities[k]

        return C

    def gravity_vector(self, joint_positions):
        """Calculate gravity vector g(q)"""
        # Calculate gravity effects in joint space
        n = len(joint_positions)
        g = np.zeros(n)

        # This would involve transforming gravitational forces
        # through the kinematic chain
        return g

    def christoffel_symbol(self, i, j, k, q):
        """Calculate Christoffel symbol for dynamic equations"""
        # Christoffel symbols are calculated from partial derivatives
        # of the mass matrix
        # Gamma^i_jk = 0.5 * (dM_ik/dq_j + dM_jk/dq_i - dM_ij/dq_k)

        # This is a simplified placeholder
        return 0.0

    def calculate_inertial_coupling(self, i, j, q):
        """Calculate inertial coupling between joints i and j"""
        # This would involve complex calculations based on
        # robot kinematics and mass distribution
        return 0.0 if i != j else 1.0  # Simplified
```

### Bipedal Dynamics and Balance

Humanoid robots have unique dynamic considerations for bipedal locomotion:

```python
# Bipedal dynamics and balance control
class BipedalDynamics:
    def __init__(self):
        self.gravity = 9.81
        self.total_mass = 70.0  # kg, approximate humanoid mass
        self.com_height = 0.8  # m, approximate CoM height
        self.step_length = 0.6  # m, typical step length

    def linear_inverted_pendulum_model(self, com_position, com_velocity, zmp_position):
        """Linear Inverted Pendulum Model for balance"""
        # LIPM: x_ddot = omega^2 * (x - x_zmp)
        # where omega^2 = g / h (h = CoM height)

        omega_sq = self.gravity / self.com_height

        # Calculate desired CoM acceleration for balance
        com_acceleration = omega_sq * (com_position - zmp_position)

        return com_acceleration

    def zero_moment_point(self, com_position, com_velocity, com_acceleration):
        """Calculate Zero Moment Point (ZMP) for balance"""
        # ZMP = CoM position - g/CoM_ddot * (CoM_height - foot_height)
        # Simplified for 2D case

        zmp_x = com_position[0] - (self.gravity / com_acceleration[0]) * (
            self.com_height - 0  # assuming foot is at ground level
        )

        zmp_y = com_position[1] - (self.gravity / com_acceleration[1]) * (
            self.com_height - 0
        )

        return np.array([zmp_x, zmp_y, 0])

    def capture_point(self, com_position, com_velocity):
        """Calculate Capture Point for balance recovery"""
        # Capture Point = CoM position + CoM velocity / omega
        # where omega = sqrt(g / CoM_height)

        omega = np.sqrt(self.gravity / self.com_height)

        capture_point = com_position + com_velocity / omega

        return capture_point

    def balance_controller(self, current_com, desired_com, current_zmp, support_polygon):
        """Balance controller using ZMP and Capture Point concepts"""
        # Calculate error
        com_error = desired_com - current_com
        zmp_error = self.calculate_zmp_error(current_zmp, support_polygon)

        # PID control for balance
        kp = 10.0  # Proportional gain
        ki = 1.0   # Integral gain
        kd = 5.0   # Derivative gain

        # Calculate control output
        control_output = kp * com_error + kd * zmp_error

        return control_output

    def calculate_zmp_error(self, current_zmp, support_polygon):
        """Calculate ZMP error relative to support polygon"""
        # Check if ZMP is within support polygon
        if self.is_zmp_in_support_polygon(current_zmp, support_polygon):
            return np.zeros(2)  # No error if in support

        # Calculate closest point in support polygon
        closest_point = self.closest_point_in_polygon(current_zmp, support_polygon)
        zmp_error = closest_point - current_zmp[:2]

        return zmp_error

    def is_zmp_in_support_polygon(self, zmp, polygon):
        """Check if ZMP is within support polygon"""
        # Use ray casting algorithm or other point-in-polygon test
        return self.point_in_polygon(zmp[:2], polygon)

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

    def closest_point_in_polygon(self, point, polygon):
        """Find closest point on polygon boundary to given point"""
        # This would find the closest point on the polygon boundary
        # to the given ZMP position
        pass
```

## Control System Implementation

### PD Control for Humanoid Robots

```python
# Control systems for humanoid robots
class HumanoidController:
    def __init__(self):
        self.joint_controllers = {}
        self.balance_controller = BipedalDynamics()
        self.trajectory_generators = {}

    def setup_joint_controller(self, joint_name, kp=100, ki=0, kd=10):
        """Setup PD controller for specific joint"""
        self.joint_controllers[joint_name] = {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'prev_error': 0,
            'integral_error': 0
        }

    def compute_joint_control(self, joint_name, desired_position, current_position,
                            desired_velocity=0, current_velocity=0, dt=0.001):
        """Compute control output for single joint"""
        controller = self.joint_controllers[joint_name]

        # Calculate errors
        position_error = desired_position - current_position
        velocity_error = desired_velocity - current_velocity

        # Update integral (with anti-windup)
        controller['integral_error'] += position_error * dt
        max_integral = 10.0  # Anti-windup limit
        controller['integral_error'] = np.clip(
            controller['integral_error'], -max_integral, max_integral
        )

        # Calculate derivatives
        derivative_error = (position_error - controller['prev_error']) / dt
        controller['prev_error'] = position_error

        # Compute control output
        p_term = controller['kp'] * position_error
        i_term = controller['ki'] * controller['integral_error']
        d_term = controller['kd'] * (velocity_error + derivative_error)

        control_output = p_term + i_term + d_term

        return control_output

    def operational_space_control(self, task_jacobian, task_desired, task_current):
        """Operational space control for task-space control"""
        # Calculate task error
        task_error = task_desired - task_current

        # Calculate joint velocities using Jacobian pseudo-inverse
        # q_dot = J# * x_dot + (I - J# * J) * q_dot_null
        # where J# is pseudo-inverse of Jacobian

        j_pinv = np.linalg.pinv(task_jacobian)
        joint_velocities = j_pinv @ task_error

        return joint_velocities

    def inverse_dynamics_control(self, joint_positions, joint_velocities,
                               joint_accelerations, gravity_compensation=True):
        """Compute required joint torques using inverse dynamics"""
        # Use recursive Newton-Euler algorithm or other inverse dynamics methods
        # to calculate required joint torques

        # This would implement the full inverse dynamics equation:
        # tau = M(q) * q_ddot + C(q, q_dot) * q_dot + g(q)

        # Simplified version
        dynamics_model = HumanoidDynamics({})  # Would use actual robot description
        coriolis_centrifugal = dynamics_model.coriolis_matrix(
            joint_positions, joint_velocities
        ) @ joint_velocities
        gravity_effects = dynamics_model.gravity_vector(joint_positions)
        inertial_effects = dynamics_model.mass_matrix(joint_positions) @ joint_accelerations

        required_torques = inertial_effects + coriolis_centrifugal + gravity_effects

        return required_torques
```

## Practical Implementation Examples

### Walking Pattern Generation

```python
# Walking pattern generation using kinematics and dynamics
class WalkingPatternGenerator:
    def __init__(self):
        self.step_height = 0.05  # m
        self.step_length = 0.6   # m
        self.step_duration = 1.0 # s
        self.zmp_reference = np.array([0.0, 0.0])  # Desired ZMP

    def generate_walk_trajectory(self, num_steps, step_width=0.2):
        """Generate complete walking trajectory"""
        trajectory = []

        for step in range(num_steps):
            # Generate double support phase
            dsp_trajectory = self.generate_double_support_phase()

            # Generate single support phase (left foot support)
            if step % 2 == 0:  # Left foot support for even steps
                ssp_trajectory = self.generate_single_support_phase(
                    'left', step_width
                )
            else:  # Right foot support for odd steps
                ssp_trajectory = self.generate_single_support_phase(
                    'right', step_width
                )

            trajectory.extend(dsp_trajectory)
            trajectory.extend(ssp_trajectory)

        return trajectory

    def generate_double_support_phase(self, duration=0.1):
        """Generate double support phase trajectory"""
        # Both feet on ground - transfer weight
        steps = int(duration / 0.01)  # Assuming 100Hz control
        phase_trajectory = []

        for i in range(steps):
            t = i / steps  # Normalized time (0 to 1)

            # Smooth weight transfer from one foot to another
            # This would generate CoM trajectory, ZMP trajectory, etc.
            phase_trajectory.append({
                'time': t,
                'com_position': self.calculate_com_trajectory(t),
                'zmp_position': self.calculate_zmp_trajectory(t),
                'joint_angles': self.calculate_joint_trajectory(t)
            })

        return phase_trajectory

    def generate_single_support_phase(self, support_foot, step_width):
        """Generate single support phase trajectory"""
        steps = int(self.step_duration / 0.01)
        phase_trajectory = []

        for i in range(steps):
            t = i / steps  # Normalized time

            # Calculate CoM trajectory following inverted pendulum model
            com_x = self.calculate_swing_foot_position(t, support_foot)
            com_y = self.balance_lateral_motion(t, support_foot, step_width)
            com_z = self.step_height_profile(t)  # Maintain CoM height

            phase_trajectory.append({
                'time': t,
                'com_position': np.array([com_x, com_y, com_z]),
                'support_foot': support_foot,
                'swing_foot': 'right' if support_foot == 'left' else 'left'
            })

        return phase_trajectory

    def calculate_swing_foot_position(self, t, support_foot):
        """Calculate swing foot position during walking"""
        # Calculate where the swing foot should be at time t
        if support_foot == 'left':
            # Right foot is swinging forward
            swing_pos = self.step_length * t  # Linear progression
        else:
            # Left foot is swinging forward
            swing_pos = self.step_length * t

        return swing_pos

    def balance_lateral_motion(self, t, support_foot, step_width):
        """Calculate lateral CoM motion for balance"""
        # Shift CoM over support foot
        if support_foot == 'left':
            target_y = step_width / 2  # CoM over left foot
        else:
            target_y = -step_width / 2  # CoM over right foot

        # Smooth transition using sine function
        smooth_factor = np.sin(t * np.pi)  # 0 to 1 to 0
        return target_y * smooth_factor

    def step_height_profile(self, t):
        """Calculate step height profile for swing foot"""
        # Create parabolic step trajectory
        height_factor = 4 * t * (1 - t)  # Parabolic: 0->1->0
        return self.com_height + self.step_height * height_factor
```

## Simulation and Validation

### Dynamics Simulation

```python
# Dynamics simulation for humanoid robots
class HumanoidDynamicsSimulator:
    def __init__(self, robot_description):
        self.dynamics_model = HumanoidDynamics(robot_description)
        self.integration_dt = 0.001  # 1ms integration step

    def simulate_step(self, current_state, joint_torques):
        """Simulate one time step of robot dynamics"""
        # Extract state variables
        joint_positions = current_state['joint_positions']
        joint_velocities = current_state['joint_velocities']

        # Calculate joint accelerations using inverse dynamics
        joint_accelerations = self.dynamics_model.euler_lagrange_dynamics(
            joint_positions, joint_velocities, joint_torques
        )

        # Integrate to get new velocities and positions
        new_velocities = joint_velocities + joint_accelerations * self.integration_dt
        new_positions = joint_positions + new_velocities * self.integration_dt

        # Calculate forward kinematics for end-effector positions
        fk_calculator = HumanoidFK()
        end_effector_poses = fk_calculator.calculate_humanoid_pose(new_positions)

        # Calculate center of mass
        com_position = self.calculate_com_position(new_positions)

        # Calculate ZMP
        zmp_position = self.calculate_zmp(com_position, new_positions, new_velocities)

        new_state = {
            'joint_positions': new_positions,
            'joint_velocities': new_velocities,
            'end_effector_poses': end_effector_poses,
            'com_position': com_position,
            'zmp_position': zmp_position
        }

        return new_state

    def calculate_com_position(self, joint_positions):
        """Calculate center of mass position"""
        # This would use the robot's mass properties and kinematics
        # to calculate the overall center of mass
        pass

    def calculate_zmp(self, com_position, joint_positions, joint_velocities):
        """Calculate Zero Moment Point"""
        # Use the dynamic equations to calculate ZMP
        # ZMP = CoM - (g / CoM_z_ddot) * (CoM - foot_position)
        pass

    def validate_stability(self, state_trajectory):
        """Validate stability of motion trajectory"""
        stability_metrics = {
            'zmp_in_support': [],
            'com_bounded': [],
            'energy_consumption': []
        }

        for state in state_trajectory:
            # Check if ZMP remains in support polygon
            zmp_in_support = self.is_zmp_stable(state['zmp_position'])
            stability_metrics['zmp_in_support'].append(zmp_in_support)

            # Check if CoM remains bounded
            com_bounded = self.is_com_stable(state['com_position'])
            stability_metrics['com_bounded'].append(com_bounded)

            # Calculate energy consumption
            energy = self.calculate_energy(state)
            stability_metrics['energy_consumption'].append(energy)

        return stability_metrics

    def is_zmp_stable(self, zmp_position):
        """Check if ZMP is within stable region"""
        # This would check against support polygon
        return True  # Simplified

    def is_com_stable(self, com_position):
        """Check if CoM is within stable bounds"""
        # Check if CoM is within reasonable bounds
        return True  # Simplified

    def calculate_energy(self, state):
        """Calculate energy consumption"""
        # Calculate kinetic and potential energy
        # This would involve joint velocities and positions
        pass
```

## Hands-On Exercise: Implementing Humanoid Kinematics and Dynamics

### Exercise Objectives
- Implement forward and inverse kinematics for a simplified humanoid model
- Create dynamic model and simulate basic movements
- Validate stability using ZMP and balance criteria
- Analyze the relationship between kinematics and dynamics

### Step-by-Step Instructions

1. **Create a simplified humanoid model** with basic kinematic chains
2. **Implement forward kinematics** for arm and leg movements
3. **Develop inverse kinematics** solver for reaching tasks
4. **Create dynamic model** using Euler-Lagrange formulation
5. **Simulate basic movements** like reaching and stepping
6. **Analyze stability** using ZMP and CoM criteria
7. **Validate results** against expected behavior

### Expected Outcomes
- Working kinematics implementation
- Dynamic simulation capability
- Understanding of stability criteria
- Practical experience with humanoid control

## Knowledge Check

1. What are the key differences between forward and inverse kinematics?
2. Explain the concept of Zero Moment Point (ZMP) in humanoid balance.
3. How does the Linear Inverted Pendulum Model simplify balance control?
4. What are the main challenges in humanoid robot dynamics?

## Summary

This chapter covered the fundamental kinematics and dynamics principles essential for humanoid robot control. Understanding both forward and inverse kinematics, along with dynamic modeling, is crucial for creating stable and capable humanoid robots. The integration of kinematic solutions with dynamic considerations enables the complex movements required for bipedal locomotion and human-like manipulation.

## Next Steps

In Chapter 14, we'll explore bipedal locomotion and balance control in detail, building upon the kinematics and dynamics foundation established here.

