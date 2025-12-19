---
sidebar_position: 12
title: "Chapter 12: Sim-to-Real Transfer Techniques"
---

# Chapter 12: Sim-to-Real Transfer Techniques

## Learning Objectives
- Understand the principles of sim-to-real transfer in robotics
- Identify challenges in simulation-to-reality deployment
- Implement domain randomization techniques
- Apply best practices for successful transfer

## Introduction to Sim-to-Real Transfer

Sim-to-real transfer, also known as domain transfer, is the process of transferring skills, behaviors, or policies learned in simulation to real-world robotic systems. This is a critical challenge in robotics, as training in the real world is often expensive, time-consuming, and potentially dangerous, while simulation provides a safe and efficient alternative for initial development.

### The Reality Gap

The "reality gap" refers to the differences between simulated and real environments that can cause policies learned in simulation to fail when deployed on real robots:

- **Visual Differences**: Lighting, textures, and rendering differences
- **Physical Differences**: Mass, friction, and dynamics variations
- **Sensor Differences**: Noise, latency, and accuracy variations
- **Actuator Differences**: Response time and precision variations
- **Environmental Differences**: Unmodeled objects and disturbances

### Why Sim-to-Real Transfer Matters

1. **Safety**: Train dangerous behaviors in simulation first
2. **Efficiency**: Faster training in simulation than in reality
3. **Cost**: Reduced wear and tear on real hardware
4. **Scalability**: Train on multiple simulated robots simultaneously
5. **Reproducibility**: Controlled experimental conditions

## Domain Randomization

### Concept and Implementation

Domain randomization is a technique that randomizes various aspects of the simulation environment to make policies robust to differences between simulation and reality.

```python
# domain_randomization.py
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DomainParams:
    """Parameters for domain randomization"""
    # Visual parameters
    lighting_intensity_range: tuple = (0.5, 2.0)
    color_variance: float = 0.1
    texture_randomization: bool = True

    # Physical parameters
    mass_variance: float = 0.1
    friction_range: tuple = (0.1, 1.0)
    damping_range: tuple = (0.01, 0.1)

    # Sensor parameters
    noise_std: float = 0.01
    delay_range: tuple = (0.0, 0.05)

    # Environmental parameters
    gravity_variance: float = 0.01
    wind_force_range: tuple = (-0.1, 0.1)

class DomainRandomizer:
    def __init__(self, params: DomainParams = None):
        self.params = params or DomainParams()
        self.current_params = self.randomize_domain()

    def randomize_domain(self) -> Dict[str, Any]:
        """Generate random domain parameters"""
        randomized_params = {}

        # Visual randomization
        randomized_params['lighting'] = np.random.uniform(
            self.params.lighting_intensity_range[0],
            self.params.lighting_intensity_range[1]
        )

        randomized_params['color_offset'] = np.random.uniform(
            -self.params.color_variance,
            self.params.color_variance,
            size=3
        )

        # Physical randomization
        randomized_params['mass_multiplier'] = 1.0 + np.random.uniform(
            -self.params.mass_variance,
            self.params.mass_variance
        )

        randomized_params['friction'] = np.random.uniform(
            self.params.friction_range[0],
            self.params.friction_range[1]
        )

        randomized_params['damping'] = np.random.uniform(
            self.params.damping_range[0],
            self.params.damping_range[1]
        )

        # Sensor randomization
        randomized_params['sensor_noise'] = np.random.uniform(
            0,
            self.params.noise_std
        )

        randomized_params['sensor_delay'] = np.random.uniform(
            self.params.delay_range[0],
            self.params.delay_range[1]
        )

        # Environmental randomization
        randomized_params['gravity_offset'] = np.random.uniform(
            -self.params.gravity_variance,
            self.params.gravity_variance
        )

        randomized_params['wind_force'] = np.random.uniform(
            self.params.wind_force_range[0],
            self.params.wind_force_range[1],
            size=3
        )

        return randomized_params

    def apply_to_simulation(self, sim_env):
        """Apply randomized parameters to simulation environment"""
        # Apply visual changes
        sim_env.set_lighting_intensity(self.current_params['lighting'])
        sim_env.add_color_offset(self.current_params['color_offset'])

        # Apply physical changes
        sim_env.set_mass_multiplier(self.current_params['mass_multiplier'])
        sim_env.set_friction(self.current_params['friction'])
        sim_env.set_damping(self.current_params['damping'])

        # Apply sensor changes
        sim_env.set_sensor_noise(self.current_params['sensor_noise'])
        sim_env.set_sensor_delay(self.current_params['sensor_delay'])

        # Apply environmental changes
        sim_env.set_gravity_offset(self.current_params['gravity_offset'])
        sim_env.set_wind_force(self.current_params['wind_force'])

    def update_domain(self):
        """Update domain parameters during training"""
        self.current_params = self.randomize_domain()
        return self.current_params

# Example usage in training loop
def train_with_domain_randomization(env, agent, episodes=1000):
    randomizer = DomainRandomizer()

    for episode in range(episodes):
        # Randomize domain every few episodes
        if episode % 10 == 0:
            randomizer.update_domain()

        # Apply current domain to simulation
        randomizer.apply_to_simulation(env)

        # Train agent in randomized environment
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
```

### Advanced Domain Randomization Techniques

```python
# advanced_domain_randomization.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

class TextureRandomizer:
    """Randomize textures and materials in simulation"""

    def __init__(self, texture_library_path):
        self.texture_library = self.load_texture_library(texture_library_path)
        self.current_textures = {}

    def load_texture_library(self, path):
        """Load texture library from path"""
        # In practice, this would load textures from a directory
        return ["texture1.jpg", "texture2.jpg", "texture3.jpg"]  # Placeholder

    def randomize_object_texture(self, object_id):
        """Apply random texture to object"""
        texture = np.random.choice(self.texture_library)
        # Apply texture to object in simulation
        return texture

    def add_random_backgrounds(self, image):
        """Add random backgrounds to training images"""
        # Apply random background to simulate different environments
        if np.random.random() < 0.3:  # 30% chance of background change
            # Add random background pattern
            background = np.random.randint(0, 255, size=image.shape, dtype=np.uint8)
            # Blend with original image
            alpha = np.random.uniform(0.1, 0.3)
            image = cv2.addWeighted(image, 1-alpha, background, alpha, 0)

        return image

class DynamicsRandomizer:
    """Randomize robot dynamics parameters"""

    def __init__(self):
        self.base_dynamics = {
            'mass': 50.0,  # kg
            'inertia': [1.0, 1.0, 1.0],  # kg*m^2
            'friction': 0.1,
            'damping': 0.05
        }

    def randomize_dynamics(self):
        """Generate randomized dynamics parameters"""
        randomized = {}

        # Randomize mass (±20%)
        randomized['mass'] = self.base_dynamics['mass'] * np.random.uniform(0.8, 1.2)

        # Randomize inertia (±30%)
        randomized['inertia'] = [
            i * np.random.uniform(0.7, 1.3) for i in self.base_dynamics['inertia']
        ]

        # Randomize friction (±50%)
        randomized['friction'] = self.base_dynamics['friction'] * np.random.uniform(0.5, 1.5)

        # Randomize damping (±50%)
        randomized['damping'] = self.base_dynamics['damping'] * np.random.uniform(0.5, 1.5)

        return randomized

    def apply_dynamics_to_robot(self, robot, dynamics_params):
        """Apply dynamics parameters to robot model"""
        robot.set_mass(dynamics_params['mass'])
        robot.set_inertia(dynamics_params['inertia'])
        robot.set_friction(dynamics_params['friction'])
        robot.set_damping(dynamics_params['damping'])

class SensorRandomizer:
    """Randomize sensor characteristics"""

    def __init__(self):
        self.base_noise_params = {
            'imu': {'std': 0.01, 'bias': 0.001},
            'camera': {'noise': 0.02, 'distortion': 0.1},
            'lidar': {'noise': 0.05, 'dropout': 0.01}
        }

    def randomize_sensor_params(self):
        """Generate randomized sensor parameters"""
        randomized = {}

        for sensor_type, params in self.base_noise_params.items():
            randomized[sensor_type] = {}
            for param_name, base_value in params.items():
                # Randomize by ±50%
                randomized[sensor_type][param_name] = base_value * np.random.uniform(0.5, 1.5)

        return randomized

    def add_sensor_noise(self, sensor_data, sensor_type, noise_params):
        """Add noise to sensor data"""
        if sensor_type == 'imu':
            # Add Gaussian noise and bias
            noise = np.random.normal(0, noise_params['std'], size=sensor_data.shape)
            bias = np.random.uniform(-noise_params['bias'], noise_params['bias'])
            return sensor_data + noise + bias

        elif sensor_type == 'camera':
            # Add noise to camera image
            noise = np.random.normal(0, noise_params['noise'], size=sensor_data.shape)
            return sensor_data + noise

        elif sensor_type == 'lidar':
            # Add noise and occasional dropouts to LIDAR
            noise = np.random.normal(0, noise_params['noise'], size=sensor_data.shape)
            data_with_noise = sensor_data + noise

            # Apply dropouts
            dropout_mask = np.random.random(size=sensor_data.shape) < noise_params['dropout']
            data_with_noise[dropout_mask] = 0  # Zero out dropped measurements

            return data_with_noise

        return sensor_data

# System-level domain randomization
class SystemRandomizer:
    def __init__(self):
        self.texture_randomizer = TextureRandomizer("textures/")
        self.dynamics_randomizer = DynamicsRandomizer()
        self.sensor_randomizer = SensorRandomizer()

    def full_randomization_step(self, sim_env):
        """Apply full system randomization"""
        # Randomize textures
        for obj_id in sim_env.get_objects():
            self.texture_randomizer.randomize_object_texture(obj_id)

        # Randomize dynamics
        dynamics_params = self.dynamics_randomizer.randomize_dynamics()
        self.dynamics_randomizer.apply_dynamics_to_robot(
            sim_env.get_robot(), dynamics_params
        )

        # Randomize sensors
        sensor_params = self.sensor_randomizer.randomize_sensor_params()

        return {
            'dynamics': dynamics_params,
            'sensors': sensor_params
        }
```

## System Identification and Parameter Estimation

### Identifying Real-World Parameters

```python
# system_identification.py
import numpy as np
from scipy.optimize import minimize
from scipy import signal
import matplotlib.pyplot as plt

class SystemIdentifier:
    """Identify real-world system parameters through excitation"""

    def __init__(self, robot):
        self.robot = robot
        self.excitation_signals = []
        self.measurements = []

    def generate_excitation_signal(self, duration=10.0, freq_range=(0.1, 5.0)):
        """Generate multi-frequency excitation signal"""
        t = np.linspace(0, duration, int(duration * 100))  # 100Hz sampling

        # Generate random signal with multiple frequencies
        signal = np.zeros_like(t)
        frequencies = np.random.uniform(freq_range[0], freq_range[1], 5)

        for freq in frequencies:
            amplitude = np.random.uniform(0.1, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            signal += amplitude * np.sin(2*np.pi*freq*t + phase)

        # Apply low-pass filter to avoid excessive high-frequency content
        b, a = signal.butter(4, 0.1, 'low')
        signal = signal.filtfilt(b, a, signal)

        # Normalize to safe amplitude range
        signal = signal / np.max(np.abs(signal)) * 0.5  # Limit to ±0.5

        return t, signal

    def excite_system(self, joint_name, excitation_signal, time_vector):
        """Apply excitation signal to robot joint and collect data"""
        self.excitation_signals = []
        self.measurements = []

        for t, u in zip(time_vector, excitation_signal):
            # Apply control signal
            self.robot.set_joint_torque(joint_name, u)

            # Collect measurements
            position = self.robot.get_joint_position(joint_name)
            velocity = self.robot.get_joint_velocity(joint_name)
            torque = self.robot.get_joint_torque(joint_name)

            self.excitation_signals.append(u)
            self.measurements.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'torque': torque
            })

    def estimate_model_parameters(self, model_type='second_order'):
        """Estimate system parameters from excitation data"""
        if model_type == 'second_order':
            return self.estimate_second_order_model()
        elif model_type == 'first_order':
            return self.estimate_first_order_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def estimate_second_order_model(self):
        """Estimate second-order system parameters: J*ddq + B*dq + K*q = T"""
        # Convert to numpy arrays
        times = np.array([m['time'] for m in self.measurements])
        positions = np.array([m['position'] for m in self.measurements])
        velocities = np.array([m['velocity'] for m in self.measurements])
        torques = np.array([m['torque'] for m in self.measurements])

        # Estimate accelerations using finite differences
        dt = np.diff(times)
        accelerations = np.diff(velocities) / dt
        accelerations = np.append(accelerations, accelerations[-1])  # Pad to match length

        # Set up regression problem: J*ddq + B*dq + K*q = T
        # We want to solve for [J, B, K] in Ax = b where:
        # A = [ddq, dq, q] and b = T

        A = np.column_stack([accelerations, velocities, positions])
        b = torques

        # Solve using least squares
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        J, B, K = params  # Inertia, damping, stiffness

        return {
            'inertia': J,
            'damping': B,
            'stiffness': K,
            'residuals': residuals,
            'condition_number': np.linalg.cond(A)
        }

    def estimate_first_order_model(self):
        """Estimate first-order system parameters: T*dy/dt + y = K*u"""
        # Similar approach but for first-order systems
        times = np.array([m['time'] for m in self.measurements])
        outputs = np.array([m['position'] for m in self.measurements])
        inputs = np.array(self.excitation_signals)

        # Estimate derivatives
        dt = np.diff(times)
        output_rates = np.diff(outputs) / dt
        output_rates = np.append(output_rates, output_rates[-1])

        # Set up regression: T*dy/dt + y = K*u
        A = np.column_stack([output_rates, outputs])
        b = inputs

        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        T_inv, K = params  # Time constant inverse, gain
        T = 1.0 / T_inv if T_inv != 0 else float('inf')

        return {
            'time_constant': T,
            'gain': K,
            'residuals': residuals
        }

class ParameterAdaptation:
    """Adapt simulation parameters based on real-world data"""

    def __init__(self, sim_env, real_robot):
        self.sim_env = sim_env
        self.real_robot = real_robot
        self.sim_params = self.get_sim_parameters()
        self.real_params = {}

    def get_sim_parameters(self):
        """Get current simulation parameters"""
        return {
            'mass': self.sim_env.get_robot_mass(),
            'friction': self.sim_env.get_robot_friction(),
            'damping': self.sim_env.get_robot_damping(),
            'inertia': self.sim_env.get_robot_inertia()
        }

    def update_sim_to_match_real(self, real_params):
        """Update simulation parameters to match real robot"""
        # Calculate parameter differences
        param_diffs = {}
        for param_name, real_value in real_params.items():
            if param_name in self.sim_params:
                sim_value = self.sim_params[param_name]
                diff = real_value - sim_value
                param_diffs[param_name] = diff

        # Apply corrections to simulation
        for param_name, diff in param_diffs.items():
            new_value = self.sim_params[param_name] + diff
            self.apply_parameter_correction(param_name, new_value)

    def apply_parameter_correction(self, param_name, new_value):
        """Apply parameter correction to simulation"""
        if param_name == 'mass':
            self.sim_env.set_robot_mass(new_value)
        elif param_name == 'friction':
            self.sim_env.set_robot_friction(new_value)
        elif param_name == 'damping':
            self.sim_env.set_robot_damping(new_value)
        elif param_name == 'inertia':
            self.sim_env.set_robot_inertia(new_value)

    def iterative_parameter_tuning(self, max_iterations=10):
        """Iteratively tune parameters to minimize sim-real gap"""
        for iteration in range(max_iterations):
            print(f"Parameter tuning iteration {iteration + 1}/{max_iterations}")

            # Collect data from both sim and real
            sim_behavior = self.collect_behavior_data(self.sim_env)
            real_behavior = self.collect_behavior_data(self.real_robot)

            # Compare behaviors and calculate corrections
            corrections = self.calculate_parameter_corrections(
                sim_behavior, real_behavior
            )

            # Apply corrections
            for param_name, correction in corrections.items():
                current_value = self.sim_params.get(param_name, 0)
                new_value = current_value + correction
                self.apply_parameter_correction(param_name, new_value)
                self.sim_params[param_name] = new_value

            # Check convergence
            error = self.calculate_behavior_error(sim_behavior, real_behavior)
            print(f"Behavior error: {error}")

            if error < 0.01:  # Convergence threshold
                print("Parameter tuning converged")
                break

def collect_behavior_data(robot, duration=5.0):
    """Collect behavioral data from robot"""
    # Implement data collection logic
    # This would involve running specific tests and recording responses
    pass

def calculate_parameter_corrections(sim_behavior, real_behavior):
    """Calculate parameter corrections based on behavior comparison"""
    # Implement correction calculation
    # This would use system identification techniques
    corrections = {}
    return corrections

def calculate_behavior_error(sim_behavior, real_behavior):
    """Calculate error between simulated and real behavior"""
    # Implement error calculation
    error = 0.0
    return error
```

## Transfer Learning and Adaptation

### Model Adaptation Techniques

```python
# transfer_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DomainAdaptationNetwork(nn.Module):
    """Neural network with domain adaptation capabilities"""

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor (shared between domains)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Task-specific output layers
        self.sim_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.real_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Domain classifier for domain adaptation
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, domain='sim'):
        features = self.feature_extractor(x)

        if domain == 'sim':
            output = self.sim_output(features)
        elif domain == 'real':
            output = self.real_output(features)
        else:
            raise ValueError(f"Unknown domain: {domain}")

        return output, features

    def classify_domain(self, features):
        """Classify whether features come from sim or real domain"""
        return self.domain_classifier(features)

class TransferLearningAgent:
    """Agent that can adapt policies from sim to real"""

    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy networks
        self.sim_policy = DomainAdaptationNetwork(state_dim, action_dim).to(self.device)
        self.real_policy = DomainAdaptationNetwork(state_dim, action_dim).to(self.device)

        # Optimizers
        self.sim_optimizer = optim.Adam(self.sim_policy.parameters(), lr=1e-3)
        self.real_optimizer = optim.Adam(self.real_policy.parameters(), lr=1e-3)
        self.domain_optimizer = optim.Adam(
            list(self.sim_policy.feature_extractor.parameters()) +
            list(self.real_policy.feature_extractor.parameters()) +
            list(self.sim_policy.domain_classifier.parameters()),
            lr=1e-3
        )

    def train_domain_adaptation(self, sim_data_loader, real_data_loader, epochs=100):
        """Train with domain adaptation"""
        for epoch in range(epochs):
            # Train on simulation data
            sim_loss = self.train_on_domain(sim_data_loader, 'sim')

            # Train on real data
            real_loss = self.train_on_domain(real_data_loader, 'real')

            # Train domain classifier to distinguish domains
            domain_loss = self.train_domain_classifier(sim_data_loader, real_data_loader)

            # Train feature extractor to fool domain classifier (domain confusion)
            confusion_loss = self.train_domain_confusion(sim_data_loader, real_data_loader)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Sim Loss: {sim_loss:.4f}, "
                      f"Real Loss: {real_loss:.4f}, Domain Loss: {domain_loss:.4f}, "
                      f"Confusion Loss: {confusion_loss:.4f}")

    def train_on_domain(self, data_loader, domain):
        """Train policy on specific domain data"""
        if domain == 'sim':
            policy = self.sim_policy
            optimizer = self.sim_optimizer
        else:
            policy = self.real_policy
            optimizer = self.real_optimizer

        total_loss = 0
        for batch_idx, (states, actions) in enumerate(data_loader):
            states, actions = states.to(self.device), actions.to(self.device)

            pred_actions, _ = policy(states, domain)
            loss = nn.MSELoss()(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def train_domain_classifier(self, sim_loader, real_loader):
        """Train domain classifier to distinguish sim vs real"""
        total_loss = 0

        for (sim_states, _), (real_states, _) in zip(sim_loader, real_loader):
            sim_states = sim_states.to(self.device)
            real_states = real_states.to(self.device)

            # Get features from both domains
            _, sim_features = self.sim_policy(sim_states, 'sim')
            _, real_features = self.real_policy(real_states, 'real')

            # Label sim as 0, real as 1
            sim_labels = torch.zeros(sim_features.size(0), 1).to(self.device)
            real_labels = torch.ones(real_features.size(0), 1).to(self.device)

            # Predict domains
            sim_pred = self.sim_policy.classify_domain(sim_features)
            real_pred = self.sim_policy.classify_domain(real_features)

            # Calculate loss
            sim_loss = nn.BCELoss()(sim_pred, sim_labels)
            real_loss = nn.BCELoss()(real_pred, real_labels)
            loss = sim_loss + real_loss

            self.domain_optimizer.zero_grad()
            loss.backward()
            self.domain_optimizer.step()

            total_loss += loss.item()

        return total_loss / min(len(sim_loader), len(real_loader))

    def train_domain_confusion(self, sim_loader, real_loader):
        """Train to confuse domain classifier (make features domain-invariant)"""
        total_loss = 0

        for (sim_states, _), (real_states, _) in zip(sim_loader, real_loader):
            sim_states = sim_states.to(self.device)
            real_states = real_states.to(self.device)

            # Get features
            _, sim_features = self.sim_policy(sim_states, 'sim')
            _, real_features = self.real_policy(real_states, 'real')

            # Try to fool classifier (want 0.5 probability for both)
            sim_pred = self.sim_policy.classify_domain(sim_features)
            real_pred = self.sim_policy.classify_domain(real_features)

            # Loss to make classifier uncertain (push towards 0.5)
            sim_loss = nn.BCELoss()(sim_pred, torch.full_like(sim_pred, 0.5))
            real_loss = nn.BCELoss()(real_pred, torch.full_like(real_pred, 0.5))
            loss = sim_loss + real_loss

            self.domain_optimizer.zero_grad()
            loss.backward()
            self.domain_optimizer.step()

            total_loss += loss.item()

        return total_loss / min(len(sim_loader), len(real_loader))

class FineTuningAgent:
    """Fine-tune policies on real robot with minimal data"""

    def __init__(self, pretrained_policy):
        self.pretrained_policy = pretrained_policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create fine-tuned version
        self.fine_tuned_policy = self.create_fine_tuned_network(pretrained_policy)

        # Only train the last few layers initially
        self.freeze_base_layers()

    def create_fine_tuned_network(self, base_network):
        """Create network for fine-tuning based on pretrained network"""
        # Copy the structure but allow for fine-tuning
        fine_tuned = type(base_network)()
        fine_tuned.load_state_dict(base_network.state_dict())
        return fine_tuned

    def freeze_base_layers(self):
        """Freeze early layers, only train later layers initially"""
        for param in self.fine_tuned_policy.feature_extractor.parameters():
            param.requires_grad = False

        # Only train the output layers initially
        for param in self.fine_tuned_policy.sim_output.parameters():
            param.requires_grad = True

    def unfreeze_layers_progressively(self, unfreeze_fraction=0.25):
        """Progressively unfreeze more layers during fine-tuning"""
        all_params = list(self.fine_tuned_policy.parameters())
        num_params = len(all_params)
        num_to_unfreeze = int(num_params * unfreeze_fraction)

        for i in range(num_to_unfreeze):
            all_params[i].requires_grad = True

    def fine_tune(self, real_data, epochs=50, initial_lr=1e-4, final_lr=1e-5):
        """Fine-tune on real robot data"""
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                    self.fine_tuned_policy.parameters()),
                              lr=initial_lr)

        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,
            end_factor=final_lr/initial_lr,
            total_iters=epochs
        )

        dataset = TensorDataset(real_data['states'], real_data['actions'])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0

            for states, actions in dataloader:
                states, actions = states.to(self.device), actions.to(self.device)

                pred_actions, _ = self.fine_tuned_policy(states, 'real')
                loss = nn.MSELoss()(pred_actions, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Progressively unfreeze layers every 10 epochs
            if epoch > 0 and epoch % 10 == 0:
                self.unfreeze_layers_progressively()
                print(f"Unfreezing more layers at epoch {epoch}")

            scheduler.step()

            if epoch % 10 == 0:
                print(f"Fine-tuning epoch {epoch}, Loss: {avg_loss:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
```

## Reality Check and Validation

### Validation Techniques

```python
# validation_techniques.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class RealityChecker:
    """Validate sim-to-real transfer performance"""

    def __init__(self):
        self.sim_performance = []
        self.real_performance = []
        self.metrics = {}

    def collect_performance_data(self, sim_env, real_robot, test_episodes=10):
        """Collect performance data from both sim and real"""

        print("Collecting simulation performance data...")
        for episode in range(test_episodes):
            sim_perf = self.evaluate_policy(sim_env)
            self.sim_performance.append(sim_perf)

        print("Collecting real robot performance data...")
        for episode in range(test_episodes):
            real_perf = self.evaluate_policy(real_robot)
            self.real_performance.append(real_perf)

    def evaluate_policy(self, env):
        """Evaluate policy performance on environment"""
        # This would run the policy and collect metrics
        # Return performance metrics like success rate, time to goal, etc.
        pass

    def calculate_transfer_metrics(self):
        """Calculate metrics for sim-to-real transfer"""
        if len(self.sim_performance) == 0 or len(self.real_performance) == 0:
            return {}

        sim_mean = np.mean(self.sim_performance)
        sim_std = np.std(self.sim_performance)
        real_mean = np.mean(self.real_performance)
        real_std = np.std(self.real_performance)

        # Calculate transfer gap
        transfer_gap = real_mean - sim_mean

        # Calculate correlation (if multiple metrics)
        if len(self.sim_performance) > 1:
            correlation, p_value = stats.pearsonr(
                self.sim_performance, self.real_performance
            )
        else:
            correlation, p_value = 0, 1

        # Calculate success rate preservation
        success_threshold = 0.8  # Define what constitutes success
        sim_success_rate = np.mean([1 if perf >= success_threshold else 0
                                   for perf in self.sim_performance])
        real_success_rate = np.mean([1 if perf >= success_threshold else 0
                                    for perf in self.real_performance])

        self.metrics = {
            'sim_mean': sim_mean,
            'sim_std': sim_std,
            'real_mean': real_mean,
            'real_std': real_std,
            'transfer_gap': transfer_gap,
            'correlation': correlation,
            'p_value': p_value,
            'sim_success_rate': sim_success_rate,
            'real_success_rate': real_success_rate,
            'success_preservation': real_success_rate / (sim_success_rate + 1e-8)
        }

        return self.metrics

    def plot_comparison(self):
        """Plot comparison between sim and real performance"""
        if len(self.sim_performance) == 0 or len(self.real_performance) == 0:
            print("No performance data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Histogram comparison
        axes[0, 0].hist(self.sim_performance, alpha=0.7, label='Simulation', bins=20)
        axes[0, 0].hist(self.real_performance, alpha=0.7, label='Real Robot', bins=20)
        axes[0, 0].set_title('Performance Distribution Comparison')
        axes[0, 0].legend()

        # Scatter plot for correlation
        axes[0, 1].scatter(self.sim_performance, self.real_performance, alpha=0.6)
        axes[0, 1].plot([min(self.sim_performance), max(self.sim_performance)],
                       [min(self.sim_performance), max(self.sim_performance)],
                       'r--', label='Perfect Transfer')
        axes[0, 1].set_xlabel('Simulation Performance')
        axes[0, 1].set_ylabel('Real Performance')
        axes[0, 1].set_title('Sim vs Real Performance Correlation')
        axes[0, 1].legend()

        # Box plot comparison
        data = [self.sim_performance, self.real_performance]
        axes[1, 0].boxplot(data, labels=['Simulation', 'Real Robot'])
        axes[1, 0].set_title('Performance Box Plot Comparison')
        axes[1, 0].set_ylabel('Performance')

        # Time series comparison
        axes[1, 1].plot(self.sim_performance, label='Simulation', alpha=0.7)
        axes[1, 1].plot(self.real_performance, label='Real Robot', alpha=0.7)
        axes[1, 1].set_title('Performance Over Episodes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Performance')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def statistical_tests(self):
        """Perform statistical tests to validate transfer"""
        if len(self.sim_performance) == 0 or len(self.real_performance) == 0:
            return {}

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(self.sim_performance, self.real_performance)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(self.sim_performance) - 1) * np.var(self.sim_performance) +
                             (len(self.real_performance) - 1) * np.var(self.real_performance)) /
                            (len(self.sim_performance) + len(self.real_performance) - 2))
        cohens_d = (np.mean(self.real_performance) - np.mean(self.sim_performance)) / pooled_std

        # Confidence intervals
        sim_ci = stats.t.interval(0.95, len(self.sim_performance) - 1,
                                 loc=np.mean(self.sim_performance),
                                 scale=stats.sem(self.sim_performance))
        real_ci = stats.t.interval(0.95, len(self.real_performance) - 1,
                                  loc=np.mean(self.real_performance),
                                  scale=stats.sem(self.real_performance))

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'sim_confidence_interval': sim_ci,
            'real_confidence_interval': real_ci
        }

class SafetyChecker:
    """Ensure safe transfer from sim to real"""

    def __init__(self, safety_thresholds):
        self.safety_thresholds = safety_thresholds
        self.safety_violations = []

    def check_safety_before_transfer(self, policy, env):
        """Check if policy is safe before real-world deployment"""
        safety_checks = [
            self.check_joint_limits(policy, env),
            self.check_balance_margins(policy, env),
            self.check_force_limits(policy, env),
            self.check_velocity_limits(policy, env)
        ]

        all_safe = all(safety_checks)

        if not all_safe:
            print("Safety checks failed! Policy not safe for real-world deployment.")
            return False

        print("All safety checks passed. Policy appears safe for transfer.")
        return True

    def check_joint_limits(self, policy, env):
        """Check if policy respects joint limits"""
        # Test policy with various states to ensure joint limits are not exceeded
        test_states = self.generate_test_states(env)

        for state in test_states:
            action = policy(state)
            # Check if resulting joint positions exceed limits
            # This would require forward kinematics
            pass

        return True  # Placeholder

    def check_balance_margins(self, policy, env):
        """Check if policy maintains balance margins"""
        # Ensure policy doesn't cause robot to fall
        pass
        return True  # Placeholder

    def check_force_limits(self, policy, env):
        """Check if policy respects force/torque limits"""
        pass
        return True  # Placeholder

    def check_velocity_limits(self, policy, env):
        """Check if policy respects velocity limits"""
        pass
        return True  # Placeholder

    def generate_test_states(self, env):
        """Generate diverse test states for safety checking"""
        # Generate various robot configurations to test
        states = []
        for _ in range(100):
            state = env.sample_random_state()
            states.append(state)
        return states

    def monitor_during_transfer(self, policy, real_robot):
        """Monitor policy during real-world execution"""
        safety_monitoring = True
        episode_count = 0

        while safety_monitoring:
            try:
                state = real_robot.get_state()
                action = policy(state)

                # Check safety constraints before executing action
                if not self.is_action_safe(action, real_robot):
                    print("Unsafe action detected! Emergency stop.")
                    real_robot.emergency_stop()
                    return False

                real_robot.execute_action(action)

                # Log for analysis
                self.log_safety_data(state, action)

                episode_count += 1

                # Stop after certain number of safe executions or user command
                if episode_count > 1000:  # Example limit
                    break

            except Exception as e:
                print(f"Safety monitoring error: {e}")
                real_robot.emergency_stop()
                return False

        return True

    def is_action_safe(self, action, robot):
        """Check if action is safe to execute"""
        # Implement safety checks
        return True

    def log_safety_data(self, state, action):
        """Log safety-related data for analysis"""
        pass
```

## Best Practices for Successful Transfer

### Implementation Guidelines

```python
# best_practices.py
class TransferBestPractices:
    """Best practices for successful sim-to-real transfer"""

    @staticmethod
    def design_for_transfer():
        """Guidelines for designing simulation for transfer"""
        practices = {
            "model_uncertainty": {
                "description": "Explicitly model uncertainties in simulation",
                "implementation": [
                    "Add realistic noise models to sensors",
                    "Include parameter variations for physical properties",
                    "Model actuator delays and limitations",
                    "Include environmental disturbances"
                ]
            },
            "rich_training_distribution": {
                "description": "Train on diverse conditions to improve robustness",
                "implementation": [
                    "Use domain randomization extensively",
                    "Include various lighting conditions",
                    "Model different terrains and surfaces",
                    "Include sensor failures and anomalies"
                ]
            },
            "gradual_complexity": {
                "description": "Start with simple tasks and gradually increase complexity",
                "implementation": [
                    "Begin with basic movements in simulation",
                    "Progress to complex behaviors step by step",
                    "Validate each level before advancing",
                    "Use curriculum learning approaches"
                ]
            },
            "extensive_validation": {
                "description": "Validate thoroughly before real-world deployment",
                "implementation": [
                    "Test on multiple simulation environments",
                    "Validate on hardware-in-the-loop setups",
                    "Use safety monitors during initial deployment",
                    "Collect and analyze performance metrics"
                ]
            }
        }
        return practices

    @staticmethod
    def transfer_checklist():
        """Checklist for sim-to-real transfer"""
        checklist = [
            {
                "category": "Simulation Fidelity",
                "items": [
                    "Physical properties match real robot",
                    "Sensor models are realistic",
                    "Actuator dynamics are accurate",
                    "Environmental conditions are representative"
                ]
            },
            {
                "category": "Training Protocol",
                "items": [
                    "Domain randomization was used",
                    "Training distribution covers operational range",
                    "Policy was tested on diverse conditions",
                    "Safety constraints were enforced"
                ]
            },
            {
                "category": "Validation",
                "items": [
                    "Policy performance was measured in simulation",
                    "Transfer metrics were calculated",
                    "Statistical significance was verified",
                    "Safety checks were performed"
                ]
            },
            {
                "category": "Real-World Deployment",
                "items": [
                    "Initial deployment with safety limits",
                    "Continuous monitoring is in place",
                    "Emergency stop procedures are ready",
                    "Gradual increase in autonomy level"
                ]
            }
        ]
        return checklist

class TransferEvaluator:
    """Evaluate transfer success and provide recommendations"""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_transfer_success(self, sim_performance, real_performance):
        """Evaluate how successful the transfer was"""
        if len(sim_performance) == 0 or len(real_performance) == 0:
            return None

        # Calculate various metrics
        sim_mean = np.mean(sim_performance)
        real_mean = np.mean(real_performance)

        # Transfer efficiency
        transfer_efficiency = real_mean / sim_mean if sim_mean != 0 else 0

        # Performance drop
        performance_drop = sim_mean - real_mean

        # Success rate preservation
        sim_success = np.mean([1 for p in sim_performance if p > 0.8])
        real_success = np.mean([1 for p in real_performance if p > 0.8])
        success_preservation = real_success / sim_success if sim_success != 0 else 0

        # Statistical significance
        _, p_value = stats.ttest_ind(sim_performance, real_performance)

        self.evaluation_results = {
            'transfer_efficiency': transfer_efficiency,
            'performance_drop': performance_drop,
            'success_preservation': success_preservation,
            'statistical_significance': p_value < 0.05,
            'sim_performance_stats': {
                'mean': sim_mean,
                'std': np.std(sim_performance),
                'min': np.min(sim_performance),
                'max': np.max(sim_performance)
            },
            'real_performance_stats': {
                'mean': real_mean,
                'std': np.std(real_performance),
                'min': np.min(real_performance),
                'max': np.max(real_performance)
            }
        }

        return self.evaluation_results

    def provide_recommendations(self):
        """Provide recommendations based on evaluation"""
        if not self.evaluation_results:
            return ["Run evaluation first to get recommendations"]

        recommendations = []

        if self.evaluation_results['transfer_efficiency'] < 0.7:
            recommendations.append(
                "Transfer efficiency is low (<70%). Consider improving simulation fidelity "
                "or using more domain randomization."
            )

        if self.evaluation_results['performance_drop'] > 0.2:
            recommendations.append(
                "Significant performance drop observed. Investigate reality gap factors "
                "and consider fine-tuning on real robot."
            )

        if self.evaluation_results['success_preservation'] < 0.8:
            recommendations.append(
                "Success rate not well preserved. Consider safety margins and "
                "conservative policy updates."
            )

        if not self.evaluation_results['statistical_significance']:
            recommendations.append(
                "Results may not be statistically significant. Collect more data "
                "or use different evaluation metrics."
            )

        return recommendations

# Example usage and integration
def complete_transfer_pipeline():
    """Complete pipeline for sim-to-real transfer"""

    print("=== Sim-to-Real Transfer Pipeline ===")

    # 1. Prepare simulation with domain randomization
    print("1. Setting up simulation with domain randomization...")
    domain_randomizer = DomainRandomizer()

    # 2. Train policy in simulation
    print("2. Training policy in randomized simulation...")
    # Training code would go here

    # 3. Validate in simulation
    print("3. Validating policy in simulation...")
    # Validation code would go here

    # 4. System identification (optional)
    print("4. Performing system identification...")
    # System identification code would go here

    # 5. Fine-tuning (if needed)
    print("5. Fine-tuning for real robot (if needed)...")
    # Fine-tuning code would go here

    # 6. Safety checks
    print("6. Performing safety checks...")
    safety_checker = SafetyChecker(safety_thresholds={})
    # Safety checking code would go here

    # 7. Real-world validation
    print("7. Validating on real robot...")
    reality_checker = RealityChecker()
    # Real-world testing code would go here

    # 8. Evaluation and analysis
    print("8. Evaluating transfer success...")
    evaluator = TransferEvaluator()
    # Evaluation code would go here

    print("Transfer pipeline completed!")

    return {
        'domain_randomizer': domain_randomizer,
        'safety_checker': safety_checker,
        'reality_checker': reality_checker,
        'evaluator': evaluator
    }

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline_components = complete_transfer_pipeline()
```

## Knowledge Check

1. What is the "reality gap" and why does it pose challenges for sim-to-real transfer?
2. How does domain randomization help improve transfer success?
3. What are the key components of a successful sim-to-real transfer pipeline?
4. How do you validate that a policy trained in simulation will work on a real robot?

## Summary

This chapter explored sim-to-real transfer techniques, covering domain randomization, system identification, transfer learning, and validation methodologies. We examined how to bridge the reality gap through various techniques including domain randomization, parameter adaptation, and safety validation. The chapter provided practical implementations for creating robust policies that can successfully transfer from simulation to real-world humanoid robots.

## Next Steps

In the next module, we'll dive into humanoid robot development, exploring kinematics, dynamics, bipedal locomotion, and manipulation techniques specifically designed for humanoid robotics applications.