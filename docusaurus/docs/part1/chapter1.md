---
title: "Weeks 1-2: Introduction to Physical AI"
sidebar_label: "Weeks 1-2: Introduction to Physical AI"
---



# Weeks 1-2: Introduction to Physical AI

## Focus and Theme: AI Systems in the Physical World. Embodied Intelligence.

## Goal
Bridging the gap between the digital brain and the physical body. Students apply their AI knowledge to control Humanoid Robots in simulated and real-world environments.

## Course Overview
The future of AI extends beyond digital spaces into the physical world. This capstone quarter introduces Physical AI—AI systems that function in reality and comprehend physical laws. Students learn to design, simulate, and deploy humanoid robots capable of natural human interactions using ROS 2, Gazebo, and NVIDIA Isaac.

## Why Physical AI Matters
Humanoid robots are poised to excel in our human-centered world because they share our physical form and can be trained with abundant data from interacting in human environments. This represents a significant transition from AI models confined to digital environments to embodied intelligence that operates in physical space.

## Learning Outcomes
- Understand Physical AI principles and embodied intelligence
- Master ROS 2 (Robot Operating System) for robotic control
- Simulate robots with Gazebo and Unity
- Develop with NVIDIA Isaac AI robot platform
- Design humanoid robots for natural interactions
- Integrate GPT models for conversational robotics

## Weekly Breakdown: Weeks 1-2
### Introduction to Physical AI
- Foundations of Physical AI and embodied intelligence
- From digital AI to robots that understand physical laws
- Overview of humanoid robotics landscape
- Sensor systems: LIDAR, cameras, IMUs, force/torque sensors

## Hardware Requirements
This course is technically demanding. It sits at the intersection of three heavy computational loads: Physics Simulation (Isaac Sim/Gazebo), Visual Perception (SLAM/Computer Vision), and Generative AI (LLMs/VLA).

### 1. The "Digital Twin" Workstation (Required per Student)
This is the most critical component. NVIDIA Isaac Sim is an Omniverse application that requires "RTX" (Ray Tracing) capabilities. Standard laptops (MacBooks or non-RTX Windows machines) will not work.

- **GPU (The Bottleneck)**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher.
  - Why: You need high VRAM to load the USD (Universal Scene Description) assets for the robot and environment, plus run the VLA (Vision-Language-Action) models simultaneously.
  - Ideal: RTX 3090 or 4090 (24GB VRAM) allows for smoother "Sim-to-Real" training.

- **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9.
  - Why: Physics calculations (Rigid Body Dynamics) in Gazebo/Isaac are CPU-intensive.

- **RAM**: 64 GB DDR5 (32 GB is the absolute minimum, but will crash during complex scene rendering).

- **OS**: Ubuntu 22.04 LTS.
  - Note: While Isaac Sim runs on Windows, ROS 2 (Humble/Iron) is native to Linux. Dual-booting or dedicated Linux machines are mandatory for a friction-free experience.

### 2. The "Physical AI" Edge Kit
Since a full humanoid robot is expensive, students learn "Physical AI" by setting up the nervous system on a desk before deploying it to a robot. This kit covers Module 3 (Isaac ROS) and Module 4 (VLA).

- **The Brain**: NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB).
  - Role: This is the industry standard for embodied AI. Students will deploy their ROS 2 nodes here to understand resource constraints vs. their powerful workstations.

- **The Eyes (Vision)**: Intel RealSense D435i or D455.
  - Role: Provides RGB (Color) and Depth (Distance) data. Essential for the VSLAM and Perception modules.

- **The Inner Ear (Balance)**: Generic USB IMU (BNO055) (Often built into the RealSense D435i or Jetson boards, but a separate module helps teach IMU calibration).

- **Voice Interface**: A simple USB Microphone/Speaker array (e.g., ReSpeaker) for the "Voice-to-Action" Whisper integration.

## Assessments
- ROS 2 package development project
- Gazebo simulation implementation
- Isaac-based perception pipeline
- Capstone: Simulated humanoid robot with conversational AI

## Summary
This introduction provides the foundational knowledge for understanding Physical AI and its applications in humanoid robotics. Students will explore the intersection of AI and physical systems, preparing them for the advanced modules that follow.

