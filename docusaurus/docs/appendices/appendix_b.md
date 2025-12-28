---
title: "Appendix B: Software Installation and Configuration"
sidebar_label: "Appendix B: Software Installation"
---



# Appendix B: Software Installation and Configuration

## Overview

This appendix provides comprehensive guidance for installing and configuring the software stack required for humanoid robot development. The software ecosystem encompasses operating systems, robotics frameworks, AI libraries, simulation environments, and development tools necessary for implementing the concepts covered in this book.

## Operating System Setup

### Ubuntu Installation
The recommended operating system for humanoid robot development is Ubuntu LTS:

```bash
# Download Ubuntu 22.04 LTS from https://ubuntu.com/download
# Create bootable USB using Rufus (Windows) or dd (Linux/Mac)
# Boot from USB and follow installation wizard
# Recommended partitioning:
# - Root (/): 100GB minimum
# - Home (/home): Remaining space
# - Swap: 2x RAM size or 16GB (whichever is smaller)
```

### System Configuration
After Ubuntu installation, configure the system for robotics development:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install build-essential cmake git curl wget vim htop iotop -y

# Install Python development packages
sudo apt install python3-dev python3-pip python3-venv python3-tk -y

# Install system dependencies for robotics
sudo apt install libeigen3-dev libopencv-dev libboost-all-dev -y
```

### Real-time Kernel Configuration
For time-critical applications, install and configure real-time kernel:

```bash
# Install real-time kernel
sudo apt install linux-image-rt-generic linux-headers-rt-generic

# Configure GRUB to boot with real-time kernel by default
sudo nano /etc/default/grub
# Add "quiet splash" to GRUB_CMDLINE_LINUX_DEFAULT
# Run update-grub
sudo update-grub

# Reboot system to use real-time kernel
sudo reboot
```

## ROS/ROS2 Installation

### ROS 2 Humble Hawksbill Installation
ROS 2 is the recommended robotics framework:

```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop-full -y

# Install additional ROS packages for humanoid robotics
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup \
    ros-humble-gazebo-ros-pkgs ros-humble-rosbridge-suite \
    ros-humble-teleop-tools ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher ros-humble-xacro -y

# Source ROS 2 setup script
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ROS 2 Workspace Setup
Create and configure your ROS 2 workspace:

```bash
# Create workspace directory
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws

# Build the workspace
colcon build --symlink-install

# Source the workspace
echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## NVIDIA CUDA and GPU Acceleration

### CUDA Installation
Install CUDA toolkit for GPU acceleration:

```bash
# Download CUDA toolkit from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Make installer executable
chmod +x cuda_12.4.0_550.54.14_linux.run

# Run installer (uncheck driver installation if driver is already installed)
sudo sh cuda_12.4.0_550.54.14_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### cuDNN Installation
Install cuDNN for deep learning acceleration:

```bash
# Download cuDNN from NVIDIA Developer website
# Extract and copy files
tar -xvf cudnn-linux-x86_64-8.9.7.29_cudaX.X-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## Python Environment Setup

### Virtual Environment Configuration
Set up a dedicated Python environment for humanoid robotics:

```bash
# Create virtual environment
python3 -m venv ~/humanoid_env
source ~/humanoid_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core Python packages
pip install numpy scipy matplotlib pandas jupyter notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install opencv-python open3d
pip install scikit-learn scikit-image
```

### Robotics-Specific Packages
Install Python packages for robotics applications:

```bash
# Activate environment
source ~/humanoid_env/bin/activate

# Install robotics libraries
pip install pybullet transforms3d
pip install openai gymnasium[box2d]
pip install stable-baselines3[extra]
pip install pyquaternion
pip install modern_robotics
pip install control
```

### Computer Vision Packages
Install computer vision and perception libraries:

```bash
# Activate environment
source ~/humanoid_env/bin/activate

# Install vision packages
pip install mediapipe
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install supervision
pip install ultralytics
pip install albumentations
pip install imgaug
```

## Simulation Environment Setup

### Gazebo Installation
Install Gazebo simulation environment:

```bash
# Install Gazebo Garden (recommended version)
sudo apt install gazebo libgazebo-dev -y

# Install ROS 2 Gazebo plugins
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins -y
```

### Isaac Sim Setup (Optional)
For advanced photorealistic simulation:

```bash
# Download Isaac Sim from NVIDIA Developer website
# Extract to desired location (e.g., /opt/isaac-sim)
# Set environment variables
echo 'export ISAACSIM_PATH=/opt/isaac-sim' >> ~/.bashrc
source ~/.bashrc

# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-* -y
```

## Development Tools Installation

### IDE and Editors
Install development tools for efficient coding:

```bash
# Install Visual Studio Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/vscode.gpg
sudo install -o root -g root -m 644 /usr/share/keyrings/vscode.gpg /etc/apt/trusted.gpg.d/
sudo apt install apt-transport-https
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/vscode.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install code

# Install ROS extension for VS Code
code --install-extension ms-iot.vscode-ros
```

### Version Control
Set up Git for version control:

```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "vim"

# Install Git extensions
sudo apt install git-gui gitk -y
```

### Build Tools
Install additional build and development tools:

```bash
# Install build tools
sudo apt install cmake build-essential pkg-config -y
sudo apt install libusb-1.0-0-dev libftdi-dev -y
sudo apt install python3-dev python3-pip -y

# Install Doxygen for documentation
sudo apt install doxygen graphviz -y
```

## AI and Machine Learning Frameworks

### TensorFlow and Keras
Install TensorFlow for neural network development:

```bash
# Activate environment
source ~/humanoid_env/bin/activate

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
pip install tensorflow-addons
pip install tensorboard
```

### Hugging Face Transformers
Install transformer models for NLP:

```bash
# Activate environment
source ~/humanoid_env/bin/activate

# Install Hugging Face libraries
pip install transformers
pip install tokenizers
pip install datasets
pip install accelerate
pip install evaluate
```

### Speech Recognition Libraries
Install speech processing libraries:

```bash
# Activate environment
source ~/humanoid_env/bin/activate

# Install speech libraries
pip install speechrecognition
pip install pyttsx3
pip install pyaudio
pip install vosk
pip install transformers
```

## Configuration Files Setup

### ROS 2 Configuration
Create configuration files for ROS 2:

```bash
# Create ROS configuration directory
mkdir -p ~/.ros

# Create environment configuration
cat > ~/.ros/setup.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/humanoid_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
export ROS_LOCALHOST_ONLY=0
EOF

chmod +x ~/.ros/setup.sh
echo "source ~/.ros/setup.sh" >> ~/.bashrc
```

### Python Path Configuration
Configure Python paths for development:

```bash
# Add to ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:/home/$USER/humanoid_ws/src"' >> ~/.bashrc
source ~/.bashrc
```

## Network Configuration

### ROS 2 Network Setup
Configure network for multi-robot systems:

```bash
# Create network configuration file
sudo nano /etc/systemd/network/01-ros.network

# Add network configuration
cat > /etc/systemd/network/01-ros.network << 'EOF'
[Match]
Name=en*

[Network]
DHCP=ipv4
Address=192.168.1.100/24
Gateway=192.168.1.1
DNS=8.8.8.8

[IPv6AcceptRA]
Enabled=true
EOF

# Set ROS network variables
echo 'export ROS_IP=192.168.1.100' >> ~/.bashrc
echo 'export ROS_MASTER_URI=http://192.168.1.100:11311' >> ~/.bashrc
```

## Testing the Installation

### Basic ROS 2 Test
Test the ROS 2 installation:

```bash
# Source ROS environment
source ~/.bashrc

# Test ROS 2 installation
ros2 run demo_nodes_cpp talker
# In another terminal: ros2 run demo_nodes_cpp listener
```

### Python Package Test
Test Python packages:

```bash
# Activate environment
source ~/humanoid_env/bin/activate

# Test basic imports
python3 -c "import rclpy; print('ROS 2 Python interface working')"
python3 -c "import torch; print(f'PyTorch working, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print('OpenCV working')"
python3 -c "import numpy as np; print('NumPy working')"
```

### GPU Acceleration Test
Verify GPU acceleration:

```bash
# Check CUDA installation
nvidia-smi

# Test PyTorch CUDA
source ~/humanoid_env/bin/activate
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Troubleshooting Common Issues

### ROS 2 Installation Issues
```bash
# If ROS packages are not found
sudo apt update
sudo apt upgrade
sudo apt install python3-colcon-common-extensions

# If there are permission issues with ROS
sudo chown -R $USER:$USER /opt/ros/humble/
```

### GPU Issues
```bash
# Check if NVIDIA drivers are properly installed
nvidia-smi

# If CUDA is not working, reinstall drivers
sudo apt remove --purge nvidia-*
sudo apt autoremove
# Reinstall from NVIDIA website
```

### Python Environment Issues
```bash
# Recreate virtual environment if corrupted
rm -rf ~/humanoid_env
python3 -m venv ~/humanoid_env
source ~/humanoid_env/bin/activate
# Reinstall packages as needed
```

## Performance Optimization

### System Tuning
Optimize system for robotics applications:

```bash
# Create system performance configuration
sudo nano /etc/sysctl.conf

# Add performance optimizations
cat >> /etc/sysctl.conf << 'EOF'
# Robotics performance optimizations
kernel.sched_rt_runtime_us = -1
vm.swappiness = 1
kernel.sched_migration_cost_ns = 5000000
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
EOF

# Apply changes
sudo sysctl -p
```

### Real-time Configuration
Configure real-time scheduling:

```bash
# Add user to real-time group
sudo usermod -a -G realtime $USER

# Configure real-time limits
sudo nano /etc/security/limits.conf

# Add real-time limits
cat >> /etc/security/limits.conf << 'EOF'
# Real-time limits for robotics
* soft rtprio 99
* hard rtprio 99
* soft memlock unlimited
* hard memlock unlimited
EOF
```

## Backup and Recovery

### System Backup Script
Create a backup script for your development environment:

```bash
# Create backup script
cat > ~/backup_ros_environment.sh << 'EOF'
#!/bin/bash
# Backup ROS 2 environment configuration

BACKUP_DIR="$HOME/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup ROS workspace
cp -r ~/humanoid_ws "$BACKUP_DIR/workspace"

# Backup virtual environment
pip freeze > "$BACKUP_DIR/requirements.txt"

# Backup configuration files
cp ~/.bashrc "$BACKUP_DIR/bashrc.backup"
cp ~/.ros/setup.sh "$BACKUP_DIR/ros_setup.backup"

# Backup system configuration
sudo cp /etc/hosts "$BACKUP_DIR/hosts.backup"
sudo cp /etc/resolv.conf "$BACKUP_DIR/resolv.backup"

echo "Backup completed to $BACKUP_DIR"
EOF

chmod +x ~/backup_ros_environment.sh
```

This comprehensive software installation guide provides the foundation for developing advanced humanoid robots. Proper setup of this software stack is essential for implementing the algorithms and techniques described throughout this book, from basic perception and control to advanced AI and human-robot interaction capabilities.

