---
title: "ضمیمہ B: سافٹ ویئر انسٹالیشن اور کنفیگریشن"
sidebar_label: "ضمیمہ B: سافٹ ویئر انسٹالیشن"
---

# ضمیمہ B: سافٹ ویئر انسٹالیشن اور کنفیگریشن

## جائزہ

یہ ضمیمہ ہیومنوائڈ روبوٹ ترقی کے لیے ضروری سافٹ ویئر اسٹیک کو انسٹال کرنے اور کنفیگر کرنے کے بارے میں جامع ہدایات فراہم کرتا ہے۔ سافٹ ویئر ایکو سسٹم آپریٹنگ سسٹم، روبوٹکس فریم ورکس، AI لائبریریز، سیمولیشن ماحول، اور ترقی کے اوزاروں پر مشتمل ہے جو اس کتاب میں بیان کردہ تصورات کو نافذ کرنے کے لیے ضروری ہے۔

## آپریٹنگ سسٹم سیٹ اپ

### اوبنٹو انسٹالیشن
ہیومنوائڈ روبوٹ ترقی کے لیے تجویز کردہ آپریٹنگ سسٹم اوبنٹو LTS ہے:

```bash
# یو بونٹو 22.04 LTS کو https://ubuntu.com/download سے ڈاؤن لوڈ کریں
# Rufus (ونڈوز) یا dd (لینکس/میک) کا استعمال کرکے بوٹ ایبل USB بنائیں
# USB سے بوٹ کریں اور انسٹالیشن ویزیڈر کو فالو کریں
# تجویز کردہ پارٹیشننگ:
# - روٹ (/): 100GB کم از کم
# - ہوم (/home): باقی جگہ
# - سواپ: 2x RAM سائز یا 16GB (جس میں سے چھوٹا ہو)
```

### سسٹم کنفیگریشن
یو بونٹو انسٹالیشن کے بعد، روبوٹکس ترقی کے لیے سسٹم کو تشکیل دیں:

```bash
# سسٹم پیکجز کو اپ ڈیٹ کریں
sudo apt update && sudo apt upgrade -y

# ضروری ترقیاتی اوزار انسٹال کریں
sudo apt install build-essential cmake git curl wget vim htop iotop -y

# پائی تھون ترقیاتی پیکجز انسٹال کریں
sudo apt install python3-dev python3-pip python3-venv python3-tk -y

# روبوٹکس کے لیے سسٹم کی وابستگیاں انسٹال کریں
sudo apt install libeigen3-dev libopencv-dev libboost-all-dev -y
```

### ریل ٹائم کرنل کنفیگریشن
وقت کے اہم ایپلی کیشنز کے لیے، ریل ٹائم کرنل انسٹال کریں اور کنفیگر کریں:

```bash
# ریل ٹائم کرنل انسٹال کریں
sudo apt install linux-image-rt-generic linux-headers-rt-generic

# GRUB کو ڈیفالٹ کے طور پر ریل ٹائم کرنل کے ساتھ بوٹ کرنے کے لیے کنفیگر کریں
sudo nano /etc/default/grub
# GRUB_CMDLINE_LINUX_DEFAULT میں "quiet splash" شامل کریں
# اپ ڈیٹ گرب چلائیں
sudo update-grub

# ریل ٹائم کرنل استعمال کرنے کے لیے سسٹم کو دوبارہ بوٹ کریں
sudo reboot
```

## ROS/ROS2 انسٹالیشن

### ROS 2 ہمبل ہاکسبل انسٹالیشن
ROS 2 تجویز کردہ روبوٹکس فریم ورک ہے:

```bash
# لوکل سیٹ کریں
locale  # UTF-8 کے لیے چیک کریں
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS 2 apt ذخیرہ کا اضافہ کریں
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ROS 2 پیکجز انسٹال کریں
sudo apt update
sudo apt install ros-humble-desktop-full -y

# ہیومنوائڈ روبوٹکس کے لیے اضافی ROS پیکجز انسٹال کریں
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup \
    ros-humble-gazebo-ros-pkgs ros-humble-rosbridge-suite \
    ros-humble-teleop-tools ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher ros-humble-xacro -y

# ROS 2 سیٹ اپ اسکرپٹ ماخذ بنائیں
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ROS 2 ورک سپیس سیٹ اپ
اپنا ROS 2 ورک سپیس بنائیں اور کنفیگر کریں:

```bash
# ورک سپیس ڈائریکٹری بنائیں
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws

# ورک سپیس بنائیں
colcon build --symlink-install

# ورک سپیس کا ماخذ بنائیں
echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## NVIDIA CUDA اور GPU ایکسلریشن

### CUDA انسٹالیشن
GPU ایکسلریشن کے لیے CUDA ٹول کٹ انسٹال کریں:

```bash
# NVIDIA سے CUDA ٹول کٹ ڈاؤن لوڈ کریں
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# انسٹالر کو قابل عمل بنائیں
chmod +x cuda_12.4.0_550.54.14_linux.run

# انسٹالر چلائیں (اگر ڈرائیور پہلے سے انسٹال ہے تو ڈرائیور انسٹالیشن کو ان چیک کریں)
sudo sh cuda_12.4.0_550.54.14_linux.run

# PATH میں CUDA کا اضافہ کریں
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### cuDNN انسٹالیشن
گہری سیکھنے کی تیزی کے لیے cuDNN انسٹال کریں:

```bash
# NVIDIA ڈیولپر ویب سائٹ سے cuDNN ڈاؤن لوڈ کریں
# فائلیں نکالیں اور کاپی کریں
tar -xvf cudnn-linux-x86_64-8.9.7.29_cudaX.X-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## پائی تھون ماحول سیٹ اپ

### مجازی ماحول کنفیگریشن
ہیومنوائڈ روبوٹکس کے لیے ایک مخصوص پائی تھون ماحول سیٹ کریں:

```bash
# مجازی ماحول بنائیں
python3 -m venv ~/humanoid_env
source ~/humanoid_env/bin/activate

# pip کو اپ گریڈ کریں
pip install --upgrade pip

# بنیادی پائی تھون پیکجز انسٹال کریں
pip install numpy scipy matplotlib pandas jupyter notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install opencv-python open3d
pip install scikit-learn scikit-image
```

### روبوٹکس کے مخصوص پیکجز
روبوٹکس ایپلی کیشنز کے لیے پائی تھون پیکجز انسٹال کریں:

```bash
# ماحول کو فعال کریں
source ~/humanoid_env/bin/activate

# روبوٹکس لائبریریز انسٹال کریں
pip install pybullet transforms3d
pip install openai gymnasium[box2d]
pip install stable-baselines3[extra]
pip install pyquaternion
pip install modern_robotics
pip install control
```

### کمپیوٹر وژن پیکجز
کمپیوٹر وژن اور ادراک لائبریریز انسٹال کریں:

```bash
# ماحول کو فعال کریں
source ~/humanoid_env/bin/activate

# وژن پیکجز انسٹال کریں
pip install mediapipe
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install supervision
pip install ultralytics
pip install albumentations
pip install imgaug
```

## سیمولیشن ماحول سیٹ اپ

### گیزبو انسٹالیشن
گیزبو سیمولیشن ماحول انسٹال کریں:

```bash
# گیزبو گارڈن انسٹال کریں (تجویز کردہ ورژن)
sudo apt install gazebo libgazebo-dev -y

# ROS 2 گیزبو پلگ انز انسٹال کریں
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins -y
```

### آئسک سیم سیٹ اپ ( اختیاری )
اعلی درجے کی فوٹو ریلزم سیمولیشن کے لیے:

```bash
# NVIDIA ڈیولپر ویب سائٹ سے آئسک سیم ڈاؤن لوڈ کریں
# مطلوبہ جگہ پر نکالیں (مثلاً، /opt/isaac-sim)
# ماحولیاتی متغیرات سیٹ کریں
echo 'export ISAACSIM_PATH=/opt/isaac-sim' >> ~/.bashrc
source ~/.bashrc

# آئسک ROS پیکجز انسٹال کریں
sudo apt install ros-humble-isaac-ros-* -y
```

## ترقی کے اوزار انسٹالیشن

### IDE اور ایڈیٹر
کارکردگی کے لیے ترقیاتی اوزار انسٹال کریں:

```bash
# ویژوئل اسٹوڈیو کوڈ انسٹال کریں
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/vscode.gpg
sudo install -o root -g root -m 644 /usr/share/keyrings/vscode.gpg /etc/apt/trusted.gpg.d/
sudo apt install apt-transport-https
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/vscode.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install code

# VS Code کے لیے ROS ایکسٹینشن انسٹال کریں
code --install-extension ms-iot.vscode-ros
```

### ورژن کنٹرول
ورژن کنٹرول کے لیے گٹ سیٹ کریں:

```bash
# گٹ کنفیگر کریں
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "vim"

# گٹ ایکسٹینشنز انسٹال کریں
sudo apt install git-gui gitk -y
```

### بلڈ ٹولز
اضافی بلڈ اور ترقیاتی اوزار انسٹال کریں:

```bash
# بلڈ ٹولز انسٹال کریں
sudo apt install cmake build-essential pkg-config -y
sudo apt install libusb-1.0-0-dev libftdi-dev -y
sudo apt install python3-dev python3-pip -y

# دستاویزات کے لیے Doxygen انسٹال کریں
sudo apt install doxygen graphviz -y
```

## AI اور مشین لرننگ فریم ورکس

### ٹینسر فلو اور کیرس
نیورل نیٹ ورک ترقی کے لیے ٹینسر فلو انسٹال کریں:

```bash
# ماحول کو فعال کریں
source ~/humanoid_env/bin/activate

# GPU سپورٹ کے ساتھ ٹینسر فلو انسٹال کریں
pip install tensorflow[and-cuda]
pip install tensorflow-addons
pip install tensorboard
```

### ہگنگ فیس ٹرانسفارمرز
NLP کے لیے ٹرانسفارمر ماڈلز انسٹال کریں:

```bash
# ماحول کو فعال کریں
source ~/humanoid_env/bin/activate

# ہگنگ فیس لائبریریز انسٹال کریں
pip install transformers
pip install tokenizers
pip install datasets
pip install accelerate
pip install evaluate
```

### اسپیچ ریکوگنیشن لائبریریز
اسپیچ پروسیسنگ لائبریریز انسٹال کریں:

```bash
# ماحول کو فعال کریں
source ~/humanoid_env/bin/activate

# اسپیچ لائبریریز انسٹال کریں
pip install speechrecognition
pip install pyttsx3
pip install pyaudio
pip install vosk
pip install transformers
```

## کنفیگریشن فائلیں سیٹ اپ

### ROS 2 کنفیگریشن
ROS 2 کے لیے کنفیگریشن فائلیں بنائیں:

```bash
# ROS کنفیگریشن ڈائریکٹری بنائیں
mkdir -p ~/.ros

# ماحول کنفیگریشن بنائیں
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

### پائی تھون پاتھ کنفیگریشن
ترقی کے لیے پائی تھون پاتھ کنفیگر کریں:

```bash
# ~/.bashrc میں شامل کریں
echo 'export PYTHONPATH="${PYTHONPATH}:/home/$USER/humanoid_ws/src"' >> ~/.bashrc
source ~/.bashrc
```

## نیٹ ورک کنفیگریشن

### ROS 2 نیٹ ورک سیٹ اپ
متعدد روبوٹ سسٹمز کے لیے نیٹ ورک کنفیگر کریں:

```bash
# نیٹ ورک کنفیگریشن فائل بنائیں
sudo nano /etc/systemd/network/01-ros.network

# نیٹ ورک کنفیگریشن شامل کریں
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

# ROS نیٹ ورک متغیرات سیٹ کریں
echo 'export ROS_IP=192.168.1.100' >> ~/.bashrc
echo 'export ROS_MASTER_URI=http://192.168.1.100:11311' >> ~/.bashrc
```

## انسٹالیشن کی جانچ

### بنیادی ROS 2 ٹیسٹ
ROS 2 انسٹالیشن کو ٹیسٹ کریں:

```bash
# ROS ماحول کا ماخذ بنائیں
source ~/.bashrc

# ROS 2 انسٹالیشن کو ٹیسٹ کریں
ros2 run demo_nodes_cpp talker
# دوسرے ٹرمنل میں: ros2 run demo_nodes_cpp listener
```

### پائی تھون پیکج ٹیسٹ
پائی تھون پیکجز کو ٹیسٹ کریں:

```bash
# ماحول کو فعال کریں
source ~/humanoid_env/bin/activate

# بنیادی درآمدات کو ٹیسٹ کریں
python3 -c "import rclpy; print('ROS 2 Python interface working')"
python3 -c "import torch; print(f'PyTorch working, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print('OpenCV working')"
python3 -c "import numpy as np; print('NumPy working')"
```

### GPU ایکسلریشن ٹیسٹ
GPU ایکسلریشن کی تصدیق کریں:

```bash
# CUDA انسٹالیشن چیک کریں
nvidia-smi

# PyTorch CUDA ٹیسٹ کریں
source ~/humanoid_env/bin/activate
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## عام مسائل کا حل

### ROS 2 انسٹالیشن کے مسائل
```bash
# اگر ROS پیکجز نہیں مل رہے
sudo apt update
sudo apt upgrade
sudo apt install python3-colcon-common-extensions

# اگر ROS کے ساتھ اجازت کے مسائل ہیں
sudo chown -R $USER:$USER /opt/ros/humble/
```

### GPU کے مسائل
```bash
# چیک کریں کہ آیا NVIDIA ڈرائیورز مناسب طریقے سے انسٹال ہیں
nvidia-smi

# اگر CUDA کام نہیں کر رہا، تو ڈرائیورز دوبارہ انسٹال کریں
sudo apt remove --purge nvidia-*
sudo apt autoremove
# NVIDIA ویب سائٹ سے دوبارہ انسٹال کریں
```

### پائی تھون ماحول کے مسائل
```bash
# اگر ماحول خراب ہو گیا ہے تو مجازی ماحول دوبارہ بنائیں
rm -rf ~/humanoid_env
python3 -m venv ~/humanoid_env
source ~/humanoid_env/bin/activate
# ضرورت کے مطابق پیکجز دوبارہ انسٹال کریں
```

## کارکردگی کی بہتری

### سسٹم ٹیوننگ
روبوٹکس ایپلی کیشنز کے لیے سسٹم کو بہتر بنائیں:

```bash
# سسٹم کارکردگی کنفیگریشن بنائیں
sudo nano /etc/sysctl.conf

# کارکردگی کی بہتری شامل کریں
cat >> /etc/sysctl.conf << 'EOF'
# روبوٹکس کارکردگی کی بہتری
kernel.sched_rt_runtime_us = -1
vm.swappiness = 1
kernel.sched_migration_cost_ns = 5000000
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
EOF

# تبدیلیاں لاگو کریں
sudo sysctl -p
```

### ریل ٹائم کنفیگریشن
ریل ٹائم شیڈولنگ کنفیگر کریں:

```bash
# صارف کو ریل ٹائم گروپ میں شامل کریں
sudo usermod -a -G realtime $USER

# ریل ٹائم حدود کنفیگر کریں
sudo nano /etc/security/limits.conf

# ریل ٹائم حدود شامل کریں
cat >> /etc/security/limits.conf << 'EOF'
# روبوٹکس کے لیے ریل ٹائم حدود
* soft rtprio 99
* hard rtprio 99
* soft memlock unlimited
* hard memlock unlimited
EOF
```

## بیک اپ اور ریکوری

### سسٹم بیک اپ اسکرپٹ
اپنے ترقیاتی ماحول کے لیے بیک اپ اسکرپٹ بنائیں:

```bash
# بیک اپ اسکرپٹ بنائیں
cat > ~/backup_ros_environment.sh << 'EOF'
#!/bin/bash
# ROS 2 ماحول کنفیگریشن کا بیک اپ

BACKUP_DIR="$HOME/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# ROS ورک سپیس کا بیک اپ لیں
cp -r ~/humanoid_ws "$BACKUP_DIR/workspace"

# مجازی ماحول کا بیک اپ لیں
pip freeze > "$BACKUP_DIR/requirements.txt"

# کنفیگریشن فائلیں کا بیک اپ لیں
cp ~/.bashrc "$BACKUP_DIR/bashrc.backup"
cp ~/.ros/setup.sh "$BACKUP_DIR/ros_setup.backup"

# سسٹم کنفیگریشن کا بیک اپ لیں
sudo cp /etc/hosts "$BACKUP_DIR/hosts.backup"
sudo cp /etc/resolv.conf "$BACKUP_DIR/resolv.backup"

echo "Backup completed to $BACKUP_DIR"
EOF

chmod +x ~/backup_ros_environment.sh
```

یہ جامع سافٹ ویئر انسٹالیشن گائیڈ اعلی درجے کے ہیومنوائڈ روبوٹس کی ترقی کے لیے بنیاد فراہم کرتا ہے۔ اس کتاب میں بیان کردہ الگورتھم اور تکنیکوں کو نافذ کرنے کے لیے اس سافٹ ویئر اسٹیک کا مناسب سیٹ اپ انتہائی ضروری ہے، بنیادی ادراک اور کنٹرول سے لے کر اعلی درجے کے AI اور انسان-روبوٹ تعامل کی صلاحیتوں تک۔

