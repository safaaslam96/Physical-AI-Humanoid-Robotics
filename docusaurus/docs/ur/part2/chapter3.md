---
title: "ہفتہ 6-7: Gazebo کے ساتھ روبوٹ کی شبیہہ سازی"
sidebar_label: "ہفتہ 6-7: Gazebo کے ساتھ روبوٹ کی شبیہہ سازی"
---

# ہفتہ 6-7: Gazebo کے ساتھ روبوٹ کی شبیہہ سازی

## ماڈیول 2: ڈیجیٹل ٹوئن (Gazebo & Unity)

### مرکز: فزیکس سیمولیشن اور ماحول تعمیر کرنا

### سیکھنے کے اہداف
- Gazebo سیمولیشن ماحول کو سیٹ اپ کرنا
- URDF اور SDF فارمیٹس کا استعمال کرتے ہوئے روبوٹ ماڈل تیار کرنا
- Gazebo میں فزیکس، گریویٹی، اور کالیژنز کو سیمولیٹ کرنا
- سینسر سیمولیشن (LiDAR، ڈیپتھ کیمرے، IMUs) کو لاگو کرنا
- روبوٹ کی ویژولزمیشن کے لیے Unity کو متعارف کرنا

## Gazebo سیمولیشن ماحول سیٹ اپ

### Gazebo کیا ہے؟

Gazebo ایک روبوٹ سیمولیٹر ہے جو حقیقی فزیکس سیمولیشن، اعلیٰ معیار کے گریفکس، اور موزوں پروگرامنگ انٹرفیس فراہم کرتا ہے۔ یہ حقیقی ہارڈ ویئر پر ڈیپلائی کرنے سے پہلے روبوٹ کے برتاؤ کو ٹیسٹ اور درست کرنے کے لیے اہم ہے۔

### تنصیب اور ضروریات

```bash
# Gazebo Garden تنصیب کریں (تجویز کردہ ورژن)
sudo apt install ros-humble-gazebo-ros-pkgs

# اضافی پلگ انز تنصیب کریں
sudo apt install ros-humble-gazebo-dev
sudo apt install ros-humble-gazebo-plugins
```

### بنیادی Gazebo لانچ

```bash
# خالی دنیا کے ساتھ Gazebo لانچ کریں
gazebo

# مخصوص دنیا فائل کے ساتھ لانچ کریں
gazebo worlds/willow.world
```

## URDF اور SDF روبوٹ تفصیل فارمیٹس

### URDF بمقابلہ SDF

- **URDF (متحدہ روبوٹ تفصیل فارمیٹ)**: XML فارمیٹ روبوٹ کنیمیٹکس اور ڈائنیمکس کی وضاحت کے لیے، زیادہ تر ROS کے ساتھ استعمال ہوتا ہے
- **SDF (سیمولیشن تفصیل فارمیٹ)**: XML فارمیٹ صرف سیمولیشن کے لیے ڈیزائن کیا گیا، Gazebo کے ذریعے استعمال ہوتا ہے

### URDF سے SDF کی تبدیلی

Gazebo URDF کو SDF میں خود بخود تبدیل کر سکتا ہے libgazebo_ros_xacro پلگ ان کا استعمال کرتے ہوئے:

```xml
<!-- لانچ فائل میں -->
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_entity.py" args="-file $(find my_robot_description)/urdf/my_robot.urdf -entity my_robot"/>
```

## فزیکس سیمولیشن اور سینسر سیمولیشن

### فزیکس انجن کنفیگریشن

Gazebo ODE (Open Dynamics Engine) کو اس کا ڈیفالٹ فزیکس انجن کے طور پر استعمال کرتا ہے:

```xml
<!-- دنیا فائل میں -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
</physics>
```

### سینسر سیمولیشن

#### LiDAR سیمولیشن
```xml
<gazebo reference="laser_link">
  <sensor type="ray" name="laser_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

#### ڈیپتھ کیمرہ سیمولیشن
```xml
<gazebo reference="camera_link">
  <sensor type="depth" name="camera_sensor">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>/camera</namespace>
        <remapping>image_raw:=image_color</remapping>
        <remapping>camera_info:=camera_info</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

#### IMU سیمولیشن
```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

## روبوٹ کی ویژولزمیشن کے لیے Unity کی معرفت

### Unity روبوٹ فریم ورکس

Unity اعلیٰ معیار کے رینڈرنگ اور انسان-روبوٹ تعامل کی صلاحیتیں فراہم کرتا ہے:

- **Unity Robotics Hub**: روبوٹکس سیمولیشن کے لیے ٹولز اور مثالوں کا مجموعہ
- **ML-Agents**: Unity ماحول میں AI ایجنٹس کو تربیت دینے کے لیے
- **ROS#**: Unity اور ROS/ROS2 کے درمیان پل

## ہفتہ وار جائزہ: ہفتہ 6-7

### ہفتہ 6: Gazebo سیمولیشن ماحول سیٹ اپ
- Gazebo تنصیب اور کنفیگریشن
- دنیا فائل تخلیق اور ماحول تعمیر کرنا
- URDF اور SDF فارمیٹس روبوٹ کی تفصیل کے لیے
- بنیادی روبوٹ کو سیمولیشن میں اسپون کرنا

### ہفتہ 7: فزیکس اور سینسر سیمولیشن
- فزیکس سیمولیشن پیرامیٹرز اور ٹیوننگ
- سینسر سیمولیشن (LiDAR، کیمرے، IMUs)
- اعلیٰ معیار کی ویژولزمیشن کے لیے Unity
- سیمولیشن میں انسان-روبوٹ تعامل کی معرفت

## جائزہ

اس ماڈیول نے Gazebo اور Unity کا استعمال کرتے ہوئے روبوٹ کی سیمولیشن کی بنیاد رکھی ہے۔ آپ نے سیکھا کہ حقیقی سیمولیشن ماحول کیسے تیار کریں، URDF/SDF کے ساتھ روبوٹ ماڈل کیسے بنائیں، اور مختلف سینسرز کو سیمولیٹ کریں۔ یہ سیمولیشن کی مہارتوں کو حقیقی ہارڈ ویئر پر ڈیپلائی کرنے سے پہلے روبوٹ کے برتاؤ کو ٹیسٹ اور درست کرنے کے لیے اہم ہے، جو فزیکل ای کے ترقی کے عمل کا ایک اہم جزو ہے۔

