---
title: "باب 4: Python کے ساتھ ROS 2 پیکجز تیار کرنا"
sidebar_label: "باب 4: ROS 2 پیکجز تیار کرنا"
---

# باب 4: Python کے ساتھ ROS 2 پیکجز تیار کرنا

## سیکھنے کے اہداف
- Python کا استعمال کرتے ہوئے ROS 2 پیکجز تیار کرنا
- Python اور rclpy لائبریری کے ساتھ نوڈز تیار کرنا
- لانچ فائلز اور پیرامیٹر مینجمنٹ نافذ کرنا
- Python-مبنی ROS 2 ترقی کے لیے بہترین طریقے اپنانا

## تعارف

Python روبوٹکس کی ترقی کے لیے سب سے مقبول زبانوں میں سے ایک بن گئی ہے اس کی سادگی، وسیع لائبریریز، اور مضبوط AI/ML ایکو سسٹم کی بدولت۔ rclpy لائبریری ROS 2 کے لیے Python بائنڈنگس فراہم کرتی ہے، جو روبوٹک اطلاقات کی تیز رفتار پروٹو ٹائپنگ اور ترقی کو فعال کرتی ہے۔ یہ باب Python کا استعمال کرتے ہوئے مضبوط ROS 2 پیکجز تیار کرنے پر مرکوز ہے، جسمانی ای آئی سسٹمز کے لیے عملی امپلیمنٹیشن اور بہترین طریقے زیر التوا دیتا ہے۔

## Python کے ساتھ ROS 2 پیکجز تیار کرنا

### پیکج سٹرکچر اور تنظیم

ایک اچھی طرح سے ترتیب شدہ ROS 2 Python پیکج ایک مخصوص تنظیم کا پیرو کرتا ہے:

```
my_robot_package/
├── CMakeLists.txt          # بِلڈ کنفیگریشن
├── package.xml            # پیکج میٹا ڈیٹا
├── setup.py               # Python پیکج کنفیگریشن
├── setup.cfg              # انسٹالیشن کنفیگریشن
├── my_robot_package/      # مرکزی Python پیکج
│   ├── __init__.py
│   ├── robot_controller.py
│   └── utils/
│       ├── __init__.py
│       └── helper_functions.py
├── launch/                # لانچ فائلز
│   └── robot_launch.py
├── config/                # کنفیگریشن فائلز
│   └── robot_params.yaml
├── test/                  # ٹیسٹ فائلز
│   └── test_robot_controller.py
└── scripts/               # قابل انجام اسکرپٹس (اختیاری)
```

### ایک نیا پیکج بنانا

Python-مبنی ROS 2 پیکج بنانے کے لیے:

```bash
# Python بِلڈ ٹائپ کے ساتھ ایک نیا پیکج بنائیں
ros2 pkg create --build-type ament_python my_robot_controller

# یا مکس C++/Python پیکجز کے لیے ament_cmake بِلڈ ٹائپ کا استعمال
ros2 pkg create --build-type ament_cmake my_mixed_robot_package
```

### package.xml کنفیگریشن

`package.xml` فائل ضروری میٹا ڈیٹا پر مشتمل ہے:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_controller</name>
  <version>0.0.0</version>
  <description>A Python-based robot controller package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py کنفیگریشن

`setup.py` فائل Python پیکج کو کنفیگر کرتی ہے:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # لانچ فائلز شامل کریں
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # کنفیگ فائلز شامل کریں
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='A Python-based robot controller package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = my_robot_controller.robot_controller:main',
            'sensor_processor = my_robot_controller.sensor_processor:main',
        ],
    },
)
```

## Python کے ساتھ نوڈز تیار کرنا

### بنیادی نوڈ سٹرکچر

ایک اچھی طرح سے ترتیب شدہ ROS 2 Python نوڈ میں درج ذیل شامل ہے:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# پیغام کی اقسام درآمد کریں
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # ڈیفالٹ ویلیوز کے ساتھ پیرامیٹرز کا اعلان کریں
        self.declare_parameter('control_frequency', 50)
        self.declare_parameter('max_velocity', 1.0)

        # پیرامیٹر ویلیوز حاصل کریں
        self.control_freq = self.get_parameter('control_frequency').value
        self.max_vel = self.get_parameter('max_velocity').value

        # پبلشرز تیار کریں
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            self._get_qos_profile()
        )

        # سبسکرائبرز تیار کریں
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            self._get_qos_profile()
        )

        # سروسز تیار کریں
        self.reset_service = self.create_service(
            Empty,  # آپ اس پیغام کی قسم کی وضاحت کریں گے
            'reset_robot',
            self.reset_callback
        )

        # کنٹرول لوپ کے لیے ٹائمر تیار کریں
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self.control_loop
        )

        # روبوٹ کی حالت شروع کریں
        self.current_state = None
        self.target_velocity = Twist()

        self.get_logger().info(f'Robot Controller initialized with {self.control_freq}Hz frequency')

    def _get_qos_profile(self):
        """حقیقی وقت کی مواصلات کے لیے ایک QoS پروفائل تیار کریں"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        return qos_profile

    def joint_state_callback(self, msg):
        """آنے والے جوائنٹ سٹیٹ پیغامات کو ہینڈل کریں"""
        self.current_state = msg
        # جوائنٹ سٹیٹس کو ضرورت کے مطابق پروسیس کریں
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def control_loop(self):
        """مخصوص فریکوئینسی پر انجام دیا جانے والا مرکزی کنٹرول لوپ"""
        if self.current_state is not None:
            # یہاں کنٹرول لاگک کو نافذ کریں
            self.cmd_vel_pub.publish(self.target_velocity)

    def reset_callback(self, request, response):
        """ری سیٹ سروس کی درخواستوں کو ہینڈل کریں"""
        self.get_logger().info('Resetting robot state')
        # یہاں ری سیٹ لاگک
        return response

    def set_target_velocity(self, linear_x, angular_z):
        """روبوٹ کے لیے ہدف کی رفتار مقرر کریں"""
        self.target_velocity.linear.x = linear_x
        self.target_velocity.angular.z = angular_z


def main(args=None):
    rclpy.init(args=args)

    robot_controller = RobotControllerNode()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down robot controller')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### اعلی درجے کے نوڈ فیچرز

#### پیرامیٹر مینجمنٹ

ROS 2 مضبوط پیرامیٹر مینجمنٹ فراہم کرتا ہے:

```python
class AdvancedRobotNode(Node):
    def __init__(self):
        super().__init__('advanced_robot')

        # وضاحت کے ساتھ پیرامیٹرز کا اعلان کریں
        self.declare_parameter(
            'robot_name',
            'default_robot',
            descriptor=rclpy.node.ParameterDescriptor(
                description='Name of the robot'
            )
        )

        self.declare_parameter('safety_limits', [1.0, 2.0, 3.0])
        self.declare_parameter('control_mode', 'velocity')

        # پیرامیٹر تبدیلیوں کے لیے کال بیک
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """پیرامیٹر تبدیلیوں کو ہینڈل کریں"""
        for param in params:
            if param.name == 'control_mode' and param.value not in ['velocity', 'position', 'effort']:
                return SetParametersResult(successful=False, reason='Invalid control mode')
        return SetParametersResult(successful=True)
```

#### لائف سائیکل نوڈز

زیادہ جٹیل اطلاقات کے لیے، لائف سائیکل نوڈز بہتر سٹیٹ مینجمنٹ فراہم کرتے ہیں:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn


class LifecycleRobotController(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_robot_controller')

    def on_configure(self, state):
        """CONFIGURING سٹیٹ میں منتقل ہونے پر کہا جاتا ہے"""
        self.get_logger().info('Configuring robot controller')
        # وسائل شروع کریں لیکن فعال آپریشن شروع نہ کریں
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """ACTIVATING سٹیٹ میں منتقل ہونے پر کہا جاتا ہے"""
        self.get_logger().info('Activating robot controller')
        # فعال آپریشن شروع کریں
        return super().on_activate(state)

    def on_deactivate(self, state):
        """DEACTIVATING سٹیٹ میں منتقل ہونے پر کہا جاتا ہے"""
        self.get_logger().info('Deactivating robot controller')
        # فعال آپریشن بند کریں لیکن وسائل رکھیں
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        """CLEANINGUP سٹیٹ میں منتقل ہونے پر کہا جاتا ہے"""
        self.get_logger().info('Cleaning up robot controller')
        # وسائل صاف کریں
        return TransitionCallbackReturn.SUCCESS
```

## Python نوڈز کے لیے لانچ فائلز

### Python-مبنی لانچ فائلز

ROS 2 لانچ فائلز Python میں لکھی جا سکتی ہیں:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessStart
from lifecycle_msgs.msg import Transition

def generate_launch_description():
    # لانچ آرگومنٹس کا اعلان کریں
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # لانچ آرگومنٹس کا اعلان کریں
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    # روبوٹ کنٹرولر نوڈ تیار کریں
    robot_controller = Node(
        package='my_robot_controller',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name}
        ],
        output='screen'
    )

    # سینسر پروسیسر نوڈ تیار کریں
    sensor_processor = Node(
        package='my_robot_controller',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # لانچ کی تفصیل لوٹائیں
    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        robot_controller,
        sensor_processor
    ])
```

### اعلی درجے کی لانچ کنفیگریشن

لانچ فائلز میں جٹیل کنفیگریشنز شامل ہو سکتے ہیں:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # دیگر لانچ فائلز شامل کریں
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ])
    )

    # مخصوص شرائط کے ساتھ نوڈز لانچ کریں
    robot_controller = Node(
        package='my_robot_controller',
        executable='robot_controller',
        name='robot_controller',
        parameters=[PathJoinSubstitution([
            FindPackageShare('my_robot_controller'),
            'config',
            'robot_config.yaml'
        ])],
        respawn=True,  # اگر نوڈ مر جاتا ہے تو دوبارہ شروع کریں
        respawn_delay=2.0
    )

    # تاخیر کے ساتھ لانچ کریں
    delayed_node = TimerAction(
        period=5.0,  # لانچ کرنے سے پہلے 5 سیکنڈ انتظار کریں
        actions=[Node(
            package='my_robot_controller',
            executable='post_processing_node',
            name='post_processor'
        )]
    )

    return LaunchDescription([
        simulation_launch,
        robot_controller,
        delayed_node
    ])
```

## پیرامیٹر مینجمنٹ

### YAML کنفیگریشن فائلز

کنفیگریشن فائلز لچکدار پیرامیٹر مینجمنٹ فراہم کرتی ہیں:

```yaml
# config/robot_config.yaml
/**:
  ros__parameters:
    # روبوٹ کی خصوصیات
    robot_name: "my_advanced_robot"
    max_velocity: 1.0
    max_angular_velocity: 1.5
    safety_distance: 0.5

    # کنٹرول پیرامیٹرز
    control_frequency: 100
    position_tolerance: 0.01
    velocity_tolerance: 0.05

    # سینسر پیرامیٹرز
    sensor_update_rate: 50
    sensor_timeout: 0.1

    # AI/ML پیرامیٹرز
    prediction_horizon: 10
    confidence_threshold: 0.8
```

### پیرامیٹرز لوڈ کرنا

پیرامیٹرز کو متعدد طریقوں سے لوڈ کیا جا سکتا ہے:

```python
class ParameterizedRobotNode(Node):
    def __init__(self):
        super().__init__('parameterized_robot')

        # YAML فائل سے پیرامیٹرز لوڈ کریں
        self.load_parameters_from_file()

        # یا مخصوص پیرامیٹرز لوڈ کریں
        self.declare_parameter('robot_config_file', 'config/robot_config.yaml')

    def load_parameters_from_file(self):
        """YAML کنفیگریشن فائل سے پیرامیٹرز لوڈ کریں"""
        config_file = self.get_parameter('robot_config_file').value

        # عمل میں، آپ ایک کنفیگریشن مینجمنٹ سسٹم استعمال کریں گے
        # یہ ایک سادہ مثال ہے
        try:
            import yaml
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                # کنفیگریشن کو پروسیس کریں
                self.get_logger().info(f'Loaded configuration from {config_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to load config file: {e}')
```

## Python-مبنی ROS 2 ترقی کے لیے بہترین طریقے

### کوڈ کی تنظیم

1. **الگ الگ امور**: پیغام ہینڈلنگ، کاروباری لاگک، اور ROS انٹرفیس کو الگ رکھیں
2. **ٹائپ ہنٹس استعمال کریں**: کوڈ کی قابلیت اور IDE سپورٹ کو بہتر بنائیں
3. **ایرر ہینڈلنگ**: مضبوط آپریشن کے لیے مناسب استثنا ہینڈلنگ نافذ کریں
4. **لاگنگ**: ڈیبگنگ اور مانیٹرنگ کے لیے مناسب لاگ سطحیں استعمال کریں

### کارکردگی کے امور

1. **کارآمد پیغام پروسیسنگ**: ڈیٹا کاپی اور پروسیسنگ اوور ہیڈ کو کم کریں
2. **میموری مینجمنٹ**: طویل مدتی سسٹمز میں میموری استعمال کا خیال رکھیں
3. **تھریڈنگ**: ROS 2 کے بلٹ ان تھریڈنگ ماڈل کو مناسب طریقے سے استعمال کریں
4. **QoS کنفیگریشن**: اپنی ایپلیکیشن کے لیے مناسب QoS ترتیبات منتخب کریں

### ٹیسٹنگ اور توثیق

```python
import unittest
from unittest.mock import Mock, patch
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


class TestRobotController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = RobotControllerNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_node_initialization(self):
        """ٹیسٹ کریں کہ نوڈ صحیح طریقے سے شروع ہوتا ہے"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'robot_controller')

    def test_parameter_setting(self):
        """پیرامیٹر سیٹنگ کی فعالیت کو ٹیسٹ کریں"""
        self.node.set_target_velocity(1.0, 0.5)
        self.assertEqual(self.node.target_velocity.linear.x, 1.0)
        self.assertEqual(self.node.target_velocity.angular.z, 0.5)
```

## جسمانی ای آئی سسٹمز کے ساتھ انضمام

### AI/ML انضمام

Python کا مضبوط AI/ML ایکو سسٹم ROS 2 کے ساتھ بلا جھگڑ انضام ہوتا ہے:

```python
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class AIPoweredRobotNode(Node):
    def __init__(self):
        super().__init__('ai_robot')

        # AI ماڈل شروع کریں
        self.model = tf.keras.models.load_model('path/to/model')
        self.bridge = CvBridge()

        # کیمرہ ڈیٹا کے لیے سبسکرائب کریں
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # AI فیصلے شائع کریں
        self.ai_cmd_pub = self.create_publisher(Twist, '/ai_cmd_vel', 10)

    def image_callback(self, msg):
        """AI ماڈل کے ساتھ تصویر کو پروسیس کریں"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ماڈل کے لیے تصویر کو پری پروسیس کریں
        input_tensor = tf.convert_to_tensor(cv_image)
        input_tensor = tf.expand_dims(input_tensor, 0)  # بیچ طول و عرض شامل کریں

        # استدلال چلائیں
        prediction = self.model(input_tensor)

        # پریڈکشن کو روبوٹ کمانڈ میں تبدیل کریں
        cmd_vel = self.convert_prediction_to_command(prediction)
        self.ai_cmd_pub.publish(cmd_vel)
```

## نالج چیک

1. ament_python اور ament_cmake بِلڈ ٹائپس کے درمیان فرق کی وضاحت کریں۔
2. لانچ فائلز کا مقصد اور وضاحت کریں کہ وہ سسٹم مینجمنٹ کو کیسے بہتر بناتے ہیں۔
3. ROS 2 ترقی میں پیرامیٹرز استعمال کرنے کے کیا فوائد ہیں؟

## ہاتھوں سے مشق

1. Python نوڈز کے ساتھ ایک مکمل ROS 2 پیکج بنائیں
2. YAML کنفیگریشن کے ساتھ پیرامیٹر مینجمنٹ نافذ کریں
3. منحصر نوڈز کے ساتھ متعدد نوڈز شروع کرنے کے لیے ایک لانچ فائل بنائیں
4. مختلف پیرامیٹر کنفیگریشنز کے ساتھ سسٹم کو ٹیسٹ کریں

## خلاصہ

Python کے ساتھ ROS 2 پیکجز تیار کرنا جسمانی ای آئی سسٹمز کی ترقی کے لیے ایک طاقتور اور لچکدار نقطہ نظر فراہم کرتا ہے۔ Python کے وسیع ایکو سسٹم اور ROS 2 کے مضبوط مواصلاتی ڈھانچے کا امتزاج جٹیل روبوٹک اطلاقات کی تیز رفتار ترقی کو فعال کرتا ہے۔ مناسب پیکج سٹرکچر، پیرامیٹر مینجمنٹ، اور لانچ کنفیگریشن برقرار رکھنے اور قابل توسیع سسٹمز تیار کرنے کے لیے ضروری ہے۔

## اگلے اقدامات

اگلے باب میں، ہم ROS 2 نوڈز، ٹاپکس، اور سروسز کو زیادہ گہرائی میں تلاش کریں گے، بشمول rclpy کا استعمال کرتے ہوئے Python ایجنٹس کو ROS کنٹرولرز سے کیسے جوڑا جائے۔