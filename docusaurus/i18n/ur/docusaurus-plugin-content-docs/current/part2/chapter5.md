---
title: "باب 5: ROS 2 نوڈز، ٹاپکس، اور سروسز میں گہرائی"
sidebar_label: "باب 5: ROS 2 مواصلات میں گہرائی"
---

# باب 5: ROS 2 نوڈز، ٹاپکس، اور سروسز میں گہرائی

## سیکھنے کے اہداف
- ROS 2 کے اعلی درجے کے مواصلاتی نمونے میں مہارت حاصل کرنا
- rclpy کا استعمال کرتے ہوئے Python ایجنٹس اور ROS کنٹرولرز کے درمیان پل نافذ کرنا
- ہیومنوائڈ روبوٹس کے لیے URDF (یونیفائیڈ روبوٹ ڈیسکرپشن فارمیٹ) کو سمجھنا
- جسمانی ای آئی سسٹمز کے لیے اعلی درجے کی مواصلاتی تکنیکس کا اطلاق

## تعارف

یہ باب ROS 2 کے اعلی درجے کے مواصلاتی میکانزم میں گہرائی سے جاتا ہے، جسمانی ای آئی سسٹمز کے سیاق میں نوڈز، ٹاپکس، اور سروسز کے عملی امپلیمنٹیشن پر توجہ مرکز کرتا ہے۔ ہم یہ تلاش کریں گے کہ Python-مبنی ای آئی ایجنٹس کو ROS-مبنی روبوٹ کنٹرولرز سے مؤثر طریقے سے کیسے جوڑا جائے، اور ہیومنوائڈ روبوٹ سسٹمز کو بیان کرنے میں URDF کے اہم کردار کا جائزہ لیں گے۔ ترقی یافتہ مواصلاتی نمونوں کو سمجھنا جسمانی ای آئی اطلاقات تیار کرنے کے لیے ضروری ہے۔

## ترقی یافتہ نوڈ امپلیمنٹیشن

### نوڈ کمپوزیشن اور مینجمنٹ

پیچیدہ جسمانی ای آئی سسٹمز میں، نوڈز کو اکثر بڑے سسٹمز کا حصہ بن کر ترتیب دیا اور منظم کیا جانا چاہیے:

```python
#!/usr/bin/env python3
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import threading


class ComposedRobotNode(Node):
    def __init__(self):
        super().__init__('composed_robot_node')

        # مختلف پروسیسنگ تھریڈز کے لیے متعدد کال بیک گروپس تیار کریں
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.ai_group = MutuallyExclusiveCallbackGroup()

        # مختلف QoS پروفائلز کے ساتھ پبلشرز
        self.sensor_pub = self.create_publisher(
            SensorData,
            'robot/sensor_data',
            qos_profile=self._get_sensor_qos()
        )

        self.control_pub = self.create_publisher(
            ControlCommand,
            'robot/control_cmd',
            qos_profile=self._get_control_qos()
        )

        # مختلف کال بیک گروپس کے ساتھ سبسکرائبرز
        self.ai_cmd_sub = self.create_subscription(
            AICommand,
            'ai/commands',
            self.ai_command_callback,
            qos_profile=self._get_ai_qos(),
            callback_group=self.ai_group
        )

        self.user_cmd_sub = self.create_subscription(
            UserCommand,
            'user/commands',
            self.user_command_callback,
            qos_profile=self._get_user_qos(),
            callback_group=self.control_group
        )

        # مختلف کال بیک گروپس کے ساتھ ٹائمرز تیار کریں
        self.sensor_timer = self.create_timer(
            0.01,  # 100Hz سینسر پروسیسنگ کے لیے
            self.sensor_processing,
            callback_group=self.sensor_group
        )

        self.control_timer = self.create_timer(
            0.02,  # 50Hz کنٹرول کے لیے
            self.control_loop,
            callback_group=self.control_group
        )

        self.ai_timer = self.create_timer(
            0.1,   # 10Hz AI پروسیسنگ کے لیے
            self.ai_processing,
            callback_group=self.ai_group
        )

    def _get_sensor_qos(self):
        """ہائی فریکوئینسی سینسر ڈیٹا کے لیے QoS"""
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

    def _get_control_qos(self):
        """ critical control commands کے لیے QoS"""
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

    def _get_ai_qos(self):
        """AI-جنریٹڈ کمانڈز کے لیے QoS"""
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

    def sensor_processing(self):
        """ہائی فریکوئینسی سینسر پروسیسنگ"""
        # سینسر ڈیٹا کو پروسیس کریں
        sensor_data = self.acquire_sensor_data()
        self.sensor_pub.publish(sensor_data)

    def control_loop(self):
        """حفاطت کی چیکس کے ساتھ کنٹرول لوپ"""
        # حفاظتی پابندیوں کے ساتھ کنٹرول لاگک نافذ کریں
        control_cmd = self.compute_control_command()

        # شائع کرنے سے پہلے حفاظت کی چیک
        if self.is_safe_to_execute(control_cmd):
            self.control_pub.publish(control_cmd)

    def ai_processing(self):
        """AI پروسیسنگ لاگک"""
        # AI فیصلوں کو پروسیس کریں
        pass

    def ai_command_callback(self, msg):
        """AI-جنریٹڈ کمانڈز کو ہینڈل کریں"""
        self.get_logger().info(f'AI Command received: {msg.command}')
        # حفاظتی توثیق کے ساتھ AI کمانڈ پروسیس کریں

    def user_command_callback(self, msg):
        """یوزر کمانڈز کو ہینڈل کریں"""
        self.get_logger().info(f'User Command received: {msg.command}')
        # ترجیح کے انتظام کے ساتھ یوزر کمانڈ پروسیس کریں
```

### نوڈ لائف سائیکل مینجمنٹ

ترقی یافتہ نوڈ مینجمنٹ لائف سائیکل کے امور کو شامل کرتا ہے:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor
import rclpy


class LifecycleRobotController(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_robot_controller')

        # وہ اجزاء شروع کریں جن کا لائف سائیکل کے ذریعے انتظام کیا جائے گا
        self.publisher = None
        self.subscriber = None
        self.service = None
        self.action_server = None
        self.timer = None

    def on_configure(self, state):
        """نوڈ کو تشکیل دیں - پبلشرز، سبسکرائبرز، وغیرہ تیار کریں"""
        self.get_logger().info('Configuring robot controller')

        # مواصلاتی انٹرفیسز تیار کریں
        self.publisher = self.create_publisher(RobotState, 'robot/state', 10)
        self.subscriber = self.create_subscription(
            RobotCommand, 'robot/command', self.command_callback, 10
        )
        self.service = self.create_service(GetRobotInfo, 'get_robot_info', self.info_service)

        # پیچیدہ کاموں کے لیے ایکشن سرور تیار کریں
        self.action_server = ActionServer(
            self,
            RobotNavigation,
            'navigate_to_pose',
            self.execute_navigation,
            goal_callback=self.navigation_goal_callback,
            cancel_callback=self.navigation_cancel_callback
        )

        # ٹائمرز تیار کریں
        self.timer = self.create_timer(0.1, self.state_publisher)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """نوڈ کو فعال کریں - فعال آپریشنز شروع کریں"""
        self.get_logger().info('Activating robot controller')

        # فعال آپریشنز کو فعال کریں
        self.publisher.enable()

        return super().on_activate(state)

    def on_deactivate(self, state):
        """نوڈ کو غیر فعال کریں - فعال آپریشنز کو روکیں"""
        self.get_logger().info('Deactivating robot controller')

        # فعال آپریشنز کو غیر فعال کریں
        self.publisher.disable()

        return super().on_deactivate(state)

    def on_cleanup(self, state):
        """وسائل صاف کریں"""
        self.get_logger().info('Cleaning up robot controller')

        # مواصلاتی انٹرفیسز کو تباہ کریں
        self.destroy_publisher(self.publisher)
        self.destroy_subscription(self.subscriber)
        self.destroy_service(self.service)
        self.destroy_timer(self.timer)
        self.action_server.destroy()

        self.publisher = None
        self.subscriber = None
        self.service = None
        self.timer = None
        self.action_server = None

        return TransitionCallbackReturn.SUCCESS

    def command_callback(self, msg):
        """روبوٹ کمانڈز کو ہینڈل کریں"""
        if self.get_current_state().id() == LifecycleState.ACTIVE.id():
            # صرف فعال ہونے پر کمانڈ پروسیس کریں
            self.execute_command(msg)

    def state_publisher(self):
        """روبوٹ کی حالت شائع کریں"""
        if self.get_current_state().id() == LifecycleState.ACTIVE.id():
            state_msg = RobotState()
            # حالت کا پیغام آباد کریں
            self.publisher.publish(state_msg)
```

## ٹاپکس اور میسج پاسنگ میں گہرائی

### ترقی یافتہ ٹاپک نمونے

#### فین-ان نمونہ
متعدد نوڈز ایک ٹاپک پر پبلش کرنا:

```python
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # متعدد سینسر سبسکرائبرز
        self.imu_sub = self.create_subscription(
            Imu, 'sensors/imu', self.imu_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, 'sensors/lidar', self.lidar_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, 'sensors/camera', self.camera_callback, 10
        )

        # واحد فیوژن آؤٹ پٹ
        self.fused_pub = self.create_publisher(
            SensorFusionOutput, 'sensors/fused_data', 10
        )

        # ڈیٹا فیوژن ٹائمر
        self.fusion_timer = self.create_timer(0.05, self.fuse_sensor_data)

        # سینسر ڈیٹا کے لیے اسٹوریج
        self.imu_data = None
        self.lidar_data = None
        self.camera_data = None

    def imu_callback(self, msg):
        """IMU ڈیٹا ذخیرہ کریں"""
        self.imu_data = msg
        self.get_logger().debug('Received IMU data')

    def lidar_callback(self, msg):
        """LIDAR ڈیٹا ذخیرہ کریں"""
        self.lidar_data = msg
        self.get_logger().debug('Received LIDAR data')

    def camera_callback(self, msg):
        """کیمرہ ڈیٹا ذخیرہ کریں"""
        self.camera_data = msg
        self.get_logger().debug('Received camera data')

    def fuse_sensor_data(self):
        """سینسر ڈیٹا کو متحدہ نمائندگی میں ضم کریں"""
        if all([self.imu_data, self.lidar_data, self.camera_data]):
            fused_msg = SensorFusionOutput()
            # سینسر فیوژن الگورتھم نافذ کریں
            fused_msg.timestamp = self.get_clock().now().to_msg()
            fused_msg.imu_data = self.imu_data
            fused_msg.lidar_data = self.lidar_data
            fused_msg.camera_data = self.camera_data

            # فیوژن الگورتھم لاگو کریں
            fused_msg.fused_state = self.apply_sensor_fusion(
                self.imu_data, self.lidar_data, self.camera_data
            )

            self.fused_pub.publish(fused_msg)
```

#### فین-آؤٹ نمونہ
واحد پبلشر سے متعدد سبسکرائبرز:

```python
class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # روبوٹ کی حالت کے لیے واحد پبلشر
        self.state_pub = self.create_publisher(
            RobotState, 'robot/state', 10
        )

        # متعدد سبسکرائبرز ایک ہی ڈیٹا وصول کریں گے
        self.state_timer = self.create_timer(0.01, self.publish_robot_state)

    def publish_robot_state(self):
        """متعدد سبسکرائبرز کو روبوٹ کی حالت شائع کریں"""
        state_msg = RobotState()
        # موجودہ روبوٹ کی حالت کے ساتھ آباد کریں
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.joint_states = self.get_joint_states()
        state_msg.odometry = self.get_odometry()
        state_msg.imu = self.get_imu_data()

        self.state_pub.publish(state_msg)
```

### کوالٹی آف سروس (QoS) ترقی یافتہ کنفیگریشن

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, Lifespan


class QoSDemonstrationNode(Node):
    def __init__(self):
        super().__init__('qos_demo')

        # مختلف استعمال کے معاملات کے لیے مختلف QoS پروفائلز

        # حقیقی وقت کا سینسر ڈیٹا (ہائی فریکوئینسی، پرانا ڈیٹا رکھنے کی ضرورت نہیں)
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0.01)  # 10ms کے اندر پہنچنا چاہیے
        )

        # حفاظت سے متعلق کمانڈز (ضرور ترسیل ہونا چاہیے، بازیافت کے لیے رکھیں)
        self.safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
            lifespan=Lifespan(seconds=30.0)  # 30 سیکنڈ تک رکھیں
        )

        # کنفیگریشن پیرامیٹرز (غیر اکثر، مستقل ہونا چاہیے)
        self.config_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # مختلف QoS کے ساتھ پبلشرز
        self.sensor_pub = self.create_publisher(SensorData, 'sensors/data', self.sensor_qos)
        self.safety_pub = self.create_publisher(SafetyCommand, 'safety/commands', self.safety_qos)
        self.config_pub = self.create_publisher(ConfigData, 'config/parameters', self.config_qos)
```

## سروسز اور ترقی یافتہ مواصلات

### ترقی یافتہ سروس امپلیمنٹیشن

```python
from rclpy.service import Service
from rclpy.callback_groups import ReentrantCallbackGroup


class AdvancedRobotServices(Node):
    def __init__(self):
        super().__init__('robot_services')

        # دوسری سروسز کو کال کر سکنے والی سروسز کے لیے ری اینٹرینٹ کال بیک گروپ استعمال کریں
        self.service_group = ReentrantCallbackGroup()

        # مختلف مقاصد کے ساتھ متعدد سروسز
        self.move_service = self.create_service(
            MoveRobot,
            'robot/move',
            self.move_robot_callback,
            callback_group=self.service_group
        )

        self.get_state_service = self.create_service(
            GetRobotState,
            'robot/get_state',
            self.get_state_callback,
            callback_group=self.service_group
        )

        self.execute_action_service = self.create_service(
            ExecuteAction,
            'robot/execute_action',
            self.execute_action_callback,
            callback_group=self.service_group
        )

    def move_robot_callback(self, request, response):
        """روبوٹ موومنٹ کی درخواستوں کو ہینڈل کریں"""
        self.get_logger().info(f'Moving robot to position: {request.target_position}')

        try:
            # درخواست کی توثیق کریں
            if not self.is_valid_target(request.target_position):
                response.success = False
                response.message = 'Invalid target position'
                return response

            # موومنٹ انجام دیں
            success = self.execute_movement(request.target_position)

            response.success = success
            response.message = 'Movement completed' if success else 'Movement failed'

        except Exception as e:
            self.get_logger().error(f'Error in move_robot: {e}')
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_state_callback(self, request, response):
        """موجودہ روبوٹ کی حالت فراہم کریں"""
        response.state = self.get_current_robot_state()
        response.timestamp = self.get_clock().now().to_msg()
        return response

    def execute_action_callback(self, request, response):
        """پیچیدہ روبوٹ ایکشن انجام دیں"""
        self.get_logger().info(f'Executing action: {request.action_name}')

        # پیچیدہ ایکشن انجام دہی
        result = self.execute_complex_action(request)

        response.success = result.success
        response.message = result.message
        response.execution_time = result.execution_time

        return response
```

## Python ایجنٹس کو ROS کنٹرولرز سے جوڑنا

### ای آئی ایجنٹ انضمام نمونہ

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge


class AIAgentBridge(Node):
    def __init__(self):
        super().__init__('ai_agent_bridge')

        # ROS انٹرفیسز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # ای آئی ایجنٹ انٹرفیسز
        self.ai_model = self.load_ai_model()
        self.cv_bridge = CvBridge()

        # ڈیٹا اسٹوریج
        self.latest_laser_data = None
        self.latest_image_data = None
        self.ai_command_queue = []

        # ای آئی پروسیسنگ ٹائمر
        self.ai_timer = self.create_timer(0.1, self.ai_processing_loop)

        # حفاظت کے پیرامیٹرز
        self.safety_distance = 0.5  # میٹر
        self.max_velocity = 1.0     # میٹر/سیکنڈ

    def load_ai_model(self):
        """نیویگیشن کے لیے ای آئی ماڈل لوڈ کریں"""
        try:
            # اپنے تربیت یافتہ ماڈل کو لوڈ کریں
            model = tf.keras.models.load_model('path/to/navigation_model')
            self.get_logger().info('AI model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load AI model: {e}')
            return None

    def laser_callback(self, msg):
        """لیزر اسکین ڈیٹا کو پروسیس کریں"""
        self.latest_laser_data = np.array(msg.ranges)
        # غلط رینج فلٹر کریں (inf, nan)
        self.latest_laser_data = np.where(
            (self.latest_laser_data == float('inf')) |
            (self.latest_laser_data == float('nan')),
            msg.range_max,
            self.latest_laser_data
        )

    def image_callback(self, msg):
        """کیمرہ امیج ڈیٹا کو پروسیس کریں"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # ماڈل ان پٹ کے لیے ری سائز کریں
            input_image = cv2.resize(cv_image, (224, 224))
            # نارملائز کریں
            input_image = input_image.astype(np.float32) / 255.0
            self.latest_image_data = input_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def ai_processing_loop(self):
        """مرکزی ای آئی پروسیسنگ لوپ"""
        if self.ai_model is None:
            return

        if self.latest_laser_data is not None and self.latest_image_data is not None:
            # ای آئی ماڈل کے لیے ان پٹ تیار کریں
            model_input = self.prepare_model_input(
                self.latest_laser_data,
                self.latest_image_data
            )

            # ای آئی پریڈکشن حاصل کریں
            ai_output = self.ai_model.predict(np.expand_dims(model_input, axis=0))

            # ای آئی آؤٹ پٹ کو ROS کمانڈ میں تبدیل کریں
            cmd_vel = self.convert_ai_output_to_cmd_vel(ai_output)

            # حفاظتی چیکس لاگو کریں
            safe_cmd_vel = self.apply_safety_constraints(cmd_vel)

            # کمانڈ شائع کریں
            self.cmd_vel_pub.publish(safe_cmd_vel)

    def prepare_model_input(self, laser_data, image_data):
        """ای آئی ماڈل ان پٹ کے لیے سینسر ڈیٹا تیار کریں"""
        # لیزر ڈیٹا کو نارملائز کریں
        normalized_laser = laser_data / np.max(laser_data) if np.max(laser_data) > 0 else laser_data

        # سینسر ڈیٹا کو ضم کریں (یہ ایک سادہ مثال ہے)
        # عمل میں، آپ زیادہ ترقی یافتہ فیوژن تکنیکس استعمال کر سکتے ہیں
        return {
            'laser': normalized_laser,
            'image': image_data
        }

    def convert_ai_output_to_cmd_vel(self, ai_output):
        """ای آئی ماڈل آؤٹ پٹ کو ٹویسٹ پیغام میں تبدیل کریں"""
        cmd_vel = Twist()

        # مثال: ای آئی آؤٹ پٹس [linear_velocity, angular_velocity]
        cmd_vel.linear.x = float(ai_output[0][0])
        cmd_vel.angular.z = float(ai_output[0][1])

        return cmd_vel

    def apply_safety_constraints(self, cmd_vel):
        """کمانڈز پر حفاظتی پابندیاں لاگو کریں"""
        # رفتار کو محدود کریں
        cmd_vel.linear.x = max(-self.max_velocity, min(self.max_velocity, cmd_vel.linear.x))
        cmd_vel.angular.z = max(-1.0, min(1.0, cmd_vel.angular.z))

        # لیزر ڈیٹا کی بنیاد پر حفاظتی چیک
        if self.latest_laser_data is not None:
            min_distance = np.min(self.latest_laser_data)
            if min_distance < self.safety_distance:
                # اگر رکاوٹ بہت قریب ہے تو ہنگامی سٹاپ
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.get_logger().warn('Safety stop: obstacle detected')

        return cmd_vel


def main(args=None):
    rclpy.init(args=args)

    ai_agent_bridge = AIAgentBridge()

    try:
        rclpy.spin(ai_agent_bridge)
    except KeyboardInterrupt:
        ai_agent_bridge.get_logger().info('Shutting down AI agent bridge')
    finally:
        ai_agent_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## ہیومنوائڈ روبوٹس کے لیے URDF

### URDF سٹرکچر کو سمجھنا

URDF (یونیفائیڈ روبوٹ ڈیسکرپشن فارمیٹ) XML-مبنی ہے اور روبوٹ کنیمیٹکس کو بیان کرتا ہے:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- مواد -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- بیس لنک -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- ٹورسو -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- جوائنٹس لنکس کو جوڑتے ہیں -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- سر -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- بائیں بازو -->
  <link name="left_shoulder">
    <visual>
      <origin xyz="0 0.1 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.1 0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

</robot>
```

### ٹرانسمیشن اور گیزبو انضمام کے ساتھ URDF

```xml
<?xml version="1.0"?>
<robot name="humanoid_with_transmissions" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- دیگر URDF فائلز شامل کریں -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.urdf.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/transmissions.urdf.xacro"/>

  <!-- روبوٹ بیس -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- گیزبو-مخصوص ٹیگز -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <!-- ROS کنٹرول کے لیے جوائنٹ کے ساتھ ٹرانسمیشن -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- ros_control کے لیے ٹرانسمیشن -->
  <transmission name="trans_joint1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="motor1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

## جسمانی ای آئی سسٹمز کے ساتھ انضمام

### متعدد ایجنٹ مواصلاتی نمونہ

```python
class MultiAgentRobotSystem(Node):
    def __init__(self):
        super().__init__('multi_agent_system')

        # ای آئی ایجنٹ مواصلات
        self.ai_command_pub = self.create_publisher(AICommand, 'ai/commands', 10)
        self.ai_feedback_sub = self.create_subscription(AIFeedback, 'ai/feedback', self.ai_feedback_callback, 10)

        # روبوٹ کنٹرولر مواصلات
        self.robot_command_pub = self.create_publisher(RobotCommand, 'robot/commands', 10)
        self.robot_state_sub = self.create_subscription(RobotState, 'robot/state', self.robot_state_callback, 10)

        # منصوبہ بندی اور انجام دہی
        self.planning_pub = self.create_publisher(Plan, 'planning/goal', 10)
        self.execution_sub = self.create_subscription(ExecutionStatus, 'execution/status', self.execution_callback, 10)

        # مرکزی پروسیسنگ ٹائمر
        self.main_timer = self.create_timer(0.05, self.main_processing_loop)

        # اندرونی حالت
        self.current_robot_state = None
        self.current_plan = None
        self.ai_goals = []

    def main_processing_loop(self):
        """AI اور روبوٹ کنٹرول کو مربوط کرنے والا مرکزی پروسیسنگ لوپ"""
        if self.current_robot_state is None:
            return

        # AI مقاصد کو پروسیس کریں اور روبوٹ کمانڈز تیار کریں
        if self.ai_goals:
            # AI مقاصد کی بنیاد پر منصوبہ بندی کریں
            plan = self.generate_plan_from_goals(self.ai_goals, self.current_robot_state)

            # منصوبہ انجام دیں
            cmd = self.plan_to_robot_command(plan)
            self.robot_command_pub.publish(cmd)

    def ai_feedback_callback(self, msg):
        """AI سسٹم سے فیڈ بیک کو ہینڈل کریں"""
        self.get_logger().info(f'AI feedback: {msg.status}')
        # AI فیڈ بیک کو پروسیس کریں اور اندرونی حالت کو اپ ڈیٹ کریں

    def robot_state_callback(self, msg):
        """موجودہ روبوٹ کی حالت کو اپ ڈیٹ کریں"""
        self.current_robot_state = msg

    def execution_callback(self, msg):
        """انجام دہی کی حالت کی اپ ڈیٹس کو ہینڈل کریں"""
        if msg.status == ExecutionStatus.COMPLETED:
            # مکمل ہونے والے ہدف کو ہٹا دیں
            if self.ai_goals:
                self.ai_goals.pop(0)
```

## نالج چیک

1. ROS 2 میں فین-ان اور فین-آؤٹ مواصلاتی نمونوں کے درمیان فرق کی وضاحت کریں۔
2. وضاحت کریں کہ کوالٹی آف سروس (QoS) پروفائلز مواصلاتی قابل اعتمادیت کو کیسے متاثر کرتے ہیں۔
3. ہیومنوائڈ روبوٹس کے لیے URDF فائل کے کلیدی اجزاء کیا ہیں؟

## ہاتھوں سے مشق

1. ایک ROS 2 نوڈ تیار کریں جو AI ایجنٹ برج نمونہ نافذ کرتا ہے
2. کم از کم 6 ڈگریوں کے ساتھ ایک سادہ ہیومنوائڈ روبوٹ کے لیے URDF فائل ڈیزائن کریں
3. ایک سروس نافذ کریں جو متعدد کلائنٹس کو روبوٹ کی حالت کی معلومات فراہم کرتی ہے
4. مختلف QoS کنفیگریشنز کے ساتھ سسٹم کو ٹیسٹ کریں تاکہ رویے کے فرق کا مشاہدہ کیا جا سکے

## خلاصہ

اس باب نے ROS 2 کے اعلی درجے کے مواصلاتی نمونوں کو تلاش کیا ہے، Python-مبنی AI ایجنٹس اور ROS-مبنی روبوٹ کنٹرولرز کے درمیان انضمام پر توجہ مرکز کرتا ہے۔ ٹاپکس، سروسز، اور URDF میں گہرائی کا جائزہ دیکھنے والے جسمانی ای آئی سسٹمز کو تعمیر کرنے کا طریقہ دکھاتا ہے جو AI فیصلہ سازی اور جسمانی روبوٹ کنٹرول کے درمیان فرق کو مؤثر طریقے سے پا سکیں۔ یہ اعلی درجے کے نمونے سمجھنا جسمانی ای آئی اطلاقات تیار کرنے کے لیے ضروری ہے۔

## اگلے اقدامات

اگلے بابوں میں، ہم سمولیشن ماحول، NVIDIA Isaac انضمام، اور ہیومنوائڈ روبوٹکس کے لیے اعلی درجے کے جسمانی ای آئی سسٹمز کے امپلیمنٹیشن کا جائزہ لیں گے۔