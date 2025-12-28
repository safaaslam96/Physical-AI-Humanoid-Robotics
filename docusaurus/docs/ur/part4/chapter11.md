---
title: "چیپٹر 11: مینوپولیشن اور گریسنگ"
sidebar_label: "چیپٹر 11: مینوپولیشن اور گریسنگ"
---

# چیپٹر 11: مینوپولیشن اور گریسنگ

## سیکھنے کے اہداف
- انسان نما ہاتھوں کے لیے مینوپولیشن کنٹرول کو سمجھنا
- گریسنگ کے الگورتھم تیار کرنا
- ROS 2 میں مینوپولیشن کا نفاذ
- انسان نما روبوٹ کے لیے گریسنگ کے حکمت عملیاں ڈیزائن کرنا

## مینوپولیشن کی معرفت

### مینوپولیشن کیا ہے؟

مینوپولیشن روبوٹ کی صلاحیت ہے کہ وہ اشیاء کو تھامے، ہیرا پھیری کرے، اور ماحول میں اشیاء کو منتقل کرے۔ یہ انسان نما روبوٹکس کا ایک اہم پہلو ہے۔

### مینوپولیشن کے اجزاء

1. **Mechanical Design**: ہاتھوں، انگلیوں، اور جوڑوں کا ڈیزائن
2. **Sensing**: چھونے، زور، اور ٹارک کا احساس
3. **Control**: مینوپولیشن کے لیے کنٹرول الگورتھم
4. **Planning**: گریسنگ اور مینوپولیشن کے لیے منصوبہ بندی

### مینوپولیشن کے اطلاقات

1. **Object Grasping**: اشیاء کو تھامنا
2. **Object Manipulation**: اشیاء کو ہیرا پھیری کرنا
3. **Tool Use**: ٹولز کو استعمال کرنا
4. **Assembly**: اجزاء کو جوڑنا
5. **Household Tasks**: گھر کے کام کاج کرنا

## گریسنگ کے حکمت عملیاں

### گریسنگ کیا ہے؟

گریسنگ ایک طریقہ ہے جس میں روبوٹ اشیاء کو تھامنے کے لیے اپنے ہاتھ کا استعمال کرتا ہے۔ یہ تین اہم اقسام میں ہے:

1. **Precision Grasp**: چھوٹی یا نازک اشیاء کے لیے
2. **Power Grasp**: بڑی یا بھاری اشیاء کے لیے
3. **Pinch Grasp**: دو انگلیوں کے ساتھ چھوٹی اشیاء کو تھامنا

### گریسنگ کے اصول

1. **Force Closure**: ہاتھ کے ذریعے اشیاء کو اس طرح تھامنا کہ وہ ہل نہ سکے
2. **Form Closure**: ہاتھ کی شکل کے ذریعے اشیاء کو تھامنا
3. **Friction**: گریسنگ کے لیے اصطکاک کا استعمال کرنا

### گریسنگ الگورتھم

```python
import numpy as np
from scipy.spatial.distance import cdist

class GraspingAlgorithm:
    def __init__(self):
        self.hand_configuration = {
            'thumb': {'position': [0.05, 0.0, 0.0], 'range': 0.05},
            'index': {'position': [0.0, 0.05, 0.0], 'range': 0.05},
            'middle': {'position': [0.0, 0.0, 0.0], 'range': 0.05},
            'ring': {'position': [0.0, -0.05, 0.0], 'range': 0.05},
            'pinky': {'position': [-0.05, -0.05, 0.0], 'range': 0.05}
        }

    def compute_grasp_quality(self, object_mesh, grasp_pose):
        """گریسنگ کی معیار کا حساب لگائیں"""
        # گریسنگ کی معیار کا حساب لگائیں
        # یہاں ہم ایک سادہ معیار کا استعمال کرتے ہیں
        contact_points = self.find_contact_points(object_mesh, grasp_pose)
        if len(contact_points) < 2:
            return 0.0

        # گریسنگ کی مستحکمی کا حساب لگائیں
        stability_score = self.evaluate_stability(contact_points)
        force_closure_score = self.evaluate_force_closure(contact_points)

        # کل معیار
        quality = 0.6 * stability_score + 0.4 * force_closure_score
        return quality

    def find_contact_points(self, object_mesh, grasp_pose):
        """گریسنگ کے کنٹیکٹ پوائنٹس تلاش کریں"""
        # ہاتھ کے ہر جوڑ کے لیے، چیک کریں کہ کیا وہ اشیاء سے ٹکرائے گا
        contact_points = []
        for finger_name, finger_config in self.hand_configuration.items():
            # ہاتھ کی جگہ کو گریسنگ پوز کے مطابق تبدیل کریں
            finger_pos_world = self.transform_point(finger_config['position'], grasp_pose)

            # چیک کریں کہ کیا ہاتھ اشیاء سے ٹکرائے گا
            if self.is_near_object(finger_pos_world, object_mesh):
                contact_points.append(finger_pos_world)

        return contact_points

    def evaluate_stability(self, contact_points):
        """گریسنگ کی مستحکمی کا جائزہ لیں"""
        if len(contact_points) < 2:
            return 0.0

        # چیک کریں کہ کیا کنٹیکٹ پوائنٹس CoM کے گرد ہیں
        com = np.mean(contact_points, axis=0)
        distances = [np.linalg.norm(cp - com) for cp in contact_points]

        # مستحکمی معیار: زیادہ فاصلے اور زیادہ کنٹیکٹ پوائنٹس = زیادہ مستحکمی
        avg_distance = np.mean(distances) if distances else 0
        stability = min(avg_distance / 0.1, 1.0)  # نارملائز کریں

        return stability

    def evaluate_force_closure(self, contact_points):
        """Force closure کا جائزہ لیں"""
        if len(contact_points) < 3:
            return 0.0

        # Force closure کے لیے، ہم چیک کرتے ہیں کہ کیا ہاتھ CoM کے گرد اشیاء کو تھام سکتا ہے
        com = np.mean(contact_points, axis=0)

        # ہر کنٹیکٹ پوائنٹ سے CoM کی طرف قوت کا تعین
        forces = []
        for cp in contact_points:
            force_dir = (com - cp) / np.linalg.norm(com - cp)
            forces.append(force_dir)

        # Force closure کے لیے، ہمیں ایسے ہاتھ چاہیے جو CoM کی طرف قوتیں لگا سکیں
        # اس کا مطلب ہے کہ ہمیں کم از کم 3 کنٹیکٹ پوائنٹس چاہیں
        force_closure_score = len(contact_points) / 5.0  # 5 fingers کا 100%
        return force_closure_score

    def transform_point(self, point, pose):
        """ایک پوائنٹ کو ٹرانسفارم کریں"""
        # pose: [x, y, z, qx, qy, qz, qw]
        x, y, z = pose[:3]
        qx, qy, qz, qw = pose[3:]

        # کوائف کو میٹرکس میں تبدیل کریں
        rotation_matrix = self.quaternion_to_rotation_matrix([qx, qy, qz, qw])

        # ٹرانس فارم
        transformed_point = np.dot(rotation_matrix, point) + np.array([x, y, z])
        return transformed_point

    def quaternion_to_rotation_matrix(self, q):
        """کوائف کو ریٹیشن میٹرکس میں تبدیل کریں"""
        w, x, y, z = q

        # میٹرکس کا حساب
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

        return rotation_matrix

    def is_near_object(self, point, object_mesh):
        """چیک کریں کہ کیا پوائنٹ اشیاء کے قریب ہے"""
        # یہ ایک سادہ چیک ہے، اصل میں آپ اشیاء کے میش کو استعمال کریں گے
        # اس کے لیے، ہم ایک نمونہ کا استعمال کرتے ہیں
        object_center = np.array([0, 0, 0])  # نمونہ
        distance = np.linalg.norm(point - object_center)
        return distance < 0.1  # 10cm کے اندر
```

## ROS 2 میں مینوپولیشن

### MoveIt! کا استعمال

MoveIt! ROS 2 کے لیے مینوپولیشن کا ایک مقبول فریم ورک ہے:

```python
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np

class ManipulationController(Node):
    def __init__(self):
        super().__init__('manipulation_controller')

        # MoveIt! سروسز کے کلائنٹس
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.fk_client = self.create_client(GetPositionFK, 'compute_fk')

        # جوڑ کے اقدار کے لیے سبسکرائبرز اور پبلشرز
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.joint_command_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # گریسنگ کے لیے سروس
        self.grasp_service = self.create_service(GraspObject, 'grasp_object', self.grasp_callback)

        # موجودہ جوڑ کی حالت
        self.current_joint_states = JointState()

    def joint_state_callback(self, msg):
        """جوڑ کی حالت کو اپ ڈیٹ کریں"""
        self.current_joint_states = msg

    def plan_grasp_trajectory(self, object_pose, grasp_type='precision'):
        """گریسنگ ٹریجیکٹری کو منصوبہ بند کریں"""
        try:
            # ابتدائی پوزیشن
            initial_pose = self.get_current_ee_pose()

            # اشیاء کے قریب جانے کا ہدف
            approach_pose = self.calculate_approach_pose(object_pose)

            # گریسنگ ہدف
            grasp_pose = self.calculate_grasp_pose(object_pose, grasp_type)

            # ٹریجیکٹری کو منصوبہ بند کریں
            trajectory = self.generate_trajectory([initial_pose, approach_pose, grasp_pose])

            return trajectory

        except Exception as e:
            self.get_logger().error(f'گریسنگ ٹریجیکٹری منصوبہ بندی میں خرابی: {e}')
            return None

    def calculate_approach_pose(self, object_pose):
        """گریسنگ سے پہلے قریب آنے کی پوزیشن کا حساب لگائیں"""
        approach_pose = PoseStamped()
        approach_pose.header = object_pose.header

        # اشیاء کے سامنے تھوڑا پیچھے کی طرف
        approach_pose.pose.position.x = object_pose.pose.position.x - 0.1
        approach_pose.pose.position.y = object_pose.pose.position.y
        approach_pose.pose.position.z = object_pose.pose.position.z

        # او رینٹیشن کو اشیاء کے سامنے کی طرف سیٹ کریں
        approach_pose.pose.orientation = object_pose.pose.orientation

        return approach_pose

    def calculate_grasp_pose(self, object_pose, grasp_type):
        """گریسنگ کی پوزیشن کا حساب لگائیں"""
        grasp_pose = PoseStamped()
        grasp_pose.header = object_pose.header

        # اشیاء کے بالکل سامنے
        grasp_pose.pose.position.x = object_pose.pose.position.x
        grasp_pose.pose.position.y = object_pose.pose.position.y
        grasp_pose.pose.position.z = object_pose.pose.position.z

        # او رینٹیشن کو گریسنگ کے لیے موزوں کریں
        grasp_pose.pose.orientation = object_pose.pose.orientation

        return grasp_pose

    def generate_trajectory(self, waypoints):
        """ایک ٹریجیکٹری تیار کریں"""
        # یہاں ہم ایک سادہ لینیئر انٹرپولیشن کا استعمال کرتے ہیں
        # اصل میں، آپ MoveIt! کے ٹریجیکٹری پلاننگ کا استعمال کریں گے
        trajectory = []

        for i in range(len(waypoints) - 1):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]

            # انٹرپولیٹ کریں
            interpolated_poses = self.interpolate_poses(start_pose, end_pose, steps=10)
            trajectory.extend(interpolated_poses)

        return trajectory

    def interpolate_poses(self, start_pose, end_pose, steps=10):
        """دو پوزیشنز کے درمیان انٹرپولیٹ کریں"""
        poses = []
        for i in range(steps + 1):
            t = i / steps

            # لینیئر انٹرپولیشن
            pose = PoseStamped()
            pose.header = start_pose.header

            pose.pose.position.x = start_pose.pose.position.x + t * (end_pose.pose.position.x - start_pose.pose.position.x)
            pose.pose.position.y = start_pose.pose.position.y + t * (end_pose.pose.position.y - start_pose.pose.position.y)
            pose.pose.position.z = start_pose.pose.position.z + t * (end_pose.pose.position.z - start_pose.pose.position.z)

            # او رینٹیشن کے لیے، سpherical linear interpolation (SLERP) کی ضرورت ہے
            pose.pose.orientation = self.slerp_orientations(start_pose.pose.orientation, end_pose.pose.orientation, t)

            poses.append(pose)

        return poses

    def slerp_orientations(self, start_quat, end_quat, t):
        """کوائف کے لیے SLERP کا استعمال کریں"""
        # یہ ایک سادہ SLERP نافذ کاری ہے
        # اصل میں، آپ ایک لائبریری کا استعمال کریں گے
        start = np.array([start_quat.w, start_quat.x, start_quat.y, start_quat.z])
        end = np.array([end_quat.w, end_quat.x, end_quat.y, end_quat.z])

        # dot product
        dot = np.dot(start, end)

        # اگر dot product منفی ہے، تو shortest path کو یقینی بنائیں
        if dot < 0.0:
            end = -end
            dot = -dot

        # اگر dot product لگ بھگ 1.0 ہے، تو linear interpolation کافی ہے
        if dot > 0.9995:
            result = start + t * (end - start)
        else:
            # SLERP کا استعمال کریں
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * t
            sin_theta = np.sin(theta)

            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            result = s0 * start + s1 * end

        # نارملائز کریں
        result = result / np.linalg.norm(result)

        from geometry_msgs.msg import Quaternion
        quat_msg = Quaternion()
        quat_msg.w = result[0]
        quat_msg.x = result[1]
        quat_msg.y = result[2]
        quat_msg.z = result[3]

        return quat_msg

    def execute_grasp(self, trajectory):
        """گریسنگ ٹریجیکٹری کو انجام دیں"""
        for pose in trajectory:
            # جوڑ کے ویلیوز کا حساب لگائیں (Inverse Kinematics)
            joint_angles = self.inverse_kinematics(pose)

            # جوڑ کے کمانڈز بھیجیں
            self.send_joint_commands(joint_angles)

            # انتظار کریں کہ کمانڈز انجام پائیں
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

    def inverse_kinematics(self, pose):
        """انورس کنیمیٹکس کا استعمال کریں"""
        # یہاں ہم ایک سادہ IK حل استعمال کریں گے
        # اصل میں، آپ MoveIt! یا ایک اور IK لائبریری کا استعمال کریں گے
        try:
            # ابتدائی کنٹریکٹ
            initial_guess = self.current_joint_states.position

            # IK کا حل تلاش کریں
            joint_angles = self.solve_ik(pose, initial_guess)

            return joint_angles

        except Exception as e:
            self.get_logger().error(f'IK حل میں خرابی: {e}')
            return self.current_joint_states.position

    def send_joint_commands(self, joint_angles):
        """جوڑ کے کمانڈز بھیجیں"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.position = joint_angles

        self.joint_command_pub.publish(joint_cmd)

    def grasp_callback(self, request, response):
        """گریسنگ سروس کال بیک"""
        self.get_logger().info(f'گریسنگ کے لیے درخواست موصول: {request.object_name}')

        try:
            # گریسنگ کی حکمت عملی کا تعین
            grasp_strategy = self.select_grasp_strategy(request.object_name, request.object_properties)

            # ٹریجیکٹری کو منصوبہ بند کریں
            trajectory = self.plan_grasp_trajectory(request.object_pose, grasp_strategy)

            if trajectory is not None:
                # ٹریجیکٹری کو انجام دیں
                self.execute_grasp(trajectory)

                # گریسنگ کو سیٹ کریں
                self.close_gripper()

                response.success = True
                response.message = "گریسنگ کامیابی کے ساتھ انجام پایا"
            else:
                response.success = False
                response.message = "گریسنگ ٹریجیکٹری منصوبہ بندی ناکام"

        except Exception as e:
            self.get_logger().error(f'گریسنگ ایکسیکیوشن میں خرابی: {e}')
            response.success = False
            response.message = f"گریسنگ ناکام: {str(e)}"

        return response

    def select_grasp_strategy(self, object_name, object_properties):
        """گریسنگ کی حکمت عملی منتخب کریں"""
        # اشیاء کی خصوصیات کے مطابق حکمت عملی منتخب کریں
        if 'small' in object_name or 'light' in object_name:
            return 'precision'
        elif 'large' in object_name or 'heavy' in object_name:
            return 'power'
        elif 'cylinder' in object_properties or 'cup' in object_name:
            return 'cylindrical'
        else:
            return 'power'

    def close_gripper(self):
        """گریپر کو بند کریں"""
        # گریپر کو بند کرنے کے لیے کمانڈ بھیجیں
        pass

    def open_gripper(self):
        """گریپر کو کھولیں"""
        # گریپر کو کھولنے کے لیے کمانڈ بھیجیں
        pass
```

## انسان نما ہاتھ کے ڈیزائن

### انسان نما ہاتھ کی خصوصیات

1. **Opposable Thumb**: انگوٹھا کا دوسری انگلیوں کے ساتھ مخالفت
2. **Multiple Joints**: ہر انگلی میں متعدد جوڑ
3. **Tactile Sensing**: چھونے کا احساس
4. **Force Sensing**: گریسنگ کے زور کا احساس

### ہاتھ کا ڈیزائن

```python
class HumanoidHand:
    def __init__(self, hand_side='right'):
        self.side = hand_side
        self.fingers = {
            'thumb': Finger(name='thumb', joints=3, range_of_motion=[-20, 90, 45]),  # degrees
            'index': Finger(name='index', joints=3, range_of_motion=[0, 90, 45]),
            'middle': Finger(name='middle', joints=3, range_of_motion=[0, 90, 45]),
            'ring': Finger(name='ring', joints=3, range_of_motion=[0, 90, 45]),
            'pinky': Finger(name='pinky', joints=3, range_of_motion=[0, 90, 45])
        }

        # ہاتھ کے سینسرز
        self.tactile_sensors = self.initialize_tactile_sensors()
        self.force_sensors = self.initialize_force_sensors()

    def initialize_tactile_sensors(self):
        """چھونے کے سینسرز شروع کریں"""
        # ہر انگلی پر چھونے کے سینسرز
        tactile_sensors = {}
        for finger_name in self.fingers.keys():
            tactile_sensors[finger_name] = TactileSensorArray(
                num_sensors=12,  # ہر انگلی پر 12 سینسرز
                sensor_locations=self.calculate_sensor_locations(finger_name)
            )
        return tactile_sensors

    def initialize_force_sensors(self):
        """زور کے سینسرز شروع کریں"""
        # ہاتھ کے جوڑوں پر زور کے سینسرز
        force_sensors = {}
        for finger_name, finger in self.fingers.items():
            force_sensors[finger_name] = {
                'proximal': ForceSensor(),
                'middle': ForceSensor(),
                'distal': ForceSensor()
            }
        return force_sensors

    def calculate_sensor_locations(self, finger_name):
        """سینسر کی جگہ کا حساب لگائیں"""
        # ہر انگلی کے لیے سینسرز کی جگہ کا حساب
        # یہ ایک نمونہ ہے
        locations = []
        for i in range(12):  # 12 سینسرز
            location = {
                'position': [i * 0.01, 0, 0],  # ہر 1cm پر
                'orientation': [0, 0, 0, 1]  # کوائف
            }
            locations.append(location)
        return locations

    def execute_grasp(self, grasp_type, object_properties):
        """گریسنگ کو انجام دیں"""
        if grasp_type == 'precision':
            return self.precision_grasp(object_properties)
        elif grasp_type == 'power':
            return self.power_grasp(object_properties)
        elif grasp_type == 'cylindrical':
            return self.cylindrical_grasp(object_properties)
        else:
            return self.default_grasp(object_properties)

    def precision_grasp(self, object_properties):
        """درست گریسنگ"""
        # انگوٹھا اور انڈیکس انگلی کا استعمال کریں
        commands = {}

        # انگوٹھا کو حرکت دیں
        commands['thumb'] = self.calculate_finger_trajectory(
            target_position=self.calculate_thumb_position(object_properties),
            grasp_type='precision'
        )

        # انڈیکس انگلی کو حرکت دیں
        commands['index'] = self.calculate_finger_trajectory(
            target_position=self.calculate_index_position(object_properties),
            grasp_type='precision'
        )

        # دیگر انگلیاں کھلی رکھیں
        for finger_name in ['middle', 'ring', 'pinky']:
            commands[finger_name] = self.calculate_finger_trajectory(
                target_position=self.open_position,
                grasp_type='open'
            )

        return commands

    def power_grasp(self, object_properties):
        """قوت گریسنگ"""
        # تمام انگلیاں اشیاء کے گرد بند کریں
        commands = {}

        for finger_name in self.fingers.keys():
            commands[finger_name] = self.calculate_finger_trajectory(
                target_position=self.calculate_power_grasp_position(finger_name, object_properties),
                grasp_type='power'
            )

        return commands

    def cylindrical_grasp(self, object_properties):
        """سلنڈرکل گریسنگ"""
        # انگوٹھا کو ایک طرف، دیگر انگلیاں دوسری طرف
        commands = {}

        # انگوٹھا کو مخالف طرف
        commands['thumb'] = self.calculate_finger_trajectory(
            target_position=self.calculate_opposing_thumb_position(object_properties),
            grasp_type='cylindrical'
        )

        # دیگر انگلیاں ایک گروپ میں
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            commands[finger_name] = self.calculate_finger_trajectory(
                target_position=self.calculate_cylindrical_finger_position(finger_name, object_properties),
                grasp_type='cylindrical'
            )

        return commands

    def calculate_finger_trajectory(self, target_position, grasp_type):
        """انگلی کی ٹریجیکٹری کا حساب لگائیں"""
        # ہدف کے پوزیشن کے مطابق انگلی کی ٹریجیکٹری
        # یہ ایک نمونہ ہے
        trajectory = {
            'start': self.get_current_finger_position(),
            'end': target_position,
            'intermediate_points': self.interpolate_trajectory(
                self.get_current_finger_position(),
                target_position,
                steps=20
            )
        }
        return trajectory

    def adjust_grasp_force(self, tactile_data, force_data):
        """گریسنگ کے زور کو ایڈجسٹ کریں"""
        # چھونے اور زور کے ڈیٹا کو استعمال کرکے گریسنگ کے زور کو ایڈجسٹ کریں
        for finger_name, sensors in self.tactile_sensors.items():
            if self.is_object_slipping(finger_name, tactile_data, force_data):
                self.increase_grasp_force(finger_name)
            elif self.is_over_force(finger_name, force_data):
                self.decrease_grasp_force(finger_name)

    def is_object_slipping(self, finger_name, tactile_data, force_data):
        """چیک کریں کہ کیا اشیاء پھسل رہی ہے"""
        # چھونے کے ڈیٹا کو استعمال کرکے پھسلن کا جائزہ
        slip_detected = False
        # یہاں پھسلن کا الگورتھم
        return slip_detected

    def is_over_force(self, finger_name, force_data):
        """چیک کریں کہ کیا زیادہ زور لگ رہا ہے"""
        # زور کے ڈیٹا کو استعمال کرکے زیادہ زور کا جائزہ
        max_safe_force = 50  # نیوٹن
        current_force = force_data[finger_name]
        return current_force > max_safe_force
```

## جائزہ

مینوپولیشن اور گریسنگ انسان نما روبوٹکس کا ایک اہم حصہ ہے۔ گریسنگ الگورتھم، MoveIt! کا استعمال، اور انسان نما ہاتھ کے ڈیزائن کو سمجھنا روبوٹ کو اشیاء کو مؤثر طریقے سے تھامنے اور ہیرا پھیری کرنے کے قابل بناتا ہے۔ ROS 2 کے ساتھ مینوپولیشن کا نفاذ روبوٹک سسٹم کے انضمام کے لیے ضروری ہے۔

