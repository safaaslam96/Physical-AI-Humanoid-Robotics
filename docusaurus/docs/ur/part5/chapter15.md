---
title: "چیپٹر 15: انسان نما روبوٹکس کے لیے کنٹرول الگورتھم"
sidebar_label: "چیپٹر 15: انسان نما روبوٹکس کے لیے کنٹرول الگورتھم"
---

# چیپٹر 15: انسان نما روبوٹکس کے لیے کنٹرول الگورتھم

## سیکھنے کے اہداف
- انسان نما روبوٹ کنٹرول کے بنیادی تصورات کو سمجھنا
- بائی پیڈل چلنے کے الگورتھم تیار کرنا
- توازن کنٹرول کے لیے ZMP (Zero Moment Point) کا استعمال کرنا
- ہاتھ کے کنٹرول کے لیے impedance control کا استعمال کرنا
- ROS 2 میں کنٹرول الگورتھم کا نفاذ

## انسان نما کنٹرول کی معرفت

### انسان نما کنٹرول کیا ہے؟

انسان نما کنٹرول وہ الگورتھم ہیں جو انسان نما روبوٹ کے جسم کو کنٹرول کرتے ہیں۔ یہ کنٹرول کے چار اہم پہلوؤں کو سنبھالتے ہیں:

1. **Balance Control**: روبوٹ کو گرنے سے بچانا
2. **Locomotion**: چلنے اور حرکت کا کنٹرول
3. **Manipulation**: چیزوں کو تھامنے اور ہیرا پھیری کا کنٹرول
4. **Coordination**: متعدد جسم کے حصوں کا ہم آہنگ کنٹرول

### انسان نما کنٹرول کے چیلنج

1. **High Degrees of Freedom**: 20+ جوڑ (joints)
2. **Dynamic Balance**: توازن کو برقرار رکھنا
3. **Complex Dynamics**: پیچیدہ ڈائنیمکس
4. **Real-time Requirements**: ریل ٹائم کنٹرول کی ضرورت

### کنٹرول کی سطحیں

1. **High-level Control**: کارروائیوں کی منصوبہ بندی
2. **Mid-level Control**: راستہ کی منصوبہ بندی
3. **Low-level Control**: جوڑ کے کنٹرول

## توازن کنٹرول

### توازن کنٹرول کیا ہے؟

توازن کنٹرول روبوٹ کو گرنے سے روکنے کے لیے کنٹرول الگورتھم ہے۔

### توازن کنٹرول کے طریقے

1. **Inverted Pendulum Model**: ایک جوڑ والا پینڈولم ماڈل
2. **Cart-Table Model**: کارٹ اور ٹیبل ماڈل
3. **Capture Point Method**: کیپچر پوائنٹ کا طریقہ
4. **ZMP Control**: Zero Moment Point کنٹرول

### ZMP (Zero Moment Point) کنٹرول

ZMP وہ نقطہ ہے جہاں کل گردش کا مومنٹ صفر ہے۔

#### ZMP کا فارمولہ

```
ZMP_x = (Σ(F_i * z_i - M_i)) / ΣF_i
ZMP_y = (Σ(F_i * x_i - M_i)) / ΣF_i
```

جہاں:
- F_i = زمین کی ردعمل کی قوت
- z_i = زمین کے کنٹیکٹ کا z کوآرڈینیٹ
- M_i = گردش کا مومنٹ
- Σ = کل

#### ZMP کنٹرول الگورتھم

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class ZMPController:
    def __init__(self):
        # کنٹرول پیرامیٹرز
        self.mass = 70.0  # kg
        self.gravity = 9.81  # m/s^2
        self.height = 0.85  # CoM کی اونچائی (m)

        # PID کنٹرولر کے پیرامیٹرز
        self.kp = 100.0  # تناسب کا پیرامیٹر
        self.ki = 10.0   # لکیر کا پیرامیٹر
        self.kd = 20.0   # اشتقاق کا پیرامیٹر

        # اسٹیٹس کو ذخیرہ کریں
        self.previous_error = 0.0
        self.integral_error = 0.0

        # سپورٹ پولی گون (قدم کی جگہ)
        self.support_polygon = self.define_support_polygon()

        # ڈیٹا کو ذخیرہ کرنے کے لیے
        self.zmp_history = []
        self.com_history = []

    def define_support_polygon(self):
        """سپورٹ پولی گون کی تعریف کریں"""
        # یہاں ہم دو قدموں کے لیے سپورٹ پولی گون کی تعریف کرتے ہیں
        # یہ ایک نمونہ ہے، حقیقی پولی گون کو حقیقی قدم کے مطابق ایڈجسٹ کریں
        foot_width = 0.1  # میٹر
        foot_length = 0.2  # میٹر

        support_polygon = np.array([
            [-foot_length/2, -foot_width/2],  # بائیں پا کا نیچے بائیں
            [-foot_length/2, foot_width/2],   # بائیں پا کا اوپر بائیں
            [foot_length/2, foot_width/2],    # بائیں پا کا اوپر دائیں
            [foot_length/2, -foot_width/2],   # بائیں پا کا نیچے دائیں
        ])

        return support_polygon

    def calculate_zmp(self, forces, moments):
        """ZMP کا حساب لگائیں"""
        total_force = np.sum(forces)

        if abs(total_force) < 1e-6:  # چھوٹا فورس
            return np.array([0.0, 0.0])

        zmp_x = np.sum(forces * moments[:, 0]) / total_force
        zmp_y = np.sum(forces * moments[:, 1]) / total_force

        return np.array([zmp_x, zmp_y])

    def calculate_com_from_zmp(self, zmp, dt):
        """ZMP سے CoM کا حساب لگائیں"""
        # لینیئر انورٹڈ پینڈولم ماڈل کا استعمال کریں
        omega = np.sqrt(self.gravity / self.height)

        # CoM کا حساب لگائیں
        com_x = zmp[0] + (self.height / self.gravity) * (omega**2) * (zmp[0] - self.com_history[-1][0] if self.com_history else 0)
        com_y = zmp[1] + (self.height / self.gravity) * (omega**2) * (zmp[1] - self.com_history[-1][1] if self.com_history else 0)

        return np.array([com_x, com_y, self.height])

    def is_stable(self, zmp, support_polygon):
        """چیک کریں کہ آیا روبوٹ مستحکم ہے"""
        # ZMP سپورٹ پولی گون کے اندر ہے؟
        return self.point_in_polygon(zmp, support_polygon)

    def point_in_polygon(self, point, polygon):
        """چیک کریں کہ کیا پوائنٹ پولی گون کے اندر ہے"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n + 1):
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

    def compute_balance_correction(self, current_zmp, desired_zmp):
        """توازن کی اصلاح کا حساب لگائیں"""
        # ZMP کی غلطی
        zmp_error = desired_zmp - current_zmp

        # PID کنٹرول کا استعمال کریں
        self.integral_error += zmp_error
        derivative_error = (zmp_error - self.previous_error)

        # کنٹرول آؤٹ پٹ
        control_output = (
            self.kp * zmp_error +
            self.ki * self.integral_error +
            self.kd * derivative_error
        )

        self.previous_error = zmp_error

        return control_output

    def generate_footstep_pattern(self, destination, current_position):
        """قدم کا پیٹرن تیار کریں"""
        # راستے کا ویکٹر
        direction_vector = destination - current_position
        distance = np.linalg.norm(direction_vector)

        if distance < 0.1:  # 10cm سے کم
            return []  # کوئی قدم نہیں

        # ہدف کی طرف یونٹ ویکٹر
        unit_direction = direction_vector / distance

        # قدم کی لمبائی
        step_length = 0.3  # 30cm
        num_steps = int(distance / step_length)

        footsteps = []
        for i in range(num_steps):
            step_position = current_position + (i + 1) * step_length * unit_direction

            # قدم کی چوڑائی کو ایڈجسٹ کریں (چلنے کا نمونہ)
            lateral_offset = (-1) ** i * 0.1  # بائیں/دائیں متبادل
            step_position[1] += lateral_offset

            footsteps.append(step_position)

        return footsteps

    def update_balance(self, current_com, current_zmp, dt):
        """توازن کو اپ ڈیٹ کریں"""
        # مطلوبہ ZMP کا حساب لگائیں (آدھے جسم کے ساتھ)
        desired_zmp = self.calculate_desired_zmp(current_com, current_zmp)

        # توازن کی اصلاح کا حساب لگائیں
        balance_correction = self.compute_balance_correction(current_zmp, desired_zmp)

        # اصلاح کو ایپلائی کریں
        corrected_com = current_com + balance_correction * dt

        # ہسٹری میں شامل کریں
        self.zmp_history.append(current_zmp)
        self.com_history.append(corrected_com)

        return corrected_com, balance_correction

    def calculate_desired_zmp(self, current_com, current_zmp):
        """مطلوبہ ZMP کا حساب لگائیں"""
        # مطلوبہ ZMP کو CoM کے قریب رکھیں
        # اس سے روبوٹ توازن برقرار رکھے گا
        com_xy = current_com[:2]  # X, Y صرف

        # کچھ اصلاح کے ساتھ مطلوبہ ZMP
        desired_zmp = com_xy * 0.9 + current_zmp * 0.1  # 90% CoM, 10% current ZMP

        return desired_zmp
```

## بائی پیڈل چلنے کے الگورتھم

### چلنے کے الگورتھم کی اقسام

1. **Predefined Gait Patterns**: پہلے سے طے شدہ چلنے کے نمونے
2. **Online Gait Generation**: آن لائن چلنے کی تولید
3. **Learning-based Gaits**: سیکھنے پر مبنی چلنے کے نمونے

### چلنے کے اہم اجزاء

1. **Foot Placement**: قدم کی جگہ
2. **Step Timing**: قدم کا وقت
3. **Swing Trajectory**: قدم کی حرکت
4. **Balance Recovery**: توازن کی بازیابی

### چلنے کا کنٹرول الگورتھم

```python
class BipedalWalkingController:
    def __init__(self):
        # چلنے کے پیرامیٹرز
        self.step_length = 0.3  # میٹر
        self.step_width = 0.2   # میٹر
        self.step_height = 0.05 # میٹر (سوئنگ فیز کے لیے)
        self.step_duration = 1.0 # سیکنڈ
        self.walk_frequency = 1.0 / self.step_duration  # Hz

        # جسم کے پیرامیٹرز
        self.leg_length = 0.9  # میٹر
        self.com_height = 0.85 # میٹر

        # چلنے کی حالت
        self.current_step = 0
        self.support_leg = 'left'  # یا 'right'
        self.swing_leg = 'right'   # یا 'left'

        # چلنے کا راستہ
        self.walk_path = []
        self.current_waypoint = 0

        # چلنے کی حالت
        self.is_walking = False
        self.walk_phase = 'stance'  # stance, swing, double_support

    def plan_walk(self, destination):
        """چلنے کا منصوبہ بنائیں"""
        # راستہ کا حساب لگائیں
        path = self.generate_walk_path(destination)
        self.walk_path = path
        self.current_waypoint = 0

        # چلنے کو شروع کریں
        self.start_walking()

    def generate_walk_path(self, destination):
        """چلنے کا راستہ تیار کریں"""
        # راستہ کو چھوٹے سیگمنٹس میں تقسیم کریں
        current_pos = self.get_current_position()
        direction = destination - current_pos
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # 10cm سے کم
            return []

        num_steps = int(distance / self.step_length)
        step_vector = direction / distance * self.step_length

        path = []
        for i in range(num_steps):
            step_pos = current_pos + (i + 1) * step_vector
            # قدم کی چوڑائی کو شامل کریں (بائیں/دائیں متبادل)
            lateral_offset = ((-1) ** i) * self.step_width / 2
            step_pos[1] += lateral_offset
            path.append(step_pos)

        return path

    def update_walking(self, dt):
        """چلنے کو اپ ڈیٹ کریں"""
        if not self.is_walking or len(self.walk_path) == 0:
            return

        # چلنے کے فیز کو اپ ڈیٹ کریں
        self.update_walk_phase(dt)

        # قدم کی پوزیشن کا حساب لگائیں
        current_step_pos = self.calculate_step_position(dt)

        # کمانڈز جاری کریں
        self.execute_walk_command(current_step_pos)

        # چیک کریں کہ کیا منزل تک پہنچ گئے
        if self.has_reached_destination():
            self.stop_walking()

    def update_walk_phase(self, dt):
        """چلنے کے فیز کو اپ ڈیٹ کریں"""
        # وقت کے مطابق فیز کو تبدیل کریں
        self.walk_phase_time += dt

        if self.walk_phase_time > self.step_duration:
            # نیا قدم
            self.walk_phase_time = 0
            self.switch_support_leg()
            self.current_step += 1

            if self.current_step >= len(self.walk_path):
                self.stop_walking()

    def switch_support_leg(self):
        """سپورٹ لیگ کو تبدیل کریں"""
        if self.support_leg == 'left':
            self.support_leg = 'right'
            self.swing_leg = 'left'
        else:
            self.support_leg = 'left'
            self.swing_leg = 'right'

    def calculate_step_position(self, dt):
        """قدم کی پوزیشن کا حساب لگائیں"""
        if self.current_waypoint >= len(self.walk_path):
            return self.get_current_foot_position(self.support_leg)

        target_position = self.walk_path[self.current_waypoint]

        # سوئنگ فیز کے لیے حرکت
        if self.walk_phase == 'swing':
            phase_ratio = self.walk_phase_time / (self.step_duration * 0.6)  # 60% سوئنگ
            if phase_ratio > 1.0:
                phase_ratio = 1.0

            # ہیرمیٹک انٹرپولیشن
            start_pos = self.get_current_foot_position(self.support_leg)
            end_pos = target_position

            # ہیرمیٹک انٹرپولیشن کا استعمال کریں
            t = phase_ratio
            h1 = 2*t**3 - 3*t**2 + 1
            h2 = -2*t**3 + 3*t**2
            h3 = t**3 - 2*t**2 + t
            h4 = t**3 - t**2

            swing_x = h1 * start_pos[0] + h2 * end_pos[0] + h3 * 0 + h4 * 0
            swing_y = h1 * start_pos[1] + h2 * end_pos[1] + h3 * 0 + h4 * 0
            swing_z = h1 * start_pos[2] + h2 * (end_pos[2] + self.step_height) + h3 * 0 + h4 * 0

            return np.array([swing_x, swing_y, swing_z])
        else:
            return self.get_current_foot_position(self.support_leg)

    def execute_walk_command(self, foot_position):
        """چلنے کے کمانڈز نافذ کریں"""
        # قدم کی پوزیشن کے مطابق جوڑ کے کمانڈز تیار کریں
        joint_commands = self.inverse_kinematics(foot_position, self.support_leg)

        # کمانڈز جاری کریں
        self.send_joint_commands(joint_commands)

    def inverse_kinematics(self, foot_position, leg_side):
        """انورس کنیمیٹکس کا استعمال کریں"""
        # ہم ایک سادہ 2D IK حل استعمال کریں گے
        # اصل میں، آپ ایک پیچیدہ 3D IK حل استعمال کریں گے

        if leg_side == 'left':
            # بائیں ٹانگ کے لیے IK
            hip_position = self.get_hip_position('left')
        else:
            # دائیں ٹانگ کے لیے IK
            hip_position = self.get_hip_position('right')

        # ٹانگ کے ویکٹر کا حساب لگائیں
        leg_vector = foot_position - hip_position
        leg_length = np.linalg.norm(leg_vector)

        # گھٹنے کے زاویہ کا حساب لگائیں (law of cosines)
        knee_angle = self.calculate_knee_angle(leg_length)

        # ہپ اور اینکل کے زاویے کا حساب لگائیں
        hip_pitch, ankle_pitch = self.calculate_hip_ankle_angles(leg_vector, knee_angle)

        return {
            'hip_pitch': hip_pitch,
            'knee': knee_angle,
            'ankle_pitch': ankle_pitch
        }

    def calculate_knee_angle(self, leg_length):
        """گھٹنے کے زاویہ کا حساب لگائیں"""
        # ہم فرض کرتے ہیں کہ ٹانگ کا اوپری حصہ اور نیچے کا حصہ برابر ہے
        upper_leg = self.leg_length / 2
        lower_leg = self.leg_length / 2

        # law of cosines کا استعمال کریں
        cos_angle = (upper_leg**2 + lower_leg**2 - leg_length**2) / (2 * upper_leg * lower_leg)
        cos_angle = np.clip(cos_angle, -1, 1)  # حد کے اندر رکھیں

        # گھٹنے کا زاویہ (180° سے گھٹنے کا زاویہ)
        knee_angle = np.pi - np.arccos(cos_angle)

        return knee_angle

    def calculate_hip_ankle_angles(self, leg_vector, knee_angle):
        """ہپ اور اینکل کے زاویے کا حساب لگائیں"""
        # گھٹنے کی پوزیشن کا حساب لگائیں
        hip_to_foot = leg_vector
        hip_to_knee_distance = self.leg_length / 2

        # ہپ کا زاویہ
        hip_angle = np.arctan2(hip_to_foot[2], hip_to_foot[0])  # pitch angle

        # اینکل کا زاویہ
        ankle_angle = -hip_angle  # ہپ کے زاویہ کے مطابق

        return hip_angle, ankle_angle

    def has_reached_destination(self):
        """چیک کریں کہ کیا منزل تک پہنچ گئے"""
        if self.current_waypoint >= len(self.walk_path):
            return True

        current_pos = self.get_current_position()
        destination = self.walk_path[-1] if self.walk_path else current_pos
        distance = np.linalg.norm(current_pos - destination)

        return distance < 0.2  # 20cm کے اندر

    def start_walking(self):
        """چلنے کو شروع کریں"""
        self.is_walking = True
        self.walk_phase_time = 0
        self.current_step = 0

    def stop_walking(self):
        """چلنے کو بند کریں"""
        self.is_walking = False
        self.walk_phase_time = 0
        self.current_step = 0
        self.walk_phase = 'stance'
```

## Impedance Control

### Impedance Control کیا ہے؟

Impedance control ایک کنٹرول کا طریقہ ہے جہاں روبوٹ کا مظروانہ (virtual) میکانیکل سسٹم کے طور پر سلوک کیا جاتا ہے جس کے پاس جڑاؤ، ڈیمپنگ، اور سٹفنس ہے۔

### Impedance Control کا فارمولہ

```
F = M(q) * (xddot_d - xddot) + B(q) * (xdot_d - xdot) + K(q) * (x_d - x)
```

جہاں:
- F = کمانڈ کی گئی فورس
- M = مظروانہ ماس میٹرکس
- B = مظروانہ ڈیمپنگ میٹرکس
- K = مظروانہ سٹفنس میٹرکس
- x = موجودہ پوزیشن
- x_d = مطلوبہ پوزیشن

### Impedance Control الگورتھم

```python
class ImpedanceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Impedance پیرامیٹرز
        self.mass_matrix = np.eye(6) * 1.0    # مظروانہ ماس
        self.damping_matrix = np.eye(6) * 5.0 # مظروانہ ڈیمپنگ
        self.stiffness_matrix = np.eye(6) * 100.0  # مظروانہ سٹفنس

        # ہیش کے لیے ڈیٹا
        self.desired_pose_history = []
        self.actual_pose_history = []

    def compute_impedance_force(self, desired_pose, actual_pose, desired_twist=None, actual_twist=None):
        """Impedance فورس کا حساب لگائیں"""
        # پوزیشن کی غلطی
        position_error = desired_pose[:3] - actual_pose[:3]

        # او رینٹیشن کی غلطی (کوائف کے ذریعے)
        desired_quat = desired_pose[3:]
        actual_quat = actual_pose[3:]
        orientation_error = self.quaternion_difference(desired_quat, actual_quat)

        # کل غلطی
        pose_error = np.concatenate([position_error, orientation_error])

        # ویلوسٹی کی غلطی (اگر دستیاب ہو)
        if desired_twist is not None and actual_twist is not None:
            velocity_error = desired_twist - actual_twist
        else:
            velocity_error = np.zeros(6)

        # Impedance فورس کا حساب لگائیں
        spring_force = self.stiffness_matrix @ pose_error
        damper_force = self.damping_matrix @ velocity_error

        # کل فورس
        total_force = spring_force + damper_force

        return total_force

    def quaternion_difference(self, q1, q2):
        """دو کوائف کے درمیان فرق"""
        # q1^-1 * q2 کا حساب لگائیں
        q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])  # conjugate
        q_diff = self.quaternion_multiply(q1_inv, q2)

        # چھوٹے زاویے کے لیے، ویکٹر حصہ زاویہ-اکس ویکٹر ہے
        angle_axis = 2 * np.arctan2(np.linalg.norm(q_diff[1:]), q_diff[0])
        if np.linalg.norm(q_diff[1:]) > 1e-6:
            axis = q_diff[1:] / np.linalg.norm(q_diff[1:])
            orientation_error = angle_axis * axis
        else:
            orientation_error = np.zeros(3)

        return orientation_error

    def quaternion_multiply(self, q1, q2):
        """دو کوائف کو ضرب دیں"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def update_impedance_parameters(self, new_mass=None, new_damping=None, new_stiffness=None):
        """Impedance پیرامیٹرز کو اپ ڈیٹ کریں"""
        if new_mass is not None:
            self.mass_matrix = new_mass
        if new_damping is not None:
            self.damping_matrix = new_damping
        if new_stiffness is not None:
            self.stiffness_matrix = new_stiffness

    def cartesian_impedance_control(self, target_pose, current_pose, target_twist=None, current_twist=None):
        """کارٹیزین impedance control"""
        # Impedance فورس کا حساب لگائیں
        force = self.compute_impedance_force(target_pose, current_pose, target_twist, current_twist)

        # جوڑ کے اسپیس میں فورس تبدیل کریں (Jacobian کے ذریعے)
        jacobian = self.robot_model.get_jacobian()
        joint_torques = jacobian.T @ force

        return joint_torques

class HybridImpedanceForceController:
    def __init__(self, robot_model):
        self.impedance_controller = ImpedanceController(robot_model)
        self.force_controller = ForceController()

        # ہائبرڈ کنٹرول کے لیے وزنیں
        self.position_weight = 0.8
        self.force_weight = 0.2

    def hybrid_control(self, target_pose, current_pose, target_force, actual_force):
        """ہائبرڈ impedance-force control"""
        # Impedance control کمانڈ
        impedance_torques = self.impedance_controller.cartesian_impedance_control(
            target_pose, current_pose
        )

        # Force control کمانڈ
        force_torques = self.force_controller.compute_force_control_torques(
            target_force, actual_force
        )

        # ہائبرڈ کمانڈ
        hybrid_torques = (
            self.position_weight * impedance_torques +
            self.force_weight * force_torques
        )

        return hybrid_torques
```

## ROS 2 میں کنٹرول الگورتھم

### کنٹرول کے لیے ROS 2 کمپوننٹس

```python
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class HumanoidControllerNode(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # سینسرز کے لیے سبسکرائبرز
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.ft_sensor_sub = self.create_subscription(
            WrenchStamped,
            '/ft_sensor/wrench',
            self.force_torque_callback,
            10
        )

        # کمانڈز کے لیے پبلشرز
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # کنٹرول کے لیے ٹائمر
        self.control_timer = self.create_timer(0.01, self.control_callback)  # 100Hz

        # روبوٹ کی حالت
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_joint_efforts = {}
        self.imu_data = None
        self.ft_data = None

        # کنٹرول کے ماڈیولز
        self.balance_controller = ZMPController()
        self.walking_controller = BipedalWalkingController()
        self.impedance_controller = ImpedanceController()

        # کنٹرول کی حالت
        self.control_mode = 'idle'  # idle, balance, walk, manipulate
        self.desired_trajectory = []

    def joint_state_callback(self, msg):
        """جوڑ کی حالت کو اپ ڈیٹ کریں"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_joint_efforts[name] = msg.effort[i]

    def imu_callback(self, msg):
        """IMU ڈیٹا کو اپ ڈیٹ کریں"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def force_torque_callback(self, msg):
        """فورس ٹورک ڈیٹا کو اپ ڈیٹ کریں"""
        self.ft_data = {
            'force': [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
            'torque': [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        }

    def control_callback(self):
        """مرکزی کنٹرول لوپ"""
        current_time = self.get_clock().now()
        dt = 0.01  # 100Hz

        if self.control_mode == 'balance':
            self.execute_balance_control(dt)
        elif self.control_mode == 'walk':
            self.execute_walking_control(dt)
        elif self.control_mode == 'manipulate':
            self.execute_manipulation_control(dt)
        elif self.control_mode == 'idle':
            self.execute_idle_control()

    def execute_balance_control(self, dt):
        """توازن کنٹرول انجام دیں"""
        # CoM اور ZMP کا حساب لگائیں
        current_com = self.estimate_center_of_mass()
        current_zmp = self.estimate_zero_moment_point()

        # توازن کو اپ ڈیٹ کریں
        corrected_com, balance_correction = self.balance_controller.update_balance(
            current_com, current_zmp, dt
        )

        # جوڑ کے کمانڈز تیار کریں
        joint_commands = self.generate_balance_joint_commands(balance_correction)

        # کمانڈز جاری کریں
        self.publish_joint_commands(joint_commands)

    def execute_walking_control(self, dt):
        """چلنے کا کنٹرول انجام دیں"""
        # چلنے کو اپ ڈیٹ کریں
        self.walking_controller.update_walking(dt)

        # چلنے کے کمانڈز تیار کریں
        walk_commands = self.generate_walking_joint_commands()

        # کمانڈز جاری کریں
        self.publish_joint_commands(walk_commands)

    def execute_manipulation_control(self, dt):
        """مینوپولیشن کنٹرول انجام دیں"""
        # مینوپولیشن ٹاسک کے لیے impedance control استعمال کریں
        if self.desired_trajectory:
            target_pose = self.desired_trajectory[0]  # پہلا ہدف
            current_pose = self.get_current_end_effector_pose()

            # Impedance control کمانڈز تیار کریں
            impedance_torques = self.impedance_controller.cartesian_impedance_control(
                target_pose, current_pose
            )

            # کمانڈز جاری کریں
            self.publish_torque_commands(impedance_torques)

    def estimate_center_of_mass(self):
        """CoM کا تخمینہ لگائیں"""
        # یہ ایک سادہ تخمینہ ہے
        # اصل میں، آپ روبوٹ کے ماڈل کا استعمال کریں گے
        com_x = 0.0
        com_y = 0.0
        com_z = 0.85  # تقریباً 85cm اوپر

        return np.array([com_x, com_y, com_z])

    def estimate_zero_moment_point(self):
        """ZMP کا تخمینہ لگائیں"""
        # یہ ایک تخمینہ ہے
        # اصل میں، آپ فورس ٹورک سینسرز کا استعمال کریں گے
        if self.ft_data:
            # فورس ٹورک ڈیٹا کا استعمال کریں
            fz = self.ft_data['force'][2]
            mx = self.ft_data['torque'][0]
            my = self.ft_data['torque'][1]

            if abs(fz) > 1e-6:  # چھوٹا فورس
                zmp_x = -my / fz
                zmp_y = mx / fz
                return np.array([zmp_x, zmp_y])

        # ڈیفالٹ: جسم کے مرکز پر
        return np.array([0.0, 0.0])

    def generate_balance_joint_commands(self, balance_correction):
        """توازن کے لیے جوڑ کے کمانڈز تیار کریں"""
        # توازن کی اصلاح کے مطابق جوڑ کے کمانڈز
        commands = Float64MultiArray()

        # ہم فرض کرتے ہیں کہ ہم جوڑ کے ناموں کو جانتے ہیں
        joint_names = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]

        positions = []
        for i, joint_name in enumerate(joint_names):
            # توازن کی اصلاح کے مطابق پوزیشن
            base_position = 0.0  # ڈیفالٹ پوزیشن
            correction = balance_correction[i % len(balance_correction)] * 0.1  # سکیل فیکٹر
            positions.append(base_position + correction)

        commands.data = positions
        return commands

    def generate_walking_joint_commands(self):
        """چلنے کے لیے جوڑ کے کمانڈز تیار کریں"""
        # چلنے کے کمانڈز
        commands = Float64MultiArray()

        # یہاں چلنے کے مطابق کمانڈز تیار کریں
        # اصل میں، آپ IK یا پری ڈیفائنڈ موشن پلان استعمال کریں گے

        # نمونہ: ہم سادہ کمانڈز استعمال کرتے ہیں
        commands.data = [0.0] * 12  # 12 جوڑ
        return commands

    def publish_joint_commands(self, commands):
        """جوڑ کے کمانڈز پبلش کریں"""
        self.joint_command_pub.publish(commands)

    def set_control_mode(self, mode):
        """کنٹرول موڈ سیٹ کریں"""
        if mode in ['idle', 'balance', 'walk', 'manipulate']:
            self.control_mode = mode
            self.get_logger().info(f'کنٹرول موڈ {mode} پر سیٹ کیا گیا')
        else:
            self.get_logger().warn(f'نامعلوم کنٹرول موڈ: {mode}')

    def start_walking_to(self, destination):
        """ منزل کی طرف چلنے کو شروع کریں"""
        dest_array = np.array([destination.x, destination.y, destination.z])
        self.walking_controller.plan_walk(dest_array)
        self.set_control_mode('walk')

    def stop_walking(self):
        """چلنے کو بند کریں"""
        self.walking_controller.stop_walking()
        self.set_control_mode('balance')  # توازن پر واپس جائیں
```

## جائزہ

انسان نما روبوٹ کنٹرول کا ایک پیچیدہ میدان ہے جس میں توازن، چلنے، اور مینوپولیشن کے کنٹرول کے الگورتھم شامل ہیں۔ ZMP کنٹرول، بائی پیڈل چلنے کے الگورتھم، اور impedance control انسان نما روبوٹس کے لیے بنیادی کنٹرول تکنیکیں ہیں۔ ROS 2 کے ساتھ ان کنٹرول الگورتھم کا نفاذ روبوٹک سسٹم کے انضمام کے لیے ضروری ہے۔

