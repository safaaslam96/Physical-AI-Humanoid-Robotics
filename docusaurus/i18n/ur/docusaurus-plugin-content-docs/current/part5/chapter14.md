---
title: "باب 14: بائی پیڈل لوکوموشن اور توازن کنٹرول"
sidebar_label: "باب 14: بائی پیڈل لوکوموشن"
---

# باب 14: بائی پیڈل لوکوموشن اور توازن کنٹرول

## سیکھنے کے اہداف
- انسان نما چلنے کی حرکت اور انسانی چلنے کے نمونوں کے اصولوں کو سمجھنا
- ہیومنوائڈ روبوٹس کے لیے توازن کنٹرول میکنزمز کا ماسٹر کرنا
- ZMP (Zero Moment Point) نظریہ اور کنٹرول کی حکمت عملیوں کا اطلاق کرنا
- چلنے کی حرکت کے اصلاح کے لیے مضبوط سیکھنے کی تکنیکوں کا اطلاق کرنا

## تعارف

دو پائوں والی لوکوموشن روبوٹکس میں سب سے چیلنجنگ مسائل میں سے ایک ہے، جس کے لیے پیچیدہ کنٹرول سسٹم، ڈائنیمک ماڈلنگ، اور ریل ٹائم اڈاپٹیشن کا انضمام درکار ہے۔ وہیلڈ یا ٹریکڈ روبوٹس کے برعکس، ہیومنوائڈ روبوٹس کو اپنے سینٹر آف ماس کو متبادل سپورٹ پوائنٹس پر منتقل کرتے ہوئے توازن برقرار رکھنے کی ضرورت ہوتی ہے، انسانی چلنے میں پائے جانے والے پیچیدہ توازن کنٹرول میکنزمز کی نقل کرتے ہوئے۔ یہ باب انسانی چلنے کی بائیومکینکس، دو پائوں والی توازن کے فزکس، اور مستحکم ہیومنوائڈ لوکوموشن کے لیے جدید کنٹرول کی حکمت عملیوں کا جائزہ لیتا ہے۔

## انسانی چلنے کی بائیومکینکس

### انسانی گیٹ سائیکل کو سمجھنا

انسانی چلنے کا ایک قابل پیش گیٹ سائیکل ہوتا ہے جس میں دو بنیادی ادوار ہوتے ہیں:

1. **اسٹینس فیز (60%)**: فوٹ زمین کے ساتھ رابطے میں ہے
2. **سوئنگ فیز (40%)**: فوٹ زمین سے باہر ہے، آگے بڑھ رہا ہے

```python
# گیٹ سائیکل کا تجزیہ
import numpy as np
import matplotlib.pyplot as plt

class HumanGaitAnalyzer:
    def __init__(self):
        self.cycle_duration = 1.0  # seconds
        self.stance_ratio = 0.6    # 60% stance, 40% swing
        self.stride_length = 0.7   # meters

    def analyze_gait_phase(self, time):
        """دیے گئے وقت پر گیٹ فیز کا تعین کریں"""
        normalized_time = (time % self.cycle_duration) / self.cycle_duration

        if normalized_time < self.stance_ratio:
            # اسٹینس فیز
            stance_time = normalized_time / self.stance_ratio
            return 'stance', stance_time
        else:
            # سوئنگ فیز
            swing_time = (normalized_time - self.stance_ratio) / (1 - self.stance_ratio)
            return 'swing', swing_time

    def human_foot_trajectory(self, phase, phase_time, leg='left'):
        """گیٹ سائیکل کے دوران انسان نما فوٹ ٹریجکٹری کا حساب لگائیں"""
        if phase == 'stance':
            # فوٹ زمین پر ہے، کم تحریک
            x = phase_time * self.stride_length  # بڑھتے ہوئے جسم کے ساتھ آگے بڑھیں
            y = 0 if leg == 'left' else -0.2    # لیٹرل پوزیشن
            z = 0  # زمین کا رابطہ
        else:  # swing
            # فوٹ کو اٹھا کر اور آگے لے جانے کے لیے ہموار ٹریجکٹری
            x = self.stance_ratio + phase_time * (1 - self.stance_ratio) * self.stride_length
            y = 0 if leg == 'left' else -0.2

            # عمودی لفٹ پروفائل (parabolic)
            lift_factor = 4 * phase_time * (1 - phase_time)  # 0->1->0
            z = 0.05 * lift_factor  # 5cm زیادہ سے زیادہ لفٹ

        return np.array([x, y, z])

    def com_horizontal_movement(self, time):
        """چلنے کے دوران CoM افقی حرکت کا ماڈل"""
        # ہموار پیشرفت کے ساتھ ساتھ ہلکا لیٹرل سوئے
        normalized_time = (time % self.cycle_duration) / self.cycle_duration

        # فارورڈ پیشرفت
        x = (time // self.cycle_duration) * self.stride_length + normalized_time * self.stride_length

        # لیٹرل سوئے (ڈبل سپورٹ پیٹرن کے مطابق)
        # CoM سپورٹ لیگ کی طرف شفٹ ہوتا ہے
        y = 0.05 * np.sin(2 * np.pi * time / self.cycle_duration)

        # عمودی اوسیلیشن (انسانی چلنے میں قدرتی ہے)
        z = 0.8 + 0.01 * np.cos(4 * np.pi * time / self.cycle_duration)

        return np.array([x, y, z])

    def step_timing_analysis(self, walking_speed):
        """چلنے کی رفتار کے مطابق اسٹیپ ٹائمنگ کا تجزیہ"""
        # گیٹ سائیکل کو رفتار کے مطابق ایڈجسٹ کریں
        cycle_time = max(0.6, 1.0 / walking_speed)  # استحکام کے لیے کم از کم 0.6s

        # رفتار کے مطابق اسٹیپ کی لمبائی ایڈجسٹ کریں
        adjusted_stride = min(self.stride_length * walking_speed, 0.9)  # زیادہ سے زیادہ 0.9m

        return cycle_time, adjusted_stride
```

### کلیدی بائیومکینکل اصول

انسانی چلنے میں کئی کلیدی بائیومکینکل اصول شامل ہیں:

1. **سینٹر آف ماس موومنٹ**: CoM ایک عدد 8 نمونے میں حرکت کرتا ہے
2. **ویٹ ٹرانسفر**: سپورٹ لیگز کے درمیان ہموار ٹرانزیشن
3. **اینکل سٹریٹجی**: توازن کے لیے اینکل ایڈجسٹمنٹس
4. **ہپ سٹریٹجی**: استحکام کے لیے ہپ موومنٹس
5. **آرم سوئنگ**: لیگ موومنٹس کے لیے کاؤنٹر بیلنس

## زیرو مومنٹ پوائنٹ (ZMP) نظریہ

### ZMP بنیادیات کو سمجھنا

زیرو مومنٹ پوائنٹ (ZMP) بائی پیڈل روبوٹکس میں ایک اہم تصور ہے جو زمین کے رد عمل کے زور کے نیٹ مومنٹ کے برابر ہوتا ہے جہاں صفر ہے۔

```python
# ZMP کیلکولیشن اور تجزیہ
class ZMPAnalyzer:
    def __init__(self, robot_height=0.8):
        self.robot_height = robot_height
        self.gravity = 9.81

    def calculate_zmp_simple(self, com_position, com_acceleration, foot_position):
        """سادہ انورٹڈ پینڈولم ماڈل کا استعمال کرتے ہوئے ZMP کا حساب لگائیں"""
        # ZMP_x = CoM_x - h/g * CoM_x_ddot
        # ZMP_y = CoM_y - h/g * CoM_y_ddot
        # جہاں h = CoM اونچائی، g = گریویٹی، CoM_ddot = ایکسلریشن

        zmp_x = com_position[0] - (self.robot_height / self.gravity) * com_acceleration[0]
        zmp_y = com_position[1] - (self.robot_height / self.gravity) * com_acceleration[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def calculate_zmp_full(self, com_position, com_velocity, com_acceleration,
                          angular_momentum, external_forces):
        """مکمل ڈائنیمک ایکویشنز کا استعمال کرتے ہوئے ZMP کا حساب لگائیں"""
        # ZMP = CoM position - g/CoM_ddot * (CoM_height - foot_height)
        # 2D کیس کے لیے سادہ کیا گیا

        zmp_x = com_position[0] - (self.gravity / com_acceleration[0]) * (
            self.com_height - 0  # فرض کریں کہ فوٹ زمین کی سطح پر ہے
        )

        zmp_y = com_position[1] - (self.gravity / com_acceleration[1]) * (
            self.com_height - 0
        )

        return np.array([zmp_x, zmp_y, 0])

    def calculate_support_polygon(self, left_foot, right_foot, foot_size=[0.2, 0.1]):
        """فوٹ کی پوزیشنز کے مطابق سپورٹ پولی گون کا حساب لگائیں"""
        # چیک کریں کہ ZMP سپورٹ پولی گون میں رہتا ہے
        if left_foot is not None and right_foot is not None:
            # ڈبل سپورٹ - دونوں فوٹس کا م convex hull
            left_vertices = self.foot_polygon(left_foot, foot_size)
            right_vertices = self.foot_polygon(right_foot, foot_size)

            all_vertices = np.vstack([left_vertices, right_vertices])
            return self.convex_hull(all_vertices)
        elif left_foot is not None:
            # لیفٹ فوٹ سپورٹ
            return self.foot_polygon(left_foot, foot_size)
        elif right_foot is not None:
            # رائٹ فوٹ سپورٹ
            return self.foot_polygon(right_foot, foot_size)
        else:
            # کوئی سپورٹ نہیں (ہوائی میں)
            return np.array([])

    def foot_polygon(self, foot_center, foot_size):
        """فوٹ کانٹیکٹ ایریا کی نمائندگی کرنے والے پولی گون کریں"""
        half_length, half_width = foot_size[0] / 2, foot_size[1] / 2
        dx, dy = foot_center[0], foot_center[1]

        return np.array([
            [dx - half_length, dy - half_width],
            [dx + half_length, dy - half_width],
            [dx + half_length, dy + half_width],
            [dx - half_length, dy + half_width]
        ])

    def is_zmp_stable(self, zmp_position, support_polygon):
        """چیک کریں کہ ZMP سپورٹ پولی گون کے اندر ہے"""
        # ray casting الگورتھم یا دیگر point-in-polygon ٹیسٹ کا استعمال کریں
        return self.point_in_polygon(zmp_position[:2], support_polygon)

    def point_in_polygon(self, point, polygon):
        """ray casting کا استعمال کرتے ہوئے چیک کریں کہ point polygon کے اندر ہے"""
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
        """polygon باؤنڈری پر point تلاش کریں جو دیے گئے ZMP position کے قریب ہو"""
        # یہ polygon باؤنڈری پر قریب ترین point تلاش کرے گا
        # دیے گئے ZMP position کے لیے
        pass
```

### ZMP-مبنی کنٹرول کی حکمت عملیاں

```python
# ZMP-based توازن کنٹرولرز
class ZMPController:
    def __init__(self, sampling_time=0.005):
        self.dt = sampling_time
        self.zmp_reference = np.zeros(2)
        self.zmp_actual = np.zeros(2)
        self.com_state = {'position': np.zeros(3), 'velocity': np.zeros(3), 'acceleration': np.zeros(3)}

        # کنٹرولر گینز
        self.kp = 10.0  # Proportional gain
        self.kd = 2.0   # Derivative gain
        self.ki = 0.5   # Integral gain

        # انٹیگرل اور ڈیریویٹیو اصطلاحات
        self.zmp_error_integral = np.zeros(2)
        self.prev_zmp_error = np.zeros(2)

    def compute_balance_control(self, measured_zmp, desired_zmp, support_polygon):
        """ZMP غلطی کے مطابق توازن کنٹرول آؤٹ پٹ کا حساب لگائیں"""
        # ZMP غلطی کا حساب لگائیں
        zmp_error = desired_zmp[:2] - measured_zmp[:2]

        # انٹیگرل ٹرم اپ ڈیٹ کریں (anti-windup کے ساتھ)
        self.zmp_error_integral += zmp_error * self.dt
        max_integral = 10.0  # Anti-windup limit
        self.zmp_error_integral = np.clip(
            self.zmp_error_integral, -max_integral, max_integral
        )

        # ڈیریویٹیو ٹرم کا حساب لگائیں
        zmp_error_derivative = (zmp_error - self.prev_zmp_error) / self.dt
        self.prev_zmp_error = zmp_error

        # کنٹرول آؤٹ پٹ کا حساب لگائیں
        p_term = self.kp * zmp_error
        i_term = self.ki * self.zmp_error_integral
        d_term = self.kd * (zmp_error_derivative)

        control_output = p_term + i_term + d_term

        return control_output

    def generate_zmp_trajectory(self, num_steps, step_width=0.2):
        """ZMP ٹریجکٹری بنائیں"""
        # ڈبل سپورٹ فیز (DS)
        ds_duration = 0.1  # 10% ڈبل سپورٹ
        ss_duration = 0.9  # 90% سنگل سپورٹ

        trajectory = []
        current_zmp = np.array([0.0, 0.0])

        for step in range(num_steps):
            # ڈبل سپورٹ فیز
            ds_steps = int(ds_duration / self.dt)
            for i in range(ds_steps):
                t = i / ds_steps  # نارملائزڈ ٹائم (0 to 1)

                # ویٹ ٹرانسفر کے لیے ZMP
                if step % 2 == 0:  # لیفٹ فوٹ سپورٹ
                    zmp_y = step_width/2 + t * (-step_width)  # لیفٹ سے رائٹ
                else:  # رائٹ فوٹ سپورٹ
                    zmp_y = -step_width/2 + t * (step_width)  # رائٹ سے لیفٹ

                trajectory.append({
                    'time': step * (ds_duration + ss_duration) + t * ds_duration,
                    'zmp_position': np.array([0.0, zmp_y, 0.0]),
                    'support_phase': 'double_support'
                })

            # سنگل سپورٹ فیز
            ss_steps = int(ss_duration / self.dt)
            for i in range(ss_steps):
                t = i / ss_steps  # نارملائزڈ ٹائم

                # ہدف ZMP: سپورٹ فوٹ کے اوپر
                support_y = step_width/2 if (step+1) % 2 == 0 else -step_width/2
                zmp_pos = np.array([0.0, support_y, 0.0])

                trajectory.append({
                    'time': step * (ds_duration + ss_duration) + ds_duration + t * ss_duration,
                    'zmp_position': zmp_pos,
                    'support_phase': 'single_support'
                })

        return trajectory
```

## توازن کنٹرول میکنزمز

### اینکل سٹریٹجی کنٹرول

اینکل سٹریٹجی توازن برقرار رکھنے کے لیے بنیادی میکنزم ہے:

```python
# اینکل سٹریٹجی توازن کنٹرول
class AnkleStrategyController:
    def __init__(self, robot_height=0.8, ankle_stiffness=1000, ankle_damping=100):
        self.robot_height = robot_height
        self.ankle_stiffness = ankle_stiffness
        self.ankle_damping = ankle_damping
        self.gravity = 9.81

    def ankle_balance_control(self, com_position, com_velocity, desired_com, foot_position):
        """اینکل سٹریٹجی کا استعمال کرتے ہوئے توازن کنٹرول"""
        # CoM کی غلطی کا حساب لگائیں
        com_error = desired_com - com_position

        # CoM کے لیے ضروری اینکل ٹورکس کا حساب لگائیں
        # اینکل سٹریٹجی: CoM position error -> ankle torque
        ankle_roll_torque = -self.ankle_stiffness * com_error[1] - self.ankle_damping * com_velocity[1]
        ankle_pitch_torque = -self.ankle_stiffness * com_error[0] - self.ankle_damping * com_velocity[0]

        return np.array([ankle_roll_torque, ankle_pitch_torque])

    def ankle_impedance_control(self, desired_position, desired_velocity,
                               actual_position, actual_velocity, dt=0.001):
        """اینکل جوائنٹس کے لیے امپیڈنس کنٹرول"""
        # F = K(x_d - x) + D(v_d - v)
        position_error = desired_position - actual_position
        velocity_error = desired_velocity - actual_velocity

        force = self.ankle_stiffness * position_error + self.ankle_damping * velocity_error

        return force
```

### ہپ سٹریٹجی کنٹرول

بڑی متغیرات کے لیے ہپ سٹریٹجی ضروری ہے:

```python
# ہپ سٹریٹجی توازن کنٹرول
class HipStrategyController:
    def __init__(self, hip_stiffness=2000, hip_damping=200):
        self.hip_stiffness = hip_stiffness
        self.hip_damping = hip_damping
        self.upper_body_controller = None

    def hip_balance_control(self, com_position, com_velocity,
                           pelvis_orientation, pelvis_angular_velocity):
        """ہپ سٹریٹجی کا استعمال کرتے ہوئے توازن کنٹرول"""
        # CoM کی غلطی کا حساب لگائیں
        com_deviation = np.linalg.norm(com_position[:2])

        # چیک کریں کہ کیا اینکل سٹریٹجی کافی نہیں ہے
        if com_deviation > 0.08:  # CoM 8cm سے زیادہ ڈویٹ ہے
            # ہپ سٹریٹجی شامل کریں
            return self.engaged_hip_control(
                com_position, com_velocity,
                pelvis_orientation, pelvis_angular_velocity
            )
        else:
            # کم از کم ہپ ایڈجسٹمنٹ
            return self.minimal_hip_adjustment(
                com_position, pelvis_orientation
            )

    def engaged_hip_control(self, com_position, com_velocity,
                           pelvis_orientation, pelvis_angular_velocity):
        """مکمل ہپ سٹریٹجی اینجیجمنٹ"""
        # CoM کو مستحکم زون میں لانے کے لیے ضروری CoM موشن کا حساب لگائیں
        desired_com_offset = -0.3 * com_position[:2]  # CoM کو سپورٹ کی طرف لے جائیں

        # ہپ کے زاویہ کا حساب لگائیں
        hip_roll = 0.1 * desired_com_offset[1]  # CoM کو مستحکم کرنے کے لیے رول
        hip_pitch = -0.1 * desired_com_offset[0]  # فارورڈ/بیک ورڈ ایڈجسٹمنٹ

        # موجودہ پیلس اوریئنٹیشن نکالیں
        current_roll = pelvis_orientation[0]
        current_pitch = pelvis_orientation[1]

        # ہپ ٹورکس کا حساب لگائیں
        roll_torque = self.hip_stiffness * (hip_roll - current_roll) - self.hip_damping * pelvis_angular_velocity[0]
        pitch_torque = self.hip_stiffness * (hip_pitch - current_pitch) - self.hip_damping * pelvis_angular_velocity[1]

        # ہپ ٹورکس کو واپس کریں [L_roll, L_pitch, R_roll, R_pitch]
        return np.array([roll_torque, pitch_torque, roll_torque, pitch_torque])

    def minimal_hip_adjustment(self, com_position, pelvis_orientation):
        """اینکل سٹریٹجی کو مکمل کرنے کے لیے کم از کم ہپ ایڈجسٹمنٹ"""
        # CoM کو ہدف کی طرف لے جانے کے لیے چھوٹا ہپ ایڈجسٹمنٹ
        hip_adjustment = -0.1 * com_position[:2]  # چھوٹا کریکٹری ایڈجسٹمنٹ
        return np.array([0.1 * hip_adjustment[1], 0, 0.1 * hip_adjustment[1], 0])
```

### کیپچر پوائنٹ کنٹرول

توازن کی بازیافت کے لیے کیپچر پوائنٹ تصور:

```python
# کیپچر پوائنٹ کے تصورات کا استعمال کرتے ہوئے توازن کنٹرول
class CapturePointController:
    def __init__(self, robot_height=0.8):
        self.robot_height = robot_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.robot_height)

    def calculate_capture_point(self, com_position, com_velocity):
        """توازن کی بازیافت کے لیے کیپچر پوائنٹ کا حساب لگائیں"""
        # کیپچر پوائنٹ = CoM position + CoM velocity / omega
        # جہاں omega = sqrt(g / CoM_height)

        capture_point = com_position[:2] + com_velocity[:2] / self.omega

        return capture_point

    def capture_point_control(self, current_com, current_com_vel, target_capture_point):
        """کیپچر پوائنٹ کے مطابق توازن کنٹرول کا حساب لگائیں"""
        # موجودہ کیپچر پوائنٹ کا حساب لگائیں
        current_capture_point = self.calculate_capture_point(current_com, current_com_vel)

        # کیپچر پوائنٹ غلطی کا حساب لگائیں
        cp_error = target_capture_point - current_capture_point

        # CoM کو ہدف کیپچر پوائنٹ کی طرف لے جانے کے لیے کنٹرول کا حساب لگائیں
        control_force = 50.0 * cp_error  # سادہ تناسب کنٹرول

        return control_force

    def balance_recovery_step(self, current_com, current_com_vel):
        """توازن کی بازیافت کے لیے ضروری اسٹیپ پوزیشن کا حساب لگائیں"""
        # کیپچر پوائنٹ کا حساب لگائیں
        capture_point = self.calculate_capture_point(current_com, current_com_vel)

        # یہ اسٹیپ کو کیپچر پوائنٹ کے قریب لینے کی کوشش کرے گا
        # تاکہ CoM بائیو میں آ جائے
        step_position = capture_point

        return step_position
```

## چلنے کی پیٹرن جنریشن

### فوٹ اسٹیپ پلاننگ

```python
# ہیومنوائڈ چلنے کے لیے فوٹ اسٹیپ پلاننگ
class FootstepPlanner:
    def __init__(self):
        self.step_length = 0.6   # meters
        self.step_width = 0.2    # meters
        self.step_height = 0.05  # meters
        self.nominal_width = 0.2

    def plan_straight_walk(self, num_steps, start_position, start_orientation):
        """سیدھی چلنے کے لیے فوٹ اسٹیپس کا منصوبہ بندی کریں"""
        footsteps = []

        current_pos = np.array(start_position)
        current_orient = start_orientation  # ریڈینز میں اورینٹیشن

        for i in range(num_steps):
            # فوٹ کی پوزیشن کا حساب لگائیں
            # جفت اسٹیپس: لیفٹ فوٹ سپورٹ، طاق اسٹیپس: رائٹ فوٹ سپورٹ
            if i % 2 == 0:  # لیفٹ فوٹ سپورٹ
                foot_offset = np.array([-self.step_width/2, 0])
            else:  # رائٹ فوٹ سپورٹ
                foot_offset = np.array([self.step_width/2, 0])

            # اورینٹیشن کے مطابق آفسیٹ کو گھمائیں
            cos_o, sin_o = np.cos(current_orient), np.sin(current_orient)
            rotation_matrix = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
            rotated_offset = rotation_matrix @ foot_offset

            # فوٹ پوزیشن کا حساب لگائیں
            foot_pos = current_pos + np.array([self.step_length * (i+1), 0])
            foot_pos[:2] += rotated_offset

            footsteps.append({
                'position': foot_pos,
                'orientation': current_orient,
                'step_number': i + 1,
                'support_leg': 'right' if i % 2 == 0 else 'left'
            })

        return footsteps

    def plan_turning_walk(self, num_steps, turn_angle, start_position, start_orientation):
        """مڑنے والی چلنے کے لیے فوٹ اسٹیپس کا منصوبہ بندی کریں"""
        footsteps = []

        current_pos = np.array(start_position)
        current_orient = start_orientation
        turn_per_step = turn_angle / num_steps

        for i in range(num_steps):
            # گھمائیں والی آفسیٹ کا حساب لگائیں
            if i % 2 == 0:  # لیفٹ فوٹ سپورٹ
                foot_offset = np.array([-self.step_width/2, 0])
            else:  # رائٹ فوٹ سپورٹ
                foot_offset = np.array([self.step_width/2, 0])

            # اورینٹیشن کے مطابق آفسیٹ کو گھمائیں
            step_angle = current_orient + turn_per_step * i
            cos_o, sin_o = np.cos(step_angle), np.sin(step_angle)
            rotation_matrix = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
            rotated_offset = rotation_matrix @ foot_offset

            # فوٹ کی پوزیشن کا حساب لگائیں
            step_progress = self.step_length * i
            dx = step_progress * np.cos(step_angle)
            dy = step_progress * np.sin(step_angle)

            foot_pos = current_pos + np.array([dx, dy, 0])
            foot_pos[:2] += rotated_offset

            footsteps.append({
                'position': foot_pos,
                'orientation': current_orient + (i + 1) * turn_per_step,
                'step_number': i + 1,
                'support_leg': 'right' if i % 2 == 0 else 'left'
            })

        return footsteps

    def plan_terrain_adaptive_steps(self, terrain_map, start_pos, goal_pos):
        """ٹیرین کنڈیشنز کے مطابق فوٹ اسٹیپس کا منصوبہ بندی کریں"""
        # یہ راستہ منصوبہ بندی کو نافذ کرے گا جو
        # رکاوٹوں، ڈھلوانوں، اور سطح کی استحکام کو مدنظر رکھے گا
        pass
```

### واکنگ ٹریجکٹری جنریشن

```python
# ہیومنوائڈ چلنے کے لیے ٹریجکٹری جنریشن
class WalkingTrajectoryGenerator:
    def __init__(self):
        self.step_height = 0.05  # m
        self.step_length = 0.6   # m
        self.step_duration = 1.0 # s
        self.zmp_reference = np.array([0.0, 0.0])  # مطلوبہ ZMP

    def generate_walk_trajectory(self, num_steps, step_width=0.2):
        """مکمل چلنے کی ٹریجکٹری بنائیں"""
        trajectory = []

        for step in range(num_steps):
            # ڈبل سپورٹ فیز جنریٹ کریں
            dsp_trajectory = self.generate_double_support_phase()

            # سنگل سپورٹ فیز (لیفٹ فوٹ سپورٹ)
            if step % 2 == 0:  # لیفٹ فوٹ سپورٹ جفت اسٹیپس کے لیے
                ssp_trajectory = self.generate_single_support_phase(
                    'left', step_width
                )
            else:  # رائٹ فوٹ سپورٹ طاق اسٹیپس کے لیے
                ssp_trajectory = self.generate_single_support_phase(
                    'right', step_width
                )

            trajectory.extend(dsp_trajectory)
            trajectory.extend(ssp_trajectory)

        return trajectory

    def generate_double_support_phase(self, duration=0.1):
        """ڈبل سپورٹ فیز ٹریجکٹری بنائیں"""
        # دونوں فوٹس زمین پر - ویٹ ٹرانسفير
        steps = int(duration / 0.01)  # فرض کریں 100Hz کنٹرول
        phase_trajectory = []

        for i in range(steps):
            t = i / steps  # نارملائزڈ ٹائم (0 سے 1)

            # فوٹ کو دوسرے فوٹ میں ہموار ویٹ ٹرانسفير
            # یہ CoM ٹریجکٹری، ZMP ٹریجکٹری، وغیرہ جنریٹ کرے گا
            phase_trajectory.append({
                'time': t,
                'com_position': self.calculate_com_trajectory(t),
                'zmp_position': self.calculate_zmp_trajectory(t),
                'joint_angles': self.calculate_joint_trajectory(t)
            })

        return phase_trajectory

    def generate_single_support_phase(self, support_foot, step_width):
        """سنگل سپورٹ فیز ٹریجکٹری بنائیں"""
        steps = int(self.step_duration / 0.01)
        phase_trajectory = []

        for i in range(steps):
            t = i / steps  # نارملائزڈ ٹائم

            # لینیئر انورٹڈ پینڈولم ماڈل کے تحت CoM ٹریجکٹری کا حساب لگائیں
            com_x = self.calculate_swing_foot_position(t, support_foot)
            com_y = self.balance_lateral_motion(t, support_foot, step_width)
            com_z = self.step_height_profile(t)  # CoM اونچائی برقرار رکھیں

            phase_trajectory.append({
                'time': t,
                'com_position': np.array([com_x, com_y, com_z]),
                'support_foot': support_foot,
                'swing_foot': 'right' if support_foot == 'left' else 'left'
            })

        return phase_trajectory

    def calculate_swing_foot_position(self, t, support_foot):
        """چلنے کے دوران سوئنگ فوٹ کی پوزیشن کا حساب لگائیں"""
        # ٹائم t پر سوئنگ فوٹ کہاں ہونا چاہیے کا حساب لگائیں
        if support_foot == 'left':
            # رائٹ فوٹ فارورڈ سوئنگ کر رہا ہے
            swing_pos = self.step_length * t  # لینیئر پروگریشن
        else:
            # لیفٹ فوٹ فارورڈ سوئنگ کر رہا ہے
            swing_pos = self.step_length * t

        return swing_pos

    def balance_lateral_motion(self, t, support_foot, step_width):
        """توازن کے لیے لیٹرل CoM موشن کا حساب لگائیں"""
        # CoM سپورٹ فوٹ پر شفٹ کریں
        if support_foot == 'left':
            target_y = step_width / 2  # CoM لیفٹ فوٹ پر
        else:
            target_y = -step_width / 2  # CoM رائٹ فوٹ پر

        # sine فنکشن کا استعمال کرتے ہوئے ہموار ٹرانزیشن
        smooth_factor = np.sin(t * np.pi)  # 0 to 1 to 0
        return target_y * smooth_factor

    def step_height_profile(self, t):
        """سوئنگ فوٹ کے لیے سٹیپ اونچائی پروفائل کا حساب لگائیں"""
        # پیرابولک سٹیپ ٹریجکٹری بنائیں
        height_factor = 4 * t * (1 - t)  # پیرابولک: 0->1->0
        return self.com_height + self.step_height * height_factor
```

## ہیومنوائڈ لوکوموشن کنٹرول سسٹم

### لینیئر انورٹڈ پینڈولم ماڈل (LIPM)

```python
# LIPM-based توازن کنٹرول
class LinearInvertedPendulumController:
    def __init__(self, robot_height=0.8):
        self.com_height = robot_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.com_height)

    def lipm_com_acceleration(self, com_position, zmp_position):
        """LIPM کا استعمال کرتے ہوئے CoM ایکسلریشن کا حساب لگائیں"""
        # LIPM: x_ddot = omega^2 * (x - x_zmp)
        com_acceleration = self.omega_sq * (com_position - zmp_position)

        return com_acceleration

    def plan_com_trajectory(self, initial_com, final_com, duration, zmp_reference=None):
        """CoM ٹریجکٹری بنائیں"""
        # Quintic polynomial interpolation کا استعمال کرتے ہوئے
        # smooth trajectory planning
        steps = int(duration / 0.01)
        trajectory = []

        for i in range(steps):
            t = i / steps  # normalized time (0 to 1)

            # Quintic polynomial coefficients
            # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            # with boundary conditions: s(0)=0, s(1)=1, s'(0)=0, s'(1)=0, s''(0)=0, s''(1)=0
            a0, a1, a2, a3, a4, a5 = 0, 0, 0, 10, -15, 6

            s = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)

            # Interpolate between initial and final CoM positions
            current_com = initial_com + s * (final_com - initial_com)

            trajectory.append({
                'time': i * 0.01,
                'com_position': current_com,
                'com_velocity': self.calculate_com_velocity(current_com, trajectory[-1] if trajectory else initial_com),
                'zmp_reference': zmp_reference if zmp_reference is not None else current_com[:2]
            })

        return trajectory

    def calculate_com_velocity(self, current_com, previous_com):
        """CoM ویلوسیٹی کا حساب لگائیں"""
        return (current_com - previous_com['com_position']) / 0.01 if previous_com else np.zeros(3)

    def compute_desired_zmp(self, com_state, com_reference):
        """CoM state کے مطابق مطلوبہ ZMP کا حساب لگائیں"""
        # Inverse LIPM dynamics: zmp = com - com_ddot / omega^2
        com_acceleration = self.calculate_com_acceleration(com_state)
        desired_zmp = com_state['position'][:2] - com_acceleration[:2] / self.omega_sq

        return desired_zmp

    def calculate_com_acceleration(self, com_state):
        """CoM ایکسلریشن کا حساب لگائیں"""
        # Central difference method for acceleration
        # This would need previous and next CoM positions
        # For now, returning zero acceleration
        return np.zeros(3)
```

### ہیومنوائڈ لوکوموشن کنٹرولر

```python
# ہیومنوائڈ لوکوموشن کنٹرولر
class HumanoidLocomotionController:
    def __init__(self):
        self.balance_controller = LinearInvertedPendulumController()
        self.footstep_planner = FootstepPlanner()
        self.trajectory_generator = WalkingTrajectoryGenerator()
        self.zmp_controller = ZMPController()

        # جوائنٹ کنٹرولرز
        self.joint_controllers = {}
        self.setup_joint_controllers()

    def setup_joint_controllers(self):
        """ہیومنوائڈ روبوٹ کے جوائنٹس کے لیے کنٹرولرز سیٹ اپ کریں"""
        # ہر جوائنٹ کے لیے PD کنٹرولر
        joint_names = [
            'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw',
            'left_knee', 'left_ankle_roll', 'left_ankle_pitch',
            'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw',
            'right_knee', 'right_ankle_roll', 'right_ankle_pitch'
        ]

        for joint_name in joint_names:
            self.joint_controllers[joint_name] = {
                'kp': 500,  # Proportional gain
                'ki': 0,    # Integral gain
                'kd': 10,   # Derivative gain
                'prev_error': 0,
                'integral_error': 0
            }

    def walk_control_loop(self, current_state, desired_velocity, dt=0.001):
        """چلنے کے کنٹرول لوپ"""
        # 1. CoM اور ZMP کا حساب لگائیں
        current_com = self.calculate_center_of_mass(current_state)
        current_zmp = self.calculate_zero_moment_point(current_state)

        # 2. ہدف CoM اور ZMP کا تعین کریں
        desired_com = self.plan_next_com_position(current_com, desired_velocity, dt)
        desired_zmp = self.balance_controller.compute_desired_zmp(
            current_com, desired_com
        )

        # 3. توازن کنٹرول کا حساب لگائیں
        balance_control = self.zmp_controller.compute_balance_control(
            current_zmp, desired_zmp, self.calculate_support_polygon(current_state)
        )

        # 4. جوائنٹ کنٹرول کا حساب لگائیں
        joint_commands = self.compute_joint_commands(balance_control, current_state)

        # 5. کنٹرول کمانڈز کو روبوٹ پر لاگو کریں
        self.apply_control_commands(joint_commands)

        return joint_commands

    def calculate_center_of_mass(self, state):
        """روبوٹ کے CoM کا حساب لگائیں"""
        # یہ روبوٹ کی ماس خصوصیات اور جوائنٹ پوزیشنز کا استعمال کرے گا
        # CoM پوزیشن کا حساب لگانے کے لیے
        pass

    def calculate_zero_moment_point(self, state):
        """روبوٹ کے ZMP کا حساب لگائیں"""
        # یہ فوٹ فورس سینسرز کی معلومات کا استعمال کرے گا
        # ZMP کا حساب لگانے کے لیے
        pass

    def calculate_support_polygon(self, state):
        """سپورٹ پولی گون کا حساب لگائیں"""
        # یہ فوٹ کی پوزیشنز کا استعمال کرے گا
        # سپورٹ پولی گون کا حساب لگانے کے لیے
        pass

    def plan_next_com_position(self, current_com, desired_velocity, dt):
        """اگلے CoM position کا منصوبہ بندی کریں"""
        # CoM کو مطلوبہ رفتار کے مطابق آگے بڑھائیں
        next_com = current_com.copy()
        next_com[0] += desired_velocity[0] * dt  # x-direction
        next_com[1] += desired_velocity[1] * dt  # y-direction (sidestepping)

        return next_com

    def compute_joint_commands(self, balance_control, current_state):
        """توازن کنٹرول سے جوائنٹ کمانڈز کا حساب لگائیں"""
        # یہ balance control signals کو joint space میں میپ کرے گا
        # inverse kinematics اور inverse dynamics کا استعمال کرتے ہوئے
        pass

    def apply_control_commands(self, joint_commands):
        """کنٹرول کمانڈز کو روبوٹ پر لاگو کریں"""
        # یہ joint torques یا positions کو روبوٹ پر لاگو کرے گا
        pass

    def adaptive_gait_control(self, terrain_type, walking_speed):
        """ٹیرین کی نوعیت کے مطابق چلنے کا پیٹرن ایڈجسٹ کریں"""
        # چلنے کی رفتار، اسٹیپ سائز، اور ٹریجکٹری کو ایڈجسٹ کریں
        # ٹیرین کی خصوصیات کے مطابق
        pass

    def disturbance_recovery(self, current_state):
        """رکاوٹوں سے بازیافت کے لیے کنٹرول کا حساب لگائیں"""
        # چیک کریں کہ CoM ZMP سے باہر ہے یا نہیں
        current_zmp = self.calculate_zero_moment_point(current_state)
        current_com = self.calculate_center_of_mass(current_state)

        com_zmp_distance = np.linalg.norm(current_com[:2] - current_zmp[:2])

        if com_zmp_distance > 0.15:  # CoM ZMP سے 15cm سے زیادہ دور ہے
            # ایمرجنسی بازیافت کنٹرول
            recovery_step = self.plan_recovery_step(current_state)
            return self.execute_recovery_step(recovery_step)

        return np.zeros(28)  # 28 DOF humanoid، ابھی تک کوئی کنٹرول نہیں
```

## سیمولیشن اور ویلیڈیشن

### ڈائنیمکس سیمولیشن

```python
# ہیومنوائڈ ڈائنیمکس سیمولیشن
class HumanoidDynamicsSimulator:
    def __init__(self, robot_description):
        self.dynamics_model = HumanoidDynamics(robot_description)
        self.integration_dt = 0.001  # 1ms integration step

    def simulate_step(self, current_state, joint_torques):
        """روبوٹ ڈائنیمکس کا ایک ٹائم سٹیپ سیمولیٹ کریں"""
        # اسٹیٹ ویریبلز نکالیں
        joint_positions = current_state['joint_positions']
        joint_velocities = current_state['joint_velocities']

        # inverse dynamics کا استعمال کرتے ہوئے joint accelerations کا حساب لگائیں
        joint_accelerations = self.dynamics_model.euler_lagrange_dynamics(
            joint_positions, joint_velocities, joint_torques
        )

        # نئی ویلوسیٹیز اور پوزیشنز حاصل کرنے کے لیے انٹیگریٹ کریں
        new_velocities = joint_velocities + joint_accelerations * self.integration_dt
        new_positions = joint_positions + new_velocities * self.integration_dt

        # end effector positions کے لیے forward kinematics کا حساب لگائیں
        fk_calculator = HumanoidFK()
        end_effector_poses = fk_calculator.calculate_humanoid_pose(new_positions)

        # Center of Mass کا حساب لگائیں
        com_position = self.calculate_com_position(new_positions)

        # ZMP کا حساب لگائیں
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
        """Center of Mass position کا حساب لگائیں"""
        # یہ روبوٹ کی ماس خصوصیات اور کنیمیٹکس کا استعمال کرے گا
        # مجموعی Center of Mass کا حساب لگانے کے لیے
        pass

    def calculate_zmp(self, com_position, joint_positions, joint_velocities):
        """Zero Moment Point کا حساب لگائیں"""
        # dynamic equations کا استعمال کرتے ہوئے ZMP کا حساب لگائیں
        # ZMP = CoM - (g / CoM_z_ddot) * (CoM - foot_position)
        pass

    def validate_stability(self, state_trajectory):
        """motion trajectory کی استحکام کی توثیق کریں"""
        stability_metrics = {
            'zmp_in_support': [],
            'com_bounded': [],
            'energy_consumption': []
        }

        for state in state_trajectory:
            # چیک کریں کہ ZMP support polygon میں رہتا ہے
            zmp_in_support = self.is_zmp_stable(state['zmp_position'])
            stability_metrics['zmp_in_support'].append(zmp_in_support)

            # چیک کریں کہ CoM bounded رہتا ہے
            com_bounded = self.is_com_stable(state['com_position'])
            stability_metrics['com_bounded'].append(com_bounded)

            # توانائی کی کھپت کا حساب لگائیں
            energy = self.calculate_energy(state)
            stability_metrics['energy_consumption'].append(energy)

        return stability_metrics

    def is_zmp_stable(self, zmp_position):
        """چیک کریں کہ ZMP مستحکم علاقے میں ہے"""
        # یہ support polygon کے خلاف چیک کرے گا
        return True  # سادہ

    def is_com_stable(self, com_position):
        """چیک کریں کہ CoM مستحکم حدود میں ہے"""
        # چیک کریں کہ CoM مناسب حدود میں ہے
        return True  # سادہ

    def calculate_energy(self, state):
        """توانائی کی کھپت کا حساب لگائیں"""
        # کائنیٹک اور پوٹینشل توانائی کا حساب لگائیں
        # یہ joint velocities اور positions میں شامل ہوگا
        pass
```

## عملی مشق: ہیومنوائڈ لوکوموشن کنٹرول نافذ کریں

### مشق کے اہداف
- ہیومنوائڈ چلنے کا بنیادی کنٹرولر نافذ کریں
- ZMP-based توازن کنٹرول نفاذ کریں
- چلنے کی ٹریجکٹری جنریٹ کریں اور سیمولیٹ کریں
- استحکام کے معیارات کے ذریعے کارکردگی کی توثیق کریں

### قدم وار ہدایات

1. **ایک سادہ ہیومنوائڈ ماڈل** بنائیں جس میں دونوں ٹانگیں ہوں
2. **ZMP کنٹرولر** LIPM ماڈل کا استعمال کرتے ہوئے نافذ کریں
3. **Footstep planner** بنائیں جو چلنے کے لیے اسٹیپس منصوبہ بند کرے
4. **CoM trajectory generator** بنائیں جو مستحکم چلنے کی ٹریجکٹریز بنائے
5. **کنٹرول الگورتھم** نافذ کریں جو ZMP tracking کرے
6. **سیمولیشن** چلائیں اور توازن کی کارکردگی کا تجزیہ کریں
7. **نتائج کی توثیق** مستحکم چلنے کے معیارات کے خلاف کریں

### متوقع نتائج
- کام کرتا ہوا ہیومنوائڈ چلنے کا کنٹرولر
- ZMP-based توازن کنٹرول کی سمجھ
- چلنے کی ٹریجکٹری جنریشن کی صلاحیت
- ہیومنوائڈ لوکوموشن کا عملی تجربہ

## علم کی چیک

1. Zero Moment Point (ZMP) کے تصور کی وضاحت کریں اور اس کی اہمیت کی وضاحت کریں۔
2. لینیئر انورٹڈ پینڈولم ماڈل (LIPM) توازن کنٹرول کو کیسے سادہ بناتا ہے؟
3. اینکل اور ہپ سٹریٹجیز کے درمیان کیا فرق ہے؟
4. ہیومنوائڈ روبوٹس کے لیے چلنے کی چیلنجوں کی وضاحت کریں۔

## خلاصہ

اس باب میں ہم نے ہیومنوائڈ روبوٹس کے لیے دو پائوں والی لوکوموشن اور توازن کنٹرول کے اہم تصورات کو کور کیا۔ ZMP نظریہ، LIPM ماڈل، اور مختلف توازن کنٹرول حکمت عملیوں کو سمجھنا ہیومنوائڈ روبوٹس کو مستحکم چلنے کی صلاحیت فراہم کرنے کے لیے ضروری ہے۔ ہیومنوائڈ لوکوموشن کے لیے کنٹرول الگورتھم کا نفاذ اور سیمولیشن کے ذریعے کارکردگی کی توثیق کرنا عملی روبوٹکس کی ترقی کے لیے اہم ہے۔

## اگلے اقدامات

باب 15 میں، ہم ہیومنوائڈ مینیپولیشن اور گریسنگ کے تصورات کو تلاش کریں گے، یہاں قائم کردہ توازن اور چلنے کی بنیاد پر تعمیر کرتے ہوئے۔