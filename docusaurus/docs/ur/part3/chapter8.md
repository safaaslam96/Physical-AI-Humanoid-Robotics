---
title: "چیپٹر 8: انسان نما کنٹرول اور توازن"
sidebar_label: "چیپٹر 8: انسان نما کنٹرول اور توازن"
---

# چیپٹر 8: انسان نما کنٹرول اور توازن

## سیکھنے کے اہداف
- انسان نما توازن کنٹرول کے اصولوں کو سمجھنا
- ZMP (Zero Moment Point) کنٹرول کا تجزیہ کرنا
- بائی پیڈل چلنے کے الگورتھم تیار کرنا
- توازن کنٹرول کے لیے PID کنٹرولرز کا استعمال کرنا

## انسان نما توازن کنٹرول

### توازن کنٹرول کیوں اہم ہے؟

انسان نما روبوٹس کے لیے توازن کنٹرول ایک اہم چیلنج ہے کیونکہ وہ صرف دو پاؤں پر کھڑے ہوتے ہیں اور چلتے ہیں، جو ایک غیر مستحکم کنفیگریشن ہے۔ توازن کو برقرار رکھنے کے لیے مستقل کنٹرول اور توازن کی حکمت عملیوں کی ضرورت ہوتی ہے۔

### توازن کے اصول

1. **Center of Mass (CoM)**: جسم کا وہ مرکز جہاں تمام ماس کو ایک نقطہ پر تصور کیا جاتا ہے
2. **Zero Moment Point (ZMP)**: زمین کا وہ نقطہ جہاں کوئی گردش کا مومنٹ نہیں ہے
3. **Support Polygon**: پاؤں کے کنٹیکٹ کے ذریعے تشکیل دی گئی جیومیٹرک علاقہ
4. **Capture Point**: CoM کو کھڑے ہونے کے لیے اس جگہ پر ڈالنے کی ضرورت ہے

### توازن کی حکمت عملیاں

1. **Ankle Strategy**: ٹخنے کے جوڑوں کا استعمال کرکے توازن برقرار رکھنا
2. **Hip Strategy**: کمر کے جوڑوں کا استعمال کرکے توازن برقرار رکھنا
3. **Stepping Strategy**: ایک قدم اٹھانا توازن بحال کرنے کے لیے
4. **Arm Strategy**: ہاتھوں کو حرکت دیکر CoM کو متوازن کرنا

## ZMP (Zero Moment Point) کنٹرول

### ZMP کیا ہے؟

Zero Moment Point (ZMP) وہ نقطہ ہے جہاں کل گردش کا مومنٹ صفر ہے۔ توازن کو برقرار رکھنے کے لیے ZMP کو سپورٹ پولی گون کے اندر رکھنا ضروری ہے۔

### ZMP کیلکولیشن

```
ZMP_x = (Σ(F_i * z_i - M_i)) / ΣF_i
ZMP_y = (Σ(F_i * x_i - M_i)) / ΣF_i
```

جہاں:
- F_i = زمین کی ردعمل قوت
- z_i = زمین کے کنٹیکٹ کا z کوآرڈینیٹ
- M_i = گردش کا مومنٹ
- Σ = کل

### ZMP کنٹرول کا تصور

ZMP کنٹرول کا تصور یہ ہے کہ اگر ZMP کو سپورٹ پولی گون کے اندر رکھا جا سکتا ہے، تو روبوٹ متوازن رہے گا۔

### ZMP کنٹرول الگورتھم

```python
import numpy as np

class ZMPController:
    def __init__(self):
        self.support_polygon = None
        self.zmp_reference = np.zeros(2)
        self.com_position = np.zeros(3)

    def calculate_zmp(self, forces, moments):
        """ZMP کا حساب لگائیں"""
        total_force = np.sum(forces)
        if total_force != 0:
            zmp_x = (forces[0] * moments[0] - moments[2]) / total_force
            zmp_y = (forces[1] * moments[1] - moments[0]) / total_force
            return np.array([zmp_x, zmp_y])
        else:
            return np.zeros(2)

    def is_balanced(self, zmp, support_polygon):
        """چیک کریں کہ آیا ZMP سپورٹ پولی گون کے اندر ہے"""
        # سادہ چیک کے لیے، ہم ایک چوکور سپورٹ پولی گون کا فرض کرتے ہیں
        x_min, x_max = support_polygon[0]
        y_min, y_max = support_polygon[1]

        return (x_min <= zmp[0] <= x_max) and (y_min <= zmp[1] <= y_max)

    def generate_footstep_pattern(self, com_trajectory, zmp_reference):
        """ZMP کے حوالے سے قدم کا پیٹرن تیار کریں"""
        footsteps = []
        # یہاں پیچیدہ فوٹ اسٹیپ جنریشن الگورتھم ہوگا
        # سادگی کے لیے، ہم ایک سادہ پیٹرن کا استعمال کرتے ہیں
        for i in range(len(com_trajectory)):
            # CoM ٹریجیکٹری کے مطابق قدم کا پیٹرن
            if i % 2 == 0:
                # دائیں قدم
                foot_pos = [com_trajectory[i][0], com_trajectory[i][1] - 0.1]
            else:
                # بائیں قدم
                foot_pos = [com_trajectory[i][0], com_trajectory[i][1] + 0.1]
            footsteps.append(foot_pos)
        return footsteps
```

## PID کنٹرول کا استعمال

### PID کنٹرول کیا ہے؟

PID (Proportional-Integral-Derivative) کنٹرول ایک کلاسیکل کنٹرول تکنیک ہے جو توازن کنٹرول کے لیے استعمال ہوتی ہے۔

### PID فارمولہ

```
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt
```

جہاں:
- u(t) = کنٹرول ان پٹ
- e(t) = غلطی (مطلوبہ قیمت - اصل قیمت)
- Kp = تناسب کا کنٹرولر
- Ki = لکیر کا کنٹرولر
- Kd = اشتقاق کا کنٹرولر

### PID کنٹرول کا استعمال

```python
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, setpoint, measured_value, dt):
        """PID کنٹرول اپ ڈیٹ"""
        error = setpoint - measured_value

        # لکیر کا حساب
        self.integral += error * dt

        # اشتقاق کا حساب
        derivative = (error - self.previous_error) / dt

        # PID آؤٹ پٹ
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.previous_error = error
        return output

class BalanceController:
    def __init__(self):
        # ٹخنے کے لیے PID کنٹرولرز
        self.ankle_pitch_pid = PIDController(kp=10.0, ki=0.1, kd=1.0)
        self.ankle_roll_pid = PIDController(kp=10.0, ki=0.1, kd=1.0)

        # کمر کے لیے PID کنٹرولرز
        self.hip_pitch_pid = PIDController(kp=5.0, ki=0.05, kd=0.5)

        self.dt = 0.01  # 100 Hz کنٹرول لوپ

    def balance_control(self, current_com, desired_com, current_orientation, desired_orientation):
        """توازن کنٹرول کا اپ ڈیٹ"""
        # CoM کی غلطی
        com_error_x = desired_com[0] - current_com[0]
        com_error_y = desired_com[1] - current_com[1]

        # او رینٹیشن غلطی
        orientation_error_pitch = desired_orientation[0] - current_orientation[0]
        orientation_error_roll = desired_orientation[1] - current_orientation[1]

        # PID کنٹرولرز کا استعمال کرکے ٹارکس کا حساب لگائیں
        ankle_torque_pitch = self.ankle_pitch_pid.update(0, orientation_error_pitch, self.dt)
        ankle_torque_roll = self.ankle_roll_pid.update(0, orientation_error_roll, self.dt)
        hip_torque_pitch = self.hip_pitch_pid.update(0, com_error_x, self.dt)

        return {
            'ankle_pitch': ankle_torque_pitch,
            'ankle_roll': ankle_torque_roll,
            'hip_pitch': hip_torque_pitch
        }
```

## بائی پیڈل چلنے کے الگورتھم

### چلنے کے مراحل

1. **Stance Phase**: ایک پاؤں زمین پر ہے، دوسرا ہوا میں
2. **Swing Phase**: دوسرا پاؤں آگے بڑھ رہا ہے
3. **Double Support Phase**: دونوں پاؤں زمین پر ہیں

### چلنے کا الگورتھم

```python
class WalkingController:
    def __init__(self):
        self.step_length = 0.3  # میٹر
        self.step_width = 0.2   # میٹر
        self.step_height = 0.05 # میٹر
        self.step_duration = 1.0 # سیکنڈ

        # چلنے کی حالت
        self.left_support = True
        self.cycle_time = 0.0
        self.phase = "stance"  # stance, swing, double_support

    def generate_walking_pattern(self, num_steps):
        """چلنے کا پیٹرن تیار کریں"""
        walking_pattern = []

        for step in range(num_steps):
            # اسٹیپ فیز کا تعین
            if step % 2 == 0:
                # بائیں پاؤں سپورٹ
                support_foot = "left"
                swing_foot = "right"
            else:
                # دائیں پاؤں سپورٹ
                support_foot = "right"
                swing_foot = "left"

            # اسٹینس فیز
            stance_start = step * self.step_duration
            stance_end = stance_start + 0.6 * self.step_duration

            # سوئنگ فیز
            swing_start = stance_end
            swing_end = step * self.step_duration + self.step_duration

            # چلنے کا پیٹرن
            pattern = {
                'step': step,
                'support_foot': support_foot,
                'swing_foot': swing_foot,
                'stance_phase': (stance_start, stance_end),
                'swing_phase': (swing_start, swing_end),
                'step_position': self.calculate_step_position(step)
            }

            walking_pattern.append(pattern)

        return walking_pattern

    def calculate_step_position(self, step):
        """قدم کی پوزیشن کا حساب لگائیں"""
        x = step * self.step_length
        y = 0 if step % 2 == 0 else self.step_width
        z = 0  # زمین کی سطح
        return [x, y, z]

    def trajectory_generation(self, start_pos, end_pos, duration, dt):
        """قدموں کے لیے ٹریجیکٹری تیار کریں"""
        num_points = int(duration / dt)
        trajectory = []

        for i in range(num_points):
            t = i * dt / duration  # 0 سے 1 تک
            # ہیرمیٹک انٹرپولیشن
            pos = self.hermitian_interpolation(start_pos, end_pos, t)
            trajectory.append(pos)

        return trajectory

    def hermitian_interpolation(self, start, end, t):
        """ہیرمیٹک انٹرپولیشن استعمال کریں"""
        # ہیرمیٹک انٹرپولیشن کا فارمولہ
        h1 = 2*t**3 - 3*t**2 + 1
        h2 = -2*t**3 + 3*t**2
        h3 = t**3 - 2*t**2 + t
        h4 = t**3 - t**2

        # شروع اور اختتام کی پوزیشن کے لیے انٹرپولیٹ کریں
        result = [0, 0, 0]
        for i in range(3):  # x, y, z
            result[i] = h1 * start[i] + h2 * end[i] + h3 * 0 + h4 * 0
        return result
```

## ہیومنوائڈ کنٹرول کی حکمت عملیاں

### 1. Inverted Pendulum Model

ان ایورٹڈ پینڈولم ماڈل انسان نما روبوٹ کو ایک ایک جوڑ والے پینڈولم کے طور پر ماڈل کرتا ہے:

```
ẍ = g/h * x - 1/h * u
```

جہاں:
- x = CoM کی پوزیشن
- ẍ = CoM کا ایکسلریشن
- g = گریویٹیشنل ایکسلریشن
- h = CoM کی اونچائی
- u = کنٹرول ان پٹ

### 2. Linear Inverted Pendulum Model (LIPM)

LIPM CoM کی اونچائی کو مستقل سمجھتا ہے:

```
ẍ = ω²(x - x_zmp)
```

جہاں:
- ω² = g/h
- x_zmp = ZMP کی پوزیشن

### 3. Cart-Table Model

کارٹ ٹیبل ماڈل CoM اور ZMP کے درمیان رشتہ کو بیان کرتا ہے:

```
x_com = x_zmp + l * tanh(ω * t)
```

## جائزہ

توازن اور چلنے کے لیے کنٹرول انسان نما روبوٹکس کا ایک اہم حصہ ہے۔ ZMP کنٹرول، PID کنٹرول، اور بائی پیڈل چلنے کے الگورتھم کو سمجھنا روبوٹ کو مستحکم چلنے کی اجازت دینے کے لیے ضروری ہے۔ ان تکنیکوں کو نافذ کرنا انسان نما روبوٹ کو انسانوں کے ماحول میں مؤثر طریقے سے کام کرنے کے قابل بناتا ہے۔

