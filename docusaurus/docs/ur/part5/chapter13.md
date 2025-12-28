---
title: "چیپٹر 13: سیم ٹو ریل منتقلی"
sidebar_label: "چیپٹر 13: سیم ٹو ریل منتقلی"
---

# چیپٹر 13: سیم ٹو ریل منتقلی

## سیکھنے کے اہداف
- سیم ٹو ریل منتقلی کے تصورات کو سمجھنا
- ڈومین رینڈمائزیشن کی تکنیکوں کو نافذ کرنا
- ڈومین ایڈاپٹیشن کے الگورتھم تیار کرنا
- ROS 2 میں سیم ٹو ریل منتقلی کے لیے کنفیگریشن
- انسان نما روبوٹ کے لیے سیم ٹو ریل منتقلی کے نظام کو ڈیزائن کرنا

## سیم ٹو ریل منتقلی کی معرفت

### سیم ٹو ریل منتقلی کیا ہے؟

سیم ٹو ریل منتقلی (Sim-to-Real Transfer) وہ عمل ہے جہاں روبوٹ کو سیمولیشن میں تربیت دی جاتی ہے اور پھر حقیقی دنیا میں کام کرنے کے قابل بنایا جاتا ہے۔ یہ روبوٹکس کا ایک اہم مسئلہ ہے کیونکہ سیمولیشن اور حقیقی دنیا کے درمیان "ریلیٹی گیپ" ہوتا ہے۔

### ریلیٹی گیپ کیا ہے؟

ریلیٹی گیپ وہ فرق ہے جو سیمولیشن اور حقیقی دنیا کے درمیان ہوتا ہے:

1. **Appearance Gap**: تصاویر کا فرق (سیمولیٹڈ vs حقیقی)
2. **Dynamics Gap**: ڈائنیمکس کا فرق (ٹکس، فریکشن، وغیرہ)
3. **Sensor Gap**: سینسر ڈیٹا کا فرق
4. **Actuation Gap**: حرکت کے درمیان فرق

### سیم ٹو ریل منتقلی کی تکنیکیں

1. **Domain Randomization**: سیمولیشن میں مختلف حالات کو رینڈمائز کرنا
2. **Domain Adaptation**: سیمولیٹڈ ڈیٹا کو حقیقی ڈیٹا کے مطابق ایڈجسٹ کرنا
3. **System Identification**: حقیقی سسٹم کے پیرامیٹرز کو شناخت کرنا
4. **Systematic Testing**: سیمولیشن اور حقیقی دنیا کے درمیان موازنہ کرنا

## ڈومین رینڈمائزیشن

### ڈومین رینڈمائزیشن کیا ہے؟

ڈومین رینڈمائزیشن وہ تکنیک ہے جہاں سیمولیشن کے مختلف پیرامیٹرز کو رینڈمائز کیا جاتا ہے تاکہ تربیت یافتہ ماڈل مختلف حالات کے لیے زیادہ متنوع ہو جائے۔

### ڈومین رینڈمائزیشن کے اجزاء

1. **Appearance Randomization**: ٹیکسچرز، رنگ، لائٹنگ
2. **Dynamics Randomization**: فریکشن، ٹکس، ماس، وغیرہ
3. **Geometry Randomization**: شکل، سائز، وغیرہ
4. **Sensor Randomization**: نوائز، ڈیلے، وغیرہ

### ڈومین رینڈمائزیشن کا الگورتھم

```python
import random
import numpy as np

class DomainRandomization:
    def __init__(self):
        # سیمولیشن پیرامیٹرز کے لیے حدود
        self.simulation_params = {
            'friction_coefficients': {'min': 0.1, 'max': 1.0},
            'gravity': {'min': 9.0, 'max': 10.0},
            'mass_variance': {'min': 0.9, 'max': 1.1},
            'lighting_conditions': ['bright', 'dim', 'shadowed', 'colored'],
            'textures': ['smooth', 'rough', 'patterned', 'random'],
            'object_sizes': {'min': 0.8, 'max': 1.2},
            'sensor_noise': {'min': 0.0, 'max': 0.1}
        }

    def randomize_environment(self):
        """سیمولیشن کے ماحول کو رینڈمائز کریں"""
        randomized_params = {}

        # فریکشن کو رینڈمائز کریں
        randomized_params['friction'] = random.uniform(
            self.simulation_params['friction_coefficients']['min'],
            self.simulation_params['friction_coefficients']['max']
        )

        # گریویٹی کو رینڈمائز کریں
        randomized_params['gravity'] = random.uniform(
            self.simulation_params['gravity']['min'],
            self.simulation_params['gravity']['max']
        )

        # ماس کو رینڈمائز کریں
        randomized_params['mass_multiplier'] = random.uniform(
            self.simulation_params['mass_variance']['min'],
            self.simulation_params['mass_variance']['max']
        )

        # لائٹنگ کو رینڈمائز کریں
        randomized_params['lighting'] = random.choice(
            self.simulation_params['lighting_conditions']
        )

        # ٹیکسچرز کو رینڈمائز کریں
        randomized_params['texture'] = random.choice(
            self.simulation_params['textures']
        )

        # اشیاء کے سائز کو رینڈمائز کریں
        randomized_params['size_multiplier'] = random.uniform(
            self.simulation_params['object_sizes']['min'],
            self.simulation_params['object_sizes']['max']
        )

        # سینسر نوائز کو رینڈمائز کریں
        randomized_params['sensor_noise'] = random.uniform(
            self.simulation_params['sensor_noise']['min'],
            self.simulation_params['sensor_noise']['max']
        )

        return randomized_params

    def apply_randomization(self, simulation_env, randomized_params):
        """رینڈمائز کردہ پیرامیٹرز کو سیمولیشن ماحول میں لاگو کریں"""
        # فریکشن کو اپ ڈیٹ کریں
        simulation_env.set_friction(randomized_params['friction'])

        # گریویٹی کو اپ ڈیٹ کریں
        simulation_env.set_gravity(randomized_params['gravity'])

        # ماس کو اپ ڈیٹ کریں
        simulation_env.set_mass_multiplier(randomized_params['mass_multiplier'])

        # لائٹنگ کو اپ ڈیٹ کریں
        simulation_env.set_lighting_condition(randomized_params['lighting'])

        # ٹیکسچر کو اپ ڈیٹ کریں
        simulation_env.set_texture(randomized_params['texture'])

        # اشیاء کے سائز کو اپ ڈیٹ کریں
        simulation_env.set_object_size_multiplier(randomized_params['size_multiplier'])

        # سینسر نوائز کو اپ ڈیٹ کریں
        simulation_env.set_sensor_noise(randomized_params['sensor_noise'])

        return simulation_env

    def adaptive_randomization(self, performance_metrics):
        """کارکردگی کے مطابق رینڈمائزیشن کو ایڈجسٹ کریں"""
        # کارکردگی کے مطابق رینڈمائزیشن کو ایڈجسٹ کریں
        # اگر کارکردگی کم ہے، تو زیادہ رینڈمائزیشن
        # اگر کارکردگی زیادہ ہے، تو کم رینڈمائزیشن

        base_range = 0.5  # بنیادی رینج
        performance_factor = 1.0 - min(performance_metrics, 1.0)  # 0 سے 1 تک

        # رینڈمائزیشن کی حد کو ایڈجسٹ کریں
        adjustment = base_range * performance_factor

        # نئی حدود کے ساتھ پیرامیٹرز کو اپ ڈیٹ کریں
        self.simulation_params['friction_coefficients']['min'] = max(0.05, 0.1 - adjustment)
        self.simulation_params['friction_coefficients']['max'] = min(1.5, 1.0 + adjustment)

        return adjustment
```

## ڈومین ایڈاپٹیشن

### ڈومین ایڈاپٹیشن کیا ہے؟

ڈومین ایڈاپٹیشن وہ تکنیک ہے جہاں ماڈل کو سیمولیٹڈ ڈیٹا کو حقیقی ڈیٹا کے مطابق تبدیل کرنے کے لیے تربیت دی جاتی ہے۔

### ڈومین ایڈاپٹیشن کے طریقے

1. **Adversarial Domain Adaptation**: ڈومین ڈسکریمنیٹر کا استعمال کرنا
2. **CycleGAN**: تصاویر کو ایک ڈومین سے دوسرے ڈومین میں تبدیل کرنا
3. **Self-supervised Learning**: حقیقی ڈیٹا سے سیکھنا

### ڈومین ایڈاپٹیشن کا الگورتھم

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, num_classes):
        super(DomainAdaptationNetwork, self).__init__()

        # فیچر ایکسٹریکٹر
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # کلاسیفائر
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # ڈومین ڈسکریمنیٹر
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=0):
        """فرووارڈ پاس"""
        # فیچر ایکسٹریکٹ کریں
        features = self.feature_extractor(x)

        # کلاسیفیکیشن
        class_output = self.classifier(features)

        # ڈومین ڈسکریمنیشن (Gradient Reverse Layer کے ساتھ)
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_discriminator(reverse_features)

        return class_output, domain_output

class ReverseLayerF(torch.autograd.Function):
    """Gradient Reverse Layer"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainAdaptationTrainer:
    def __init__(self, model, source_loader, target_loader):
        self.model = model
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(self):
        """ایک ایپوچ میں تربیت دیں"""
        self.model.train()

        total_loss = 0
        for (source_data, source_labels), (target_data, _) in zip(self.source_loader, self.target_loader):
            # ڈومین کے لیبل
            source_domain_labels = torch.zeros(len(source_data))  # 0 = source
            target_domain_labels = torch.ones(len(target_data))    # 1 = target

            # آپٹیمائزر کو صفر کریں
            self.optimizer.zero_grad()

            # سورس ڈیٹا کے لیے
            source_class_pred, source_domain_pred = self.model(source_data, alpha=1.0)

            # ٹارگیٹ ڈیٹا کے لیے
            target_class_pred, target_domain_pred = self.model(target_data, alpha=1.0)

            # کلاسیفیکیشن لو
            class_loss = self.class_criterion(source_class_pred, source_labels)

            # ڈومین ڈسکریمنیشن لو
            source_domain_loss = self.domain_criterion(source_domain_pred.squeeze(), source_domain_labels)
            target_domain_loss = self.domain_criterion(target_domain_pred.squeeze(), target_domain_labels)
            domain_loss = source_domain_loss + target_domain_loss

            # کل لو
            total_batch_loss = class_loss + domain_loss

            # بیک ورڈ
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        return total_loss / len(self.source_loader)
```

## ROS 2 میں سیم ٹو ریل منتقلی

### سیم ٹو ریل کنفیگریشن

```python
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
import numpy as np

class SimToRealTransferNode(Node):
    def __init__(self):
        super().__init__('sim_to_real_transfer')

        # سیمولیشن اور حقیقی ڈیٹا کے لیے سبسکرائبرز
        self.sim_sensor_sub = self.create_subscription(JointState, '/sim/joint_states', self.sim_sensor_callback, 10)
        self.real_sensor_sub = self.create_subscription(JointState, '/real/joint_states', self.real_sensor_callback, 10)
        self.sim_image_sub = self.create_subscription(Image, '/sim/camera/image_raw', self.sim_image_callback, 10)
        self.real_image_sub = self.create_subscription(Image, '/real/camera/image_raw', self.real_image_callback, 10)

        # کمانڈز کے لیے پبلشرز
        self.sim_cmd_pub = self.create_publisher(Twist, '/sim/cmd_vel', 10)
        self.real_cmd_pub = self.create_publisher(Twist, '/real/cmd_vel', 10)

        # کیلیبریشن کے لیے سروس
        self.calibration_service = self.create_service(String, 'calibrate_sim_to_real', self.calibration_callback)

        # سیم ٹو ریل پیرامیٹرز
        self.calibration_params = {
            'position_offset': [0.0, 0.0, 0.0],
            'orientation_offset': [0.0, 0.0, 0.0, 1.0],  # quaternions
            'velocity_scale': 1.0,
            'sensor_bias': [0.0, 0.0, 0.0],
            'actuator_deadband': 0.01
        }

        # سیم ٹو ریل کے لیے ٹائمر
        self.transfer_timer = self.create_timer(0.1, self.transfer_callback)

        # ڈیٹا کو ذخیرہ کرنے کے لیے
        self.sim_data_buffer = []
        self.real_data_buffer = []

    def sim_sensor_callback(self, msg):
        """سیمولیٹڈ سینسر ڈیٹا کو ہینڈل کریں"""
        # ڈیٹا کو کیلیبریٹ کریں
        calibrated_data = self.calibrate_sensor_data(msg, 'sim')
        self.sim_data_buffer.append(calibrated_data)

        # بفر کو محدود کریں
        if len(self.sim_data_buffer) > 100:
            self.sim_data_buffer.pop(0)

    def real_sensor_callback(self, msg):
        """حقیقی سینسر ڈیٹا کو ہینڈل کریں"""
        # ڈیٹا کو کیلیبریٹ کریں
        calibrated_data = self.calibrate_sensor_data(msg, 'real')
        self.real_data_buffer.append(calibrated_data)

        # بفر کو محدود کریں
        if len(self.real_data_buffer) > 100:
            self.real_data_buffer.pop(0)

    def sim_image_callback(self, msg):
        """سیمولیٹڈ تصویر کو ہینڈل کریں"""
        # تصویر کو حقیقی ڈیٹا کے مطابق تبدیل کریں (اگر ضروری ہو)
        processed_image = self.process_simulation_image(msg)
        # یہاں آپ ایک CycleGAN یا دیگر تکنیک کا استعمال کر سکتے ہیں

    def real_image_callback(self, msg):
        """حقیقی تصویر کو ہینڈل کریں"""
        # تصویر کو سیمولیشن ڈیٹا کے مطابق تبدیل کریں (اگر ضروری ہو)
        processed_image = self.process_real_image(msg)

    def calibrate_sensor_data(self, sensor_data, domain):
        """سینسر ڈیٹا کو کیلیبریٹ کریں"""
        calibrated_data = JointState()
        calibrated_data.header = sensor_data.header
        calibrated_data.name = sensor_data.name

        if domain == 'sim':
            # سیمولیٹڈ ڈیٹا کو حقیقی ڈیٹا کے مطابق تبدیل کریں
            calibrated_data.position = [
                pos + self.calibration_params['position_offset'][i % 3]
                for i, pos in enumerate(sensor_data.position)
            ]
            calibrated_data.velocity = [
                vel * self.calibration_params['velocity_scale']
                for vel in sensor_data.velocity
            ]
            calibrated_data.effort = [
                eff + self.calibration_params['sensor_bias'][i % 3]
                for i, eff in enumerate(sensor_data.effort)
            ]
        else:
            # حقیقی ڈیٹا کو سیمولیشن ڈیٹا کے مطابق تبدیل کریں
            calibrated_data.position = [
                pos - self.calibration_params['position_offset'][i % 3]
                for i, pos in enumerate(sensor_data.position)
            ]
            calibrated_data.velocity = [
                vel / self.calibration_params['velocity_scale']
                for vel in sensor_data.velocity
            ]
            calibrated_data.effort = [
                eff - self.calibration_params['sensor_bias'][i % 3]
                for i, eff in enumerate(sensor_data.effort)
            ]

        return calibrated_data

    def calibration_callback(self, request, response):
        """کیلیبریشن کے لیے سروس کال بیک"""
        try:
            # کیلیبریشن کا طریقہ
            if request.data == 'collect_data':
                self.collect_calibration_data()
                response.data = 'Calibration data collected'
            elif request.data == 'compute_params':
                self.compute_calibration_params()
                response.data = 'Calibration parameters computed'
            elif request.data == 'apply_calibration':
                self.apply_calibration()
                response.data = 'Calibration applied'
            else:
                response.data = 'Unknown calibration command'

        except Exception as e:
            self.get_logger().error(f'Calibration error: {e}')
            response.data = f'Calibration failed: {str(e)}'

        return response

    def collect_calibration_data(self):
        """کیلیبریشن ڈیٹا جمع کریں"""
        # ایک ہی کمانڈ دونوں ماحولوں میں بھیجیں اور ریسپانس کو جمع کریں
        test_commands = [
            [0.1, 0.0, 0.0],  # آگے جائیں
            [0.0, 0.1, 0.0],  # بائیں جائیں
            [0.0, 0.0, 0.1],  # گھومیں
        ]

        self.calibration_data = {
            'sim_responses': [],
            'real_responses': [],
            'commands': test_commands
        }

        for cmd in test_commands:
            # دونوں ماحولوں میں کمانڈ بھیجیں
            sim_cmd = Twist()
            sim_cmd.linear.x = cmd[0]
            sim_cmd.linear.y = cmd[1]
            sim_cmd.angular.z = cmd[2]
            self.sim_cmd_pub.publish(sim_cmd)

            real_cmd = Twist()
            real_cmd.linear.x = cmd[0]
            real_cmd.linear.y = cmd[1]
            real_cmd.angular.z = cmd[2]
            self.real_cmd_pub.publish(real_cmd)

            # ریسپانس کو جمع کریں (یہ ایک اسٹوریج ہے)
            # اصل میں، آپ اس کو اگلے چکر میں کریں گے
            import time
            time.sleep(1)  # کمانڈ کو انجام دینے کے لیے انتظار کریں

    def compute_calibration_params(self):
        """کیلیبریشن پیرامیٹرز کا حساب لگائیں"""
        if hasattr(self, 'calibration_data'):
            # سیمولیشن اور حقیقی ریسپانس کا موازنہ
            sim_positions = np.array(self.calibration_data['sim_responses'])
            real_positions = np.array(self.calibration_data['real_responses'])

            # پیرامیٹرز کا حساب لگائیں
            if len(sim_positions) > 0 and len(real_positions) > 0:
                # آف سیٹ کا حساب لگائیں
                position_offset = np.mean(real_positions - sim_positions, axis=0)
                self.calibration_params['position_offset'] = position_offset.tolist()

                # سکیل کا حساب لگائیں
                sim_norm = np.linalg.norm(sim_positions, axis=1)
                real_norm = np.linalg.norm(real_positions, axis=1)

                # صفر والیو کو چھوڑ دیں
                valid_indices = sim_norm > 0.01
                if np.any(valid_indices):
                    scale_factors = real_norm[valid_indices] / sim_norm[valid_indices]
                    self.calibration_params['velocity_scale'] = np.mean(scale_factors)

    def apply_calibration(self):
        """کیلیبریشن کو لاگو کریں"""
        self.get_logger().info(f'Applying calibration parameters: {self.calibration_params}')

    def transfer_callback(self):
        """سیم ٹو ریل ٹرانسفر کے لیے کال بیک"""
        # یہاں آپ اس کو نافذ کر سکتے ہیں کہ کیسے کمانڈز کو سیمولیشن سے حقیقی دنیا میں منتقل کیا جائے
        pass

    def sim_to_real_command(self, sim_command):
        """سیمولیٹڈ کمانڈ کو حقیقی کمانڈ میں تبدیل کریں"""
        real_command = Twist()

        # پوزیشن آف سیٹ کو ایڈجسٹ کریں
        real_command.linear.x = sim_command.linear.x * self.calibration_params['velocity_scale']
        real_command.linear.y = sim_command.linear.y * self.calibration_params['velocity_scale']
        real_command.linear.z = sim_command.linear.z * self.calibration_params['velocity_scale']

        real_command.angular.x = sim_command.angular.x
        real_command.angular.y = sim_command.angular.y
        real_command.angular.z = sim_command.angular.z * self.calibration_params['velocity_scale']

        return real_command

    def real_to_sim_feedback(self, real_feedback):
        """حقیقی فیڈ بیک کو سیمولیشن فیڈ بیک میں تبدیل کریں"""
        # یہاں حقیقی فیڈ بیک کو سیمولیشن کے مطابق تبدیل کریں
        sim_feedback = real_feedback

        # کیلیبریشن پیرامیٹرز کو استعمال کریں
        sim_feedback.position = [
            pos - self.calibration_params['position_offset'][i % 3]
            for i, pos in enumerate(real_feedback.position)
        ]

        return sim_feedback
```

## انسان نما روبوٹ کے لیے سیم ٹو ریل منتقلی

### انسان نما روبوٹ کے چیلنج

انسان نما روبوٹ کے لیے سیم ٹو ریل منتقلی میں خاص چیلنج ہیں:

1. **Balance Transfer**: توازن کنٹرول کو سیمولیشن سے حقیقی دنیا میں منتقل کرنا
2. **Bipedal Locomotion**: چلنے کے الگورتھم کو منتقل کرنا
3. **Manipulation**: گریسنگ اور مینوپولیشن کو منتقل کرنا
4. **Sensor Fusion**: متعدد سینسرز کو ضم کرنا

### انسان نما سیم ٹو ریل سٹریٹیجی

```python
class HumanoidSimToRealStrategy:
    def __init__(self):
        # انسان نما روبوٹ کے لیے خاص پیرامیٹرز
        self.balance_calibration = {
            'com_offset': [0.0, 0.0, 0.0],
            'ankle_impedance': 1.0,
            'hip_stiffness': 1.0
        }

        self.locomotion_calibration = {
            'step_timing': 1.0,
            'foot_placement': [0.0, 0.0],
            'swing_height': 0.05
        }

        self.manipulation_calibration = {
            'grip_force': 1.0,
            'reach_offset': [0.0, 0.0, 0.0]
        }

    def calibrate_balance_controller(self, sim_controller, real_robot):
        """توازن کنٹرولر کو کیلیبریٹ کریں"""
        # سیمولیشن کنٹرولر کے پیرامیٹرز کو حاصل کریں
        sim_params = sim_controller.get_parameters()

        # حقیقی روبوٹ کے مطابق پیرامیٹرز کو ایڈجسٹ کریں
        calibrated_params = self.adjust_balance_parameters(sim_params, real_robot)

        # کیلیبریٹ کردہ پیرامیٹرز کو حقیقی کنٹرولر میں سیٹ کریں
        real_robot.balance_controller.set_parameters(calibrated_params)

    def adjust_balance_parameters(self, sim_params, real_robot):
        """توازن کے پیرامیٹرز کو ایڈجسٹ کریں"""
        calibrated_params = {}

        # PID گین کو ایڈجسٹ کریں
        for param_name, value in sim_params.items():
            if 'gain' in param_name.lower():
                # سیمولیشن اور حقیقی دنیا کے درمیان فرق کو ایڈجسٹ کریں
                calibrated_params[param_name] = value * self.balance_calibration['ankle_impedance']
            elif 'offset' in param_name.lower():
                calibrated_params[param_name] = value + self.balance_calibration['com_offset'][0]
            else:
                calibrated_params[param_name] = value

        return calibrated_params

    def calibrate_locomotion(self, sim_locomotion, real_robot):
        """چلنے کو کیلیبریٹ کریں"""
        # چلنے کے پیرامیٹرز کو حاصل کریں
        sim_locomotion_params = sim_locomotion.get_parameters()

        # حقیقی روبوٹ کے مطابق پیرامیٹرز کو ایڈجسٹ کریں
        calibrated_locomotion_params = self.adjust_locomotion_parameters(
            sim_locomotion_params,
            real_robot
        )

        # کیلیبریٹ کردہ پیرامیٹرز کو حقیقی چلنے کے ماڈل میں سیٹ کریں
        real_robot.locomotion_controller.set_parameters(calibrated_locomotion_params)

    def adjust_locomotion_parameters(self, sim_params, real_robot):
        """چلنے کے پیرامیٹرز کو ایڈجسٹ کریں"""
        calibrated_params = {}

        for param_name, value in sim_params.items():
            if 'timing' in param_name.lower():
                calibrated_params[param_name] = value * self.locomotion_calibration['step_timing']
            elif 'placement' in param_name.lower():
                calibrated_params[param_name] = [
                    val + offset
                    for val, offset in zip(value, self.locomotion_calibration['foot_placement'])
                ]
            elif 'height' in param_name.lower():
                calibrated_params[param_name] = value + self.locomotion_calibration['swing_height']
            else:
                calibrated_params[param_name] = value

        return calibrated_params

    def calibrate_manipulation(self, sim_manipulation, real_robot):
        """مینوپولیشن کو کیلیبریٹ کریں"""
        # مینوپولیشن کے پیرامیٹرز کو حاصل کریں
        sim_manipulation_params = sim_manipulation.get_parameters()

        # حقیقی روبوٹ کے مطابق پیرامیٹرز کو ایڈجسٹ کریں
        calibrated_manipulation_params = self.adjust_manipulation_parameters(
            sim_manipulation_params,
            real_robot
        )

        # کیلیبریٹ کردہ پیرامیٹرز کو حقیقی مینوپولیشن ماڈل میں سیٹ کریں
        real_robot.manipulation_controller.set_parameters(calibrated_manipulation_params)

    def adjust_manipulation_parameters(self, sim_params, real_robot):
        """مینوپولیشن کے پیرامیٹرز کو ایڈجسٹ کریں"""
        calibrated_params = {}

        for param_name, value in sim_params.items():
            if 'force' in param_name.lower() or 'grip' in param_name.lower():
                calibrated_params[param_name] = value * self.manipulation_calibration['grip_force']
            elif 'reach' in param_name.lower() or 'offset' in param_name.lower():
                calibrated_params[param_name] = [
                    val + offset
                    for val, offset in zip(value, self.manipulation_calibration['reach_offset'])
                ]
            else:
                calibrated_params[param_name] = value

        return calibrated_params

    def validate_transfer(self, real_robot, test_scenarios):
        """منتقلی کی توثیق کریں"""
        validation_results = {}

        for scenario_name, scenario_func in test_scenarios.items():
            try:
                # ٹیسٹ سیناریو چلائیں
                result = scenario_func(real_robot)
                validation_results[scenario_name] = {
                    'success': result['success'],
                    'performance': result['performance'],
                    'metrics': result['metrics']
                }
            except Exception as e:
                validation_results[scenario_name] = {
                    'success': False,
                    'error': str(e)
                }

        return validation_results
```

## جائزہ

سیم ٹو ریل منتقلی انسان نما روبوٹکس کا ایک اہم پہلو ہے۔ ڈومین رینڈمائزیشن، ڈومین ایڈاپٹیشن، اور کیلیبریشن کے تصورات کو سمجھنا روبوٹ کو سیمولیشن میں تربیت دینے اور حقیقی دنیا میں مؤثر طریقے سے کام کرنے کے قابل بناتا ہے۔ ROS 2 کے ساتھ سیم ٹو ریل کا نفاذ روبوٹک سسٹم کے انضمام کے لیے ضروری ہے۔

