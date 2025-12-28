---
title: "ہفتہ 8-10: NVIDIA Isaac پلیٹ فارم"
sidebar_label: "ہفتہ 8-10: NVIDIA Isaac پلیٹ فارم"
---

# ہفتہ 8-10: NVIDIA Isaac پلیٹ فارم

## ماڈیول 3: AI-روبوٹ براہن (NVIDIA Isaac™)

### مرکز: اعلیٰ تر تاثر اور تربیت

### سیکھنے کے اہداف
- فوٹو ریلزم سیمولیشن کے لیے NVIDIA Isaac Sim کنفیگر کرنا
- Isaac ROS: ہارڈ ویئر ایکسلریٹڈ VSLAM (وژول SLAM) اور نیوی گیشن کو لاگو کرنا
- Nav2: بائی پیڈل انسان نما حرکت کے لیے راستہ منصوبہ بندی کا استعمال کرنا
- سیم ٹو ریل منتقلی کی تکنیکوں کو لاگو کرنا
- AI پاورڈ تاثر اور مینوپولیشن سسٹم تیار کرنا

## NVIDIA Isaac SDK اور Isaac Sim

### NVIDIA Isaac پلیٹ فارم کی معرفت

NVIDIA Isaac پلیٹ فارم AI پاورڈ روبوٹس کی ترقی، سیمولیشن، اور ڈیپلائمنٹ کے لیے ایک جامع حل ہے۔ یہ ہارڈ ویئر ایکسلریشن، سیمولیشن ٹولز، اور سافٹ ویئر فریم ورکس کو جوڑتا ہے تاکہ اعلیٰ کارکردگی والے روبوٹکس ایپلی کیشنز کو فعال کیا جا سکے۔

### Isaac Sim: فوٹو ریلزم سیمولیشن اور مصنوعی ڈیٹا جنریشن

Isaac Sim NVIDIA کے Omniverse پلیٹ فارم پر تعمیر شدہ ایک ہائی فائیڈلٹی سیمولیشن ماحول ہے۔ یہ درج ذیل فراہم کرتا ہے:

- **فوٹو ریلزم رینڈرنگ**: حقیقی سینسر سیمولیشن کے لیے فزیکلی بیسڈ رینڈرنگ
- **مصنوعی ڈیٹا جنریشن**: AI ماڈلز کی تربیت کے لیے بڑے پیمانے کے ڈیٹا سیٹس
- **فزیکس سیمولیشن**: درست ریجڈ باڈی ڈائنیمکس اور کنٹیکٹ فزکس
- **میولٹی روبوٹ سیمولیشن**: پیچیدہ میولٹی روبوٹ منظرناموں کے لیے

### Isaac ROS: ہارڈ ویئر ایکسلریٹڈ تاثر

### Isaac ROS کا جائزہ

Isaac ROS ہارڈ ویئر ایکسلریٹڈ تاثر اور نیوی گیشن کی صلاحیتیں فراہم کرتا ہے:

- **وژول SLAM (VSLAM)**: ریل ٹائم سیمولٹینیئس لوکلائزیشن اور میپنگ
- **کمپیوٹر وژن**: ہارڈ ویئر ایکسلریٹڈ امیج پروسیسنگ
- **سینسر فیوژن**: متعدد سینسر ماڈلیٹیز کا انضمام
- **ROS 2 مطابقت**: ROS 2 ایکو سسٹم کے ساتھ بلا جھگڑ انضمام

### Isaac ROS تنصیب

```bash
# Isaac ROS پیکجز تنصیب کریں
sudo apt update
sudo apt install ros-humble-isaac-ros-common
sudo apt install ros-humble-isaac-ros-visual-slam
sudo apt install ros-humble-isaac-ros-gxf
sudo apt install ros-humble-isaac-ros-augmenter
```

## Nav2: بائی پیڈل انسان نما حرکت کے لیے راستہ منصوبہ بندی

### انسان نما کے لیے Nav2 آرکیٹیکچر

Nav2 (نیوی گیشن 2) ROS 2 کے لیے نیوی گیشن فریم ورک ہے، جسے انسان نما روبوٹس کے لیے اڈاپٹ کیا گیا ہے:

- **گلوبل پلینر**: انسان نما کنیمیٹکس کو مدنظر رکھتے ہوئے راستہ منصوبہ بندی
- **مقامی پلینر**: بائی پیڈل حرکت کے لیے ڈائنا مک رکاوٹ ڈیٹکشن
- **کنٹرولر**: انسان نما مخصوص موشن کنٹرولرز
- **ریکوری بیہیویئرز**: انسان نما مناسب ریکوری ایکشنز

### انسان نما مخصوص Nav2 کنفیگریشن

```yaml
# انسان نما روبوٹس کے لیے Nav2 کنفیگریشن
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    # انسان نما مخصوص پیرامیٹرز
    humanoid_specific:
      max_step_height: 0.15  # بائی پیڈل حرکت کی زیادہ سے زیادہ اونچائی
      step_width: 0.3        # عام اقدام کی چوڑائی
      balance_margin: 0.1    # توازن کے لیے سیفٹی مارجن

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    # انسان نما کنٹرولرز
    humanoid_controller:
      type: "humanoid_base_controller/HumanoidController"
      # بائی پیڈل حرکت کے پیرامیٹرز
      step_duration: 0.8
      step_height: 0.05
      balance_control: true
```

## سیم ٹو ریل منتقلی کی تکنیکیں

### سیم ٹو ریل منتقلی کی معرفت

سیم ٹو ریل منتقلی روبوٹس کو سیمولیشن میں تربیت دینے سے حقیقی دنیا میں مؤثر طریقے سے کام کرنے کے قابل بناتی ہے۔ یہ انسان نما روبوٹکس کے لیے اہم ہے جہاں حقیقی دنیا میں تربیت مہنگی اور ممکنہ طور پر خطرناک ہو سکتی ہے۔

### ڈومین رینڈمائزیشن

```python
# سیم ٹو ریل منتقلی کے لیے ڈومین رینڈمائزیشن
import random

class DomainRandomization:
    def __init__(self):
        self.simulation_params = {
            'friction_range': [0.1, 1.0],
            'gravity_range': [9.5, 10.5],
            'lighting_conditions': ['sunny', 'cloudy', 'indoor'],
            'texture_variety': ['smooth', 'rough', 'patterned']
        }

    def randomize_environment(self):
        """سیمولیشن پیرامیٹرز کو ڈومین رینڈمائزیشن کے لیے رینڈمائز کریں"""
        # فزیکس پیرامیٹرز کو رینڈمائز کریں
        friction = random.uniform(
            self.simulation_params['friction_range'][0],
            self.simulation_params['friction_range'][1]
        )

        gravity = random.uniform(
            self.simulation_params['gravity_range'][0],
            self.simulation_params['gravity_range'][1]
        )

        # وژول پیرامیٹرز کو رینڈمائز کریں
        lighting = random.choice(self.simulation_params['lighting_conditions'])
        texture = random.choice(self.simulation_params['texture_variety'])

        return {
            'friction': friction,
            'gravity': gravity,
            'lighting': lighting,
            'texture': texture
        }
```

## ہفتہ وار جائزہ: ہفتہ 8-10

### ہفتہ 8: NVIDIA Isaac SDK اور Isaac Sim
- Isaac Sim تنصیب اور کنفیگریشن
- فوٹو ریلزم سیمولیشن ماحول سیٹ اپ
- مصنوعی ڈیٹا جنریشن تکنیکیں
- فزیکس سیمولیشن اور سینسر ماڈلنگ

### ہفتہ 9: Isaac ROS اور ہارڈ ویئر ایکسلریٹڈ تاثر
- Isaac ROS تنصیب اور سیٹ اپ
- ہارڈ ویئر ایکسلریشن کے ساتھ وژول SLAM لاگو کرنا
- GPU ایکسلریشن کے ساتھ کمپیوٹر وژن پائپ لائنز
- انسان نما روبوٹس کے لیے سینسر فیوژن

### ہفتہ 10: Nav2 اور سیم ٹو ریل منتقلی
- بائی پیڈل انسان نما حرکت کے لیے Nav2 کنفیگریشن
- توازن کی پابندیوں کے ساتھ راستہ منصوبہ بندی
- سیم ٹو ریل منتقلی کی تکنیکیں
- ڈومین رینڈمائزیشن اور سسٹم شناخت

## جائزہ

اس ماڈیول نے NVIDIA Isaac پلیٹ فارم کا جائزہ لیا، جس میں Isaac Sim کے لیے فوٹو ریلزم سیمولیشن، Isaac ROS کے لیے ہارڈ ویئر ایکسلریٹڈ تاثر، اور انسان نما نیوی گیشن کے لیے Nav2 شامل ہے۔ آپ نے سیکھا کہ سیم ٹو ریل منتقلی کی تکنیکیں کیسے لاگو کریں جو انسان نما روبوٹس کو حقیقی دنیا کے ماحول میں ڈیپلائی کرنے کے لیے ضروری ہیں۔ ترقی یافتہ سیمولیشن، تاثر، اور نیوی گیشن ٹولز کا امتزاج جسمانی AI سسٹم کی ترقی کے قابل بناتا ہے۔

