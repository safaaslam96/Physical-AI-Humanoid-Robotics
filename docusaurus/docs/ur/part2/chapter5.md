---
title: "ہفتہ 11-13: وژن-زبان-ایکشن اور گفتگو کرنے والی روبوٹکس"
sidebar_label: "ہفتہ 11-13: وژن-زبان-ایکشن اور گفتگو کرنے والی روبوٹکس"
---

# ہفتہ 11-13: وژن-زبان-ایکشن اور گفتگو کرنے والی روبوٹکس

## ماڈیول 4: وژن-زبان-ایکشن (VLA) اور کیپسٹون

### مرکز: LLMs اور روبوٹکس کا اتحاد

### سیکھنے کے اہداف
- OpenAI Whisper کا استعمال کرتے ہوئے وائس ٹو ایکشن کو لاگو کرنا
- کوگنیٹو پلاننگ کا استعمال کرتے ہوئے قدرتی زبان کو ROS 2 ایکشن سیکوئنس میں تبدیل کرنا
- کیپسٹون پروجیکٹ مکمل کرنا: خود مختار انسان نما جو وائس کمانڈز موصول کرتا ہے، راستے کا منصوبہ بندی کرتا ہے، رکاوٹوں سے گزرتا ہے، اشیاء کی شناخت کرتا ہے، اور ان کو مینوپولیٹ کرتا ہے
- گفتگو کرنے والی روبوٹکس کے لیے GPT ماڈلز کو ضم کرنا
- ملٹی ماڈل تعامل سسٹم (speech، gesture، vision) تیار کرنا

## وائس ٹو ایکشن: OpenAI Whisper کا استعمال کرتے ہوئے وائس کمانڈز

### وائس کمانڈ پروسیسنگ کی معرفت

وائس ٹو ایکشن سسٹم روبوٹس کو قدرتی زبان کمانڈز کو سمجھنے اور انجام دینے کے قابل بناتا ہے۔ اس میں درج ذیل شامل ہے:

1. **speech recognition**: بولی گئی زبان کو متن میں تبدیل کرنا
2. **natural language understanding**: کمانڈز کا مطلب سمجھنا
3. **action mapping**: کمانڈز کو روبوٹ ایکشنز میں تبدیل کرنا
4. **execution**: درخواست کردہ ایکشنز انجام دینا

### OpenAI Whisper کے لیے تنصیب

```bash
# Whisper تنصیب کریں
pip install openai-whisper
```

### وائس کمانڈ پروسیسنگ نوڈ

```python
# وائس کمانڈ پروسیسنگ نوڈ
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import whisper
import speech_recognition as sr
import threading
import queue
import numpy as np

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # روبوٹ کمانڈز کے لیے پبلشرز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.voice_response_pub = self.create_publisher(String, '/voice_response', 10)

        # Whisper ماڈل کو لوڈ کریں
        self.whisper_model = whisper.load_model("base")  # یا "small", "medium", "large"

        # speech recognition کو شروع کریں
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # سائلنس ڈیٹکشن کے لیے انرجی تھریشولڈ سیٹ کریں
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True

        # پروسیسنگ کے لیے کمانڈ کیو
        self.command_queue = queue.Queue()

        # وائس ریکوگنیشن تھریڈ شروع کریں
        self.voice_thread = threading.Thread(target=self.voice_recognition_loop)
        self.voice_thread.daemon = True
        self.voice_thread.start()

        # کمانڈز پروسیسنگ کے لیے ٹائمر
        self.process_timer = self.create_timer(0.1, self.process_commands)

    def voice_recognition_loop(self):
        """مسلسل وائس ریکوگنیشن لوپ"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.get_logger().info("وائس ریکوگنیشن شروع ہو گیا۔ سن رہا ہے...")

        while rclpy.ok():
            try:
                with self.microphone as source:
                    # ٹائم آؤٹ کے ساتھ آڈیو سنیں
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=5)

                    # آڈیو کو numpy ارے میں تبدیل کریں Whisper کے لیے
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                    # Whisper کا استعمال کرتے ہوئے speech recognition کریں
                    result = self.whisper_model.transcribe(audio_data.astype(np.float32) / 32768.0)
                    text = result['text'].strip()

                    if text:
                        self.get_logger().info(f"شناخت شدہ: {text}")
                        self.command_queue.put(text)

            except sr.WaitTimeoutError:
                # یہ متوقع ہے جب کوئی speech نہیں ملتی
                continue
            except Exception as e:
                self.get_logger().warn(f"وائس ریکوگنیشن خرابی: {e}")
                continue

    def process_commands(self):
        """شناخت شدہ کمانڈز کو پروسیس کریں"""
        try:
            while not self.command_queue.empty():
                command_text = self.command_queue.get_nowait()
                self.process_voice_command(command_text)
        except queue.Empty:
            pass

    def process_voice_command(self, command_text):
        """وائس کمانڈ کو پروسیس کریں اور روبوٹ ایکشنز میں تبدیل کریں"""
        # کمانڈ ٹیکسٹ کو نارملائز کریں
        command = command_text.lower().strip()

        # تصدیق بھیجیں
        response_msg = String()

        if "move forward" in command or "go forward" in command:
            self.move_forward()
            response_msg.data = "آگے بڑھ رہا ہے"
        elif "move backward" in command or "go backward" in command:
            self.move_backward()
            response_msg.data = "پیچھے جا رہا ہے"
        elif "turn left" in command:
            self.turn_left()
            response_msg.data = "بائیں مڑ رہا ہے"
        elif "turn right" in command:
            self.turn_right()
            response_msg.data = "دائیں مڑ رہا ہے"
        elif "stop" in command:
            self.stop_robot()
            response_msg.data = "روبوٹ روک رہا ہے"
        elif "come to me" in command or "come here" in command:
            self.navigate_to_user()
            response_msg.data = "آپ کی طرف جا رہا ہے"
        elif "pick up" in command or "grasp" in command:
            self.initiate_grasping()
            response_msg.data = "اشیاء کو تھامنے کی کوشش کر رہا ہے"
        elif "clean the room" in command:
            self.start_cleaning_routine()
            response_msg.data = "صاف کرنے کی روتین شروع ہو رہی ہے"
        else:
            response_msg.data = f"کمانڈ تسلیم نہیں ہوئی: {command_text}"
            self.get_logger().warn(f"نامعلوم کمانڈ: {command_text}")

        self.voice_response_pub.publish(response_msg)
```

## کوگنیٹو پلاننگ: LLMs کا استعمال کرتے ہوئے قدرتی زبان کو ROS 2 ایکشنز میں تبدیل کرنا

### کوگنیٹو پلاننگ کی معرفت

کوگنیٹو پلاننگ قدرتی زبان کمانڈز اور کم سطح کے روبوٹ ایکشنز کے درمیان پل کو جوڑتا ہے۔ اس میں درج ذیل شامل ہے:

1. **command parsing**: کمانڈز کا سیمینٹک مطلب سمجھنا
2. **task decomposition**: پیچیدہ کمانڈز کو ذیلی کاموں میں توڑنا
3. **action sequencing**: ایکشنز کو منطقی طور پر ترتیب دینا
4. **context awareness**: ماحولیاتی پابندیوں کو مدنظر رکھنا

### LLM-بیسڈ کمانڈ ٹرانسلیشن

```python
import openai
from rclpy.node import Node
from std_msgs.msg import String
import json

class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner')

        # منصوبہ بند ایکشنز کے لیے پبلشر
        self.action_pub = self.create_publisher(String, '/planned_actions', 10)

        # OpenAI کلائنٹ کو شروع کریں
        # نوٹ: عملی طور پر، آپ رازداری کے لیے مقامی ماڈلز جیسے Ollama کا استعمال کر سکتے ہیں
        openai.api_key = "your-api-key-here"

        # ٹاسک ڈیکمپوزیشن پیٹرنز
        self.task_patterns = {
            "room کو صاف کریں": [
                {"action": "detect_objects", "params": {"object_types": ["trash", "dirt"]}},
                {"action": "navigate_to", "params": {"target": "object_location"}},
                {"action": "grasp_object", "params": {"object": "trash"}},
                {"action": "navigate_to", "params": {"target": "trash_bin"}},
                {"action": "release_object", "params": {}}
            ],
            "میرے لیے پانی لائیں": [
                {"action": "detect_object", "params": {"object": "water_bottle"}},
                {"action": "navigate_to", "params": {"target": "water_location"}},
                {"action": "grasp_object", "params": {"object": "water_bottle"}},
                {"action": "navigate_to", "params": {"target": "user_location"}},
                {"action": "release_object", "params": {}}
            ]
        }

    def plan_from_natural_language(self, command):
        """LLM کا استعمال کرتے ہوئے قدرتی زبان کمانڈ سے ایکشنز کا منصوبہ بند کریں"""
        try:
            # LLM کے لیے ایک پرومپٹ تیار کریں
            prompt = f"""
            درج ذیل قدرتی زبان کمانڈ کو روبوٹ ایکشنز کی ترتیب میں تبدیل کریں۔
            روبوٹ ایک انسان نما روبوٹ ہے جس میں نیوی گیشن، مینوپولیشن، اور تاثر کی صلاحیتیں ہیں۔
            JSON کی فہرست واپس کریں ایکشنز کے ساتھ درج ذیل فارمیٹ میں:
            [
                {{"action": "action_name", "params": {{"param1": "value1", "param2": "value2"}}}}
            ]

            دستیاب ایکشنز:
            - navigate_to: مقام پر جانا (params: target)
            - detect_object: ایک شے تلاش کرنا (params: object_type)
            - grasp_object: ایک شے اٹھانا (params: object)
            - release_object: ایک شے چھوڑنا (params: target)
            - speak: کچھ کہنا (params: text)
            - wait: مدت تک انتظار کرنا (params: seconds)

            کمانڈ: {command}

            جواب (صرف JSON):
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # جواب کو پارس کریں
            action_sequence = json.loads(response.choices[0].message.content)
            return action_sequence

        except Exception as e:
            self.get_logger().error(f"LLM پلاننگ خرابی: {e}")
            # پیٹرن میچنگ کے لیے فال بیک
            return self.fallback_plan(command)
```

## کیپسٹون پروجیکٹ: خود مختار انسان نما

### پروجیکٹ کا جائزہ

کیپسٹون پروجیکٹ تمام ماڈیولز کو ایک مکمل خود مختار انسان نما سسٹم میں ضم کرتا ہے:

1. **وائس کمانڈز موصول کرتا ہے** Whisper speech recognition کے ذریعے
2. **ایکشنز کا منصوبہ بند کرتا ہے** LLM-بیسڈ کوگنیٹو پلاننگ کے ذریعے
3. **رکاوٹوں سے گزرتا ہے** Isaac Sim اور Nav2 کے ذریعے
4. **اشیاء کی شناخت کرتا ہے** کمپیوٹر وژن کے ذریعے
5. **اشیاء کو مینوپولیٹ کرتا ہے** انسان نما بازوں کے ذریعے

## GPT ماڈلز کو گفتگو کرنے والی روبوٹکس کے لیے ضم کرنا

### گفتگو کرنے والی AI نوڈ

```python
import openai
from rclpy.node import Node
from std_msgs.msg import String
import json

class ConversationalAINode(Node):
    def __init__(self):
        super().__init__('conversational_ai')

        # سبسکرائبرز اور پبلشرز
        self.user_input_sub = self.create_subscription(String, '/user_input', self.user_input_callback, 10)
        self.voice_response_pub = self.create_publisher(String, '/voice_response', 10)
        self.robot_state_sub = self.create_subscription(String, '/robot_state', self.robot_state_callback, 10)

        # گفتگو کی تاریخ
        self.conversation_history = []

        # روبوٹ کا ماحول
        self.robot_state = {}

    def user_input_callback(self, msg):
        """صارف کے ان پٹ کو ہینڈل کریں اور جواب تیار کریں"""
        user_input = msg.data
        self.get_logger().info(f"صارف کا ان پٹ: {user_input}")

        # گفتگو کی تاریخ میں شامل کریں
        self.conversation_history.append({"role": "user", "content": user_input})

        # GPT کا استعمال کرتے ہوئے جواب تیار کریں
        response = self.generate_response(user_input)

        # جواب پبلش کریں
        response_msg = String()
        response_msg.data = response
        self.voice_response_pub.publish(response_msg)

        # گفتگو کی تاریخ میں شامل کریں
        self.conversation_history.append({"role": "assistant", "content": response})

        # گفتگو کی تاریخ کو آخری 10 تبدیلیوں تک محدود کریں
        if len(self.conversation_history) > 20:  # 10 تبدیلیاں = 20 پیغامات
            self.conversation_history = self.conversation_history[-20:]

    def generate_response(self, user_input):
        """GPT ماڈل کا استعمال کرتے ہوئے جواب تیار کریں"""
        try:
            # گفتگو کا ماحول تیار کریں
            messages = [
                {"role": "system", "content": f"""
                آپ ایک مددگار انسان نما روبوٹ ہیں۔ آپ حرکت کر سکتے ہیں، نیوی گیٹ کر سکتے ہیں، اشیاء کو تلاش کر سکتے ہیں، اور چیزیں ہیرا پھیری کر سکتے ہیں۔
                موجودہ روبوٹ کی حالت: {json.dumps(self.robot_state)}
                صارف کی درخواستوں کا قدرتی اور مددگار انداز میں جواب دیں۔ اگر ایکشنز کرنے کے لیے کہا جائے تو،
                تصدیق کریں اور اشارہ دیں کہ آپ درخواست کو پروسیس کریں گے۔
                """}
            ]

            # گفتگو کی تاریخ شامل کریں
            messages.extend(self.conversation_history[-10:])  # آخری 5 تبدیلیاں

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            self.get_logger().error(f"GPT جواب تیاری کی خرابی: {e}")
            return "معذرت، مجھے ابھی جواب دینے میں پریشانی ہو رہی ہے۔"
```

## ملٹی ماڈل تعامل: speech، gesture، vision

### ملٹی ماڈل تاثر نوڈ

```python
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import cv2
from cv_bridge import CvBridge
import numpy as np

class MultiModalPerceptionNode(Node):
    def __init__(self):
        super().__init__('multi_modal_perception')

        # CV برج کو شروع کریں
        self.cv_bridge = CvBridge()

        # مختلف ماڈلیٹیز کے لیے سبسکرائبرز
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.voice_sub = self.create_subscription(String, '/voice_commands', self.voice_callback, 10)

        # پروسیسڈ ڈیٹا کے لیے پبلشرز
        self.gesture_pub = self.create_publisher(String, '/detected_gestures', 10)
        self.object_pub = self.create_publisher(String, '/detected_objects', 10)
        self.attention_pub = self.create_publisher(PointStamped, '/attention_target', 10)

        # داخلی حالت
        self.latest_image = None
        self.latest_depth = None
        self.voice_commands = []

    def image_callback(self, msg):
        """کیمرہ امیج کو اشیاء اور اشارے کے لیے پروسیس کریں"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image

            # اشیاء کو تلاش کریں
            objects = self.detect_objects(cv_image)
            if objects:
                obj_msg = String()
                obj_msg.data = json.dumps(objects)
                self.object_pub.publish(obj_msg)

            # اشارے کو تلاش کریں
            gestures = self.detect_gestures(cv_image)
            if gestures:
                gesture_msg = String()
                gesture_msg.data = json.dumps(gestures)
                self.gesture_pub.publish(gesture_msg)

        except Exception as e:
            self.get_logger().error(f"امیج پروسیسنگ کی خرابی: {e}")

    def depth_callback(self, msg):
        """ڈیپتھ امیج کو 3D معلومات کے لیے پروسیس کریں"""
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = depth_image
        except Exception as e:
            self.get_logger().error(f"ڈیپتھ پروسیسنگ کی خرابی: {e}")

    def voice_callback(self, msg):
        """بصری معلومات کے تناظر میں وائس کمانڈز کو پروسیس کریں"""
        command = msg.data
        self.voice_commands.append(command)

        # موجودہ بصری منظر کے تناظر میں کمانڈ پروسیس کریں
        self.process_multimodal_command(command)

    def detect_objects(self, image):
        """تصویر میں اشیاء کو تلاش کریں کمپیوٹر وژن کا استعمال کرتے ہوئے"""
        # یہ ایک تربیت یافتہ اشیاء کی شناخت ماڈل کا استعمال کرے گا
        # نمائش کے لیے، سادہ رنگ-بیسڈ ڈیٹیکشن کا استعمال کر رہا ہے
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # مختلف اشیاء کے لیے رنگ کی حدیں کیفیت کریں
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        detected_objects = []

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # چھوٹے کنٹورز کو فلٹر کریں
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_objects.append({
                        'type': color_name,
                        'bbox': [x, y, x+w, y+h],
                        'center': [x + w//2, y + h//2]
                    })

        return detected_objects
```

## ہفتہ وار جائزہ: ہفتہ 11-13

### ہفتہ 11: انسان نما روبوٹکس کی ترقی
- انسان نما روبوٹ کنیمیٹکس اور ڈائنیمکس
- بائی پیڈل چلنے اور توازن کنٹرول
- انسان نما ہاتھوں کے ساتھ مینوپولیشن اور تھامنا
- قدرتی انسان-روبوٹ تعامل کا ڈیزائن

### ہفتہ 12: وژن-زبان-ایکشن انضمام
- Whisper کے ساتھ وائس ٹو ایکشن لاگو کرنا
- LLMs کے ساتھ کوگنیٹو پلاننگ
- ملٹی ماڈل تاثر انضمام
- ایکشن سیکوئنسنگ اور ایگزیکیوشن

### ہفتہ 13: گفتگو کرنے والی روبوٹکس اور کیپسٹون
- گفتگو کرنے والی AI کے لیے GPT ماڈلز کو ضم کرنا
- speech recognition اور قدرتی زبان کی سمجھ
- ملٹی ماڈل تعامل: speech، gesture، vision
- کیپسٹون پروجیکٹ: خود مختار انسان نما ڈیمو

## جائزہ

اس ماڈیول نے وژن، زبان، اور ایکشن سسٹم کا انضمام انسان نما روبوٹکس میں دکھایا ہے۔ آپ نے سیکھا کہ OpenAI Whisper کا استعمال کرتے ہوئے وائس ٹو ایکشن سسٹم کیسے لاگو کریں، LLMs کے ساتھ کوگنیٹو پلینر تخلیق کریں، اور ان کو کمپیوٹر وژن اور مینوپولیشن کی صلاحیتوں کے ساتھ ضم کریں۔ کیپسٹون پروجیکٹ قدرتی زبان کی سمجھ اور جسمانی ایکشن ایگزیکیوشن کے مکمل پائپ لائن کو ظاہر کرتا ہے، جو جسمانی AI سسٹم میں متعدد AI ماڈلیٹیز کا اتحاد کا نمائندہ ہے۔

