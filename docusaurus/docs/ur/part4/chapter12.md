---
title: "چیپٹر 12: انسان-روبوٹ تعامل"
sidebar_label: "چیپٹر 12: انسان-روبوٹ تعامل"
---

# چیپٹر 12: انسان-روبوٹ تعامل

## سیکھنے کے اہداف
- انسان-روبوٹ تعامل کے تصورات کو سمجھنا
- گفتگو کے نظام کو ROS 2 میں نافذ کرنا
- اشاروں اور اظہار کا نفاذ
- قدرتی زبان کے انٹرفیسز کو ڈیزائن کرنا
- انسان نما روبوٹ کے لیے تعامل کے نظام کو لاگو کرنا

## انسان-روبوٹ تعامل (HRI) کی معرفت

### HRI کیا ہے؟

انسان-روبوٹ تعامل (Human-Robot Interaction) وہ میدان ہے جو انسانوں اور روبوٹس کے درمیان مواصلت، تعاون، اور تعامل کے طریقوں کو تلاش کرتا ہے۔

### HRI کے اجزاء

1. **Communication**: الفاظ، اشارے، اظہار
2. **Collaboration**: کام کرنے کے لیے انسان اور روبوٹ کا تعاون
3. **Trust**: انسان کا روبوٹ پر اعتماد
4. **Social Norms**: معاشرتی قواعد کا احترام

### HRI کے اطلاقات

1. **Service Robots**: ہوٹل، ریستوراں، اور دیکھ بھال کے روبوٹس
2. **Educational Robots**: سکولوں اور تعلیمی اداروں میں روبوٹس
3. **Companion Robots**: گھریلو معاونت اور دیکھ بھال کے روبوٹس
4. **Industrial Robots**: انسان-روبوٹ تعاون کے لیے صنعتی روبوٹس

## گفتگو کے نظام

### گفتگو کا سسٹم کیا ہے؟

گفتگو کا سسٹم روبوٹ کے لیے ایک ایسی سہولت ہے جو انسانوں کے ساتھ قدرتی زبان کے ذریعے بات چیت کرنے کے قابل بناتا ہے۔

### گفتگو کے نظام کے اجزاء

1. **Speech Recognition**: انسان کی بولی کو متن میں تبدیل کرنا
2. **Natural Language Understanding (NLU)**: متن کو سمجھنا
3. **Dialog Manager**: بات چیت کا انتظام
4. **Natural Language Generation (NLG)**: جواب تیار کرنا
5. **Speech Synthesis**: متن کو بولی میں تبدیل کرنا

### ROS 2 میں گفتگو کا نظام

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import speech_recognition as sr
import pyttsx3
import openai
import json

class ConversationalAISystem(Node):
    def __init__(self):
        super().__init__('conversational_ai_system')

        # گفتگو کے لیے سبسکرائبرز اور پبلشرز
        self.user_input_sub = self.create_subscription(String, '/user_input', self.user_input_callback, 10)
        self.voice_command_pub = self.create_publisher(String, '/voice_commands', 10)
        self.response_pub = self.create_publisher(String, '/robot_response', 10)

        # گفتگو کی تاریخ
        self.dialog_history = []

        # گفتگو کے ایجینٹ
        self.dialog_manager = DialogManager()
        self.nlu = NaturalLanguageUnderstanding()
        self.nlg = NaturalLanguageGeneration()

        # اسپیچ ریکوگنیشن
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # اسپیچ سنتھیسائزیشن
        self.tts_engine = pyttsx3.init()
        self.setup_tts()

        # گفتگو کے لیے ٹائمر
        self.dialog_timer = self.create_timer(0.1, self.dialog_processing_callback)

    def setup_tts(self):
        """ٹیکسٹ ٹو اسپیچ سسٹم سیٹ اپ کریں"""
        # TTS انجن کو کنفیگر کریں
        rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', rate - 50)  # آواز کو سست کریں
        volume = self.tts_engine.getProperty('volume')
        self.tts_engine.setProperty('volume', volume + 0.25)

    def user_input_callback(self, msg):
        """صارف کا ان پٹ ہینڈل کریں"""
        user_input = msg.data
        self.get_logger().info(f'صارف: {user_input}')

        # گفتگو کی تاریخ میں شامل کریں
        self.dialog_history.append({'role': 'user', 'content': user_input})

        # جواب تیار کریں
        response = self.generate_response(user_input)

        # جواب کو پبلش کریں
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # گفتگو کی تاریخ میں شامل کریں
        self.dialog_history.append({'role': 'assistant', 'content': response})

        # گفتگو کی تاریخ کو محدود کریں
        if len(self.dialog_history) > 20:  # 10 تبدیلیاں
            self.dialog_history = self.dialog_history[-20:]

        # جواب بولیں
        self.speak_response(response)

    def generate_response(self, user_input):
        """جواب تیار کریں"""
        try:
            # گفتگو کا ماحول تیار کریں
            system_prompt = f"""
            آپ ایک مددگار انسان نما روبوٹ ہیں۔ آپ انسانوں کے ساتھ قدرتی طریقے سے بات چیت کر سکتے ہیں۔
            آپ کے پاس یہ صلاحیتیں ہیں:
            - حرکت: آگے، پیچھے، بائیں، دائیں جا سکتے ہیں
            - گریسنگ: چیزیں تھام سکتے ہیں
            - تلاش: اشیاء تلاش کر سکتے ہیں
            - گفتگو: قدرتی طریقے سے بات چیت کر سکتے ہیں
            """

            # گفتگو کا مکمل متن تیار کریں
            messages = [
                {'role': 'system', 'content': system_prompt}
            ]
            messages.extend(self.dialog_history[-10:])  # آخری 5 تبدیلیاں

            # OpenAI API کو کال کریں
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.get_logger().error(f'جواب تیاری میں خرابی: {e}')
            return "معذرت، مجھے ابھی جواب دینے میں پریشانی ہو رہی ہے۔"

    def speak_response(self, response_text):
        """جواب کو بولیں"""
        try:
            # اسپیچ سنتھیسائزیشن کے لیے تھریڈ استعمال کریں
            self.tts_engine.say(response_text)
            self.tts_engine.runAndWait()
        except Exception as e:
            self.get_logger().error(f'بولنے میں خرابی: {e}')

    def listen_for_speech(self):
        """بولی کو سنیں"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5.0)

            # بولی کو متن میں تبدیل کریں
            text = self.recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            self.get_logger().info('کوئی بولی نہیں سنی گئی')
            return None
        except sr.UnknownValueError:
            self.get_logger().info('بولی کو سمجھنا ممکن نہیں تھا')
            return None
        except Exception as e:
            self.get_logger().error(f'بولی کو سمجھنے میں خرابی: {e}')
            return None

    def process_speech_command(self, speech_text):
        """بولی کمانڈ کو پروسیس کریں"""
        # بولی کو سمجھیں
        understood_command = self.nlu.understand_command(speech_text)

        if understood_command:
            # کمانڈ کو انجام دیں
            result = self.execute_command(understood_command)

            # جواب تیار کریں
            response = self.nlg.generate_response(result, speech_text)
            return response

        return "میں آپ کے کمانڈ کو نہیں سمجھ سکا۔ براہ کرم دوبارہ کہیں۔"

    def execute_command(self, command):
        """کمانڈ کو انجام دیں"""
        command_type = command.get('type', '')
        params = command.get('params', {})

        if command_type == 'move':
            return self.execute_move_command(params)
        elif command_type == 'grasp':
            return self.execute_grasp_command(params)
        elif command_type == 'find':
            return self.execute_find_command(params)
        else:
            return {'success': False, 'message': f'نامعلوم کمانڈ: {command_type}'}

    def execute_move_command(self, params):
        """Move کمانڈ کو انجام دیں"""
        direction = params.get('direction', 'forward')
        distance = params.get('distance', 1.0)

        # روبوٹ کو حرکت کمانڈ بھیجیں
        cmd_msg = String()
        cmd_msg.data = f'move_{direction}_{distance}'
        self.voice_command_pub.publish(cmd_msg)

        return {'success': True, 'message': f'{direction} کی طرف {distance} میٹر چل رہا ہوں'}

    def execute_grasp_command(self, params):
        """Grasp کمانڈ کو انجام دیں"""
        object_name = params.get('object', '')

        # گریسنگ کمانڈ بھیجیں
        cmd_msg = String()
        cmd_msg.data = f'grasp_{object_name}'
        self.voice_command_pub.publish(cmd_msg)

        return {'success': True, 'message': f'{object_name} کو تھام رہا ہوں'}

    def execute_find_command(self, params):
        """Find کمانڈ کو انجام دیں"""
        object_name = params.get('object', '')

        # تلاش کمانڈ بھیجیں
        cmd_msg = String()
        cmd_msg.data = f'find_{object_name}'
        self.voice_command_pub.publish(cmd_msg)

        return {'success': True, 'message': f'{object_name} تلاش کر رہا ہوں'}
```

## اشارے اور اظہار

### اشارے کیا ہیں؟

اشارے روبوٹ کی حرکت ہے جو کسی چیز یا جگہ کی طرف اشارہ کرنے یا کسی چیز کو نمایاں کرنے کے لیے استعمال ہوتی ہے۔

### اظہار کیا ہیں؟

اظہار روبوٹ کی حرکت ہے جو کوئی جذبات یا معلومات ظاہر کرنے کے لیے استعمال ہوتی ہے۔

### ROS 2 میں اشارے اور اظہار

```python
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
import math

class GestureExpressionSystem(Node):
    def __init__(self):
        super().__init__('gesture_expression_system')

        # اشارے اور اظہار کے لیے پبلشرز
        self.gesture_pub = self.create_publisher(String, '/robot_gestures', 10)
        self.expression_pub = self.create_publisher(String, '/robot_expressions', 10)
        self.pointing_pub = self.create_publisher(PointStamped, '/robot_pointing', 10)

        # بات چیت کے لیے سبسکرائبرز
        self.dialog_sub = self.create_subscription(String, '/robot_response', self.dialog_callback, 10)

        # روبوٹ کی حالت
        self.robot_state = {
            'head_position': [0, 0, 1.5],  # x, y, z
            'arm_position': [0, 0, 0],     # x, y, z
            'gaze_target': None
        }

    def dialog_callback(self, msg):
        """گفتگو کے جواب کے مطابق اشارے اور اظہار کریں"""
        response_text = msg.data

        # جواب کو تجزیہ کریں اور مناسب اشارے یا اظہار منتخب کریں
        gesture = self.select_appropriate_gesture(response_text)
        expression = self.select_appropriate_expression(response_text)

        if gesture:
            self.perform_gesture(gesture)
        if expression:
            self.perform_expression(expression)

    def select_appropriate_gesture(self, text):
        """متن کے مطابق مناسب اشارہ منتخب کریں"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['there', 'over there', 'that', 'point']):
            return 'pointing'
        elif any(word in text_lower for word in ['yes', 'agree', 'ok', 'good']):
            return 'nodding'
        elif any(word in text_lower for word in ['no', 'disagree', 'wrong']):
            return 'shaking_head'
        elif any(word in text_lower for word in ['hello', 'hi', 'greetings']):
            return 'waving'
        elif any(word in text_lower for word in ['think', 'consider', 'contemplate']):
            return 'thinking'
        elif any(word in text_lower for word in ['surprise', 'wow', 'amazing']):
            return 'surprised'
        elif any(word in text_lower for word in ['listen', 'hear', 'attention']):
            return 'attentive'
        elif any(word in text_lower for word in ['help', 'assistance', 'aid']):
            return 'offering_help'
        elif any(word in text_lower for word in ['wait', 'hold', 'stop']):
            return 'stopping'
        else:
            return 'neutral'

    def select_appropriate_expression(self, text):
        """متن کے مطابق مناسب اظہار منتخب کریں"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['happy', 'glad', 'pleased', 'joy']):
            return 'happy'
        elif any(word in text_lower for word in ['sad', 'sorry', 'unhappy', 'disappointed']):
            return 'sad'
        elif any(word in text_lower for word in ['angry', 'frustrated', 'annoyed']):
            return 'angry'
        elif any(word in text_lower for word in ['confused', 'uncertain', 'puzzled']):
            return 'confused'
        elif any(word in text_lower for word in ['surprise', 'wow', 'amazing']):
            return 'surprised'
        elif any(word in text_lower for word in ['thinking', 'considering', 'analyzing']):
            return 'thinking'
        elif any(word in text_lower for word in ['greeting', 'hello', 'welcome']):
            return 'friendly'
        elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'grateful'
        else:
            return 'neutral'

    def perform_gesture(self, gesture_type):
        """ایک اشارہ کریں"""
        gesture_msg = String()
        gesture_msg.data = gesture_type
        self.gesture_pub.publish(gesture_msg)

        # اشارے کے لیے مخصوص حرکتیں
        if gesture_type == 'pointing':
            self.perform_pointing_gesture()
        elif gesture_type == 'nodding':
            self.perform_nodding_gesture()
        elif gesture_type == 'waving':
            self.perform_waving_gesture()
        elif gesture_type == 'thinking':
            self.perform_thinking_gesture()

    def perform_expression(self, expression_type):
        """ایک اظہار کریں"""
        expression_msg = String()
        expression_msg.data = expression_type
        self.expression_pub.publish(expression_msg)

    def perform_pointing_gesture(self):
        """اشارہ کریں"""
        # ہم فرض کرتے ہیں کہ ہدف کی پوزیشن دی گئی ہے
        target_point = self.get_current_gaze_target()

        if target_point:
            # اشارہ کریں کی طرف
            pointing_msg = PointStamped()
            pointing_msg.header.frame_id = 'base_link'
            pointing_msg.header.stamp = self.get_clock().now().to_msg()
            pointing_msg.point.x = target_point[0]
            pointing_msg.point.y = target_point[1]
            pointing_msg.point.z = target_point[2]

            self.pointing_pub.publish(pointing_msg)

    def perform_nodding_gesture(self):
        """سر ہلائیں (ہاں)"""
        # یہاں ہم سر کی ہلکی گردش کا کمانڈ بھیجیں گے
        gesture_msg = String()
        gesture_msg.data = 'nodding'
        self.gesture_pub.publish(gesture_msg)

    def perform_waving_gesture(self):
        """ہاتھ ہلائیں (ہیلو)"""
        # ہاتھ کو ہلکا ہلائیں
        gesture_msg = String()
        gesture_msg.data = 'waving'
        self.gesture_pub.publish(gesture_msg)

    def perform_thinking_gesture(self):
        """سوچنے کا اشارہ"""
        # سر کو ہلکا جھکائیں، یا ہاتھ چبھائیں
        gesture_msg = String()
        gesture_msg.data = 'thinking'
        self.gesture_pub.publish(gesture_msg)

    def calculate_pointing_direction(self, target_position, robot_position):
        """اشارہ کرنے کی سمت کا حساب لگائیں"""
        # ہدف کی طرف ویکٹر
        direction_vector = [
            target_position[0] - robot_position[0],
            target_position[1] - robot_position[1],
            target_position[2] - robot_position[2]
        ]

        # ویکٹر کو نارملائز کریں
        magnitude = math.sqrt(sum(v**2 for v in direction_vector))
        if magnitude > 0:
            normalized_direction = [v / magnitude for v in direction_vector]
        else:
            normalized_direction = [0, 0, 0]

        return normalized_direction

    def get_current_gaze_target(self):
        """موجودہ دیکھنے کا ہدف حاصل کریں"""
        # یہاں ہم ایک نمونہ ہدف کا استعمال کریں گے
        # اصل میں، آپ کو دیکھنے کے سسٹم سے ہدف حاصل کرنا ہوگا
        if self.robot_state['gaze_target']:
            return self.robot_state['gaze_target']
        else:
            # ڈیفالٹ ہدف (1 میٹر آگے)
            return [
                self.robot_state['head_position'][0] + 1.0,
                self.robot_state['head_position'][1],
                self.robot_state['head_position'][2]
            ]
```

## نیوی گیشن اور تعامل

### تعامل کے ساتھ نیوی گیشن

انسان نما روبوٹ کو انسانوں کے ساتھ تعامل کے دوران نیوی گیٹ کرنا بھی چاہیے:

```python
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
import numpy as np

class InteractiveNavigationSystem(Node):
    def __init__(self):
        super().__init__('interactive_navigation_system')

        # نیوی گیشن کے لیے سبسکرائبرز اور پبلشرز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # تعامل کے لیے سبسکرائبرز اور پبلشرز
        self.voice_command_sub = self.create_subscription(String, '/voice_commands', self.voice_command_callback, 10)
        self.user_proximity_sub = self.create_subscription(String, '/user_proximity', self.user_proximity_callback, 10)

        # روبوٹ کی حالت
        self.current_pose = None
        self.is_navigating = False
        self.following_user = False
        self.user_follow_position = None

        # تعامل کے لیے ٹائمر
        self.interaction_timer = self.create_timer(0.1, self.interaction_callback)

    def voice_command_callback(self, msg):
        """وائس کمانڈز کو ہینڈل کریں"""
        command = msg.data.lower()

        if 'follow me' in command or 'come with me' in command:
            self.start_following_user()
        elif 'stop following' in command or 'stay here' in command:
            self.stop_following_user()
        elif 'go to' in command or 'move to' in command:
            self.parse_and_execute_navigation_command(command)
        elif 'come to me' in command or 'come here' in command:
            self.navigate_to_user()
        else:
            self.get_logger().info(f'نامعلوم کمانڈ: {command}')

    def user_proximity_callback(self, msg):
        """صارف کی قربت کو ہینڈل کریں"""
        proximity_data = eval(msg.data)  # نوٹ: صرف ٹیسٹ کے لیے، اصل میں JSON استعمال کریں
        self.user_follow_position = proximity_data.get('position')

    def start_following_user(self):
        """صارف کو فالو کرنا شروع کریں"""
        self.following_user = True
        self.get_logger().info('صارف کو فالو کرنا شروع کیا')

    def stop_following_user(self):
        """صارف کو فالو کرنا بند کریں"""
        self.following_user = False
        self.get_logger().info('صارف کو فالو کرنا بند کیا')

    def navigate_to_user(self):
        """صارف کی طرف جائیں"""
        if self.user_follow_position:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = self.user_follow_position[0]
            goal_pose.pose.position.y = self.user_follow_position[1]
            goal_pose.pose.position.z = 0.0

            # اورینٹیشن (صارف کی طرف دیکھیں)
            direction = np.array([
                self.user_follow_position[0] - self.current_pose.position.x,
                self.user_follow_position[1] - self.current_pose.position.y
            ])
            distance = np.linalg.norm(direction)
            if distance > 0.1:  # 10cm کے اندر نہ جائیں
                direction_normalized = direction / distance
                yaw = np.arctan2(direction_normalized[1], direction_normalized[0])

                # yaw سے کوائف
                goal_pose.pose.orientation.z = np.sin(yaw / 2.0)
                goal_pose.pose.orientation.w = np.cos(yaw / 2.0)

                self.goal_pub.publish(goal_pose)

    def parse_and_execute_navigation_command(self, command):
        """نیوی گیشن کمانڈ کو پارس کریں اور انجام دیں"""
        # کمانڈ کو پارس کریں (مثال: "go to kitchen", "move to living room")
        if 'kitchen' in command:
            self.go_to_kitchen()
        elif 'living room' in command:
            self.go_to_living_room()
        elif 'bedroom' in command:
            self.go_to_bedroom()
        elif 'office' in command:
            self.go_to_office()
        else:
            self.get_logger().info(f'نامعلوم منزل: {command}')

    def go_to_kitchen(self):
        """کچن کی طرف جائیں"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = 2.0
        goal_pose.pose.position.y = 3.0
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)

    def go_to_living_room(self):
        """لیونگ روم کی طرف جائیں"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = 0.0
        goal_pose.pose.position.y = 0.0
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)

    def go_to_bedroom(self):
        """بیڈ روم کی طرف جائیں"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = -2.0
        goal_pose.pose.position.y = 1.0
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)

    def go_to_office(self):
        """دفتر کی طرف جائیں"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = 1.0
        goal_pose.pose.position.y = -2.0
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)

    def interaction_callback(self):
        """تعامل کا کال بیک"""
        if self.following_user and self.user_follow_position:
            # صارف کو فالو کریں
            self.follow_user()

    def follow_user(self):
        """صارف کو فالو کریں"""
        if self.user_follow_position:
            # صارف کی طرف جائیں لیکن ایک محفوظ فاصلہ رکھیں
            safe_distance = 1.0  # میٹر

            current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])
            user_pos = np.array(self.user_follow_position[:2])

            direction = user_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > safe_distance + 0.5:  # کافی دور ہیں
                # صارف کی طرف جائیں
                cmd_vel = Twist()
                cmd_vel.linear.x = min(0.5, (distance - safe_distance) * 0.5)  # 0.5 m/s max
                cmd_vel.angular.z = np.arctan2(direction[1], direction[0]) * 0.5  # Turn towards user

                self.cmd_vel_pub.publish(cmd_vel)
            elif distance < safe_distance - 0.2:  # بہت قریب ہیں
                # پیچھے ہٹیں
                cmd_vel = Twist()
                cmd_vel.linear.x = -0.2  # Slowly move back
                self.cmd_vel_pub.publish(cmd_vel)
            else:
                # جگہ پر رکیں
                cmd_vel = Twist()
                self.cmd_vel_pub.publish(cmd_vel)
```

## جائزہ

انسان-روبوٹ تعامل انسان نما روبوٹکس کا ایک اہم حصہ ہے۔ گفتگو کے نظام، اشارے، اظہار، اور تعامل کے ذرائع کو سمجھنا روبوٹ کو انسانوں کے ساتھ مؤثر طریقے سے بات چیت کرنے اور تعامل کرنے کے قابل بناتا ہے۔ ROS 2 کے ساتھ HRI کا نفاذ روبوٹک سسٹم کے انضمام کے لیے ضروری ہے۔

