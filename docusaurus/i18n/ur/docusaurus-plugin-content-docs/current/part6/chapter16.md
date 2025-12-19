---
title: "باب 16: قدرتی انسان-روبوٹ انٹرایکشن"
sidebar_label: "باب 16: انسان-روبوٹ انٹرایکشن"
---

# باب 16: قدرتی انسان-روبوٹ انٹرایکشن

## سیکھنے کے اہداف
- انسان-روبوٹ انٹرایکشن اور سوشل روبوٹکس کے اصولوں کو سمجھنا
- متعدد ماڈلٹی کمیونیکیشن سسٹم (speech, gesture, facial expressions) نافذ کرنا
- سوشل ویئر نیویگیشن اور انٹرایکشن کے رویے کا ڈیزائن کرنا
- انسان-روبوٹ انٹرایکشن کی کوالٹی اور صارف کے تجربے کا جائزہ لینا

## تعارف

قدرتی انسان-روبوٹ انٹرایکشن (NHRI) ہیومنوائڈ روبوٹکس کا ایک اہم جزو ہے، جس کی اجازت دیتا ہے کہ انسان روزمرہ کے ماحول میں مؤثر طریقے سے مربوط اور تعاون کریں۔ روایتی صنعتی روبوٹس کے برعکس جو معزول ماحول میں کام کرتے ہیں، ہیومنوائڈ روبوٹس کو پیچیدہ سماجی سیاق و سباق کو نیویگیٹ کرنا، انسانی ارادے کی تشریح کرنا، اور سماجی اشاروں کے جواب میں مناسب طریقے سے جواب دینا ہوتا ہے۔ یہ باب اصولوں، ٹیکنالوجیز، اور قدرتی اور ذاتی انسان-روبوٹ انٹرایکشنز کے لیے ڈیزائن کے ا consideration کو تلاش کرتا ہے۔

## انسان-روبوٹ انٹرایکشن کے اصول

### سوشل روبوٹکس کی بنیادیں

سماجی روبوٹکس روبوٹس کی ڈیزائن اور نفاذ کو احاطہ کرتا ہے جو انسانوں کے ساتھ سماجی طور پر معنی خیز انداز میں ملوث ہوتے ہیں:

1. **سماجی اشاروں کی شناخت**: انسانی سماجی سگنلز کو سمجھنے اور تشریح کرنے کی صلاحیت
2. **مناسب جواب کی تیاری**: سیاق و سباق کے مطابق متعلقہ جوابات کا تیار کرنا
3. **سماجی نارملز کی پابندی**: ثقافتی اور سماجی رواجوں کو فالو کرنا
4. **ایمان کی تعمیر**: انسان-روبوٹ ایمان کے تعلقات کو قائم کرنا اور برقرار رکھنا
5. ** ذاتی نوعیت**: انفرادی صارف کی ترجیحات اور صلاحیتوں کے مطابق ایڈجسٹ کرنا

### انٹرایکشن ماڈلٹیز

موثر HRI متعدد کمیونیکیشن چینلز کا استعمال کرتا ہے:

```python
# متعدد ماڈلٹی انٹرایکشن فریم ورک
class MultimodalInteraction:
    def __init__(self):
        self.modalities = {
            'speech': SpeechInterface(),
            'gesture': GestureRecognition(),
            'facial_expression': FacialExpressionSystem(),
            'gaze': GazeTracking(),
            'proxemics': ProxemicsManager()
        }

    def process_interaction(self, human_input):
        """
        متعدد ماڈلٹیز سے ان پٹ کو ایک ساتھ پروسیس کریں
        """
        processed_inputs = {}

        for modality_name, modality_system in self.modalities.items():
            processed_inputs[modality_name] = modality_system.process_input(human_input)

        # ماڈلٹیز کے درمیان معلومات کو ضم کریں
        integrated_input = self.integrate_modalities(processed_inputs)

        return integrated_input

    def integrate_modalities(self, modality_inputs):
        """
        مختلف ماڈلٹیز سے معلومات کو ضم کریں
        """
        # متعدد دلائل کو وزن دینے کے لیے توجہ کے میکنزمز کا استعمال کریں
        # سیاق و سباق اور قابل اعتمادی کے مطابق
        integrated_output = {}

        # مثال: روبوٹ کے سامنے ہونے پر speech کو زیادہ وزن دیں
        if self.is_facing_human():
            integrated_output['speech_weight'] = 0.8
        else:
            integrated_output['gesture_weight'] = 0.7

        return integrated_output
```

### روبوٹس میں ذہن کی تھیوری

ذہن کی تھیوری روبوٹس کو انسانوں کی ذہنی حالت کو نسبت دینے کی اجازت دیتی ہے:

```python
# ذہن کی تھیوری نفاذ
class TheoryOfMind:
    def __init__(self):
        self.belief_model = BeliefModel()
        self.intention_recognizer = IntentionRecognizer()
        self.mind_state_predictor = MindStatePredictor()

    def attribute_mental_state(self, human_behavior):
        """
        انسانی رویے کو ارادے، خواہشات، اور ارادوں کو نسبت دیں
        """
        # مشاہدہ شدہ اعمال سے ارادے کو پہچانیں
        intention = self.intention_recognizer.recognize(human_behavior['actions'])

        # انسانی نقطہ نظر سے دنیا کے بارے میں عقائد کو سمجھیں
        beliefs = self.belief_model.infer(human_behavior['observations'])

        # ذہنی حالت کے مطابق مستقبل کے اعمال کی پیشن گوئی کریں
        predicted_actions = self.mind_state_predictor.predict(beliefs, intention)

        return {
            'intention': intention,
            'beliefs': beliefs,
            'predicted_actions': predicted_actions
        }

    def predict_human_response(self, robot_action):
        """
        یہ دیکھنے کے لیے کہ انسان روبوٹ کے ایکشن کا کیا جواب دے گا
        """
        # روبوٹ کے بارے میں انسان کا ذہنی ماڈل بنائیں
        human_model_of_robot = self.belief_model.create_model(robot_action)

        # انسان کے ذہنی ماڈل کے مطابق انسان کی ردعمل کی پیشن گوئی کریں
        predicted_response = self.mind_state_predictor.predict(
            human_model_of_robot, robot_action
        )

        return predicted_response
```

## گفتگو اور زبان کی انٹرایکشن

### قدرتی زبان کی تفہیم

قدرتی زبان کی تفہیم (NLU) روبوٹس کو انسانی گفتگو کی تشریح کرنے کی اجازت دیتی ہے:

```python
# قدرتی زبان کی تفہیم سسٹم
import speech_recognition as sr
from transformers import pipeline
import spacy

class NaturalLanguageUnderstanding:
    def __init__(self):
        self.speech_recognizer = sr.Recognizer()
        self.language_model = pipeline("question-answering")
        self.nlp_processor = spacy.load("en_core_web_sm")
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()

    def process_speech_input(self, audio_input):
        """
        گفتگو کو پروسیس کریں اور معنی نکالیں
        """
        # گفتگو کو ٹیکسٹ میں تبدیل کریں
        text = self.speech_recognizer.recognize_google(audio_input)

        # جملے کی ساخت کو پارس کریں اور اجزاء نکالیں
        doc = self.nlp_processor(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # ارادے کی شناخت کریں
        intent = self.intent_classifier.classify(text)

        # سیمینٹک معنی نکالیں
        semantic_meaning = self.extract_semantic_meaning(text, entities, intent)

        return {
            'text': text,
            'entities': entities,
            'intent': intent,
            'semantic_meaning': semantic_meaning
        }

    def speech_to_text(self, audio):
        """
        گفتگو کو ٹیکسٹ میں تبدیل کریں
        """
        try:
            text = self.speech_recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service سے نتائج نہیں مل سکے; {e}")
            return None

    def extract_semantic_meaning(self, text, entities, intent):
        """
        ٹیکسٹ، اجزاء، اور ارادے سے سیمینٹک معنی نکالیں
        """
        # ارادے اور اجزاء کو جوڑ کر سیمینٹک نمائندگی تشکیل دیں
        semantic_meaning = {
            'action': intent,
            'objects': [ent[0] for ent in entities if ent[1] in ['OBJECT', 'PRODUCT']],
            'locations': [ent[0] for ent in entities if ent[1] in ['GPE', 'LOC']],
            'people': [ent[0] for ent in entities if ent[1] in ['PERSON']],
            'quantities': [ent[0] for ent in entities if ent[1] in ['CARDINAL', 'MONEY']]
        }

        return semantic_meaning
```

### گفتگو کی ترکیب اور تیاری

قدرتی گفتگو کی ترکیب انسان نما روبوٹ کے جوابات تیار کرتی ہے:

```python
# گفتگو کی ترکیب اور تیاری
import pyttsx3
import os
from gtts import gTTS

class SpeechSynthesis:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.voice_styles = ['neutral', 'friendly', 'professional', 'enthusiastic']
        self.current_voice = 'neutral'

    def generate_response(self, semantic_response, context):
        """
        سیمینٹک معنی کے مطابق قدرتی زبان کا جواب تیار کریں
        """
        # سیاق و سباق کے مطابق مناسب جواب ٹیمپلیٹ منتخب کریں
        template = self.select_response_template(semantic_response['intent'], context)

        # مخصوص اجزاء کے ساتھ ٹیمپلیٹ کو بھریں
        response_text = self.fill_template(template, semantic_response['entities'])

        return response_text

    def select_response_template(self, intent, context):
        """
        ارادے اور سیاق و سباق کے مطابق مناسب جواب ٹیمپلیٹ منتخب کریں
        """
        templates = {
            'greeting': [
                "ہیلو! میں آج آپ کی کس طرح مدد کر سکتا ہوں؟",
                "ہی! یہ دیکھ کر اچھا لگا! کہ آپ کیسے ہیں؟",
                "اچھا دن! میں آپ کی کس طرح مدد کر سکتا ہوں؟"
            ],
            'navigation_request': [
                "میں آپ کو {location} کی طرف نیویگیٹ کرنے میں مدد کر سکتا ہوں.",
                "ضرور، میں آپ کو {location} تک لے جاؤں گا.",
                "میں آپ کو {location} لے جاؤں گا. مجھے فالو کریں!"
            ],
            'manipulation_request': [
                "میں آپ کے ساتھ اس {object} کے کام میں مدد کر سکتا ہوں.",
                "ضرور، میں آپ کے لیے وہ {object} لاتا ہوں.",
                "میں آپ کے لیے {object} حاصل کر لوں گا."
            ]
        }

        import random
        return random.choice(templates.get(intent, ["میں سمجھتا ہوں."]))

    def fill_template(self, template, entities):
        """
        جواب ٹیمپلیٹ کو مخصوص اجزاء کے ساتھ بھریں
        """
        # متعلقہ اجزاء نکالیں
        locations = entities.get('locations', [])
        objects = entities.get('objects', [])

        # اجزاء کے ساتھ ٹیمپلیٹ کو بھریں
        if locations:
            return template.format(location=locations[0])
        elif objects:
            return template.format(object=objects[0])
        else:
            return template

    def speak_text(self, text):
        """
        ٹیکسٹ کو گفتگو میں تبدیل کریں اور چلائیں
        """
        # مطلوبہ انداز کے مطابق وائس کی خصوصیات سیٹ کریں
        self.set_voice_properties(self.current_voice)

        # ٹیکسٹ کو گفتگو میں تبدیل کریں
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def set_voice_properties(self, style):
        """
        مطلوبہ انداز کے مطابق وائس کی خصوصیات سیٹ کریں
        """
        if style == 'friendly':
            self.tts_engine.setProperty('rate', 150)  # ذرا سست
            self.tts_engine.setProperty('volume', 0.9)
        elif style == 'professional':
            self.tts_engine.setProperty('rate', 180)  # معیاری رفتار
            self.tts_engine.setProperty('volume', 0.8)
        elif style == 'enthusiastic':
            self.tts_engine.setProperty('rate', 160)  # تیز
            self.tts_engine.setProperty('volume', 1.0)
```

## اشارہ کی شناخت اور مواصلات

### اشارہ کی شناخت کے سسٹم

اشارہ کی شناخت روبوٹس کو انسانی بدن کی زبان کی تشریح کرنے کی اجازت دیتی ہے:

```python
# اشارہ کی شناخت کا سسٹم
import cv2
import mediapipe as mp
import numpy as np

class GestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7
        )
        self.gesture_classifier = GestureClassifier()
        self.gesture_library = self.load_gesture_library()

    def recognize_gestures(self, image):
        """
        تصویری ان پٹ سے اشاروں کی شناخت کریں
        """
        # ہاتھ کے نشانات کے لیے تصویر کو پروسیس کریں
        hand_results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pose_results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        gestures = []

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # ہاتھ کی پوز کی خصوصیات نکالیں
                features = self.extract_hand_features(hand_landmarks)

                # اشارہ کی شناخت کریں
                gesture = self.gesture_classifier.classify(features)
                gestures.append(gesture)

        if pose_results.pose_landmarks:
            # بدن کی پوز کی خصوصیات نکالیں
            body_features = self.extract_body_features(pose_results.pose_landmarks)

            # بدن کا اشارہ کی شناخت کریں
            body_gesture = self.gesture_classifier.classify_body_gesture(body_features)
            gestures.append(body_gesture)

        return gestures

    def extract_hand_features(self, hand_landmarks):
        """
        اشارہ کی شناخت کے لیے ہاتھ کے نشانات سے خصوصیات نکالیں
        """
        # کلیدی نقاط کے درمیان فاصلے کا حساب لگائیں
        features = []

        # پالم سینٹر (کلائی اور مڈل فنگر کے MCP کی تقریبی)
        palm_center = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
        ])

        # فنگر ٹپ کی پوزیشنز کو پالم کے حوالے سے نکالیں
        for finger_tip in [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]:
            tip_pos = np.array([
                hand_landmarks.landmark[finger_tip].x,
                hand_landmarks.landmark[finger_tip].y
            ])
            relative_pos = tip_pos - palm_center
            features.extend(relative_pos)

        return np.array(features)

    def generate_robot_gestures(self, context):
        """
        سیاق و سباق کے مطابق مناسب روبوٹ اشارے تیار کریں
        """
        # انٹرایکشن کے سیاق و سباق کے مطابق اشارہ منتخب کریں
        if context['interaction_type'] == 'greeting':
            return self.select_greeting_gesture()
        elif context['interaction_type'] == 'navigation':
            return self.select_navigation_gesture()
        elif context['interaction_type'] == 'attention':
            return self.select_attention_gesture()
        else:
            return self.select_neutral_gesture()

    def select_greeting_gesture(self):
        """
        مناسب جیسے کا انتخاب کریں
        """
        return {
            'type': 'wave',
            'duration': 2.0,
            'amplitude': 0.3,
            'frequency': 1.0
        }
```

### روبوٹ کے لیے اشارہ تیاری

روبوٹس انسانوں کے ساتھ بات چیت کے لیے معنی خیز اشارے تیار کر سکتے ہیں:

```python
# روبوٹ اشارہ تیاری
class RobotGestureGenerator:
    def __init__(self):
        self.gesture_sequences = {
            'greeting': ['raise_arm', 'wave_hand', 'lower_arm'],
            'attention': ['point', 'look_at_human', 'wait_for_response'],
            'navigation': ['point_direction', 'step_forward', 'gesture_path'],
            'acknowledgment': ['nod', 'smile_led', 'wait']
        }

    def execute_gesture_sequence(self, sequence_name, parameters=None):
        """
        از قبل طے شدہ اشارہ ترتیب نافذ کریں
        """
        sequence = self.gesture_sequences.get(sequence_name, [])

        for gesture_name in sequence:
            self.execute_single_gesture(gesture_name, parameters)

    def execute_single_gesture(self, gesture_name, parameters):
        """
        پیرامیٹرز کے ساتھ واحد اشارہ نافذ کریں
        """
        if gesture_name == 'raise_arm':
            self.raise_arm(parameters.get('arm', 'right'), parameters.get('angle', 90))
        elif gesture_name == 'wave_hand':
            self.wave_hand(parameters.get('arm', 'right'),
                         parameters.get('amplitude', 0.3),
                         parameters.get('frequency', 1.0))
        elif gesture_name == 'point':
            self.point_to_location(parameters.get('target_location'))
        elif gesture_name == 'nod':
            self.nod_head(parameters.get('amplitude', 0.2), parameters.get('duration', 1.0))

    def raise_arm(self, arm, angle):
        """
        مخصوص اینگل تک مخصوص بازو اٹھائیں
        """
        # بازو کے جوائنٹ اینگلز کو کنٹرول کریں تاکہ مطلوبہ پوزیشن حاصل ہو
        if arm == 'right':
            # دائیں بازو جوائنٹس کو حرکت دیں
            pass
        elif arm == 'left':
            # بائیں بازو جوائنٹس کو حرکت دیں
            pass

    def wave_hand(self, arm, amplitude, frequency):
        """
        مخصوص امپلی ٹیوڈ اور فریکوینسی کے ساتھ ہاتھ ہلائیں
        """
        import time
        import math

        start_time = time.time()
        duration = 2.0  # 2 سیکنڈ مکمل ہلک
        while time.time() - start_time < duration:
            # جیبیویئر حرکت تیار کریں
            wave_angle = amplitude * math.sin(2 * math.pi * frequency * (time.time() - start_time))

            # ہلک کو ہاتھ پر لاگو کریں
            if arm == 'right':
                # دائیں ہاتھ پر ہلک لاگو کریں
                pass
            elif arm == 'left':
                # بائیں ہاتھ پر ہلک لاگو کریں
                pass

            time.sleep(0.01)  # 100 Hz اپ ڈیٹ کی شرح

    def point_to_location(self, target_location):
        """
        مخصوص مقام کی طرف اشارہ کریں
        """
        # ہدف کی سمت کا حساب لگائیں
        # اشارہ کرنے کے لیے بازو حرکت دیں
        pass

    def nod_head(self, amplitude, duration):
        """
        مخصوص امپلی ٹیوڈ اور مدت کے ساتھ سر ہلائیں
        """
        import time
        start_time = time.time()
        current_time = start_time

        while current_time - start_time < duration:
            # ہلک حرکت تیار کریں
            progress = (current_time - start_time) / duration
            angle = amplitude * math.sin(2 * math.pi * progress * 2)  # 2 ہلک مدت کے دوران

            # سر کی حرکت لاگو کریں
            current_time = time.time()
            time.sleep(0.01)  # 100 Hz اپ ڈیٹ کی شرح
```

## چہرے کی اظہار اور جذباتی مواصلات

### چہرے کے اظہار کے سسٹم

چہرے کے اظہار روبوٹس کو جذبات اور سماجی سگنلز کو ظاہر کرنے کی اجازت دیتے ہیں:

```python
# چہرے کے اظہار کا سسٹم
class FacialExpressionSystem:
    def __init__(self):
        self.expression_library = {
            'happy': {'eyes': 'smile', 'mouth': 'smile', 'eyebrows': 'raised'},
            'sad': {'eyes': 'droop', 'mouth': 'frown', 'eyebrows': 'lowered'},
            'surprised': {'eyes': 'wide', 'mouth': 'open', 'eyebrows': 'raised'},
            'angry': {'eyes': 'narrow', 'mouth': 'frown', 'eyebrows': 'furrowed'},
            'neutral': {'eyes': 'normal', 'mouth': 'neutral', 'eyebrows': 'normal'}
        }
        self.current_expression = 'neutral'
        self.expression_intensity = 1.0

    def set_expression(self, expression_name, intensity=1.0):
        """
        مخصوص شدت کے ساتھ چہرے کا اظہار سیٹ کریں
        """
        if expression_name in self.expression_library:
            self.current_expression = expression_name
            self.expression_intensity = intensity

            # اظہار کے مطابق چہرے کی خصوصیات کو اپ ڈیٹ کریں
            expression_config = self.expression_library[expression_name]
            self.update_facial_features(expression_config, intensity)

    def update_facial_features(self, expression_config, intensity):
        """
        اظہار کی ترتیب کے مطابق چہرے کی خصوصیات کو اپ ڈیٹ کریں
        """
        # ہر چہرے کے فیچر کو اپ ڈیٹ کریں
        for feature, setting in expression_config.items():
            self.set_facial_feature(feature, setting, intensity)

    def set_facial_feature(self, feature, setting, intensity):
        """
        مخصوص چہرے کے فیچر کو دی گئی ترتیب اور شدت کے ساتھ سیٹ کریں
        """
        # چہرے کا اظہار بنانے کے لیے LED اریز، سرvo، یا ڈسپلے عناصر کو کنٹرول کریں
        pass

    def animate_transition(self, from_expression, to_expression, duration=1.0):
        """
        اظہار کے درمیان ہموار ٹرانزیشن اینیمیٹ کریں
        """
        import time

        start_time = time.time()

        while time.time() - start_time < duration:
            # اظہار کے درمیان انٹرپولیٹ کریں
            progress = (time.time() - start_time) / duration
            self.interpolate_expressions(from_expression, to_expression, progress)
            time.sleep(0.05)  # 20 Hz اپ ڈیٹ کی شرح

    def interpolate_expressions(self, from_expr, to_expr, progress):
        """
        پیش رفت کے مطابق دو اظہار کے درمیان انٹرپولیٹ کریں
        """
        from_config = self.expression_library[from_expr]
        to_config = self.expression_library[to_expr]

        # ہر فیچر کے لیے لکیری انٹرپولیشن
        for feature in from_config.keys():
            # حقیقی نفاذ میں یہ حقیقی فیچر ویلیوز کو بلینڈ کرے گا
            pass
```

### جذباتی حالت کی ماڈلنگ

جذباتی حالتیں روبوٹ کے رویے اور انٹرایکشن کو متاثر کرتی ہیں:

```python
# جذباتی حالت کی ماڈلنگ
class EmotionalStateModel:
    def __init__(self):
        self.emotional_states = {
            'happiness': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0
        }
        self.mood = 'neutral'
        self.arousal = 0.5  # 0.0 سے 1.0
        self.valence = 0.5  # 0.0 سے 1.0 (منفی سے مثبت)

    def update_emotional_state(self, event):
        """
        انٹرایکشن ایونٹس کے مطابق جذباتی حالت کو اپ ڈیٹ کریں
        """
        # مثبت/منفی ایونٹس کے مطابق اپ ڈیٹ کریں
        if event['type'] == 'positive':
            self.emotional_states['happiness'] += 0.2
            self.emotional_states['sadness'] -= 0.1
        elif event['type'] == 'negative':
            self.emotional_states['sadness'] += 0.2
            self.emotional_states['happiness'] -= 0.1
        elif event['type'] == 'surprising':
            self.emotional_states['surprise'] += 0.3

        # جذباتی حالتیں نارملائز کریں
        self.normalize_emotional_states()

        # جذباتی حالتیں کے مطابق موڈ اور اروسال کو اپ ڈیٹ کریں
        self.update_mood()
        self.update_arousal()

    def normalize_emotional_states(self):
        """
        یقینی بنائیں کہ جذباتی حالتیں درست حدود [0, 1] میں ہیں
        """
        for emotion, value in self.emotional_states.items():
            self.emotional_states[emotion] = max(0.0, min(1.0, value))

    def update_mood(self):
        """
        جذباتی حالتیں کے مطابق مجموعی موڈ کو اپ ڈیٹ کریں
        """
        # جذباتی حالتیں کا وزنی اوسط حساب لگائیں
        total_weight = sum(self.emotional_states.values())

        if total_weight > 0:
            weighted_mood = sum(
                value * self.get_emotion_weight(emotion)
                for emotion, value in self.emotional_states.items()
            ) / total_weight

            if weighted_mood > 0.6:
                self.mood = 'positive'
            elif weighted_mood < 0.4:
                self.mood = 'negative'
            else:
                self.mood = 'neutral'

    def get_emotion_weight(self, emotion):
        """
        موڈ کے حساب کے لیے جذبے کا وزن حاصل کریں
        """
        weights = {
            'happiness': 1.0,
            'sadness': -1.0,
            'anger': -0.8,
            'fear': -0.6,
            'surprise': 0.3,
            'disgust': -0.9
        }
        return weights.get(emotion, 0.0)

    def get_appropriate_response(self, human_emotion):
        """
        روبوٹ کی جذباتی حالت اور انسان کے جذبے کے مطابق مناسب جواب حاصل کریں
        """
        if self.mood == 'positive' and human_emotion == 'happy':
            return 'match_positive_emotion'
        elif self.mood == 'negative' and human_emotion == 'sad':
            return 'show_empathy'
        elif self.mood == 'neutral' and human_emotion == 'angry':
            return 'show_calm_reassurance'
        else:
            return 'standard_response'
```

## پروکسمکس اور سپیشل انٹرایکشن

### ذاتی جگہ کا نظم کرنا

پروکسمکس انسانوں اور روبوٹس کے درمیان مناسب جگہ کے رشتے کا تعین کرتا ہے:

```python
# پروکسمکس مینجمنٹ سسٹم
class ProxemicsManager:
    def __init__(self):
        self.personal_space_zones = {
            'intimate': 0.45,    # 0-1.5 فٹ
            'personal': 1.2,     # 1.5-4 فٹ
            'social': 3.6,       # 4-12 فٹ
            'public': 7.6        # 12+ فٹ
        }
        self.current_distance = 2.0  # ڈیفالٹ سوشل فاصلہ
        self.appropriate_distance = self.personal_space_zones['social']

    def calculate_appropriate_distance(self, interaction_type, human_culture):
        """
        انٹرایکشن کی قسم اور ثقافت کے مطابق مناسب فاصلہ کا حساب لگائیں
        """
        base_distance = self.personal_space_zones['social']

        # انٹرایکشن کی قسم کے مطابق ایڈجسٹ کریں
        if interaction_type == 'greeting':
            base_distance = self.personal_space_zones['personal']
        elif interaction_type == 'intimate_conversation':
            base_distance = self.personal_space_zones['intimate']
        elif interaction_type == 'presentation':
            base_distance = self.personal_space_zones['public']

        # ثقافتی ترجیحات کے مطابق ایڈجسٹ کریں
        cultural_factor = self.get_cultural_distance_factor(human_culture)
        appropriate_distance = base_distance * cultural_factor

        return appropriate_distance

    def get_cultural_distance_factor(self, culture):
        """
        فاصلے کی ترجیحات کے لیے ثقافتی عنصر حاصل کریں
        """
        cultural_factors = {
            'mediterranean': 0.8,  # قریبی انٹرایکشن
            'north_american': 1.0,  # معیاری فاصلہ
            'east_asian': 1.2,     # زیادہ فاصلہ
            'latin_american': 0.9, # قریبی انٹرایکشن
            'middle_eastern': 0.85 # قریبی انٹرایکشن
        }
        return cultural_factors.get(culture, 1.0)

    def maintain_appropriate_distance(self, human_position, robot_position):
        """
        انسان سے مناسب فاصلہ برقرار رکھیں
        """
        import math

        # موجودہ فاصلہ کا حساب لگائیں
        current_distance = math.sqrt(
            (human_position.x - robot_position.x)**2 +
            (human_position.y - robot_position.y)**2
        )

        # چیک کریں کہ ایڈجسٹمنٹ کی ضرورت ہے یا نہیں
        if current_distance < self.appropriate_distance * 0.8:
            # بہت قریب - دور جائیں
            self.move_away_from_human(human_position, robot_position)
        elif current_distance > self.appropriate_distance * 1.2:
            # بہت دور - قریب جائیں
            self.move_towards_human(human_position, robot_position)

    def move_away_from_human(self, human_pos, robot_pos):
        """
        روبوٹ کو انسان سے مناسب فاصلے تک دور کریں
        """
        # انسان سے روبوٹ کی طرف سمت ویکٹر کا حساب لگائیں
        direction_x = robot_pos.x - human_pos.x
        direction_y = robot_pos.y - human_pos.y

        # سمت کو noramlize کریں
        magnitude = math.sqrt(direction_x**2 + direction_y**2)
        if magnitude > 0:
            direction_x /= magnitude
            direction_y /= magnitude

        # مناسب فاصلے پر نئی پوزیشن کا حساب لگائیں
        new_x = human_pos.x + direction_x * self.appropriate_distance
        new_y = human_pos.y + direction_y * self.appropriate_distance

        # نئی پوزیشن پر جائیں
        self.navigate_to_position(new_x, new_y)

    def move_towards_human(self, human_pos, robot_pos):
        """
        روبوٹ کو انسان کی طرف مناسب فاصلے تک لے جائیں
        """
        # روبوٹ سے انسان کی طرف سمت ویکٹر کا حساب لگائیں
        direction_x = human_pos.x - robot_pos.x
        direction_y = human_pos.y - robot_pos.y

        # سمت کو noramlize کریں
        magnitude = math.sqrt(direction_x**2 + direction_y**2)
        if magnitude > 0:
            direction_x /= magnitude
            direction_y /= magnitude

        # مناسب فاصلے پر نئی پوزیشن کا حساب لگائیں
        new_x = human_pos.x - direction_x * self.appropriate_distance
        new_y = human_pos.y - direction_y * self.appropriate_distance

        # نئی پوزیشن پر جائیں
        self.navigate_to_position(new_x, new_y)

    def navigate_to_position(self, x, y):
        """
        روبوٹ کو مخصوص پوزیشن پر لے جائیں
        """
        # پوزیشن پر جانے کے لیے نیویگیشن سٹیک کا استعمال کریں
        pass
```

## سوشل نیویگیشن اور ویفائنڈنگ

### سوشل-ویئر نیویگیشن

سماجی نیویگیشن انسانی موجودگی اور سماجی رواج کو مدنظر رکھتی ہے:

```python
# سوشل-ویئر نیویگیشن
class SocialNavigation:
    def __init__(self):
        self.social_rules = {
            'avoid_back': True,           # پیچھے سے مت اپروچ کریں
            'respect_personal_space': True,  # مناسب فاصلہ برقرار رکھیں
            'yield_to_humans': True,      # انسانوں کو راستہ دیں
            'face_humans': True,          # ممکن ہو تو انسانوں کا سامنا کریں
            'avoid_interrupting': True    # گفتگو میں مت رکاوٹ ڈالیں
        }
        self.navigation_planner = SocialPathPlanner()

    def plan_social_path(self, start_pose, goal_pose, human_positions):
        """
        سماجی رواج کو مدنظر رکھتے ہوئے راستہ منصوبہ بندی کریں
        """
        # انسانوں کے مقامات کے مطابق سوشل کوسٹ میپ تیار کریں
        social_cost_map = self.create_social_cost_map(human_positions)

        # سوشل کوسٹ میپ کا استعمال کرتے ہوئے راستہ منصوبہ بندی کریں
        path = self.navigation_planner.plan_path_with_cost_map(
            start_pose, goal_pose, social_cost_map
        )

        return path

    def create_social_cost_map(self, human_positions):
        """
        سماجی طور پر غلط علاقوں کو مدنظر رکھتے ہوئے کوسٹ میپ تیار کریں
        """
        # کوسٹ میپ کو شروع کریں
        cost_map = np.zeros((100, 100))  # مثال کا سائز

        for human_pos in human_positions:
            # انسانوں کے گرد کوسٹ زونز شامل کریں
            self.add_social_cost_zones(cost_map, human_pos)

        return cost_map

    def add_social_cost_zones(self, cost_map, human_pos):
        """
        سماجی قوانین کے مطابق انسان کے گرد کوسٹ زونز شامل کریں
        """
        # ذاتی جگہ کی خلاف ورزی کے لیے زیادہ کوسٹ
        self.add_cost_around_point(cost_map, human_pos, radius=1.0, cost=100)

        # پیچھے سے اپروچ کے لیے میڈیم کوسٹ
        behind_pos = self.calculate_behind_position(human_pos)
        self.add_cost_around_point(cost_map, behind_pos, radius=0.5, cost=50)

        # چہرہ-چہرہ کے اپروچ کے لیے کم کوسٹ
        face_pos = self.calculate_face_position(human_pos)
        self.add_cost_around_point(cost_map, face_pos, radius=0.3, cost=10)

    def calculate_behind_position(self, human_pos):
        """
        انسان کے نقطہ نظر اور سمت کے مطابق پیچھے کی پوزیشن کا حساب لگائیں
        """
        # انسان کی سمت کا اندازا لگائیں
        # پیچھے کی پوزیشن کا حساب لگائیں
        pass

    def calculate_face_position(self, human_pos):
        """
        چہرہ-چہرہ انٹرایکشن کے لیے مناسب پوزیشن کا حساب لگائیں
        """
        # چہرہ-چہرہ انٹرایکشن کے لیے پوزیشن کا حساب لگائیں
        pass

    def execute_social_navigation(self, path, human_positions):
        """
        سماجی رکاوٹوں کو مدنظر رکھتے ہوئے نیویگیشن انجام دیں
        """
        for waypoint in path:
            # چیک کریں کہ حرکت سماجی قوانین کی خلاف ورزی کرے گی یا نہیں
            if self.would_violate_social_rules(waypoint, human_positions):
                # راستہ ایڈجسٹ کریں یا انتظار کریں
                adjusted_waypoint = self.adjust_for_social_rules(waypoint, human_positions)
                self.move_to_pose(adjusted_waypoint)
            else:
                self.move_to_pose(waypoint)

    def would_violate_social_rules(self, pose, human_positions):
        """
        چیک کریں کہ پوز کے ساتھ حرکت سماجی قوانین کی خلاف ورزی کرے گی یا نہیں
        """
        for human_pos in human_positions:
            distance = self.calculate_distance(pose, human_pos)

            # ذاتی جگہ کی خلاف ورزی چیک کریں
            if distance < 1.0:  # ذاتی جگہ کی حد
                return True

            # پیچھے سے اپروچ چیک کریں
            if self.is_approaching_from_behind(pose, human_pos):
                return True

        return False
```

## انسان-روبوٹ انٹرایکشن میں اعتماد اور قبولیت

### اعتماد کی تعمیر کے میکنزمز

اعتماد مؤثر انسان-روبوٹ تعاون کے لیے ضروری ہے:

```python
# اعتماد کی تعمیر کے میکنزمز
class TrustBuilder:
    def __init__(self):
        self.trust_model = {
            'competence': 0.5,      # روبوٹ کی ثابت شدہ صلاحیت
            'reliability': 0.5,     # کارکردگی کی مسلسل
            'predictability': 0.5,  # رویے کی پیشن گوئی کی صلاحیت
            'benevolence': 0.5,     # اچھی نیت کا احساس
            'transparency': 0.5     # روبوٹ کی حالت اور ارادوں کی وضاحت
        }
        self.trust_history = []

    def update_trust_after_interaction(self, interaction_outcome):
        """
        انٹرایکشن کے نتیجے کے مطابق اعتماد ماڈل کو اپ ڈیٹ کریں
        """
        # کامیابی/ناکامی کے مطابق صلاحیت کو اپ ڈیٹ کریں
        if interaction_outcome['success']:
            self.trust_model['competence'] += 0.1
        else:
            self.trust_model['competence'] -= 0.05

        # مسلسل کارکردگی کے مطابق قابل اعتمادی کو اپ ڈیٹ کریں
        if interaction_outcome['consistent']:
            self.trust_model['reliability'] += 0.05

        # امیدوں کو پورا کرنے کے مطابق قابل پیشن گوئی کو اپ ڈیٹ کریں
        if interaction_outcome['expected']:
            self.trust_model['predictability'] += 0.05

        # یقینی بنائیں کہ ویلیوز حدود میں رہیں [0, 1]
        for key in self.trust_model:
            self.trust_model[key] = max(0.0, min(1.0, self.trust_model[key]))

        # تاریخ میں انٹرایکشن کو ریکارڈ کریں
        self.trust_history.append({
            'outcome': interaction_outcome,
            'timestamp': time.time(),
            'trust_values': self.trust_model.copy()
        })

    def calculate_overall_trust(self):
        """
        وزنی اوسط کے طور پر مجموعی اعتماد کی سطح کا حساب لگائیں
        """
        weights = {
            'competence': 0.3,
            'reliability': 0.25,
            'predictability': 0.2,
            'benevolence': 0.15,
            'transparency': 0.1
        }

        overall_trust = sum(
            self.trust_model[key] * weights[key]
            for key in self.trust_model
        )

        return overall_trust

    def adapt_behavior_for_trust_building(self, user_trust_level):
        """
        صارف کی اعتماد کی سطح کے مطابق روبوٹ کا رویہ ایڈجسٹ کریں
        """
        if user_trust_level < 0.3:
            # کم اعتماد - انتہائی احتیاط اور وضاحت سے کام کریں
            return {
                'speed': 'slow',
                'transparency': 'high',
                'explanation': 'detailed',
                'autonomy': 'low'
            }
        elif user_trust_level < 0.7:
            # درمیانی اعتماد - کارکردگی اور احتیاط کو متوازن کریں
            return {
                'speed': 'moderate',
                'transparency': 'medium',
                'explanation': 'brief',
                'autonomy': 'medium'
            }
        else:
            # زیادہ اعتماد - زیادہ کارآمد اور خود مختار
            return {
                'speed': 'normal',
                'transparency': 'low',
                'explanation': 'minimal',
                'autonomy': 'high'
            }

    def provide_transparency_mechanisms(self):
        """
        اعتماد کی تعمیر کے لیے وضاحت کے میکنزمز فراہم کریں
        """
        return {
            'intent_explanation': self.explain_intention(),
            'action_explanation': self.explain_current_action(),
            'plan_explanation': self.explain_planned_actions(),
            'uncertainty_communication': self.communicate_uncertainty()
        }

    def explain_intention(self):
        """
        روبوٹ کا موجودہ ارادہ بیان کریں
        """
        return f"میں فی الحال {self.current_task} کر رہا ہوں تاکہ {self.goal_reason}"

    def communicate_uncertainty(self):
        """
        صارف کو روبوٹ کی عدم یقینی سے مطلع کریں
        """
        if self.uncertainty_level > 0.5:
            return "میں اس ایکشن کے بارے میں مکمل طور پر یقینی نہیں ہوں. کیا آپ چاہیں گے کہ میں جاری رکھوں؟"
        else:
            return "میں اس ایکشن میں یقینی ہوں."
```

## HRI میں ثقافتی ایڈاپٹیشن

### کراس-کلچرل انٹرایکشن

روبوٹس کو مختلف ثقافتی نارملز اور امکانات کے مطابق ایڈجسٹ کرنا چاہیے:

```python
# کراس-کلچرل انٹرایکشن ایڈاپٹیشن
class CulturalAdaptation:
    def __init__(self):
        self.cultural_models = {
            'japanese': {
                'personal_space': 1.5,
                'eye_contact': 'moderate',
                'greeting': 'bow',
                'formality': 'high',
                'directness': 'low'
            },
            'american': {
                'personal_space': 1.0,
                'eye_contact': 'high',
                'greeting': 'handshake',
                'formality': 'medium',
                'directness': 'high'
            },
            'middle_eastern': {
                'personal_space': 1.2,
                'eye_contact': 'moderate',
                'greeting': 'handshake',
                'formality': 'high',
                'directness': 'medium'
            }
        }
        self.current_cultural_model = 'american'  # ڈیفالٹ

    def adapt_to_culture(self, detected_culture):
        """
        تسلیم کردہ ثقافت کے مطابق انٹرایکشن انداز ایڈجسٹ کریں
        """
        if detected_culture in self.cultural_models:
            self.current_cultural_model = detected_culture
            cultural_params = self.cultural_models[detected_culture]

            # proxemics ایڈجسٹ کریں
            self.adjust_personal_space(cultural_params['personal_space'])

            # نظر کے رابطے کا رویہ ایڈجسٹ کریں
            self.adjust_eye_contact(cultural_params['eye_contact'])

            # رسمیت کی سطح ایڈجسٹ کریں
            self.adjust_formality(cultural_params['formality'])

            # رابطے کی براہ راست کو ایڈجسٹ کریں
            self.adjust_directness(cultural_params['directness'])

    def adjust_personal_space(self, distance_factor):
        """
        ثقافت کے مطابق مناسب ذاتی جگہ ایڈجسٹ کریں
        """
        # proxemics مینیجر کی ترتیبات کو تبدیل کریں
        self.proxemics_manager.appropriate_distance *= distance_factor

    def adjust_eye_contact(self, level):
        """
        ثقافتی ترجیحات کے مطابق نظر کے رابطے کا رویہ ایڈجسٹ کریں
        """
        if level == 'high':
            self.gaze_tracker.set_attention_level(0.8)
        elif level == 'moderate':
            self.gaze_tracker.set_attention_level(0.5)
        elif level == 'low':
            self.gaze_tracker.set_attention_level(0.2)

    def adjust_formality(self, level):
        """
        زبان اور رویے کی رسمیت کو ایڈجسٹ کریں
        """
        if level == 'high':
            self.speech_synthesizer.set_voice_style('formal')
            self.gesture_generator.set_gesture_intensity(0.3)
        elif level == 'medium':
            self.speech_synthesizer.set_voice_style('neutral')
            self.gesture_generator.set_gesture_intensity(0.5)
        elif level == 'low':
            self.speech_synthesizer.set_voice_style('casual')
            self.gesture_generator.set_gesture_intensity(0.7)

    def detect_cultural_background(self, human_behavior):
        """
        انسانی رویے کے نمونوں سے ثقافتی پس منظر کا تعین کریں
        """
        # رویے کے نمونوں، زبان، اشاروں، وغیرہ کا تجزیہ کریں
        cultural_indicators = {
            'greeting_style': self.analyze_greeting(human_behavior),
            'personal_space_preference': self.analyze_space_behavior(human_behavior),
            'communication_style': self.analyze_communication(human_behavior)
        }

        # ثقافتی ماڈلز سے میچ کریں
        best_match = self.match_to_cultural_model(cultural_indicators)

        return best_match

    def match_to_cultural_model(self, indicators):
        """
        مشاہدہ شدہ اشاروں کو ثقافتی ماڈلز سے میچ کریں
        """
        scores = {}

        for culture, model in self.cultural_models.items():
            score = 0
            for indicator, value in indicators.items():
                if value == model.get(indicator):
                    score += 1
            scores[culture] = score

        # سب سے زیادہ اسکور والی ثقافت لوٹائیں
        return max(scores, key=scores.get) if scores else 'american'
```

## جائزہ اور صارف کا تجربہ

### HRI جائزہ میٹرکس

انسان-روبوٹ انٹرایکشن کی کارکردگی کا جائزہ لینا:

```python
# HRI جائزہ فریم ورک
class HRIEvaluation:
    def __init__(self):
        self.metrics = {
            'task_success_rate': 0.0,
            'interaction_time': 0.0,
            'user_satisfaction': 0.0,
            'trust_level': 0.0,
            'social_acceptance': 0.0,
            'naturalness': 0.0
        }
        self.evaluation_sessions = []

    def evaluate_interaction(self, interaction_session):
        """
        متعدد میٹرکس کا استعمال کرتے ہوئے انٹرایکشن سیشن کا جائزہ لیں
        """
        evaluation = {}

        # کامیابی کی شرح
        evaluation['task_success_rate'] = self.calculate_task_success(
            interaction_session['tasks']
        )

        # انٹرایکشن کی کارکردگی
        evaluation['interaction_time'] = self.calculate_interaction_efficiency(
            interaction_session['duration']
        )

        # صارف کی مطمئنی (سوال نامہ یا رویے کے اشاروں سے)
        evaluation['user_satisfaction'] = self.assess_user_satisfaction(
            interaction_session['user_feedback']
        )

        # اعتماد کی سطح (سوال نامہ یا رویے کے تجزیے سے)
        evaluation['trust_level'] = self.assess_trust_level(
            interaction_session['trust_indicators']
        )

        # سماجی قبولیت (قریبی، ملوث ہونے کے وقت، وغیرہ سے)
        evaluation['social_acceptance'] = self.assess_social_acceptance(
            interaction_session['social_behavior']
        )

        # قدرتی نوعیت (انٹرایکشن کے نمونوں سے)
        evaluation['naturalness'] = self.assess_naturalness(
            interaction_session['interaction_patterns']
        )

        # جائزہ محفوظ کریں
        self.evaluation_sessions.append({
            'session_id': interaction_session['id'],
            'metrics': evaluation,
            'timestamp': time.time()
        })

        return evaluation

    def calculate_task_success(self, tasks):
        """
        کامیابی کی شرح کا حساب لگائیں
        """
        successful_tasks = sum(1 for task in tasks if task['success'])
        total_tasks = len(tasks)

        return successful_tasks / total_tasks if total_tasks > 0 else 0.0

    def calculate_interaction_efficiency(self, duration):
        """
        انٹرایکشن کی کارکردگی کا حساب لگائیں
        """
        # کم مدت = زیادہ کارکردگی (الٹا)
        # 0-1 اسکیل پر نارملائز کریں
        max_expected_duration = 300  # 5 منٹ
        efficiency = max(0, 1 - (duration / max_expected_duration))

        return efficiency

    def assess_user_satisfaction(self, feedback):
        """
        فیڈ بیک سے صارف کی مطمئنی کا جائزہ لیں
        """
        if 'questionnaire' in feedback:
            # سوال نامہ کے اسکورز کا اوسط
            scores = feedback['questionnaire']
            return sum(scores) / len(scores) if scores else 0.5
        elif 'behavioral_indicators' in feedback:
            # مطمئنی کے اشاروں کا تجزیہ کریں
            return self.analyze_satisfaction_indicators(
                feedback['behavioral_indicators']
            )
        else:
            return 0.5  # ڈیفالٹ نیوٹرل

    def assess_trust_level(self, trust_indicators):
        """
        مختلف اشاروں سے اعتماد کی سطح کا جائزہ لیں
        """
        trust_score = 0.0
        count = 0

        if 'physical_proximity' in trust_indicators:
            # قریبی فاصلہ زیادہ اعتماد کی علامت ہے
            trust_score += self.proximity_to_trust(trust_indicators['physical_proximity'])
            count += 1

        if 'interaction_frequency' in trust_indicators:
            # زیادہ تعدد زیادہ اعتماد کی علامت ہے
            trust_score += self.frequency_to_trust(trust_indicators['interaction_frequency'])
            count += 1

        if 'task_delegation' in trust_indicators:
            # کام کو سونپنے کی خواہش اعتماد کی علامت ہے
            trust_score += self.delegation_to_trust(trust_indicators['task_delegation'])
            count += 1

        return trust_score / count if count > 0 else 0.5

    def generate_evaluation_report(self):
        """
        جامع جائزہ رپورٹ تیار کریں
        """
        report = {
            'average_metrics': {},
            'trends_over_time': {},
            'recommendations': []
        }

        # اوسط میٹرکس کا حساب لگائیں
        for metric in self.metrics.keys():
            values = [session['metrics'][metric] for session in self.evaluation_sessions]
            if values:
                report['average_metrics'][metric] = sum(values) / len(values)

        # رجحانات کی شناخت کریں
        report['trends_over_time'] = self.analyze_trends()

        # تجاویز تیار کریں
        report['recommendations'] = self.generate_recommendations()

        return report
```

## عملی مشق: سوشل انٹرایکشن سسٹم کا نفاذ

### مشق کے اہداف
- گفتگو اور اشارے کے ساتھ بنیادی سوشل انٹرایکشن سسٹم نافذ کریں
- مناسب فاصلے کے نظم کے لیے proxemics مینجمنٹ ضم کریں
- مختلف ثقافتی ترتیبات کے ساتھ انٹرایکشن ٹیسٹ کریں
- صارف کے تجربے اور اعتماد کی تعمیر کا جائزہ لیں

### قدم وار ہدایات

1. **بیسک انٹرایکشن فریم ورک** کو سیٹ اپ کریں گفتگو کی شناخت اور ترکیب کے ساتھ
2. **اشارہ کی شناخت** کمپیوٹر وژن کا استعمال کرتے ہوئے نافذ کریں
3. **proxemics مینجمنٹ** مناسب جگہیں بات چیت کے لیے ضم کریں
4. **ثقافتی ایڈاپٹیشن** صارف کی شناخت کے مطابق نافذ کریں
5. **متعدد صارفین** کے ساتھ ٹیسٹ کریں اور جائزہ میٹرکس جمع کریں
6. **نتائج کا تجزیہ** کریں اور انٹرایکشن پیرامیٹرز کو بہتر بنائیں

### متوقع نتائج
- کام کرتا ہوا سوشل انٹرایکشن سسٹم
- HRI میں ثقافتی ایڈاپٹیشن کی سمجھ
- جائزہ میٹرکس کا تجربہ
- بہتر بنائے گئے انٹرایکشن پیرامیٹرز

## علم کی چیک

1. متعدد ماڈلٹی انسان-روبوٹ انٹرایکشن میں کون سی کلیدی ماڈلٹیز ہیں؟
2. سوشل روبوٹکس میں ذہن کی تھیوری کے تصور کی وضاحت کریں۔
3. proxemics انسان-روبوٹ انٹرایکشن کو کیسے متاثر کرتا ہے؟
4. HRI میں اعتماد کی تعمیر میں کون سے عوامل شامل ہیں؟

## خلاصہ

اس باب نے قدرتی انسان-روبوٹ انٹرایکشن کے پیچیدہ میدان کو تلاش کیا، گفتگو اور زبان کی پروسیسنگ، اشارہ کی شناخت، جذباتی مواصلات، سوشل نیویگیشن، اور اعتماد کی تعمیر کو کور کیا۔ مؤثر HRI کو متعدد حسی ماڈلٹیز کے پیچیدہ انضمام، ثقافتی آگاہی، اور انسانی سماجی اشاروں اور امکانات کے جواب میں مطیع رویے کی ضرورت ہوتی ہے۔ جیسے جیسے ہیومنوائڈ روبوٹس انسانی ماحول میں زیادہ عام ہوتے جا رہے ہیں، قدرتی اور مناسب طریقے سے بات چیت کرنے کی صلاحیت کامیابی کے لیے ناگزیر ہے۔

## اگلے اقدامات

باب 17 میں، ہم روبوٹس میں گفتگو کے لیے بڑے زبانی ماڈلز (LLMs) کے انضمام کو تلاش کریں گے، یہ دیکھتے ہوئے کہ اعلی درجے کا AI روبوٹ کی ذہانت اور بات چیت کی صلاحیتوں کو کیسے بہتر بنا سکتا ہے۔