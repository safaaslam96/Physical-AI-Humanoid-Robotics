---
title: "باب 18: ہیومنوائڈ روبوٹس کے لیے اسپیچ ریکوگنیشن اور نیچرل لینگویج انڈر اسٹینڈنگ"
sidebar_label: "باب 18: اسپیچ ریکوگنیشن اور NLU"
---

# باب 18: ہیومنوائڈ روبوٹس کے لیے اسپیچ ریکوگنیشن اور نیچرل لینگویج انڈر اسٹینڈنگ

## سیکھنے کے اہداف
- ہیومنوائڈ روبوٹس کے لیے اسپیچ ریکوگنیشن سسٹم کے بنیادی اصولوں کو سمجھنا
- روبوٹ کمانڈ کی تشریح کے لیے نیچرل لینگویج انڈر اسٹینڈنگ (NLU) نافذ کرنا
- حقیقی دنیا کے ماحول کے لیے مستحکم اسپیچ پروسیسنگ پائپ لائنز ڈیزائن کرنا
- شور کے ماحول میں اسپیچ ریکوگنیشن کارکردگی کا جائزہ لینا اور بہتر بنانا

## تعارف

اسپیچ ریکوگنیشن اور نیچرل لینگویج انڈر اسٹینڈنگ (NLU) قدرتی انسان-روبوٹ انٹرایکشن کی بنیاد ہے۔ ہیومنوائڈ روبوٹس کے لیے مؤثر طور پر انسانوں کے ساتھ بات چیت کرنے کے لیے، انہیں گفتگو کے حکم کو درست طور پر پہچاننا اور سیاق و سباق میں ان کا مطلب سمجھنا چاہیے۔ یہ باب ہیومنوائڈ روبوٹس کے لیے خاص طور پر ڈیزائن کردہ اسپیچ ریکوگنیشن اور NLU سسٹمزم کو تلاش کرتا ہے، حقیقی دنیا کے ماحول، شور، اور ریل ٹائم پروسیسنگ کی چیلنجوں کا مقابلہ کرتے ہوئے۔

## اسپیچ ریکوگنیشن کے بنیادی اصول

### آٹومیٹک اسپیچ ریکوگنیشن (ASR) سسٹم

آٹومیٹک اسپیچ ریکوگنیشن (ASR) سسٹم منہ سے بولی گئی زبان کو ٹیکسٹ میں تبدیل کرتے ہیں۔ ہیومنوائڈ روبوٹس کے لیے، ASR کو ماحولیاتی حالات کے لیے مستحکم ہونا چاہیے اور ریل ٹائم قابل ہونا چاہیے:

```python
# ہیومنوائڈ روبوٹس کے لیے اسپیچ ریکوگنیشن سسٹم
import speech_recognition as sr
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import webrtcvad
import pyaudio
import threading
import queue

class RobotSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.vad = webrtcvad.Vad(2)  # VAD (Voice Activity Detection) level 2
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # امبیئنٹ نوائز کے لیے ایڈجسٹ کریں
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # وائس ایکٹیویشن کے لیے انرگی تھریشولڈ سیٹ کریں
        self.recognizer.energy_threshold = 4000

        # اسپیچ ڈیٹیکشن کے لیے منیم پاز تھریشولڈ سیٹ کریں
        self.recognizer.pause_threshold = 0.8

    def listen_for_speech(self):
        """
        مائیکروفون کے ذریعے اسپیچ کے لیے سنیں اور آڈیو ڈیٹا لوٹائیں
        """
        try:
            with self.microphone as source:
                print("اسپیچ کے لیے سن رہے ہیں...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return audio
        except sr.WaitTimeoutError:
            print("وقت کی حد کے اندر کوئی گفتگو نہیں پکڑی گئی")
            return None
        except sr.UnknownValueError:
            print("گفتگو کو سمجھ نہیں سکے")
            return None

    def recognize_speech_google(self, audio):
        """
        گوگل کی اسپیچ ریکوگنیشن سروس کا استعمال کرتے ہوئے اسپیچ کو پہچانیں
        """
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.RequestError:
            print("API دستیاب نہیں ہے یا غیر جواب دہ")
            return None
        except sr.UnknownValueError:
            print("گفتگو کو سمجھ نہیں سکے")
            return None

    def recognize_speech_wav2vec2(self, audio):
        """
        آف لائن ریکوگنیشن کے لیے Wav2Vec2 ماڈل کا استعمال کریں
        """
        try:
            # آڈیو کو رو ڈیٹا میں تبدیل کریں
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

            # آڈیو کو پروسیس کریں
            inputs = self.processor(raw_data, sampling_rate=16000, return_tensors="pt", padding=True)

            # انفریسنگ کریں
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits

            # پریڈکشنز کو ڈیکوڈ کریں
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription
        except Exception as e:
            print(f"Wav2Vec2 ریکوگنیشن میں خامی: {e}")
            return None

    def noise_robust_recognition(self, audio):
        """
        ریکوگنیشن سے پہلے نوائز ریڈکشن تکنیکس لاگو کریں
        """
        # نوائز ریڈکشن لاگو کریں
        # یہ ایک سادہ ا approach ہے - عمل میں، اعلی درجے کے نوائز ریڈکشن الگورتھم استعمال کریں
        try:
            # رو ڈیٹا میں تبدیل کریں
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

            # بنیادی نوائز ریڈکشن (اسپیکٹرل سب ٹریکشن ایپروچ) لاگو کریں
            # آڈیو کے شروع میں نوائز پروفائل کا حساب لگائیں
            noise_profile = np.mean(np.abs(raw_data[:1000]))

            # نوائز ریڈکشن لاگو کریں
            cleaned_data = raw_data - noise_profile
            cleaned_data = np.clip(cleaned_data, -32768, 32767)

            # نیا آڈیو ڈیٹا بنائیں
            cleaned_audio = sr.AudioData(
                cleaned_data.tobytes(),
                audio.sample_rate,
                audio.sample_width
            )

            # صاف کردہ آڈیو سے گفتگو کو پہچانیں
            text = self.recognizer.recognize_google(cleaned_audio)
            return text
        except Exception as e:
            print(f"نوائز روبسٹ ریکوگنیشن میں خامی: {e}")
            return None

    def continuous_listening(self, callback_func):
        """
        اسپیچ کے لیے جاری طور پر سنیں اور پہچانے گئے ٹیکسٹ کے ساتھ کال بیک فنکشن کال کریں
        """
        self.is_listening = True

        def listen_thread():
            while self.is_listening:
                try:
                    with self.microphone as source:
                        print("روبوٹ سن رہا ہے...")
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                        # گفتگو کو پہچانیں
                        text = self.recognize_speech_google(audio)

                        if text:
                            print(f"پہچانا گیا: {text}")
                            callback_func(text)
                except sr.WaitTimeoutError:
                    continue  # سننا جاری رکھیں
                except sr.UnknownValueError:
                    print("گفتگو کو سمجھ نہیں سکے")
                    continue
                except sr.RequestError as e:
                    print(f"اسپیچ ریکوگنیشن سروس کے ساتھ خامی: {e}")
                    continue

        # سننے والے تھریڈ کو شروع کریں
        listener_thread = threading.Thread(target=listen_thread)
        listener_thread.daemon = True
        listener_thread.start()

        return listener_thread

    def stop_listening(self):
        """
        جاری سننے کا عمل بند کریں
        """
        self.is_listening = False
```

### ریل ٹائم اسپیچ پروسیسنگ پائپ لائن

ہیومنوائڈ روبوٹس کے لیے ریل ٹائم اسپیچ پروسیسنگ پائپ لائن کا نفاذ:

```python
# ریل ٹائم اسپیچ پروسیسنگ پائپ لائن
import threading
import time
import collections
import numpy as np

class RealTimeSpeechPipeline:
    def __init__(self, robot_speech_recognizer):
        self.recognizer = robot_speech_recognizer
        self.is_running = False
        self.pipeline_thread = None
        self.speech_buffer = collections.deque(maxlen=10)  # آخری 10 اسپیچ سیگمنٹس اسٹور کریں
        self.active_listening = False
        self.listening_callback = None

    def start_pipeline(self):
        """
        ریل ٹائم اسپیچ پروسیسنگ پائپ لائن شروع کریں
        """
        self.is_running = True
        self.pipeline_thread = threading.Thread(target=self._pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()

    def stop_pipeline(self):
        """
        ریل ٹائم اسپیچ پروسیسنگ پائپ لائن بند کریں
        """
        self.is_running = False
        if self.pipeline_thread:
            self.pipeline_thread.join()

    def set_callback(self, callback):
        """
        پہچانی گئی گفتگو کے لیے کال بیک فنکشن سیٹ کریں
        """
        self.listening_callback = callback

    def _pipeline_loop(self):
        """
        ریل ٹائم اسپیچ کے لیے مرکزی پروسیسنگ لوپ
        """
        while self.is_running:
            if self.active_listening:
                try:
                    audio = self.recognizer.listen_for_speech()
                    if audio:
                        # اسپیچ ریکوگنیشن پروسیس کریں
                        recognized_text = self.recognizer.recognize_speech_google(audio)

                        if recognized_text:
                            # بفر میں شامل کریں
                            self.speech_buffer.append(recognized_text)

                            # اگر سیٹ ہو تو کال بیک کال کریں
                            if self.listening_callback:
                                self.listening_callback(recognized_text)

                            # NLU کے لیے پروسیس کریں
                            intent = self._process_nlu(recognized_text)

                            # بفر میں اسٹور کریں
                            self.speech_buffer.append({
                                'text': recognized_text,
                                'intent': intent,
                                'timestamp': time.time()
                            })
                except Exception as e:
                    print(f"اسپیچ پائپ لائن میں خامی: {e}")

            time.sleep(0.1)  # زیادہ CPU استعمال کو روکنے کے لیے چھوٹا وقفہ

    def activate_listening(self):
        """
        سننے کا موڈ چالو کریں
        """
        self.active_listening = True

    def deactivate_listening(self):
        """
        سننے کا موڈ غیر فعال کریں
        """
        self.active_listening = False

    def _process_nlu(self, text):
        """
        پہچانے گئے ٹیکسٹ پر نیچرل لینگویج انڈر اسٹینڈنگ کو پروسیس کریں
        """
        # سادہ ارادہ کیسیفکیشن
        text_lower = text.lower()

        # ارادہ پیٹرنز کی وضاحت کریں
        intent_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'navigation': ['go to', 'move to', 'navigate', 'walk to', 'go', 'move', 'walk'],
            'manipulation': ['pick up', 'get', 'bring', 'take', 'grasp', 'grab', 'lift'],
            'information_request': ['what', 'where', 'when', 'how', 'who', 'why', 'tell me'],
            'stop': ['stop', 'halt', 'pause', 'wait'],
            'follow': ['follow', 'come with', 'accompany', 'follow me'],
            'action': ['help', 'assist', 'do', 'perform', 'execute']
        }

        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent

        return 'unknown'

    def get_recent_speech(self, count=5):
        """
        سب سے حالیہ پہچانی گئی اسپیچ سیگمنٹس حاصل کریں
        """
        recent = list(self.speech_buffer)[-count:]
        return recent
```

## نیچرل لینگویج انڈر اسٹینڈنگ (NLU)

### ارادہ کی شناخت اور کیسیفکیشن

نیچرل لینگویج انڈر اسٹینڈنگ سسٹمزم کو صارف کے ارادے کو درست طور پر کیسیفائز کرنا اور متعلقہ معلومات نکالنا چاہیے:

```python
# نیچرل لینگویج انڈر اسٹینڈنگ سسٹم
import re
from typing import Dict, List, Tuple
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

class RobotNLU:
    def __init__(self):
        # NLP پروسیسنگ کے لیے spaCy ماڈل لوڈ کریں
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy انگریزی ماڈل نہیں ملا. مندرجہ ذیل کمانڈ کے ساتھ انسٹال کریں: python -m spacy download en_core_web_sm")
            self.nlp = None

        # ارادہ کلاسیفائر کو شروع کریں
        self.intent_classifier = MultinomialNB()
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.is_trained = False

        # اینٹیٹی پیٹرنز کی وضاحت کریں
        self.entity_patterns = {
            'LOCATION': [
                r'\b(?:to|at|in|near|by|on)\s+([A-Za-z\s]+?)(?:\.|,|$)',
                r'\b(?:the\s+|a\s+|an\s+)(kitchen|bedroom|living room|office|bathroom|garden|door|window)\b'
            ],
            'OBJECT': [
                r'\b(?:the\s+|a\s+|an\s+|some\s+)([A-Za-z\s]+?)\b',
                r'\b(?:pick up|get|take|bring|grab|lift|move)\s+(?:the\s+|a\s+|an\s+)?([A-Za-z\s]+?)\b'
            ],
            'PERSON': [
                r'\b(?:to|with|see|find)\s+([A-Za-z\s]+?)\b',
                r'\b(?:call|tell|ask)\s+([A-Za-z\s]+?)\b'
            ],
            'NUMBER': [r'\b(\d+)\b']
        }

        # تربیت کے لیے ارادہ ٹیمپلیٹس کی وضاحت کریں
        self.intent_training_data = {
            'greeting': [
                'hello robot', 'hi there', 'good morning', 'hey robot', 'hello', 'hi', 'good evening'
            ],
            'navigation': [
                'go to the kitchen', 'move to the bedroom', 'walk to the office',
                'navigate to the living room', 'go there', 'move forward', 'turn left'
            ],
            'manipulation': [
                'pick up the cup', 'get the book', 'bring me the water',
                'take the pen', 'grasp the object', 'lift the box'
            ],
            'information_request': [
                'what time is it', 'where are you', 'how are you',
                'what can you do', 'tell me about yourself', 'what is your name'
            ],
            'stop': [
                'stop', 'halt', 'pause', 'wait', 'stop moving', 'freeze'
            ],
            'follow': [
                'follow me', 'come with me', 'accompany me', 'follow', 'come along'
            ],
            'action': [
                'help me', 'assist me', 'do something', 'perform a task',
                'execute action', 'help', 'assist'
            ]
        }

        # کلاسیفائر کو تربیت دیں
        self._train_classifier()

    def _train_classifier(self):
        """
        ازلا_defined ڈیٹا کے ساتھ ارادہ کلاسیفائر کو تربیت دیں
        """
        texts = []
        labels = []

        for intent, examples in self.intent_training_data.items():
            for example in examples:
                texts.append(example)
                labels.append(intent)

        # ٹیکسٹس کو ویکٹرائز کریں
        X = self.tfidf_vectorizer.fit_transform(texts)

        # کلاسیفائر کو تربیت دیں
        self.intent_classifier.fit(X, labels)
        self.is_trained = True

    def classify_intent(self, text: str) -> str:
        """
        دی گئی ٹیکسٹ کے ارادہ کی کلاسیفکیشن کریں
        """
        if not self.is_trained:
            return 'unknown'

        # ان پٹ ٹیکسٹ کو ویکٹرائز کریں
        X = self.tfidf_vectorizer.transform([text])

        # ارادہ کی پریڈکٹ کریں
        predicted_intent = self.intent_classifier.predict(X)[0]

        # پریڈکشن کی اعتماد کا حساب لگائیں
        prediction_probs = self.intent_classifier.predict_proba(X)[0]
        max_prob = max(prediction_probs)

        # صرف اس وقت پریڈکشن لوٹائیں جب اعتماد کافی ہو
        if max_prob > 0.3:  # اعتماد کی حد
            return predicted_intent
        else:
            return 'unknown'

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        ٹیکسٹ سے نامزد اجزاء نکالیں
        """
        entities = {}

        # ایکسٹریکٹ اینٹیٹیز کے لیے regex پیٹرنز کا استعمال کریں
        for entity_type, patterns in self.entity_patterns.items():
            entity_matches = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entity_matches.extend(matches)

            # میچز کو صاف کریں
            cleaned_matches = [match.strip() for match in entity_matches if match.strip()]
            entities[entity_type] = list(set(cleaned_matches))  # ڈوپلیکیٹس ہٹائیں

        # اگر spaCy دستیاب ہے تو، زیادہ ترقی یافتہ NER کے لیے اس کا استعمال کریں
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'TIME', 'DATE']:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    entities[ent.label_].append(ent.text)

        return entities

    def parse_command(self, text: str) -> Dict:
        """
        ارادہ اور اجزاء کے ساتھ کمانڈ کو سٹرکچرڈ فارمیٹ میں پارس کریں
        """
        intent = self.classify_intent(text)
        entities = self.extract_entities(text)

        return {
            'text': text,
            'intent': intent,
            'entities': entities,
            'confidence': self._get_intent_confidence(text)
        }

    def _get_intent_confidence(self, text: str) -> float:
        """
        ارادہ کیسیفکیشن کے لیے اعتماد اسکور حاصل کریں
        """
        if not self.is_trained:
            return 0.0

        X = self.tfidf_vectorizer.transform([text])
        prediction_probs = self.intent_classifier.predict_proba(X)[0]
        max_prob = max(prediction_probs)

        return float(max_prob)

    def process_contextual_command(self, text: str, context: Dict) -> Dict:
        """
        اضافی سیاق و سباق کی معلومات کے ساتھ کمانڈ کو پروسیس کریں
        """
        parsed = self.parse_command(text)

        # سیاق و سباق کے ساتھ بڑھائیں
        parsed['context'] = context

        # سیاق و سباق کی بنیاد پر ضمیر اور حوالہ جات کو حل کریں
        if 'it' in text.lower() and context.get('last_object'):
            # 'it' کو آخری ذکر کردہ چیز کے ساتھ بدلیں
            parsed['resolved_entities'] = {
                'object': [context['last_object']]
            }

        if 'there' in text.lower() and context.get('last_location'):
            # 'there' کو آخری ذکر کردہ مقام کے ساتھ بدلیں
            parsed['resolved_entities'] = parsed.get('resolved_entities', {})
            parsed['resolved_entities']['location'] = [context['last_location']]

        return parsed
```

### سیاق و سباق کے خیال رکھنے والا لینگویج انڈر اسٹینڈنگ

زیادہ قدرتی انٹرایکشنز کے لیے سیاق و سباق کے خیال رکھنے والا تصور:

```python
# سیاق و سباق کے خیال رکھنے والا لینگویج انڈر اسٹینڈنگ
class ContextualNLU:
    def __init__(self):
        self.nlu = RobotNLU()
        self.context_memory = {}
        self.conversation_history = []
        self.max_history = 10  # آخری 10 انٹرایکشنز رکھیں

    def process_with_context(self, text: str, robot_state: Dict, environment: Dict) -> Dict:
        """
        سیاق و سباق کے آگاہی کے ساتھ ٹیکسٹ کو پروسیس کریں
        """
        # سیاق و سباق بنائیں
        context = {
            'robot_state': robot_state,
            'environment': environment,
            'conversation_history': self.conversation_history[-3:],  # آخری 3 انٹرایکشنز
            'current_time': time.time(),
            'last_entities': self.context_memory.get('last_entities', {})
        }

        # کمانڈ کو سیاق و سباق کے ساتھ پارس کریں
        parsed = self.nlu.process_contextual_command(text, context)

        # گفتگو کی تاریخ میں اسٹور کریں
        self.conversation_history.append({
            'text': text,
            'parsed': parsed,
            'timestamp': time.time()
        })

        # ہسٹری کا سائز محدود کریں
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # سیاق و سباق میموری کو اپ ڈیٹ کریں
        self._update_context_memory(parsed)

        return parsed

    def _update_context_memory(self, parsed: Dict):
        """
        پارسڈ کمانڈ سے متعلقہ معلومات کے ساتھ سیاق و سباق میموری کو اپ ڈیٹ کریں
        """
        # آخری اجزاء کو اسٹور کریں
        entities = parsed.get('entities', {})
        if entities:
            self.context_memory['last_entities'] = entities

            # مخصوص اجزاء کو اسٹور کریں
            if entities.get('OBJECT'):
                self.context_memory['last_object'] = entities['OBJECT'][0]
            if entities.get('LOCATION'):
                self.context_memory['last_location'] = entities['LOCATION'][0]
            if entities.get('PERSON'):
                self.context_memory['last_person'] = entities['PERSON'][0]

    def resolve_references(self, text: str) -> str:
        """
        سیاق و سباق کی بنیاد پر ٹیکسٹ میں ضمیروں اور حوالہ جات کو حل کریں
        """
        resolved_text = text

        # 'it' کو آخری چیز کے مطابق حل کریں
        if 'it' in text.lower() and self.context_memory.get('last_object'):
            resolved_text = resolved_text.replace('it', self.context_memory['last_object'])

        # 'there' کو آخری مقام کے مطابق حل کریں
        if 'there' in text.lower() and self.context_memory.get('last_location'):
            resolved_text = resolved_text.replace('there', self.context_memory['last_location'])

        return resolved_text

    def get_context_summary(self) -> Dict:
        """
        موجودہ سیاق و سباق کا خلاصہ حاصل کریں
        """
        return {
            'last_entities': self.context_memory.get('last_entities', {}),
            'conversation_history_count': len(self.conversation_history),
            'context_memory_keys': list(self.context_memory.keys())
        }
```

## مضبوط اسپیچ پروسیسنگ

### نوائز ریڈکشن اور فلٹرنگ

حقیقی دنیا کے ماحول کے لیے نوائز ریڈکشن تکنیکس کا نفاذ:

```python
# اسپیچ ریکوگنیشن کے لیے نوائز ریڈکشن اور فلٹرنگ
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, wiener
import librosa

class NoiseReductionSystem:
    def __init__(self):
        # نوائز ریڈکشن کے لیے پیرامیٹر
        self.sample_rate = 16000
        self.frame_length = 2048
        self.hop_length = 512
        self.noise_threshold = 0.01

    def spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        نوائز ریڈکشن کے لیے اسپیکٹرل سب ٹریکشن لاگو کریں
        """
        # STFT کا حساب لگائیں
        stft = librosa.stft(audio_data, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # نوائز اسپیکٹرم کا حساب لگائیں (پہلے 0.5 سیکنڈ کو نوائز ریفرنس کے طور پر)
        noise_frames = int(0.5 * self.sample_rate / self.hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # اسپیکٹرل سب ٹریکشن لاگو کریں
        enhanced_magnitude = magnitude - self.noise_threshold * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0)  # غیر منفی یقینی بنائیں

        # سگنل کو دوبارہ تعمیر کریں
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)

        return enhanced_audio

    def wiener_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        نوائز کم کرنے کے لیے وینر فلٹر لاگو کریں
        """
        # وینر فلٹر لاگو کریں
        filtered_audio = wiener(audio_data)
        return filtered_audio

    def bandpass_filter(self, audio_data: np.ndarray, low_freq: float = 300, high_freq: float = 3400) -> np.ndarray:
        """
        انسانی گفتگو کی فریکوینسیز پر توجہ مرکوز کرنے کے لیے بینڈ پاس فلٹر لاگو کریں
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # بٹر ورتھ بینڈ پاس فلٹر ڈیزائن کریں
        b, a = butter(4, [low, high], btype='band', analog=False)

        # فلٹر لاگو کریں
        filtered_audio = filtfilt(b, a, audio_data)

        return filtered_audio

    def voice_activity_detection(self, audio_data: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """
        آڈیو سگنل میں وائس ایکٹیویٹی کا پتہ لگائیں
        """
        # سگنل کی انرگی کا حساب لگائیں
        energy = np.array([
            np.sum(np.abs(audio_data[i:i+self.hop_length]**2))
            for i in range(0, len(audio_data), self.hop_length)
        ])

        # انرگی کو نارملائز کریں
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy

        # وائس ایکٹیویٹی ماسک بنائیں
        voice_mask = energy > threshold

        # صرف وائس ایکٹو سیگمنٹس کے ساتھ آڈیو کو دوبارہ تعمیر کریں
        result = np.zeros_like(audio_data)
        for i, is_voice in enumerate(voice_mask):
            start_idx = i * self.hop_length
            end_idx = min(start_idx + self.hop_length, len(audio_data))
            if is_voice:
                result[start_idx:end_idx] = audio_data[start_idx:end_idx]

        return result

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        آڈیو کے لیے مکمل پری پروسیسنگ پائپ لائن لاگو کریں
        """
        # بینڈ پاس فلٹر لاگو کریں
        filtered_audio = self.bandpass_filter(audio_data)

        # نوائز ریڈکشن لاگو کریں
        reduced_audio = self.spectral_subtraction(filtered_audio)

        # ویئر فلٹر لاگو کریں
        final_audio = self.wiener_filter(reduced_audio)

        return final_audio
```

### ملٹی-مائیکروفون پروسیسنگ

بہتر اسپیچ ریکوگنیشن کے لیے ملٹی-مائیکروفون پروسیسنگ کا نفاذ:

```python
# ہیومنوائڈ روبوٹس کے لیے ملٹی-مائیکروفون پروسیسنگ
import numpy as np
from scipy import signal
import pyaudio

class MultiMicrophoneProcessor:
    def __init__(self, num_mics=4, sample_rate=16000):
        self.num_mics = num_mics
        self.sample_rate = sample_rate
        self.audio = pyaudio.PyAudio()

        # مائیکروفون کی پوزیشنز (سادہ - عمل میں، ان کی کیلیبریشن ہوگی)
        self.mic_positions = np.array([
            [-0.1, 0.0, 0.0],   # بائیں
            [0.1, 0.0, 0.0],    # دائیں
            [0.0, -0.1, 0.0],   # سامنے
            [0.0, 0.1, 0.0]     # پیچھے
        ])

    def beamforming(self, multi_channel_audio: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        مخصوص سمت پر توجہ مرکوز کرنے کے لیے بیم فارمنگ لاگو کریں
        """
        if direction is None:
            # ڈیفالٹ طور پر سامنے کی سمت کو
            direction = np.array([0, -1, 0])  # سامنے کی سمت

        # سمت کو نارملائز کریں
        direction = direction / np.linalg.norm(direction)

        # ہر مائیکروفون کے لیے ٹائم ڈیلیز کا حساب لگائیں
        delays = []
        speed_of_sound = 343.0  # m/s
        mic_distance = 0.2  # m (مائیکروفونز کے درمیان تقریبی فاصلہ)

        for pos in self.mic_positions:
            delay = np.dot(pos, direction) / speed_of_sound
            delays.append(delay)

        # اشاروں کو ہموار کرنے کے لیے ڈیلیز لاگو کریں
        aligned_signals = []
        for i, audio_channel in enumerate(multi_channel_audio):
            delay_samples = int(delays[i] * self.sample_rate)
            if delay_samples > 0:
                # ڈیلے کو شفٹ کر کے لاگو کریں
                delayed_signal = np.concatenate([np.zeros(delay_samples), audio_channel[:-delay_samples]])
            else:
                delayed_signal = audio_channel
            aligned_signals.append(delayed_signal)

        # بیم فارمنگ کے لیے اشاروں کا مجموعہ
        beamformed_signal = np.sum(aligned_signals, axis=0)

        return beamformed_signal

    def noise_suppression(self, multi_channel_audio: np.ndarray) -> np.ndarray:
        """
        متعدد مائیکروفون ان پٹس کا استعمال کرتے ہوئے نوائز سپریشن لاگو کریں
        """
        # مائیکروفونز کے درمیان سپیشل کاریلیشن کا حساب لگائیں
        correlations = []
        for i in range(self.num_mics):
            for j in range(i+1, self.num_mics):
                correlation = np.corrcoef(multi_channel_audio[i], multi_channel_audio[j])[0, 1]
                correlations.append(correlation)

        # اوسط کاریلیشن
        avg_correlation = np.mean(correlations)

        # کاریلیشن کی بنیاد پر سپیشل فلٹرنگ لاگو کریں
        if avg_correlation > 0.3:  # زیادہ کاریلیشن کوہیرنٹ سگنل کی علامت ہے
            # کوہیرنٹ سگنل کو بڑھانے کے لیے بیم فارمنگ استعمال کریں
            enhanced_signal = self.beamforming(multi_channel_audio)
        else:
            # سگنل کو بڑھانے کے لیے سگنل کی بہترین مائیکروفون استعمال کریں
            snrs = [self._calculate_snr(channel) for channel in multi_channel_audio]
            best_mic_idx = np.argmax(snrs)
            enhanced_signal = multi_channel_audio[best_mic_idx]

        return enhanced_signal

    def _calculate_snr(self, audio_signal: np.ndarray) -> float:
        """
        سگنل کے لیے سگنل-ٹو-نوائز ریشیو کا حساب لگائیں
        """
        signal_power = np.mean(audio_signal ** 2)
        noise_power = np.var(audio_signal)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def adaptive_filtering(self, multi_channel_audio: np.ndarray) -> np.ndarray:
        """
        نوائز کم کرنے کے لیے اڈاپٹیو فلٹرنگ لاگو کریں
        """
        # آخری مائیکروفون کو نوائز کے حوالے کے طور پر استعمال کریں
        reference = multi_channel_audio[-1]

        # دیگر مائیکروفونز پر اڈاپٹیو فلٹرنگ لاگو کریں
        filtered_signals = []
        for i in range(self.num_mics - 1):
            # اڈاپٹیو فلٹر بنائیں
            filtered = self._adaptive_filter(multi_channel_audio[i], reference)
            filtered_signals.append(filtered)

        # فلٹرڈ سگنلز کو ملانا
        combined = np.mean(filtered_signals, axis=0)
        return combined

    def _adaptive_filter(self, primary_signal: np.ndarray, reference_signal: np.ndarray,
                         filter_length: int = 64) -> np.ndarray:
        """
        LMS الگورتھم کا استعمال کرتے ہوئے اڈاپٹیو فلٹرنگ لاگو کریں
        """
        # فلٹر کوائف کو شروع کریں
        w = np.zeros(filter_length)
        mu = 0.01  # لرننگ ریٹ

        output = np.zeros(len(primary_signal))
        error = np.zeros(len(primary_signal))

        for n in range(filter_length, len(primary_signal)):
            # حوالہ ان پٹ حاصل کریں
            ref_input = reference_signal[n-filter_length:n][::-1]

            # فلٹر آؤٹ پٹ
            y = np.dot(w, ref_input)

            # خامی
            error[n] = primary_signal[n] - y

            # فلٹر کوائف کو اپ ڈیٹ کریں
            w = w + mu * error[n] * ref_input

            # آؤٹ پٹ
            output[n] = error[n]

        return output
```

## روبوٹ سسٹمزم کے ساتھ انضمام

### اسپیچ-سے-ایکشن پائپ لائن

اسپیچ ریکوگنیشن سے روبوٹ ایکشن تک مکمل پائپ لائن کا تخلیق:

```python
# مکمل اسپیچ-سے-ایکشن پائپ لائن
class SpeechToActionPipeline:
    def __init__(self, robot_interface):
        self.speech_recognizer = RobotSpeechRecognizer()
        self.nlu = RobotNLU()
        self.contextual_nlu = ContextualNLU()
        self.robot_interface = robot_interface
        self.pipeline = RealTimeSpeechPipeline(self.speech_recognizer)

        # روبوٹ اسٹیٹ اور ماحول
        self.robot_state = {}
        self.environment = {}

    def start_listening(self):
        """
        اسپیچ-سے-ایکشن پائپ لائن شروع کریں
        """
        # پہچانی گئی گفتگو کے لیے کال بیک سیٹ کریں
        def speech_callback(text):
            self.process_speech_command(text)

        self.pipeline.set_callback(speech_callback)
        self.pipeline.start_pipeline()
        self.pipeline.activate_listening()

        print("اسپیچ-سے-ایکشن پائپ لائن شروع ہو گئی. روبوٹ سن رہا ہے...")

    def stop_listening(self):
        """
        اسپیچ-سے-ایکشن پائپ لائن بند کریں
        """
        self.pipeline.deactivate_listening()
        self.pipeline.stop_pipeline()
        print("اسپیچ-سے-ایکشن پائپ لائن بند کر دی گئی.")

    def process_speech_command(self, text: str):
        """
        مکمل پائپ لائن کے ذریعے اسپیچ کمانڈ کو پروسیس کریں
        """
        print(f"اسپیچ کمانڈ کو پروسیس کر رہا ہے: {text}")

        # سیاق و سباق کے ساتھ NLU کا استعمال کریں
        parsed_command = self.contextual_nlu.process_with_context(
            text,
            self.robot_state,
            self.environment
        )

        # ارادہ کے مطابق ایکسیکیوٹ کریں
        intent = parsed_command['intent']
        entities = parsed_command['entities']

        if intent == 'navigation':
            self._execute_navigation(entities)
        elif intent == 'manipulation':
            self._execute_manipulation(entities)
        elif intent == 'greeting':
            self._execute_greeting()
        elif intent == 'information_request':
            self._execute_information_request(text, entities)
        elif intent == 'stop':
            self._execute_stop()
        elif intent == 'follow':
            self._execute_follow(entities)
        elif intent == 'action':
            self._execute_generic_action(entities)
        else:
            self._execute_unknown_command()

    def _execute_navigation(self, entities: Dict):
        """
        نیویگیشن کمانڈ ایکسیکیوٹ کریں
        """
        target_locations = entities.get('LOCATION', [])

        if target_locations:
            target = target_locations[0]
            print(f"روبوٹ یہاں جا رہا ہے: {target}")

            # چیک کریں کہ مقام ماحول میں موجود ہے
            if target in self.environment.get('locations', []):
                success = self.robot_interface.navigate_to_location(target)
                if success:
                    print(f"کامیابی کے ساتھ {target} پر پہنچا")
                else:
                    print(f"{target} پر جانے میں ناکامی")
            else:
                print(f"نامعلوم مقام: {target}. دستیاب مقامات: {self.environment.get('locations', [])}")
        else:
            print("نیویگیشن کمانڈ میں کوئی ہدف مقام متعین نہیں کیا گیا")

    def _execute_manipulation(self, entities: Dict):
        """
        مینیپولیشن کمانڈ ایکسیکیوٹ کریں
        """
        target_objects = entities.get('OBJECT', [])

        if target_objects:
            target = target_objects[0]
            print(f"روبوٹ چیز کو مینیپولیٹ کرنے کی کوشش کر رہا ہے: {target}")

            # چیک کریں کہ چیز ماحول میں موجود ہے
            if target in self.environment.get('objects', []):
                success = self.robot_interface.manipulate_object(target)
                if success:
                    print(f"کامیابی کے ساتھ {target} کو مینیپولیٹ کیا")
                else:
                    print(f"{target} کو مینیپولیٹ کرنے میں ناکامی")
            else:
                print(f"نامعلوم چیز: {target}. دستیاب چیزیں: {self.environment.get('objects', [])}")
        else:
            print("مینیپولیشن کمانڈ میں کوئی ہدف چیز متعین نہیں کی گئی")

    def _execute_greeting(self):
        """
        گریٹنگ ایکشن ایکسیکیوٹ کریں
        """
        print("روبوٹ گریٹنگ انجام دے رہا ہے")
        self.robot_interface.perform_greeting()

    def _execute_information_request(self, text: str, entities: Dict):
        """
        معلومات کی درخواست ایکسیکیوٹ کریں
        """
        print(f"روبوٹ معلومات کی درخواست کو پروسیس کر رہا ہے: {text}")

        # مخصوص معلومات کی درخواستوں کے لیے چیک کریں
        text_lower = text.lower()

        if 'time' in text_lower:
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M")
            response = f"موجودہ وقت {current_time} ہے"
        elif 'name' in text_lower or 'you' in text_lower:
            response = "میں آپ کا ہیومنوائڈ روبوٹ اسسٹنٹ ہوں. مجھے اسسٹنٹ کہہ سکتے ہیں."
        elif 'location' in text_lower or 'where' in text_lower and 'you' in text_lower:
            location = self.robot_state.get('location', 'نامعلوم مقام')
            response = f"میں فی الحال {location} پر ہوں"
        elif 'capabilities' in text_lower or 'can you' in text_lower:
            capabilities = self.robot_state.get('capabilities', [])
            capability_list = ', '.join(capabilities) if capabilities else 'میں مختلف کاموں میں مدد کر سکتا ہوں'
            response = f"میں مندرجہ ذیل کام کر سکتا ہوں: {capability_list}"
        else:
            response = "میں وقت، نام، مقام، اور صلاحیات کے بارے میں معلومات فراہم کر سکتا ہوں. میں اور کیسے مدد کر سکتا ہوں؟"

        print(f"روبوٹ ریسپانس: {response}")
        self.robot_interface.speak_text(response)

    def _execute_stop(self):
        """
        اسٹاپ کمانڈ ایکسیکیوٹ کریں
        """
        print("روبوٹ اسٹاپ کمانڈ انجام دے رہا ہے")
        self.robot_interface.stop_current_action()

    def _execute_follow(self, entities: Dict):
        """
        فالو کمانڈ ایکسیکیوٹ کریں
        """
        target_persons = entities.get('PERSON', [])

        if target_persons:
            target = target_persons[0]
            print(f"روبوٹ {target} کو فالو کرنے کی کوشش کر رہا ہے")
            success = self.robot_interface.follow_person(target)
            if success:
                print(f"{target} کو فالو کرنا شروع کر دیا")
            else:
                print(f"{target} کو فالو کرنے میں ناکامی")
        else:
            print("روبوٹ ڈیفالٹ رویہ (اسپیکر کو فالو کرنا) کر رہا ہے")
            success = self.robot_interface.follow_person("speaker")
            if success:
                print("اسپیکر کو فالو کرنا شروع کر دیا")

    def _execute_generic_action(self, entities: Dict):
        """
        جنرک ایکشن کو اینٹیٹیز کے مطابق ایکسیکیوٹ کریں
        """
        print(f"روبوٹ جنرک ایکشن کر رہا ہے اینٹیٹیز کے ساتھ: {entities}")
        # یہ سیاق و سباق کے مطابق مختلف ایکشنز کو فعال کر سکتا ہے
        self.robot_interface.perform_generic_action(entities)

    def _execute_unknown_command(self):
        """
        نامعلوم کمانڈ کو ہینڈل کریں
        """
        response = "معاف کیجیے، مجھے وہ کمانڈ سمجھ میں نہیں آئی. کیا آپ براہ کرم دوبارہ بیان کر سکتے ہیں؟"
        print(f"روبوٹ ریسپانس: {response}")
        self.robot_interface.speak_text(response)

    def update_robot_state(self, new_state: Dict):
        """
        روبوٹ اسٹیٹ کی معلومات کو اپ ڈیٹ کریں
        """
        self.robot_state.update(new_state)

    def update_environment(self, new_environment: Dict):
        """
        ماحول کی معلومات کو اپ ڈیٹ کریں
        """
        self.environment.update(new_environment)
```

## کارکردگی کی بہتری

### ریل ٹائم پروسیسنگ کے لیے کارکردگی کے اعتبارات

ہیومنوائڈ روبوٹس کے لیے ریل ٹائم اسپیچ ریکوگنیشن کے لیے کارکردگی کو بہتر بنانا:

```python
# اسپیچ پروسیسنگ کے لیے کارکردگی کی بہتری
import time
import threading
from queue import Queue, Empty
import psutil

class OptimizedSpeechProcessor:
    def __init__(self):
        self.input_queue = Queue(maxsize=10)  # میموری مسائل کو روکنے کے لیے کیو کا سائز محدود کریں
        self.output_queue = Queue(maxsize=10)
        self.is_running = False
        self.processing_thread = None
        self.cpu_threshold = 80  # کارکردگی کے لیے CPU استعمال کی حد
        self.processing_delay = 0.0  # ریٹ لیمنٹنگ کے لیے پروسیسنگ تاخیر

        # کمپوننٹس کو شروع کریں
        self.speech_recognizer = RobotSpeechRecognizer()
        self.nlu = RobotNLU()

    def start_processing(self):
        """
        بہتر کردہ پروسیسنگ پائپ لائن شروع کریں
        """
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """
        بہتر کردہ پروسیسنگ پائپ لائن بند کریں
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()

    def add_audio_input(self, audio_data):
        """
        پروسیسنگ کیو میں آڈیو ان پٹ شامل کریں
        """
        try:
            self.input_queue.put_nowait(audio_data)
            return True
        except:
            # کیو بھر گئی ہے، ان پٹ چھوڑ دیں
            return False

    def get_processed_output(self):
        """
        کیو سے پروسیسڈ آؤٹ پٹ حاصل کریں
        """
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None

    def _processing_loop(self):
        """
        کارکردگی کی بہتری کے ساتھ مرکزی پروسیسنگ لوپ
        """
        while self.is_running:
            try:
                # سسٹم وسائل چیک کریں
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # CPU استعمال کے مطابق پروسیسنگ ایڈجسٹ کریں
                if cpu_percent > self.cpu_threshold:
                    # CPU زیادہ ہونے پر پروسیسنگ کی شرح کم کریں
                    time.sleep(0.1)
                    continue

                # آڈیو ان پٹ حاصل کریں
                try:
                    audio_data = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue

                # آڈیو کو پروسیس کریں
                start_time = time.time()

                # گفتگو کو پہچانیں
                recognized_text = self.speech_recognizer.recognize_speech_google(audio_data)

                if recognized_text:
                    # NLU کے ساتھ پارس کریں
                    parsed_result = self.nlu.parse_command(recognized_text)

                    # پروسیسنگ ٹائم کی معلومات شامل کریں
                    processing_time = time.time() - start_time
                    parsed_result['processing_time'] = processing_time

                    # آؤٹ پٹ کیو میں نتیجہ ڈالیں
                    try:
                        self.output_queue.put_nowait(parsed_result)
                    except:
                        # آؤٹ پٹ کیو بھر گئی ہے، نتیجہ چھوڑ دیں
                        pass

                # زیادہ CPU استعمال کو روکنے کے لیے ریٹ لیمنٹنگ
                processing_time = time.time() - start_time
                if processing_time < 0.01:  # منیم پروسیسنگ ٹائم
                    time.sleep(0.01 - processing_time)

            except Exception as e:
                print(f"پروسیسنگ لوپ میں خامی: {e}")
                time.sleep(0.1)  # جاری رکھنے سے پہلے مختصر وقفہ

    def get_performance_metrics(self):
        """
        پروسیسر کے لیے کارکردگی کے میٹرکس حاصل کریں
        """
        return {
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'is_running': self.is_running
        }
```

## عملی مشق: اسپیچ ریکوگنیشن سسٹم کا نفاذ

### مشق کے اہداف
- ہیومنوائڈ روبوٹ کے لیے اسپیچ ریکوگنیشن پائپ لائن سیٹ اپ کریں
- کمانڈ کی تشریح کے لیے نیچرل لینگویج انڈر اسٹینڈنگ نافذ کریں
- مختلف گفتگو ان پٹس کے ساتھ سسٹم کی جانچ کریں
- پہچان کی درستگی اور ریسپانس ٹائم کا جائزہ لیں

### قدم وار ہدایات

1. **مائیکروفون سیٹ اپ** کے ساتھ اسپیچ ریکوگنیشن سسٹم کو شروع کریں
2. **NLU کمپوننٹ** کو ارادہ کیسیفکیشن اور اینٹیٹی ایکسٹریکشن کے لیے نافذ کریں
3. **انٹیگریشن پائپ لائن** کو اسپیچ ریکوگنیشن کو روبوٹ ایکشنز سے جوڑنے کے لیے تخلیق کریں
4. **مختلف گفتگو ان پٹس** کے ساتھ جانچ کریں اور کارکردگی کا جائزہ لیں
5. **ریل ٹائم کارکردگی** کے لیے نوائز ریڈکشن کے لیے آپٹیمائز کریں
6. **نتائج کا تجزیہ** کریں اور سسٹم کو بہتر بنائیں

### متوقع نتائج
- کام کرتا ہوا ہیومنوائڈ روبوٹ کے لیے اسپیچ ریکوگنیشن سسٹم
- NLU نفاذ کی سمجھ
- ریل ٹائم پروسیسنگ کی مہارت
- کارکردگی کے جائزہ کی مہارت

## علم کی چیک

1. حقیقی دنیا کے ماحول میں ہیومنوائڈ روبوٹس کے لیے اسپیچ ریکوگنیشن کو نافذ کرنے میں کیا کلیدی چیلنج ہیں؟
2. آٹومیٹک اسپیچ ریکوگنیشن (ASR) اور نیچرل لینگویج انڈر اسٹینڈنگ (NLU) کے درمیان فرق وضاحت کریں۔
3. سیاق و سباق کے خیال رکھنے والا لینگویج انڈر اسٹینڈنگ روبوٹ انٹرایکشن کو کیسے بہتر بناتا ہے؟
4. شوری ماحول میں اسپیچ ریکوگنیشن کی درستگی کو بہتر بنانے کے لیے کون سی تکنیکس استعمال کی جا سکتی ہیں؟

## خلاصہ

اس باب میں ہیومنوائڈ روبوٹس کے لیے اسپیچ ریکوگنیشن اور نیچرل لینگویج انڈر اسٹینڈنگ سسٹمزم کا نفاذ کور کیا گیا۔ ہم نے ASR کے بنیادی اصولوں کو تلاش کیا، NLU کے لیے ارادہ کیسیفکیشن اور اینٹیٹی ایکسٹریکشن نافذ کیا، نوائز ریڈکشن تکنیکس کو پتہ کیا، اور گفتگو ان پٹ سے روبوٹ ایکشن تک مکمل پائپ لائن تخلیق کی۔ مضبوط اسپیچ پروسیسنگ سسٹمزم کا انضمام زیادہ قدرتی اور بے تکلف انسان-روبوٹ انٹرایکشن کو فعال کرتا ہے، جو ہیومنوائڈ روبوٹس کو روزمرہ کی ایپلیکیشنزم میں زیادہ قابل رسائی اور مفید بناتا ہے۔

## اگلے اقدامات

باب 19 میں، ہم LLMs کے ساتھ کوگنیٹو پلاننگ کو تلاش کریں گے، یہ دیکھتے ہوئے کہ بڑے زبانی ماڈلز ہیومنوائڈ روبوٹک سسٹمزم میں ہائی-لیول ریزننگ اور پلاننگ کے لیے کیسے استعمال کیے جا سکتے ہیں، زیادہ ترقی یافتہ خود مختار رویے کو فعال کرتے ہوئے۔