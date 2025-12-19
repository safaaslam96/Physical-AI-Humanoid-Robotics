---
sidebar_position: 18
title: "Chapter 18: Speech Recognition and Natural Language Understanding"
---

# Chapter 18: Speech Recognition and Natural Language Understanding

## Learning Objectives
- Implement voice command processing for humanoid robots
- Master natural language understanding in robotics
- Design multi-modal interaction systems
- Create robust dialogue management for robot communication

## Introduction to Speech Recognition in Robotics

Speech recognition is a critical component of natural human-robot interaction, enabling robots to understand and respond to voice commands. For humanoid robots, speech recognition systems must handle the challenges of real-world environments, including background noise, multiple speakers, and varying acoustic conditions.

### Key Challenges in Robot Speech Recognition

1. **Acoustic Environment**: Robots operate in noisy, dynamic environments
2. **Real-time Processing**: Commands must be processed with minimal latency
3. **Speaker Variability**: Systems must handle different voices and accents
4. **Domain Specificity**: Robot commands often use specific terminology
5. **Robustness**: Systems must work reliably in various conditions

### Speech Recognition Architecture

The typical speech recognition pipeline for robotics includes:

1. **Audio Input**: Microphone array for sound capture
2. **Preprocessing**: Noise reduction and signal enhancement
3. **Feature Extraction**: Converting audio to recognizable features
4. **Recognition**: Converting features to text
5. **Natural Language Understanding**: Interpreting the meaning
6. **Action Mapping**: Connecting to robot actions

## Voice Command Processing Systems

### Real-time Audio Processing

```python
# audio_processing.py
import pyaudio
import numpy as np
import webrtcvad
import collections
import threading
import queue
import time
from scipy import signal
import librosa

class AudioProcessor:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, num_channels=1):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.num_channels = num_channels
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2

        # Audio buffer
        self.audio_buffer = collections.deque(maxlen=30)  # 30 frames = 900ms
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_end_time = None

        # Noise reduction parameters
        self.noise_threshold = 0.01
        self.speech_threshold = 0.05

        # Audio processing parameters
        self.pre_emphasis = 0.97
        self.frame_stride = 0.01  # 10ms stride

    def record_audio_stream(self, callback, duration=None):
        """Record audio stream with real-time processing"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.num_channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )

        print("Starting audio recording...")

        start_time = time.time()
        while True:
            if duration and time.time() - start_time > duration:
                break

            # Read audio data
            audio_data = stream.read(self.frame_size, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer
            self.audio_buffer.append(audio_array)

            # Check for voice activity
            is_voice_active = self.is_voice_active(audio_array)

            if is_voice_active and not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = time.time()
                print("Speech detected - started")

            elif not is_voice_active and self.is_speaking:
                # Speech ended
                self.is_speaking = False
                self.speech_end_time = time.time()
                print("Speech ended")

                # Process the collected speech segment
                speech_segment = self.get_speech_segment()
                if len(speech_segment) > 0:
                    callback(speech_segment, self.speech_start_time, self.speech_end_time)

            # Process with callback if continuously speaking
            if self.is_speaking:
                callback(audio_array, time.time(), None, partial=True)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def is_voice_active(self, audio_frame):
        """Check if voice is active in the audio frame using WebRTC VAD"""
        # Convert to bytes for WebRTC VAD
        audio_bytes = (audio_frame * 32767).astype(np.int16).tobytes()

        # Check voice activity
        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except:
            # Fallback: simple energy-based detection
            energy = np.mean(np.abs(audio_frame))
            return energy > self.speech_threshold

    def get_speech_segment(self):
        """Get the complete speech segment from the buffer"""
        if not self.audio_buffer:
            return np.array([])

        # Concatenate buffered audio
        speech_data = np.concatenate(list(self.audio_buffer))

        # Apply preprocessing
        processed_speech = self.preprocess_audio(speech_data)

        return processed_speech

    def preprocess_audio(self, audio_data):
        """Apply preprocessing to audio data"""
        # Apply pre-emphasis filter
        emphasized_audio = self.pre_emphasis_filter(audio_data)

        # Noise reduction
        denoised_audio = self.reduce_noise(emphasized_audio)

        # Normalize
        normalized_audio = self.normalize_audio(denoised_audio)

        return normalized_audio

    def pre_emphasis_filter(self, audio_data, pre_emphasis=0.97):
        """Apply pre-emphasis filter to enhance high frequencies"""
        return np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

    def reduce_noise(self, audio_data):
        """Simple noise reduction using spectral subtraction"""
        # For simplicity, using a basic noise reduction approach
        # In practice, more sophisticated methods would be used

        # Estimate noise profile from beginning of audio
        noise_profile = np.mean(np.abs(audio_data[:int(self.sample_rate * 0.1)]))  # First 100ms

        # Apply soft thresholding
        threshold = max(self.noise_threshold, noise_profile * 0.5)

        # Soft thresholding
        denoised = np.sign(audio_data) * np.maximum(np.abs(audio_data) - threshold, 0)

        return denoised

    def normalize_audio(self, audio_data):
        """Normalize audio to consistent level"""
        if len(audio_data) == 0:
            return audio_data

        # RMS normalization
        rms = np.sqrt(np.mean(audio_data ** 2))
        target_rms = 0.1  # Target RMS level

        if rms > 0:
            gain = target_rms / rms
            normalized = audio_data * gain
            # Clip to prevent overflow
            return np.clip(normalized, -1.0, 1.0)
        else:
            return audio_data

    def detect_wake_word(self, audio_data, wake_word_model=None):
        """Detect wake word in audio stream"""
        # Simple energy-based wake word detection
        # In practice, this would use a trained wake word model

        energy = np.mean(np.abs(audio_data))

        # For demonstration, assume wake word is detected if energy is high
        # and has a specific pattern
        if energy > self.speech_threshold * 2:  # Higher threshold for wake word
            # Additional checks could include:
            # - Spectral analysis for specific word patterns
            # - ML model inference
            # - Keyword spotting algorithms
            return True

        return False

class WakeWordDetector:
    """Advanced wake word detection system"""
    def __init__(self):
        self.wake_words = ["robot", "hey robot", "assistant", "listen"]
        self.detected_wake_word = None
        self.wake_word_confidence = 0.0

    def detect_wake_word(self, audio_data, sample_rate=16000):
        """Detect wake word in audio data"""
        # This would typically use a trained model
        # For this example, we'll use a simple approach

        # In practice, you might use:
        # - Porcupine wake word engine
        # - Custom trained wake word model
        # - Audio keyword spotting with ML

        # Simple approach: look for specific audio patterns
        # This is a placeholder - real implementation would be more sophisticated

        # Calculate audio features
        rms_energy = np.sqrt(np.mean(audio_data ** 2))

        # Check for specific patterns (simplified)
        if rms_energy > 0.05:  # Energy threshold
            # In a real system, this would analyze spectral features
            # and use ML models to identify wake words
            self.detected_wake_word = "robot"
            self.wake_word_confidence = 0.8
            return True

        return False

    def reset_detection(self):
        """Reset wake word detection state"""
        self.detected_wake_word = None
        self.wake_word_confidence = 0.0

# Example usage
def example_audio_processing():
    processor = AudioProcessor()
    wake_detector = WakeWordDetector()

    def audio_callback(audio_segment, start_time, end_time, partial=False):
        if partial:
            # Check for wake word in partial stream
            if wake_detector.detect_wake_word(audio_segment):
                print(f"Wake word detected at {start_time:.2f}s!")
        else:
            # Process complete speech segment
            print(f"Complete speech segment processed: {len(audio_segment)} samples")

    print("Starting audio processing example...")
    print("Speak to test voice activity detection")

    # Note: This would run indefinitely in a real system
    # For this example, we'll just show the structure

if __name__ == "__main__":
    example_audio_processing()
```

### Speech-to-Text Integration

```python
# speech_to_text.py
import speech_recognition as sr
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import Dict, List, Optional, Callable

class SpeechToTextEngine:
    def __init__(self, language="en-US", model_type="default"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language
        self.model_type = model_type

        # Configuration
        self.recognizer.energy_threshold = 300  # Adjust for environment
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Pause duration to consider phrase complete

        # Initialize microphone settings
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)

        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Recognition history
        self.recognition_history = []
        self.max_history = 50

    def recognize_speech(self, audio_data, use_api=True):
        """Recognize speech from audio data"""
        try:
            if use_api:
                # Use Google Web Speech API (requires internet)
                text = self.recognizer.recognize_google(audio_data, language=self.language)
            else:
                # Use offline recognition (if available)
                text = self.recognizer.recognize_sphinx(audio_data, language=self.language)

            # Add to history
            self.add_to_history(text, confidence=0.9)  # Default confidence

            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

    def add_to_history(self, text, confidence=0.0):
        """Add recognition result to history"""
        entry = {
            'text': text,
            'confidence': confidence,
            'timestamp': time.time()
        }

        self.recognition_history.append(entry)

        # Keep history size manageable
        if len(self.recognition_history) > self.max_history:
            self.recognition_history = self.recognition_history[-self.max_history:]

    def get_recognition_context(self):
        """Get recent recognition context"""
        recent_entries = self.recognition_history[-5:]  # Last 5 recognitions
        return [entry['text'] for entry in recent_entries if entry['text']]

    def continuous_listening(self, callback: Callable[[str], None], timeout=5.0):
        """Continuously listen for speech and process results"""
        with self.microphone as source:
            print("Listening for speech...")

            while True:
                try:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5.0)

                    # Recognize speech
                    text = self.recognize_speech(audio)

                    if text:
                        print(f"Recognized: {text}")
                        callback(text)
                    else:
                        print("Could not understand audio")

                except sr.WaitTimeoutError:
                    # No speech detected within timeout, continue listening
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    time.sleep(1)  # Brief pause before retrying

class AdvancedSpeechRecognizer:
    """Advanced speech recognition with multiple engine support"""
    def __init__(self, language="en-US"):
        self.language = language

        # Multiple recognition engines
        self.engines = {
            'google': self.google_recognize,
            'wit': self.wit_recognize,
            'houndify': self.houndify_recognize,
            'bing': self.bing_recognize
        }

        self.active_engine = 'google'
        self.fallback_engines = ['google', 'bing']  # Priority order for fallback

        # Recognition confidence thresholds
        self.confidence_threshold = 0.7
        self.low_confidence_threshold = 0.5

    def set_credentials(self, engine, **credentials):
        """Set API credentials for recognition engines"""
        if engine == 'wit':
            self.wit_key = credentials.get('key')
        elif engine == 'houndify':
            self.houndify_client_id = credentials.get('client_id')
            self.houndify_client_key = credentials.get('client_key')
        elif engine == 'bing':
            self.bing_key = credentials.get('key')

    def recognize_with_fallback(self, audio_data, confidence_boost=False):
        """Recognize speech with fallback engines"""
        results = []

        for engine_name in self.fallback_engines:
            try:
                result = self.engines[engine_name](audio_data)
                if result and result['confidence'] >= self.low_confidence_threshold:
                    results.append(result)

                    # If high confidence result found, return immediately
                    if result['confidence'] >= self.confidence_threshold:
                        return result
            except Exception as e:
                print(f"Engine {engine_name} failed: {e}")
                continue

        # If no results with acceptable confidence, return the best one
        if results:
            return max(results, key=lambda x: x['confidence'])

        return None

    def google_recognize(self, audio_data):
        """Google Web Speech API recognition"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            # Get confidence using alternative method
            text = recognizer.recognize_google(audio_data, language=self.language)

            # Estimate confidence (Google API doesn't provide confidence directly)
            # This is a simplified estimation
            word_count = len(text.split())
            confidence = min(0.9, 0.6 + (word_count * 0.05))  # Basic confidence estimation

            return {
                'text': text,
                'confidence': confidence,
                'engine': 'google'
            }
        except Exception as e:
            raise e

    def wit_recognize(self, audio_data):
        """Wit.ai recognition (requires API key)"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            text = recognizer.recognize_wit(audio_data, key=self.wit_key, language=self.language)

            # Wit.ai may provide confidence information
            return {
                'text': text,
                'confidence': 0.8,  # Placeholder
                'engine': 'wit'
            }
        except Exception as e:
            raise e

    def houndify_recognize(self, audio_data):
        """Houndify recognition (requires credentials)"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            text = recognizer.recognize_houndify(
                audio_data,
                client_id=self.houndify_client_id,
                client_key=self.houndify_client_key
            )

            return {
                'text': text,
                'confidence': 0.8,  # Placeholder
                'engine': 'houndify'
            }
        except Exception as e:
            raise e

    def bing_recognize(self, audio_data):
        """Microsoft Bing recognition"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            text = recognizer.recognize_bing(audio_data, key=self.bing_key, language=self.language)

            return {
                'text': text,
                'confidence': 0.8,  # Placeholder
                'engine': 'bing'
            }
        except Exception as e:
            raise e

class VoiceCommandProcessor:
    """Process voice commands and map to robot actions"""
    def __init__(self, speech_recognizer):
        self.speech_recognizer = speech_recognizer
        self.command_mapping = self.initialize_command_mapping()
        self.context = {}
        self.user_preferences = {}

    def initialize_command_mapping(self):
        """Initialize mapping from voice commands to robot actions"""
        return {
            # Navigation commands
            'navigate_to': {
                'patterns': ['go to', 'navigate to', 'move to', 'walk to', 'head to'],
                'action': 'navigation',
                'required_params': ['destination']
            },
            'move_forward': {
                'patterns': ['move forward', 'go forward', 'step forward', 'forward'],
                'action': 'move',
                'required_params': ['direction', 'distance']
            },
            'turn': {
                'patterns': ['turn left', 'turn right', 'turn around', 'rotate'],
                'action': 'rotate',
                'required_params': ['direction']
            },

            # Manipulation commands
            'grasp_object': {
                'patterns': ['grasp', 'grab', 'pick up', 'take', 'hold'],
                'action': 'grasp',
                'required_params': ['object']
            },
            'release_object': {
                'patterns': ['release', 'let go', 'drop', 'put down'],
                'action': 'release',
                'required_params': []
            },

            # Information commands
            'get_time': {
                'patterns': ['what time is it', 'time', 'current time', 'tell me the time'],
                'action': 'time_query',
                'required_params': []
            },
            'get_date': {
                'patterns': ['what date is it', 'date', 'current date', 'tell me the date'],
                'action': 'date_query',
                'required_params': []
            },

            # Social interaction
            'greeting': {
                'patterns': ['hello', 'hi', 'good morning', 'good afternoon', 'good evening'],
                'action': 'greet',
                'required_params': []
            },
            'farewell': {
                'patterns': ['goodbye', 'bye', 'see you', 'farewell'],
                'action': 'farewell',
                'required_params': []
            }
        }

    def process_voice_command(self, text):
        """Process voice command and extract action"""
        text_lower = text.lower().strip()

        # Check for command patterns
        for command_type, config in self.command_mapping.items():
            for pattern in config['patterns']:
                if pattern in text_lower:
                    # Extract parameters
                    params = self.extract_parameters(text_lower, command_type, config)

                    # Validate required parameters
                    missing_params = self.validate_parameters(params, config['required_params'])

                    if not missing_params:
                        return {
                            'action': config['action'],
                            'command_type': command_type,
                            'parameters': params,
                            'raw_text': text
                        }
                    else:
                        # Ask for missing parameters
                        return {
                            'action': 'request_info',
                            'missing_params': missing_params,
                            'original_command': command_type
                        }

        # If no specific command found, treat as general query
        return {
            'action': 'general_query',
            'query': text,
            'command_type': 'unknown'
        }

    def extract_parameters(self, text, command_type, config):
        """Extract parameters from command text"""
        params = {}

        if command_type == 'navigate_to':
            # Extract destination from text like "go to kitchen"
            for pattern in config['patterns']:
                if pattern in text:
                    remaining_text = text.replace(pattern, '').strip()
                    if remaining_text:
                        params['destination'] = remaining_text
                    break

        elif command_type == 'grasp_object':
            # Extract object to grasp
            for pattern in config['patterns']:
                if pattern in text:
                    remaining_text = text.replace(pattern, '').strip()
                    if remaining_text:
                        # Remove common words like "the", "a", "an"
                        remaining_text = remaining_text.replace('the ', '').replace('a ', '').replace('an ', '').strip()
                        params['object'] = remaining_text
                    break

        elif command_type == 'move_forward':
            # Extract distance if specified
            import re
            distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(meters?|m|feet|ft)', text)
            if distance_match:
                params['distance'] = float(distance_match.group(1))
                params['unit'] = distance_match.group(2)
            else:
                params['distance'] = 1.0  # Default distance
                params['unit'] = 'meters'

        elif command_type == 'turn':
            if 'left' in text:
                params['direction'] = 'left'
            elif 'right' in text:
                params['direction'] = 'right'
            elif 'around' in text:
                params['direction'] = 'around'
            else:
                params['direction'] = 'left'  # Default

        return params

    def validate_parameters(self, params, required_params):
        """Validate that required parameters are present"""
        missing = []
        for param in required_params:
            if param not in params or params[param] is None:
                missing.append(param)
        return missing

    def handle_command(self, command_result):
        """Handle the processed command result"""
        action = command_result['action']

        if action == 'request_info':
            # Request missing information
            missing = command_result['missing_params']
            original_cmd = command_result['original_command']

            return f"I need more information: {', '.join(missing)} for {original_cmd.replace('_', ' ')} command."

        elif action == 'general_query':
            # Handle as general query
            query = command_result['query']
            return f"I received your query: '{query}'. How can I help?"

        else:
            # Execute specific action
            command_type = command_result['command_type']
            params = command_result.get('parameters', {})

            return f"Executing {command_type.replace('_', ' ')} with parameters: {params}"

# Example usage
def example_speech_recognition():
    print("Speech Recognition Example")

    # Initialize speech recognizer
    stt_engine = SpeechToTextEngine()

    # Initialize command processor
    command_processor = VoiceCommandProcessor(stt_engine)

    # Test commands
    test_commands = [
        "Navigate to the kitchen",
        "Grasp the red cup",
        "What time is it?",
        "Hello robot",
        "Turn left"
    ]

    print("Processing test commands:")
    for command in test_commands:
        print(f"\nInput: {command}")
        result = command_processor.process_voice_command(command)
        response = command_processor.handle_command(result)
        print(f"Output: {response}")

if __name__ == "__main__":
    try:
        example_speech_recognition()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install SpeechRecognition pyaudio")
```

## Natural Language Understanding

### Intent Recognition and Entity Extraction

```python
# nlu_engine.py
import re
import spacy
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class IntentType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INFORMATION = "information"
    SOCIAL = "social"
    SYSTEM = "system"

@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

@dataclass
class Intent:
    """Represents a recognized intent"""
    name: str
    confidence: float
    entities: List[Entity]
    parameters: Dict[str, str]

class RuleBasedNLUEngine:
    """Rule-based Natural Language Understanding engine"""
    def __init__(self):
        # Load spaCy model (small English model)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.intent_patterns = self.initialize_intent_patterns()
        self.entity_patterns = self.initialize_entity_patterns()

    def initialize_intent_patterns(self):
        """Initialize patterns for intent recognition"""
        return {
            IntentType.NAVIGATION: [
                (r'go to (\w+)', 0.9),
                (r'move to (\w+)', 0.9),
                (r'navigate to (\w+)', 0.9),
                (r'walk to (\w+)', 0.9),
                (r'head to (\w+)', 0.8),
                (r'bring me to (\w+)', 0.8),
            ],
            IntentType.MANIPULATION: [
                (r'grasp (\w+)', 0.9),
                (r'grab (\w+)', 0.9),
                (r'pick up (\w+)', 0.9),
                (r'take (\w+)', 0.9),
                (r'hold (\w+)', 0.8),
                (r'lift (\w+)', 0.8),
                (r'release (\w+)', 0.9),
                (r'put down (\w+)', 0.9),
                (r'drop (\w+)', 0.8),
            ],
            IntentType.INFORMATION: [
                (r'what time', 0.9),
                (r'what date', 0.9),
                (r'tell me about', 0.8),
                (r'how (?:is|are|do)', 0.7),
                (r'what (?:is|are|can)', 0.7),
                (r'when', 0.7),
                (r'where', 0.7),
                (r'who', 0.7),
            ],
            IntentType.SOCIAL: [
                (r'hello', 0.9),
                (r'hi', 0.9),
                (r'good morning', 0.9),
                (r'good afternoon', 0.9),
                (r'good evening', 0.9),
                (r'goodbye', 0.9),
                (r'bye', 0.9),
                (r'thank you', 0.8),
                (r'thanks', 0.8),
            ],
            IntentType.SYSTEM: [
                (r'stop', 0.9),
                (r'cancel', 0.9),
                (r'abort', 0.8),
                (r'help', 0.8),
                (r'reset', 0.7),
                (r'restart', 0.7),
            ]
        }

    def initialize_entity_patterns(self):
        """Initialize patterns for entity extraction"""
        return {
            'LOCATION': [
                (r'\b(kitchen|living room|bedroom|office|bathroom|dining room|hallway|garage|garden|outside)\b', 0.9),
            ],
            'OBJECT': [
                (r'\b(cup|bottle|book|phone|keys|ball|box|toy|food|water|coffee)\b', 0.8),
            ],
            'PERSON': [
                (r'\b(mom|dad|mother|father|brother|sister|friend|person|man|woman|child)\b', 0.7),
            ],
            'TIME': [
                (r'\b(\d{1,2}:\d{2})\b', 0.9),  # Time like 10:30
                (r'\b(today|tomorrow|yesterday|now|later|morning|afternoon|evening|night)\b', 0.8),
            ],
            'NUMBER': [
                (r'\b(\d+(?:\.\d+)?)\b', 0.9),
            ]
        }

    def process_text(self, text: str) -> Intent:
        """Process text and extract intent and entities"""
        # First, try to recognize intent
        intent = self.recognize_intent(text)

        # Then, extract entities
        entities = self.extract_entities(text)

        # Update intent with entities
        intent.entities = entities

        # Extract parameters from entities
        intent.parameters = self.entities_to_parameters(entities)

        return intent

    def recognize_intent(self, text: str) -> Intent:
        """Recognize intent from text using pattern matching"""
        best_intent = None
        best_confidence = 0.0

        for intent_type, patterns in self.intent_patterns.items():
            for pattern, base_confidence in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    # Calculate confidence based on match quality
                    confidence = base_confidence

                    # Boost confidence if it's an exact match
                    if match.group(0).lower() == text.lower().strip():
                        confidence = min(1.0, confidence + 0.1)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = Intent(
                            name=intent_type.value,
                            confidence=confidence,
                            entities=[],
                            parameters={}
                        )

        if best_intent is None:
            # Default to general query intent
            best_intent = Intent(
                name='general_query',
                confidence=0.3,
                entities=[],
                parameters={}
            )

        return best_intent

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using pattern matching"""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern, base_confidence in patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    entity = Entity(
                        text=match.group(1) if len(match.groups()) > 0 else match.group(0),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=base_confidence
                    )
                    entities.append(entity)

        # Sort entities by position in text
        entities.sort(key=lambda x: x.start)

        return entities

    def entities_to_parameters(self, entities: List[Entity]) -> Dict[str, str]:
        """Convert entities to action parameters"""
        params = {}

        for entity in entities:
            if entity.label == 'LOCATION':
                params['destination'] = entity.text
            elif entity.label == 'OBJECT':
                params['object'] = entity.text
            elif entity.label == 'PERSON':
                params['person'] = entity.text
            elif entity.label == 'TIME':
                params['time'] = entity.text
            elif entity.label == 'NUMBER':
                params['number'] = entity.text

        return params

class MLBasedNLUEngine:
    """Machine Learning-based NLU engine (simplified example)"""
    def __init__(self):
        # In practice, this would load a trained model
        # For this example, we'll simulate with rule-based approach
        self.rule_engine = RuleBasedNLUEngine()

        # Pre-trained intent classification (simulated)
        self.intent_classifier = self.train_intent_classifier()

        # Named Entity Recognition (simulated)
        self.ner_model = self.train_ner_model()

    def train_intent_classifier(self):
        """Train intent classification model (simulated)"""
        # This would typically involve:
        # 1. Training data with labeled intents
        # 2. Feature extraction (TF-IDF, embeddings, etc.)
        # 3. Training classifier (SVM, neural network, etc.)

        # For simulation, return a function that mimics classification
        def classify_intent(text):
            # Simulate ML classification with confidence scores
            text_lower = text.lower()

            scores = {
                'navigation': 0.1,
                'manipulation': 0.1,
                'information': 0.1,
                'social': 0.1,
                'system': 0.1
            }

            # Boost scores based on keywords
            if any(word in text_lower for word in ['go', 'move', 'navigate', 'walk', 'head']):
                scores['navigation'] = 0.9
            if any(word in text_lower for word in ['grasp', 'grab', 'pick', 'take', 'hold', 'release']):
                scores['manipulation'] = 0.9
            if any(word in text_lower for word in ['what', 'when', 'where', 'how', 'time', 'date']):
                scores['information'] = 0.8
            if any(word in text_lower for word in ['hello', 'hi', 'good', 'bye', 'thank']):
                scores['social'] = 0.9
            if any(word in text_lower for word in ['stop', 'cancel', 'help']):
                scores['system'] = 0.8

            # Return best scoring intent
            best_intent = max(scores.items(), key=lambda x: x[1])
            return best_intent[0], best_intent[1]

        return classify_intent

    def train_ner_model(self):
        """Train Named Entity Recognition model (simulated)"""
        # This would typically use models like BERT, SpaCy, etc.
        # For simulation, return a function that mimics NER

        def recognize_entities(text):
            # Simulate entity recognition
            entities = []

            # Location entities
            location_pattern = r'\b(kitchen|living room|bedroom|office|bathroom)\b'
            for match in re.finditer(location_pattern, text.lower()):
                entities.append(Entity(
                    text=match.group(0),
                    label='LOCATION',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))

            # Object entities
            object_pattern = r'\b(cup|bottle|book|phone|keys)\b'
            for match in re.finditer(object_pattern, text.lower()):
                entities.append(Entity(
                    text=match.group(0),
                    label='OBJECT',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                ))

            return entities

        return recognize_entities

    def process_text(self, text: str) -> Intent:
        """Process text using ML-based NLU"""
        # Classify intent
        intent_name, intent_confidence = self.intent_classifier(text)

        # Recognize entities
        entities = self.ner_model(text)

        # Create intent object
        intent = Intent(
            name=intent_name,
            confidence=intent_confidence,
            entities=entities,
            parameters=self.entities_to_parameters(entities)
        )

        return intent

    def entities_to_parameters(self, entities: List[Entity]) -> Dict[str, str]:
        """Convert entities to action parameters"""
        params = {}

        for entity in entities:
            if entity.label == 'LOCATION':
                params['destination'] = entity.text
            elif entity.label == 'OBJECT':
                params['object'] = entity.text

        return params

class ContextualNLUEngine:
    """Context-aware NLU engine that considers conversation history"""
    def __init__(self):
        self.ml_engine = MLBasedNLUEngine()
        self.conversation_context = []
        self.max_context_length = 10

    def process_text_with_context(self, text: str, user_id: str = "default") -> Intent:
        """Process text considering conversation context"""
        # Get recent context
        recent_context = self.get_recent_context(user_id)

        # Process with context
        intent = self.ml_engine.process_text(text)

        # Enhance with context
        enhanced_intent = self.enhance_with_context(intent, recent_context, text)

        # Add to context
        self.add_to_context(user_id, text, enhanced_intent)

        return enhanced_intent

    def get_recent_context(self, user_id: str) -> List[Dict]:
        """Get recent conversation context for user"""
        # In practice, this would access a database or memory system
        user_context = []

        # Filter context for specific user
        for entry in self.conversation_context[-self.max_context_length:]:
            if entry.get('user_id') == user_id:
                user_context.append(entry)

        return user_context

    def enhance_with_context(self, intent: Intent, context: List[Dict], current_text: str) -> Intent:
        """Enhance intent recognition with context information"""
        # If intent confidence is low, use context to improve
        if intent.confidence < 0.6 and context:
            # Look for related entities in context
            for entry in reversed(context[-3:]):  # Check last 3 exchanges
                prev_intent = entry.get('intent', {})
                prev_entities = prev_intent.get('entities', [])

                # If previous interaction was about navigation,
                # and current text is ambiguous, assume navigation context
                if (prev_intent.get('name') == 'navigation' and
                    any(word in current_text.lower() for word in ['there', 'it', 'that', 'the'])):

                    # Try to resolve ambiguous references
                    for entity in prev_entities:
                        if entity['label'] == 'LOCATION':
                            # Current text might refer to this location
                            if 'go' in current_text.lower() or 'move' in current_text.lower():
                                intent.parameters['destination'] = entity['text']
                                intent.confidence = max(intent.confidence, 0.7)
                                intent.name = 'navigation'

        return intent

    def add_to_context(self, user_id: str, text: str, intent: Intent):
        """Add interaction to conversation context"""
        context_entry = {
            'user_id': user_id,
            'text': text,
            'intent': {
                'name': intent.name,
                'confidence': intent.confidence,
                'entities': [{'text': e.text, 'label': e.label, 'confidence': e.confidence}
                           for e in intent.entities],
                'parameters': intent.parameters.copy()
            },
            'timestamp': time.time()
        }

        self.conversation_context.append(context_entry)

        # Keep context size manageable
        if len(self.conversation_context) > 100:  # Maximum 100 entries
            self.conversation_context = self.conversation_context[-50:]

# Example usage
def example_nlu_processing():
    print("Natural Language Understanding Example")

    # Initialize NLU engines
    rule_engine = RuleBasedNLUEngine()
    ml_engine = MLBasedNLUEngine()
    contextual_engine = ContextualNLUEngine()

    # Test sentences
    test_sentences = [
        "Navigate to the kitchen",
        "Grasp the red cup",
        "What time is it?",
        "Hello robot",
        "Go to the place we talked about",
        "Pick up that object"
    ]

    print("Testing Rule-Based NLU:")
    for sentence in test_sentences:
        intent = rule_engine.process_text(sentence)
        print(f"Input: '{sentence}' -> Intent: {intent.name} (conf: {intent.confidence:.2f})")
        if intent.entities:
            print(f"  Entities: {[(e.text, e.label) for e in intent.entities]}")
        if intent.parameters:
            print(f"  Parameters: {intent.parameters}")

    print("\nTesting ML-Based NLU:")
    for sentence in test_sentences:
        intent = ml_engine.process_text(sentence)
        print(f"Input: '{sentence}' -> Intent: {intent.name} (conf: {intent.confidence:.2f})")
        if intent.entities:
            print(f"  Entities: {[(e.text, e.label) for e in intent.entities]}")

if __name__ == "__main__":
    import time
    example_nlu_processing()
```

## Multi-Modal Interaction

### Combining Speech with Other Modalities

```python
# multi_modal_interaction.py
import numpy as np
import cv2
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import queue

@dataclass
class MultiModalInput:
    """Represents input from multiple modalities"""
    speech: Optional[str] = None
    vision: Optional[np.ndarray] = None
    gesture: Optional[str] = None
    touch: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0

@dataclass
class MultiModalOutput:
    """Represents output across multiple modalities"""
    speech: Optional[str] = None
    action: Optional[str] = None
    visual_feedback: Optional[np.ndarray] = None
    haptic_feedback: Optional[str] = None

class MultiModalFusionEngine:
    """Fuses information from multiple modalities"""
    def __init__(self):
        self.speech_processor = self.initialize_speech_processor()
        self.vision_processor = self.initialize_vision_processor()
        self.gesture_processor = self.initialize_gesture_processor()

        # Confidence weights for different modalities
        self.modality_weights = {
            'speech': 0.6,
            'vision': 0.3,
            'gesture': 0.1
        }

        # Fusion strategies
        self.fusion_strategies = {
            'early': self.early_fusion,
            'late': self.late_fusion,
            'intermediate': self.intermediate_fusion
        }

        self.current_fusion_strategy = 'late'

    def initialize_speech_processor(self):
        """Initialize speech processing components"""
        # This would integrate with the speech recognition system
        return {
            'recognizer': None,  # Would be connected to speech recognizer
            'nlu': None          # Would be connected to NLU engine
        }

    def initialize_vision_processor(self):
        """Initialize vision processing components"""
        # This would connect to computer vision systems
        return {
            'object_detector': None,
            'pose_estimator': None,
            'scene_analyzer': None
        }

    def initialize_gesture_processor(self):
        """Initialize gesture processing components"""
        # This would connect to gesture recognition systems
        return {
            'hand_tracker': None,
            'gesture_classifier': None
        }

    def early_fusion(self, inputs: MultiModalInput) -> Dict[str, Any]:
        """Combine raw inputs from different modalities before processing"""
        fused_features = {}

        # For early fusion, we would combine raw features
        # This is complex and often not practical for different data types
        # So we'll simulate by creating a combined representation

        if inputs.speech:
            # Convert speech to features
            speech_features = self.extract_speech_features(inputs.speech)
            fused_features['speech'] = speech_features

        if inputs.vision is not None:
            # Extract visual features
            vision_features = self.extract_vision_features(inputs.vision)
            fused_features['vision'] = vision_features

        if inputs.gesture:
            # Convert gesture to features
            gesture_features = self.extract_gesture_features(inputs.gesture)
            fused_features['gesture'] = gesture_features

        return fused_features

    def late_fusion(self, inputs: MultiModalInput) -> Dict[str, Any]:
        """Process each modality separately and combine results"""
        results = {}

        if inputs.speech:
            speech_result = self.process_speech(inputs.speech)
            results['speech'] = speech_result

        if inputs.vision is not None:
            vision_result = self.process_vision(inputs.vision)
            results['vision'] = vision_result

        if inputs.gesture:
            gesture_result = self.process_gesture(inputs.gesture)
            results['gesture'] = gesture_result

        # Combine results based on confidence and weights
        combined_result = self.combine_results(results)

        return combined_result

    def intermediate_fusion(self, inputs: MultiModalInput) -> Dict[str, Any]:
        """Combine modalities at intermediate processing levels"""
        # This would involve sharing representations between modalities
        # during processing, which is more complex

        # For this example, we'll use a weighted combination
        # of partially processed information
        results = self.late_fusion(inputs)

        # Apply cross-modal influence
        refined_result = self.apply_cross_modal_influence(results, inputs)

        return refined_result

    def extract_speech_features(self, speech: str) -> Dict[str, Any]:
        """Extract features from speech input"""
        # In practice, this would use acoustic and linguistic features
        return {
            'text': speech,
            'word_count': len(speech.split()),
            'key_phrases': self.extract_key_phrases(speech),
            'sentiment': self.estimate_sentiment(speech)
        }

    def extract_vision_features(self, vision: np.ndarray) -> Dict[str, Any]:
        """Extract features from visual input"""
        # In practice, this would use CNN features, object detection, etc.
        return {
            'objects_present': [],  # Would come from object detection
            'scene_type': 'indoor',  # Would be classified
            'people_present': 0,     # Would come from person detection
            'prominent_colors': []   # Would come from color analysis
        }

    def extract_gesture_features(self, gesture: str) -> Dict[str, Any]:
        """Extract features from gesture input"""
        return {
            'gesture_type': gesture,
            'confidence': 0.9,
            'temporal_pattern': 'single'
        }

    def process_speech(self, speech: str) -> Dict[str, Any]:
        """Process speech input"""
        # Use NLU to extract meaning
        # This would connect to the NLU system
        return {
            'intent': 'unknown',
            'entities': [],
            'confidence': 0.8,
            'raw_text': speech
        }

    def process_vision(self, vision: np.ndarray) -> Dict[str, Any]:
        """Process visual input"""
        # This would use computer vision algorithms
        return {
            'detected_objects': [],
            'scene_description': 'unknown',
            'confidence': 0.7
        }

    def process_gesture(self, gesture: str) -> Dict[str, Any]:
        """Process gesture input"""
        return {
            'gesture_meaning': gesture,
            'confidence': 0.9
        }

    def combine_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine results from different modalities"""
        # Calculate weighted confidence
        total_confidence = 0
        weighted_intent = None
        max_confidence = 0

        for modality, result in results.items():
            weight = self.modality_weights.get(modality, 0.1)
            confidence = result.get('confidence', 0.5) * weight

            total_confidence += confidence

            if confidence > max_confidence:
                max_confidence = confidence
                weighted_intent = result.get('intent', 'unknown')

        # Normalize confidence
        normalized_confidence = min(1.0, total_confidence)

        return {
            'intent': weighted_intent or 'unknown',
            'confidence': normalized_confidence,
            'modality_contributions': {
                mod: results[mod].get('confidence', 0.5) if mod in results else 0
                for mod in self.modality_weights.keys()
            },
            'individual_results': results
        }

    def apply_cross_modal_influence(self, results: Dict, inputs: MultiModalInput) -> Dict[str, Any]:
        """Apply cross-modal influence to refine results"""
        # Example: If speech says "that object" and vision shows an object,
        # link them together
        refined = results.copy()

        speech_result = results['individual_results'].get('speech', {})
        vision_result = results['individual_results'].get('vision', {})

        # If speech contains reference to visual object
        if (speech_result.get('raw_text') and
            any(word in speech_result['raw_text'].lower() for word in ['that', 'there', 'it']) and
            vision_result.get('detected_objects')):

            # Resolve the reference
            refined['resolved_reference'] = vision_result['detected_objects'][0] if vision_result['detected_objects'] else None
            refined['intent'] = 'referenced_object_interaction'

        return refined

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple keyword extraction
        keywords = ['navigate', 'grasp', 'move', 'go to', 'pick up', 'hello', 'help']
        found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
        return found_keywords

    def estimate_sentiment(self, text: str) -> str:
        """Estimate sentiment from text"""
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'please', 'thank', 'nice']
        negative_words = ['bad', 'terrible', 'stop', 'don\'t', 'not']

        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'

class MultiModalInteractionManager:
    """Manages multi-modal interaction flow"""
    def __init__(self):
        self.fusion_engine = MultiModalFusionEngine()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Active interaction state
        self.current_interaction = None
        self.interaction_history = []

        # Modalities
        self.speech_recognizer = None  # Would be connected to speech system
        self.vision_system = None      # Would be connected to vision system
        self.gesture_system = None     # Would be connected to gesture system

    def process_multi_modal_input(self, inputs: MultiModalInput) -> MultiModalOutput:
        """Process multi-modal input and generate response"""
        # Fuse the inputs
        fused_result = self.fusion_engine.late_fusion(inputs)

        # Generate appropriate response based on fused result
        response = self.generate_response(fused_result, inputs)

        return response

    def generate_response(self, fused_result: Dict[str, Any], inputs: MultiModalInput) -> MultiModalOutput:
        """Generate multi-modal response"""
        output = MultiModalOutput()

        intent = fused_result.get('intent', 'unknown')
        confidence = fused_result.get('confidence', 0.0)

        if confidence < 0.3:
            output.speech = "I'm not sure I understood that correctly. Could you please repeat?"
        elif intent == 'navigation':
            output.speech = "I'll navigate to the location for you."
            output.action = "navigate_to_location"
        elif intent == 'manipulation':
            output.speech = "I'll grasp that object for you."
            output.action = "grasp_object"
        elif intent == 'referenced_object_interaction':
            output.speech = "I see what you mean. I'll interact with that object."
            output.action = "interact_with_object"
        else:
            output.speech = f"I understand you want me to {intent.replace('_', ' ')}."

        # Add visual feedback if vision input was used
        if inputs.vision is not None:
            output.visual_feedback = self.generate_visual_feedback(inputs.vision, fused_result)

        return output

    def generate_visual_feedback(self, vision_input: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        """Generate visual feedback based on input and analysis"""
        # Create a simple visual feedback (in practice, this would be more sophisticated)
        feedback = vision_input.copy()

        # Draw confidence indicator
        height, width = feedback.shape[:2]
        conf_text = f"Confidence: {analysis.get('confidence', 0):.2f}"

        cv2.putText(feedback, conf_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return feedback

    def start_interaction_loop(self):
        """Start the multi-modal interaction loop"""
        def interaction_worker():
            while True:
                try:
                    # Get input from queue
                    inputs = self.input_queue.get(timeout=1.0)

                    # Process the multi-modal input
                    output = self.process_multi_modal_input(inputs)

                    # Put output in queue
                    self.output_queue.put(output)

                    # Add to interaction history
                    self.interaction_history.append({
                        'input': inputs,
                        'output': output,
                        'timestamp': time.time()
                    })

                    # Keep history size manageable
                    if len(self.interaction_history) > 100:
                        self.interaction_history = self.interaction_history[-50:]

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in interaction loop: {e}")

        # Start worker thread
        worker_thread = threading.Thread(target=interaction_worker, daemon=True)
        worker_thread.start()

    def add_input(self, inputs: MultiModalInput):
        """Add multi-modal input to processing queue"""
        self.input_queue.put(inputs)

    def get_output(self) -> Optional[MultiModalOutput]:
        """Get processed output from queue"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

# Example usage
def example_multi_modal_interaction():
    print("Multi-Modal Interaction Example")

    # Initialize the multi-modal system
    manager = MultiModalInteractionManager()

    # Start the interaction loop
    manager.start_interaction_loop()

    # Simulate multi-modal inputs
    test_inputs = [
        MultiModalInput(
            speech="Navigate to the kitchen",
            vision=np.random.rand(480, 640, 3),  # Simulated image
            timestamp=time.time()
        ),
        MultiModalInput(
            speech="Grasp the red cup",
            vision=np.random.rand(480, 640, 3),
            gesture="pointing",
            timestamp=time.time()
        ),
        MultiModalInput(
            speech="Go to that location",
            vision=np.random.rand(480, 640, 3),
            gesture="pointing",
            timestamp=time.time()
        )
    ]

    print("Processing multi-modal inputs:")
    for i, inputs in enumerate(test_inputs):
        print(f"\nTest {i+1}:")
        print(f"Speech: {inputs.speech}")
        print(f"Vision: {'Present' if inputs.vision is not None else 'Absent'}")
        print(f"Gesture: {inputs.gesture}")

        # Process the input
        output = manager.process_multi_modal_input(inputs)

        print(f"Response: {output.speech}")
        print(f"Action: {output.action}")

if __name__ == "__main__":
    import time
    example_multi_modal_interaction()
```

## Dialogue Management

### Conversational Flow Control

```python
# dialogue_management.py
import re
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

class DialogueState(Enum):
    GREETING = "greeting"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    CONFIRMING = "confirming"
    HANDLING_ERROR = "handling_error"
    IDLE = "idle"

class ConversationType(Enum):
    TASK_ORIENTED = "task_oriented"
    INFORMATION_SEEKING = "information_seeking"
    SOCIAL_CHITCHAT = "social_chitchat"
    COMMAND_EXECUTION = "command_execution"

@dataclass
class DialogueContext:
    """Context information for the current dialogue"""
    user_id: str
    conversation_type: ConversationType
    current_topic: Optional[str] = None
    previous_utterances: List[Dict[str, str]] = None
    task_stack: List[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = None
    session_start_time: float = 0.0
    last_activity_time: float = 0.0

class DialogueManager:
    """Manages conversational flow and dialogue state"""
    def __init__(self):
        self.current_state = DialogueState.IDLE
        self.context = None
        self.state_handlers = self.initialize_state_handlers()
        self.response_templates = self.initialize_response_templates()
        self.conversation_history = []
        self.max_history_length = 50

    def initialize_state_handlers(self):
        """Initialize handlers for different dialogue states"""
        return {
            DialogueState.GREETING: self.handle_greeting_state,
            DialogueState.LISTENING: self.handle_listening_state,
            DialogueState.PROCESSING: self.handle_processing_state,
            DialogueState.RESPONDING: self.handle_responding_state,
            DialogueState.CONFIRMING: self.handle_confirming_state,
            DialogueState.HANDLING_ERROR: self.handle_error_state
        }

    def initialize_response_templates(self):
        """Initialize response templates for different situations"""
        return {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Good day! How may I help you?"
            ],
            'confirmation': [
                "I'll {action} for you. Is that correct?",
                "So you want me to {action}. Should I proceed?",
                "I understand you want {action}. Is this right?"
            ],
            'error': [
                "I'm sorry, I didn't understand that.",
                "Could you please rephrase that?",
                "I'm having trouble understanding. Could you try again?"
            ],
            'task_complete': [
                "I've completed {task}.",
                "Task {task} is done.",
                "{task} has been completed successfully."
            ]
        }

    def start_conversation(self, user_id: str, conversation_type: ConversationType = ConversationType.TASK_ORIENTED):
        """Start a new conversation with a user"""
        self.context = DialogueContext(
            user_id=user_id,
            conversation_type=conversation_type,
            previous_utterances=[],
            task_stack=[],
            user_preferences={},
            session_start_time=time.time(),
            last_activity_time=time.time()
        )

        self.current_state = DialogueState.GREETING
        return self.generate_greeting()

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response"""
        if not self.context:
            # Start default conversation
            self.start_conversation("default_user")

        self.context.last_activity_time = time.time()

        # Add user input to history
        self.add_to_conversation_history("user", user_input)

        # Process based on current state
        response = self.state_handlers[self.current_state](user_input)

        # Add system response to history
        self.add_to_conversation_history("system", response.get('text', ''))

        return response

    def handle_greeting_state(self, user_input: str) -> Dict[str, Any]:
        """Handle the greeting state"""
        response = {
            'text': self.generate_greeting(),
            'state': DialogueState.LISTENING,
            'action': 'greeting'
        }
        self.current_state = DialogueState.LISTENING
        return response

    def handle_listening_state(self, user_input: str) -> Dict[str, Any]:
        """Handle the listening state - analyze user input"""
        # Analyze the user input to determine next state
        analysis = self.analyze_user_input(user_input)

        if analysis['intent'] == 'greeting':
            response = {
                'text': "Hello! How can I help you?",
                'state': DialogueState.LISTENING,
                'action': 'greeting_response'
            }
        elif analysis['intent'] == 'task_request':
            # Push task onto stack
            task = {
                'type': 'task',
                'description': user_input,
                'parameters': analysis.get('parameters', {}),
                'status': 'pending'
            }
            self.context.task_stack.append(task)

            response = {
                'text': f"I can help with that. You want me to {user_input.lower()}. Is that correct?",
                'state': DialogueState.CONFIRMING,
                'action': 'task_confirmation',
                'task_id': len(self.context.task_stack) - 1
            }
            self.current_state = DialogueState.CONFIRMING
        else:
            # Default to processing
            response = {
                'text': f"I'll process your request: {user_input}",
                'state': DialogueState.PROCESSING,
                'action': 'processing_request'
            }
            self.current_state = DialogueState.PROCESSING

        return response

    def handle_processing_state(self, user_input: str) -> Dict[str, Any]:
        """Handle the processing state"""
        # In a real system, this would process the request
        # For this example, we'll just acknowledge and return to listening

        response = {
            'text': "I'm processing your request. This may take a moment.",
            'state': DialogueState.RESPONDING,
            'action': 'processing_acknowledgment'
        }
        self.current_state = DialogueState.RESPONDING
        return response

    def handle_responding_state(self, user_input: str) -> Dict[str, Any]:
        """Handle the responding state"""
        # Generate response based on context
        if self.context.task_stack:
            current_task = self.context.task_stack[-1]
            if current_task['status'] == 'completed':
                response_text = self.response_templates['task_complete'][0].format(
                    task=current_task['description']
                )
            else:
                response_text = "I've processed your request."
        else:
            response_text = "I've handled your request."

        response = {
            'text': response_text,
            'state': DialogueState.LISTENING,
            'action': 'task_completion'
        }
        self.current_state = DialogueState.LISTENING
        return response

    def handle_confirming_state(self, user_input: str) -> Dict[str, Any]:
        """Handle the confirming state"""
        user_input_lower = user_input.lower()

        if any(word in user_input_lower for word in ['yes', 'yep', 'sure', 'okay', 'correct', 'right']):
            # User confirmed, proceed with task
            if self.context.task_stack:
                task = self.context.task_stack[-1]
                task['status'] = 'confirmed'

                response = {
                    'text': f"Great! I'll work on {task['description']} now.",
                    'state': DialogueState.PROCESSING,
                    'action': 'task_execution',
                    'execute_task': True
                }
                self.current_state = DialogueState.PROCESSING
            else:
                response = {
                    'text': "I'm ready to help. What would you like me to do?",
                    'state': DialogueState.LISTENING,
                    'action': 'request_task'
                }
                self.current_state = DialogueState.LISTENING
        elif any(word in user_input_lower for word in ['no', 'nope', 'wrong', 'cancel', 'stop']):
            # User rejected, cancel task
            if self.context.task_stack:
                cancelled_task = self.context.task_stack.pop()

                response = {
                    'text': f"I've cancelled the task: {cancelled_task['description']}. What else can I help with?",
                    'state': DialogueState.LISTENING,
                    'action': 'task_cancelled'
                }
            else:
                response = {
                    'text': "Okay, what would you like to do instead?",
                    'state': DialogueState.LISTENING,
                    'action': 'request_alternative'
                }
            self.current_state = DialogueState.LISTENING
        else:
            # Unclear response, ask for clarification
            response = {
                'text': "I'm not sure if you confirmed or rejected the task. Please say yes or no.",
                'state': DialogueState.CONFIRMING,
                'action': 'request_clarification'
            }
            # Stay in confirming state

        return response

    def handle_error_state(self, user_input: str) -> Dict[str, Any]:
        """Handle the error state"""
        response = {
            'text': self.response_templates['error'][0],
            'state': DialogueState.LISTENING,
            'action': 'error_recovery'
        }
        self.current_state = DialogueState.LISTENING
        return response

    def analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to determine intent and parameters"""
        user_lower = user_input.lower()

        analysis = {
            'intent': 'unknown',
            'confidence': 0.0,
            'parameters': {}
        }

        # Intent classification
        if any(greeting in user_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            analysis['intent'] = 'greeting'
            analysis['confidence'] = 0.9
        elif any(task_word in user_lower for task_word in ['go', 'navigate', 'move', 'walk', 'head', 'take', 'grasp', 'pick up', 'get']):
            analysis['intent'] = 'task_request'
            analysis['confidence'] = 0.8

            # Extract parameters
            # Simple parameter extraction
            import re
            destination_match = re.search(r'to (\w+)', user_lower)
            if destination_match:
                analysis['parameters']['destination'] = destination_match.group(1)

            object_match = re.search(r'(?:grasp|pick up|take) (\w+)', user_lower)
            if object_match:
                analysis['parameters']['object'] = object_match.group(1)
        else:
            analysis['intent'] = 'general_request'
            analysis['confidence'] = 0.6

        return analysis

    def generate_greeting(self) -> str:
        """Generate an appropriate greeting"""
        import random
        return random.choice(self.response_templates['greeting'])

    def add_to_conversation_history(self, speaker: str, text: str):
        """Add an utterance to the conversation history"""
        utterance = {
            'speaker': speaker,
            'text': text,
            'timestamp': time.time()
        }

        self.conversation_history.append(utterance)

        # Keep history size manageable
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def get_conversation_context(self, num_utterances: int = 3) -> List[Dict[str, str]]:
        """Get recent conversation context"""
        return self.conversation_history[-num_utterances:]

    def reset_conversation(self):
        """Reset the conversation state"""
        self.current_state = DialogueState.IDLE
        self.context = None
        self.conversation_history = []

class ContextualDialogueManager(DialogueManager):
    """Enhanced dialogue manager with context awareness"""
    def __init__(self):
        super().__init__()
        self.user_models = {}  # Store user-specific models
        self.topic_tracker = TopicTracker()
        self.follow_up_manager = FollowUpManager()

    def process_user_input(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """Enhanced input processing with user context"""
        # Load or create user model
        if user_id not in self.user_models:
            self.user_models[user_id] = UserModel(user_id)

        user_model = self.user_models[user_id]

        # Update user model with new input
        user_model.update_with_input(user_input)

        # Track topics
        new_topics = self.topic_tracker.extract_topics(user_input)
        self.topic_tracker.update_context(new_topics)

        # Handle follow-ups
        follow_up_response = self.follow_up_manager.check_for_follow_up(user_input, self.conversation_history)
        if follow_up_response:
            return follow_up_response

        # Process with enhanced context
        if not self.context or self.context.user_id != user_id:
            self.start_conversation(user_id, ConversationType.TASK_ORIENTED)

        self.context.last_activity_time = time.time()

        # Add user input to history
        self.add_to_conversation_history("user", user_input)

        # Process based on current state with user context
        response = self.state_handlers[self.current_state](user_input)

        # Personalize response based on user model
        response['text'] = self.personalize_response(response['text'], user_model)

        # Add system response to history
        self.add_to_conversation_history("system", response.get('text', ''))

        return response

    def personalize_response(self, response: str, user_model: 'UserModel') -> str:
        """Personalize response based on user model"""
        # Simple personalization
        if user_model.preferred_greeting:
            response = response.replace("Hello", user_model.preferred_greeting)

        # Add user-specific information
        if user_model.last_interaction_time:
            time_diff = time.time() - user_model.last_interaction_time
            if time_diff > 3600:  # More than an hour
                response = f"It's good to see you again! {response}"

        return response

class TopicTracker:
    """Tracks and manages conversation topics"""
    def __init__(self):
        self.current_topics = []
        self.topic_history = []
        self.topic_keywords = {
            'navigation': ['go', 'move', 'navigate', 'walk', 'to', 'toward', 'kitchen', 'bedroom', 'office'],
            'manipulation': ['grasp', 'pick', 'take', 'hold', 'cup', 'object', 'grab', 'lift'],
            'time': ['time', 'date', 'when', 'now', 'schedule', 'calendar'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'hot', 'cold'],
            'greeting': ['hello', 'hi', 'good', 'morning', 'afternoon', 'evening', 'hey']
        }

    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        text_lower = text.lower()
        detected_topics = []

        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics

    def update_context(self, new_topics: List[str]):
        """Update topic context"""
        self.current_topics = new_topics
        self.topic_history.extend(new_topics)

class FollowUpManager:
    """Manages follow-up questions and responses"""
    def __init__(self):
        self.last_utterance = ""
        self.follow_up_patterns = {
            r'do (?:you|I) need to.*': 'request_clarification',
            r'what.*next': 'request_next_step',
            r'how.*do.*that': 'request_explanation',
            r'can you.*': 'request_capability',
        }

    def check_for_follow_up(self, current_input: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """Check if current input is a follow-up to previous utterances"""
        if not history:
            return None

        # Check for follow-up patterns
        for pattern, intent in self.follow_up_patterns.items():
            if re.search(pattern, current_input.lower()):
                # This is a follow-up question
                return {
                    'text': self.generate_follow_up_response(intent, history[-1]),
                    'state': DialogueState.LISTENING,
                    'action': 'follow_up_response'
                }

        return None

    def generate_follow_up_response(self, intent: str, previous_utterance: Dict) -> str:
        """Generate appropriate response for follow-up"""
        responses = {
            'request_clarification': "Let me clarify: I can help with navigation, object manipulation, and information retrieval.",
            'request_next_step': "The next step would be to execute the task I described.",
            'request_explanation': "I can explain how I plan to complete this task.",
            'request_capability': "Yes, I can perform various tasks including navigation and manipulation."
        }
        return responses.get(intent, "I can help you with that.")

class UserModel:
    """Model of a specific user for personalization"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferred_greeting = None
        self.last_interaction_time = None
        self.conversation_style = 'formal'  # or 'casual'
        self.topic_preferences = []
        self.response_preferences = {
            'verbosity': 'medium',  # 'brief', 'medium', 'detailed'
            'tone': 'helpful'      # 'formal', 'friendly', 'helpful'
        }

    def update_with_input(self, user_input: str):
        """Update user model based on new input"""
        self.last_interaction_time = time.time()

        # Learn preferences from input
        if user_input.lower().startswith('hey') or user_input.lower().startswith('hi'):
            self.conversation_style = 'casual'
        elif user_input.lower().startswith('hello') or user_input.lower().startswith('good'):
            self.conversation_style = 'formal'

# Example usage
def example_dialogue_management():
    print("Dialogue Management Example")

    # Initialize dialogue manager
    dialogue_manager = ContextualDialogueManager()

    # Simulate a conversation
    conversation = [
        "Hello robot",
        "Navigate to the kitchen",
        "Yes, that's correct",
        "Grasp the red cup",
        "No, I meant the blue one",
        "What time is it?",
        "Thank you"
    ]

    print("Simulating conversation:")
    for i, user_input in enumerate(conversation):
        print(f"\n{i+1}. User: {user_input}")

        response = dialogue_manager.process_user_input(user_input, f"user_{i%3}")
        print(f"   Robot: {response['text']}")
        print(f"   State: {response['state'].value}")
        print(f"   Action: {response['action']}")

if __name__ == "__main__":
    example_dialogue_management()
```

## Knowledge Check

1. What are the key components of a speech recognition system for robotics?
2. How does natural language understanding differ from speech recognition?
3. What are the benefits of multi-modal interaction in robotics?
4. How does dialogue management contribute to natural conversation flow?

## Summary

This chapter covered speech recognition and natural language understanding for humanoid robots. We explored voice command processing systems, natural language understanding engines, multi-modal interaction techniques, and dialogue management systems. The chapter provided practical implementations for processing speech input, understanding user intent, combining multiple modalities, and managing conversational flow for natural human-robot interaction.

## Next Steps

In the next chapter, we'll explore cognitive planning with LLMs, covering how to use Large Language Models to translate natural language commands into executable robotic actions and create sophisticated task planning systems for humanoid robots.