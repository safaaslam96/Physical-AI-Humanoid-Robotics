---
title: "Chapter 18: Speech Recognition and Natural Language Understanding for Humanoid Robots"
sidebar_label: "Chapter 18: Speech Recognition and NLU"
---



# Chapter 18: Speech Recognition and Natural Language Understanding for Humanoid Robots

## Learning Objectives
- Understand the fundamentals of speech recognition systems for humanoid robots
- Implement Natural Language Understanding (NLU) for robot command interpretation
- Design robust speech processing pipelines for real-world environments
- Evaluate and optimize speech recognition performance in noisy conditions

## Introduction

Speech recognition and Natural Language Understanding (NLU) form the foundation of natural human-robot interaction. For humanoid robots to effectively communicate with humans, they must accurately recognize spoken commands and understand their semantic meaning in context. This chapter explores the integration of speech recognition and NLU systems specifically designed for humanoid robots, addressing the unique challenges of real-world environments, noise, and the need for real-time processing.

## Speech Recognition Fundamentals

### Automatic Speech Recognition (ASR) Systems

Automatic Speech Recognition (ASR) systems convert spoken language into text. For humanoid robots, ASR must be robust to environmental conditions and real-time capable:

```python
# Speech recognition system for humanoid robots
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

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Set energy threshold for voice activation
        self.recognizer.energy_threshold = 4000

        # Set minimum pause threshold for speech detection
        self.recognizer.pause_threshold = 0.8

    def listen_for_speech(self):
        """
        Listen for speech using the microphone and return audio data
        """
        try:
            with self.microphone as source:
                print("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return audio
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            print("Could not understand the speech")
            return None

    def recognize_speech_google(self, audio):
        """
        Recognize speech using Google's speech recognition service
        """
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.RequestError:
            print("API unavailable or unresponsive")
            return None
        except sr.UnknownValueError:
            print("Could not understand the speech")
            return None

    def recognize_speech_wav2vec2(self, audio):
        """
        Recognize speech using Wav2Vec2 model for offline recognition
        """
        try:
            # Convert audio to raw data
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

            # Process audio
            inputs = self.processor(raw_data, sampling_rate=16000, return_tensors="pt", padding=True)

            # Perform inference
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription
        except Exception as e:
            print(f"Error in Wav2Vec2 recognition: {e}")
            return None

    def noise_robust_recognition(self, audio):
        """
        Apply noise reduction techniques before recognition
        """
        # Apply noise reduction
        # This is a simplified approach - in practice, use advanced noise reduction algorithms
        try:
            # Convert to raw data
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

            # Apply basic noise reduction (spectral subtraction approach)
            # Calculate noise profile from beginning of audio
            noise_profile = np.mean(np.abs(raw_data[:1000]))

            # Apply noise reduction
            cleaned_data = raw_data - noise_profile
            cleaned_data = np.clip(cleaned_data, -32768, 32767)

            # Create new audio data
            cleaned_audio = sr.AudioData(
                cleaned_data.tobytes(),
                audio.sample_rate,
                audio.sample_width
            )

            # Recognize speech from cleaned audio
            text = self.recognizer.recognize_google(cleaned_audio)
            return text
        except Exception as e:
            print(f"Error in noise robust recognition: {e}")
            return None

    def continuous_listening(self, callback_func):
        """
        Continuously listen for speech and call the callback function with recognized text
        """
        self.is_listening = True

        def listen_thread():
            while self.is_listening:
                try:
                    with self.microphone as source:
                        print("Robot is listening...")
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                        # Recognize speech
                        text = self.recognize_speech_google(audio)

                        if text:
                            print(f"Recognized: {text}")
                            callback_func(text)
                except sr.WaitTimeoutError:
                    continue  # Keep listening
                except sr.UnknownValueError:
                    print("Could not understand speech")
                    continue
                except sr.RequestError as e:
                    print(f"Error with speech recognition service: {e}")
                    continue

        # Start the listening thread
        listener_thread = threading.Thread(target=listen_thread)
        listener_thread.daemon = True
        listener_thread.start()

        return listener_thread

    def stop_listening(self):
        """
        Stop the continuous listening process
        """
        self.is_listening = False
```

### Real-Time Speech Processing Pipeline

Implementing a real-time speech processing pipeline for humanoid robots:

```python
# Real-time speech processing pipeline
import threading
import time
import collections
import numpy as np

class RealTimeSpeechPipeline:
    def __init__(self, robot_speech_recognizer):
        self.recognizer = robot_speech_recognizer
        self.is_running = False
        self.pipeline_thread = None
        self.speech_buffer = collections.deque(maxlen=10)  # Store last 10 speech segments
        self.active_listening = False
        self.listening_callback = None

    def start_pipeline(self):
        """
        Start the real-time speech processing pipeline
        """
        self.is_running = True
        self.pipeline_thread = threading.Thread(target=self._pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()

    def stop_pipeline(self):
        """
        Stop the real-time speech processing pipeline
        """
        self.is_running = False
        if self.pipeline_thread:
            self.pipeline_thread.join()

    def set_callback(self, callback):
        """
        Set callback function for recognized speech
        """
        self.listening_callback = callback

    def _pipeline_loop(self):
        """
        Main processing loop for real-time speech
        """
        while self.is_running:
            if self.active_listening:
                try:
                    audio = self.recognizer.listen_for_speech()
                    if audio:
                        # Process speech recognition
                        recognized_text = self.recognizer.recognize_speech_google(audio)

                        if recognized_text:
                            # Add to buffer
                            self.speech_buffer.append(recognized_text)

                            # Call callback if set
                            if self.listening_callback:
                                self.listening_callback(recognized_text)

                            # Process for NLU
                            intent = self._process_nlu(recognized_text)

                            # Store in buffer
                            self.speech_buffer.append({
                                'text': recognized_text,
                                'intent': intent,
                                'timestamp': time.time()
                            })
                except Exception as e:
                    print(f"Error in speech pipeline: {e}")

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def activate_listening(self):
        """
        Activate the listening mode
        """
        self.active_listening = True

    def deactivate_listening(self):
        """
        Deactivate the listening mode
        """
        self.active_listening = False

    def _process_nlu(self, text):
        """
        Process Natural Language Understanding on recognized text
        """
        # Simple intent classification
        text_lower = text.lower()

        # Define intent patterns
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
        Get the most recent recognized speech segments
        """
        recent = list(self.speech_buffer)[-count:]
        return recent
```

## Natural Language Understanding (NLU)

### Intent Recognition and Classification

Natural Language Understanding systems must accurately classify user intents and extract relevant information:

```python
# Natural Language Understanding system
import re
from typing import Dict, List, Tuple
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

class RobotNLU:
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize intent classifier
        self.intent_classifier = MultinomialNB()
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.is_trained = False

        # Define entity patterns
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

        # Define intent templates for training
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

        # Train the classifier
        self._train_classifier()

    def _train_classifier(self):
        """
        Train the intent classifier with predefined data
        """
        texts = []
        labels = []

        for intent, examples in self.intent_training_data.items():
            for example in examples:
                texts.append(example)
                labels.append(intent)

        # Vectorize the texts
        X = self.tfidf_vectorizer.fit_transform(texts)

        # Train the classifier
        self.intent_classifier.fit(X, labels)
        self.is_trained = True

    def classify_intent(self, text: str) -> str:
        """
        Classify the intent of the given text
        """
        if not self.is_trained:
            return 'unknown'

        # Vectorize the input text
        X = self.tfidf_vectorizer.transform([text])

        # Predict the intent
        predicted_intent = self.intent_classifier.predict(X)[0]

        # Get prediction confidence
        prediction_probs = self.intent_classifier.predict_proba(X)[0]
        max_prob = max(prediction_probs)

        # Only return the prediction if confidence is high enough
        if max_prob > 0.3:  # Confidence threshold
            return predicted_intent
        else:
            return 'unknown'

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from the text
        """
        entities = {}

        # Use regex patterns to extract entities
        for entity_type, patterns in self.entity_patterns.items():
            entity_matches = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entity_matches.extend(matches)

            # Clean up matches
            cleaned_matches = [match.strip() for match in entity_matches if match.strip()]
            entities[entity_type] = list(set(cleaned_matches))  # Remove duplicates

        # If spaCy is available, use it for more sophisticated NER
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
        Parse a command into structured format with intent and entities
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
        Get confidence score for intent classification
        """
        if not self.is_trained:
            return 0.0

        X = self.tfidf_vectorizer.transform([text])
        prediction_probs = self.intent_classifier.predict_proba(X)[0]
        max_prob = max(prediction_probs)

        return float(max_prob)

    def process_contextual_command(self, text: str, context: Dict) -> Dict:
        """
        Process a command with additional context information
        """
        parsed = self.parse_command(text)

        # Enhance with context
        parsed['context'] = context

        # Resolve pronouns and references based on context
        if 'it' in text.lower() and context.get('last_object'):
            # Replace 'it' with the last mentioned object
            parsed['resolved_entities'] = {
                'object': [context['last_object']]
            }

        if 'there' in text.lower() and context.get('last_location'):
            # Replace 'there' with the last mentioned location
            parsed['resolved_entities'] = parsed.get('resolved_entities', {})
            parsed['resolved_entities']['location'] = [context['last_location']]

        return parsed
```

### Context-Aware Language Understanding

Implementing context-aware understanding for more natural interactions:

```python
# Context-aware language understanding
class ContextualNLU:
    def __init__(self):
        self.nlu = RobotNLU()
        self.context_memory = {}
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 interactions

    def process_with_context(self, text: str, robot_state: Dict, environment: Dict) -> Dict:
        """
        Process text with context awareness
        """
        # Build context
        context = {
            'robot_state': robot_state,
            'environment': environment,
            'conversation_history': self.conversation_history[-3:],  # Last 3 interactions
            'current_time': time.time(),
            'last_entities': self.context_memory.get('last_entities', {})
        }

        # Parse the command with context
        parsed = self.nlu.process_contextual_command(text, context)

        # Store in conversation history
        self.conversation_history.append({
            'text': text,
            'parsed': parsed,
            'timestamp': time.time()
        })

        # Limit history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # Update context memory
        self._update_context_memory(parsed)

        return parsed

    def _update_context_memory(self, parsed: Dict):
        """
        Update context memory with relevant information from parsed command
        """
        # Store last entities mentioned
        entities = parsed.get('entities', {})
        if entities:
            self.context_memory['last_entities'] = entities

            # Store specific entities
            if entities.get('OBJECT'):
                self.context_memory['last_object'] = entities['OBJECT'][0]
            if entities.get('LOCATION'):
                self.context_memory['last_location'] = entities['LOCATION'][0]
            if entities.get('PERSON'):
                self.context_memory['last_person'] = entities['PERSON'][0]

    def resolve_references(self, text: str) -> str:
        """
        Resolve pronouns and references in text based on context
        """
        resolved_text = text

        # Resolve 'it' based on last object
        if 'it' in text.lower() and self.context_memory.get('last_object'):
            resolved_text = resolved_text.replace('it', self.context_memory['last_object'])

        # Resolve 'there' based on last location
        if 'there' in text.lower() and self.context_memory.get('last_location'):
            resolved_text = resolved_text.replace('there', self.context_memory['last_location'])

        return resolved_text

    def get_context_summary(self) -> Dict:
        """
        Get a summary of the current context
        """
        return {
            'last_entities': self.context_memory.get('last_entities', {}),
            'conversation_history_count': len(self.conversation_history),
            'context_memory_keys': list(self.context_memory.keys())
        }
```

## Robust Speech Processing

### Noise Reduction and Filtering

Implementing noise reduction techniques for real-world environments:

```python
# Noise reduction and filtering for speech recognition
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, wiener
import librosa

class NoiseReductionSystem:
    def __init__(self):
        # Parameters for noise reduction
        self.sample_rate = 16000
        self.frame_length = 2048
        self.hop_length = 512
        self.noise_threshold = 0.01

    def spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction
        """
        # Compute STFT
        stft = librosa.stft(audio_data, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise spectrum (first 0.5 seconds as noise reference)
        noise_frames = int(0.5 * self.sample_rate / self.hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Apply spectral subtraction
        enhanced_magnitude = magnitude - self.noise_threshold * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0)  # Ensure non-negative

        # Reconstruct the signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)

        return enhanced_audio

    def wiener_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction
        """
        # Apply Wiener filter
        filtered_audio = wiener(audio_data)
        return filtered_audio

    def bandpass_filter(self, audio_data: np.ndarray, low_freq: float = 300, high_freq: float = 3400) -> np.ndarray:
        """
        Apply bandpass filter to focus on human speech frequencies
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Design Butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band', analog=False)

        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)

        return filtered_audio

    def voice_activity_detection(self, audio_data: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """
        Detect voice activity in audio signal
        """
        # Calculate energy of the signal
        energy = np.array([
            np.sum(np.abs(audio_data[i:i+self.hop_length]**2))
            for i in range(0, len(audio_data), self.hop_length)
        ])

        # Normalize energy
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy

        # Create voice activity mask
        voice_mask = energy > threshold

        # Reconstruct audio with only voice-active segments
        result = np.zeros_like(audio_data)
        for i, is_voice in enumerate(voice_mask):
            start_idx = i * self.hop_length
            end_idx = min(start_idx + self.hop_length, len(audio_data))
            if is_voice:
                result[start_idx:end_idx] = audio_data[start_idx:end_idx]

        return result

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to audio
        """
        # Apply bandpass filter
        filtered_audio = self.bandpass_filter(audio_data)

        # Apply noise reduction
        reduced_audio = self.spectral_subtraction(filtered_audio)

        # Apply Wiener filter
        final_audio = self.wiener_filter(reduced_audio)

        return final_audio
```

### Multi-Microphone Processing

Implementing multi-microphone processing for better speech recognition:

```python
# Multi-microphone processing for humanoid robots
import numpy as np
from scipy import signal
import pyaudio

class MultiMicrophoneProcessor:
    def __init__(self, num_mics=4, sample_rate=16000):
        self.num_mics = num_mics
        self.sample_rate = sample_rate
        self.audio = pyaudio.PyAudio()

        # Microphone positions (simplified - in practice, these would be calibrated)
        self.mic_positions = np.array([
            [-0.1, 0.0, 0.0],   # Left
            [0.1, 0.0, 0.0],    # Right
            [0.0, -0.1, 0.0],   # Front
            [0.0, 0.1, 0.0]     # Back
        ])

    def beamforming(self, multi_channel_audio: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        Apply beamforming to focus on a specific direction
        """
        if direction is None:
            # Default to front direction
            direction = np.array([0, -1, 0])  # Front direction

        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Calculate time delays for each microphone
        delays = []
        speed_of_sound = 343.0  # m/s
        mic_distance = 0.2  # m (approximate distance between mics)

        for pos in self.mic_positions:
            delay = np.dot(pos, direction) / speed_of_sound
            delays.append(delay)

        # Apply delays to align signals
        aligned_signals = []
        for i, audio_channel in enumerate(multi_channel_audio):
            delay_samples = int(delays[i] * self.sample_rate)
            if delay_samples > 0:
                # Apply delay by shifting
                delayed_signal = np.concatenate([np.zeros(delay_samples), audio_channel[:-delay_samples]])
            else:
                delayed_signal = audio_channel
            aligned_signals.append(delayed_signal)

        # Sum aligned signals for beamforming
        beamformed_signal = np.sum(aligned_signals, axis=0)

        return beamformed_signal

    def noise_suppression(self, multi_channel_audio: np.ndarray) -> np.ndarray:
        """
        Apply noise suppression using multiple microphone inputs
        """
        # Calculate spatial correlation between microphones
        correlations = []
        for i in range(self.num_mics):
            for j in range(i+1, self.num_mics):
                correlation = np.corrcoef(multi_channel_audio[i], multi_channel_audio[j])[0, 1]
                correlations.append(correlation)

        # Average correlation
        avg_correlation = np.mean(correlations)

        # Apply spatial filtering based on correlation
        if avg_correlation > 0.3:  # High correlation indicates coherent signal
            # Use beamforming to enhance coherent signal
            enhanced_signal = self.beamforming(multi_channel_audio)
        else:
            # Use single best microphone (highest SNR)
            snrs = [self._calculate_snr(channel) for channel in multi_channel_audio]
            best_mic_idx = np.argmax(snrs)
            enhanced_signal = multi_channel_audio[best_mic_idx]

        return enhanced_signal

    def _calculate_snr(self, audio_signal: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio for a signal
        """
        signal_power = np.mean(audio_signal ** 2)
        noise_power = np.var(audio_signal)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def adaptive_filtering(self, multi_channel_audio: np.ndarray) -> np.ndarray:
        """
        Apply adaptive filtering to reduce noise
        """
        # Use one microphone as reference for noise
        reference = multi_channel_audio[-1]  # Last mic as reference

        # Apply adaptive filtering to other microphones
        filtered_signals = []
        for i in range(self.num_mics - 1):
            # Create adaptive filter
            filtered = self._adaptive_filter(multi_channel_audio[i], reference)
            filtered_signals.append(filtered)

        # Combine filtered signals
        combined = np.mean(filtered_signals, axis=0)
        return combined

    def _adaptive_filter(self, primary_signal: np.ndarray, reference_signal: np.ndarray,
                         filter_length: int = 64) -> np.ndarray:
        """
        Apply adaptive filtering using LMS algorithm
        """
        # Initialize filter coefficients
        w = np.zeros(filter_length)
        mu = 0.01  # Learning rate

        output = np.zeros(len(primary_signal))
        error = np.zeros(len(primary_signal))

        for n in range(filter_length, len(primary_signal)):
            # Get reference input
            ref_input = reference_signal[n-filter_length:n][::-1]

            # Filter output
            y = np.dot(w, ref_input)

            # Error
            error[n] = primary_signal[n] - y

            # Update filter coefficients
            w = w + mu * error[n] * ref_input

            # Output
            output[n] = error[n]

        return output
```

## Integration with Robot Systems

### Speech-to-Action Pipeline

Creating a complete pipeline from speech recognition to robot action:

```python
# Complete speech-to-action pipeline
class SpeechToActionPipeline:
    def __init__(self, robot_interface):
        self.speech_recognizer = RobotSpeechRecognizer()
        self.nlu = RobotNLU()
        self.contextual_nlu = ContextualNLU()
        self.robot_interface = robot_interface
        self.pipeline = RealTimeSpeechPipeline(self.speech_recognizer)

        # Robot state and environment
        self.robot_state = {}
        self.environment = {}

    def start_listening(self):
        """
        Start the speech-to-action pipeline
        """
        # Set up callback for recognized speech
        def speech_callback(text):
            self.process_speech_command(text)

        self.pipeline.set_callback(speech_callback)
        self.pipeline.start_pipeline()
        self.pipeline.activate_listening()

        print("Speech-to-action pipeline started. Robot is listening...")

    def stop_listening(self):
        """
        Stop the speech-to-action pipeline
        """
        self.pipeline.deactivate_listening()
        self.pipeline.stop_pipeline()
        print("Speech-to-action pipeline stopped.")

    def process_speech_command(self, text: str):
        """
        Process a speech command through the entire pipeline
        """
        print(f"Processing speech command: {text}")

        # Process with contextual NLU
        parsed_command = self.contextual_nlu.process_with_context(
            text,
            self.robot_state,
            self.environment
        )

        # Execute based on intent
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
        Execute navigation command
        """
        target_locations = entities.get('LOCATION', [])

        if target_locations:
            target = target_locations[0]
            print(f"Robot navigating to: {target}")

            # Check if location exists in environment
            if target in self.environment.get('locations', []):
                success = self.robot_interface.navigate_to_location(target)
                if success:
                    print(f"Successfully navigated to {target}")
                else:
                    print(f"Failed to navigate to {target}")
            else:
                print(f"Unknown location: {target}. Available locations: {self.environment.get('locations', [])}")
        else:
            print("No target location specified in navigation command")

    def _execute_manipulation(self, entities: Dict):
        """
        Execute manipulation command
        """
        target_objects = entities.get('OBJECT', [])

        if target_objects:
            target = target_objects[0]
            print(f"Robot attempting to manipulate: {target}")

            # Check if object exists in environment
            if target in self.environment.get('objects', []):
                success = self.robot_interface.manipulate_object(target)
                if success:
                    print(f"Successfully manipulated {target}")
                else:
                    print(f"Failed to manipulate {target}")
            else:
                print(f"Unknown object: {target}. Available objects: {self.environment.get('objects', [])}")
        else:
            print("No target object specified in manipulation command")

    def _execute_greeting(self):
        """
        Execute greeting action
        """
        print("Robot executing greeting")
        self.robot_interface.perform_greeting()

    def _execute_information_request(self, text: str, entities: Dict):
        """
        Execute information request
        """
        print(f"Robot processing information request: {text}")

        # Check for specific information requests
        text_lower = text.lower()

        if 'time' in text_lower:
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M")
            response = f"The current time is {current_time}"
        elif 'name' in text_lower or 'you' in text_lower:
            response = "I am your humanoid robot assistant. You can call me Assistant."
        elif 'location' in text_lower or 'where' in text_lower and 'you' in text_lower:
            location = self.robot_state.get('location', 'an unknown location')
            response = f"I am currently at {location}"
        elif 'capabilities' in text_lower or 'can you' in text_lower:
            capabilities = self.robot_state.get('capabilities', [])
            capability_list = ', '.join(capabilities) if capabilities else 'I can assist with various tasks'
            response = f"I can perform the following tasks: {capability_list}"
        else:
            response = "I can provide information about time, my name, location, and capabilities. How else can I help?"

        print(f"Robot response: {response}")
        self.robot_interface.speak_text(response)

    def _execute_stop(self):
        """
        Execute stop command
        """
        print("Robot executing stop command")
        self.robot_interface.stop_current_action()

    def _execute_follow(self, entities: Dict):
        """
        Execute follow command
        """
        target_persons = entities.get('PERSON', [])

        if target_persons:
            target = target_persons[0]
            print(f"Robot attempting to follow: {target}")
            success = self.robot_interface.follow_person(target)
            if success:
                print(f"Successfully started following {target}")
            else:
                print(f"Failed to follow {target}")
        else:
            print("Robot following default behavior (follow speaker)")
            success = self.robot_interface.follow_person("speaker")
            if success:
                print("Successfully started following speaker")

    def _execute_generic_action(self, entities: Dict):
        """
        Execute generic action based on entities
        """
        print(f"Robot executing generic action with entities: {entities}")
        # This could trigger various actions based on context
        self.robot_interface.perform_generic_action(entities)

    def _execute_unknown_command(self):
        """
        Handle unknown command
        """
        response = "I'm sorry, I didn't understand that command. Could you please rephrase?"
        print(f"Robot response: {response}")
        self.robot_interface.speak_text(response)

    def update_robot_state(self, new_state: Dict):
        """
        Update robot state information
        """
        self.robot_state.update(new_state)

    def update_environment(self, new_environment: Dict):
        """
        Update environment information
        """
        self.environment.update(new_environment)
```

## Performance Optimization

### Real-Time Processing Considerations

Optimizing speech recognition for real-time performance on humanoid robots:

```python
# Performance optimization for speech processing
import time
import threading
from queue import Queue, Empty
import psutil

class OptimizedSpeechProcessor:
    def __init__(self):
        self.input_queue = Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.output_queue = Queue(maxsize=10)
        self.is_running = False
        self.processing_thread = None
        self.cpu_threshold = 80  # CPU usage threshold for optimization
        self.processing_delay = 0.0  # Processing delay for rate limiting

        # Initialize components
        self.speech_recognizer = RobotSpeechRecognizer()
        self.nlu = RobotNLU()

    def start_processing(self):
        """
        Start the optimized processing pipeline
        """
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """
        Stop the optimized processing pipeline
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()

    def add_audio_input(self, audio_data):
        """
        Add audio input to the processing queue
        """
        try:
            self.input_queue.put_nowait(audio_data)
            return True
        except:
            # Queue is full, drop the input
            return False

    def get_processed_output(self):
        """
        Get processed output from the queue
        """
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None

    def _processing_loop(self):
        """
        Main processing loop with performance optimization
        """
        while self.is_running:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Adjust processing based on CPU usage
                if cpu_percent > self.cpu_threshold:
                    # Reduce processing rate if CPU is high
                    time.sleep(0.1)
                    continue

                # Get audio input
                try:
                    audio_data = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Process the audio
                start_time = time.time()

                # Recognize speech
                recognized_text = self.speech_recognizer.recognize_speech_google(audio_data)

                if recognized_text:
                    # Process with NLU
                    parsed_result = self.nlu.parse_command(recognized_text)

                    # Add processing time information
                    processing_time = time.time() - start_time
                    parsed_result['processing_time'] = processing_time

                    # Put result in output queue
                    try:
                        self.output_queue.put_nowait(parsed_result)
                    except:
                        # Output queue is full, drop the result
                        pass

                # Rate limiting to prevent excessive CPU usage
                processing_time = time.time() - start_time
                if processing_time < 0.01:  # Minimum processing time
                    time.sleep(0.01 - processing_time)

            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)  # Brief pause before continuing

    def get_performance_metrics(self):
        """
        Get performance metrics for the processor
        """
        return {
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'is_running': self.is_running
        }
```

## Hands-On Exercise: Implementing Speech Recognition System

### Exercise Objectives
- Set up a speech recognition pipeline for a humanoid robot
- Implement Natural Language Understanding for command interpretation
- Test the system with various speech inputs
- Evaluate recognition accuracy and response time

### Step-by-Step Instructions

1. **Initialize the speech recognition system** with proper microphone setup
2. **Implement the NLU component** for intent classification and entity extraction
3. **Create the integration pipeline** connecting speech recognition to robot actions
4. **Test with various speech inputs** and evaluate performance
5. **Optimize for real-time performance** and noise reduction
6. **Analyze results** and refine the system

### Expected Outcomes
- Working speech recognition system for humanoid robot
- Understanding of NLU implementation
- Experience with real-time processing optimization
- Performance evaluation skills

## Knowledge Check

1. What are the key challenges in implementing speech recognition for humanoid robots in real-world environments?
2. Explain the difference between Automatic Speech Recognition (ASR) and Natural Language Understanding (NLU).
3. How does context-aware language understanding improve robot interactions?
4. What techniques can be used to improve speech recognition accuracy in noisy environments?

## Summary

This chapter covered the implementation of speech recognition and Natural Language Understanding systems for humanoid robots. We explored the fundamentals of ASR, implemented NLU for intent classification and entity extraction, addressed noise reduction techniques, and created a complete pipeline from speech input to robot action. The integration of robust speech processing systems enables more natural and intuitive human-robot interaction, making humanoid robots more accessible and useful in everyday applications.

## Next Steps

In Chapter 19, we'll examine Cognitive Planning with LLMs, exploring how Large Language Models can be used for high-level reasoning and planning in humanoid robotic systems, enabling more sophisticated autonomous behaviors.

