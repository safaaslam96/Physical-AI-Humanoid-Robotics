---
sidebar_position: 16
title: "Chapter 16: Natural Human-Robot Interaction"
---

# Chapter 16: Natural Human-Robot Interaction

## Learning Objectives
- Design natural communication paradigms for human-robot interaction
- Implement user experience optimization for humanoid robots
- Master advanced interaction techniques for humanoid robotics
- Understand social robotics principles and best practices

## Introduction to Natural Human-Robot Interaction

Natural Human-Robot Interaction (NHRI) is the study and design of interfaces and behaviors that allow humans to interact with robots in intuitive, familiar ways. For humanoid robots, this involves creating interactions that feel natural to humans by leveraging our understanding of human communication patterns, social cues, and expectations.

### Key Principles of Natural Interaction

1. **Intuitive Communication**: Using familiar modalities like speech, gestures, and facial expressions
2. **Context Awareness**: Understanding the environment and situation
3. **Social Norms**: Following human social conventions and expectations
4. **Predictability**: Behaving in ways that humans can anticipate
5. **Feedback**: Providing clear, timely responses to human actions
6. **Safety**: Ensuring interactions are safe and comfortable

### Communication Modalities

Human communication is multi-modal, involving:
- **Verbal**: Speech and language
- **Non-verbal**: Gestures, facial expressions, posture
- **Paralinguistic**: Tone, pitch, volume
- **Spatial**: Proxemics and personal space
- **Temporal**: Timing and rhythm of interactions

## Communication Paradigms

### Voice-Based Interaction

```python
# voice_interaction.py
import speech_recognition as sr
import pyttsx3
import asyncio
import threading
from queue import Queue
import time

class VoiceInteractionManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()

        # Configuration
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level

        # Voice interaction state
        self.is_listening = False
        self.conversation_context = {}
        self.response_queue = Queue()
        self.command_handlers = {}

        # Initialize speech recognition
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.setup_command_handlers()

    def setup_command_handlers(self):
        """Setup command handlers for different robot capabilities"""
        self.command_handlers.update({
            'move': self.handle_move_command,
            'grasp': self.handle_grasp_command,
            'navigation': self.handle_navigation_command,
            'information': self.handle_information_request,
            'social': self.handle_social_interaction
        })

    def start_voice_system(self):
        """Start the voice interaction system"""
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self.voice_loop, daemon=True)
        self.listen_thread.start()

    def voice_loop(self):
        """Main voice interaction loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for speech with timeout
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=5.0)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                print(f"Heard: {text}")

                # Process the command
                self.process_voice_command(text)

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except sr.UnknownValueError:
                self.speak_response("Sorry, I didn't understand that.")
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                self.speak_response("Sorry, I'm having trouble understanding.")

    def process_voice_command(self, text):
        """Process a voice command using NLP techniques"""
        # Simple command parsing (in practice, use more sophisticated NLP)
        text_lower = text.lower()

        # Check for wake word (optional)
        if 'robot' in text_lower or 'hey' in text_lower:
            # Extract the actual command
            command_start = max(text_lower.find('robot') + 6, text_lower.find('hey') + 4)
            command = text_lower[command_start:].strip()
        else:
            command = text_lower

        # Classify command type
        command_type = self.classify_command(command)

        if command_type in self.command_handlers:
            response = self.command_handlers[command_type](command)
            if response:
                self.speak_response(response)
        else:
            self.speak_response(f"I can help with movement, grasping, navigation, information, or social interaction. What would you like to do?")

    def classify_command(self, command):
        """Classify command type based on keywords"""
        if any(word in command for word in ['move', 'go', 'walk', 'navigate', 'to', 'toward']):
            return 'navigation'
        elif any(word in command for word in ['grasp', 'pick', 'take', 'grab', 'hold', 'lift']):
            return 'grasp'
        elif any(word in command for word in ['move', 'step', 'forward', 'backward', 'left', 'right']):
            return 'move'
        elif any(word in command for word in ['what', 'how', 'when', 'where', 'who', 'tell me', 'information']):
            return 'information'
        elif any(word in command for word in ['hello', 'hi', 'good', 'morning', 'afternoon', 'evening', 'bye', 'goodbye', 'nice', 'meet']):
            return 'social'
        else:
            return 'unknown'

    def handle_navigation_command(self, command):
        """Handle navigation-related commands"""
        # Extract destination from command
        destinations = ['kitchen', 'living room', 'bedroom', 'office', 'dining room']

        for dest in destinations:
            if dest in command:
                # Simulate navigation
                self.speak_response(f"Okay, I'm navigating to the {dest}. Please follow me.")
                # In practice, this would trigger navigation system
                return f"Moving toward {dest}"

        # If no specific destination, ask for clarification
        return "Where would you like me to navigate to?"

    def handle_grasp_command(self, command):
        """Handle grasping-related commands"""
        # Extract object reference
        objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'box']

        for obj in objects:
            if obj in command:
                # Simulate grasping action
                self.speak_response(f"Okay, I'll grasp the {obj} for you.")
                # In practice, this would trigger grasping system
                return f"Attempting to grasp {obj}"

        return "What object would you like me to grasp?"

    def handle_move_command(self, command):
        """Handle movement commands"""
        if 'forward' in command or 'ahead' in command:
            self.speak_response("Moving forward.")
            return "Moving forward"
        elif 'backward' in command or 'back' in command:
            self.speak_response("Moving backward.")
            return "Moving backward"
        elif 'left' in command:
            self.speak_response("Turning left.")
            return "Turning left"
        elif 'right' in command:
            self.speak_response("Turning right.")
            return "Turning right"
        else:
            return "In what direction would you like me to move?"

    def handle_information_request(self, command):
        """Handle information requests"""
        if 'time' in command:
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M")
            return f"The current time is {current_time}."
        elif 'date' in command:
            import datetime
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            return f"Today's date is {current_date}."
        elif 'weather' in command:
            return "I don't have access to weather information right now, but I can help you find it if you'd like."
        else:
            return "I can tell you the time, date, or help with other information. What would you like to know?"

    def handle_social_interaction(self, command):
        """Handle social interaction commands"""
        if 'hello' in command or 'hi' in command:
            return "Hello! It's nice to meet you. How can I help you today?"
        elif 'bye' in command or 'goodbye' in command:
            return "Goodbye! It was nice interacting with you."
        elif 'thank' in command:
            return "You're welcome! I'm happy to help."
        elif 'name' in command:
            return "I'm your humanoid robot assistant. You can call me ARIA - Autonomous Robot Interaction Assistant."
        else:
            return "Hello! How can I assist you today?"

    def speak_response(self, text):
        """Generate speech response"""
        print(f"Robot says: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def get_response(self, text):
        """Get response without speaking (for internal use)"""
        command_type = self.classify_command(text.lower())
        if command_type in self.command_handlers:
            return self.command_handlers[command_type](text.lower())
        return "I'm not sure how to respond to that."

    def stop_voice_system(self):
        """Stop the voice interaction system"""
        self.is_listening = False

class ConversationalContextManager:
    """Manage conversation context and state"""
    def __init__(self):
        self.context = {
            'current_topic': None,
            'previous_utterances': [],
            'user_preferences': {},
            'task_state': {},
            'time_started': time.time()
        }

    def update_context(self, user_input, robot_response):
        """Update conversation context"""
        self.context['previous_utterances'].append({
            'user': user_input,
            'robot': robot_response,
            'timestamp': time.time()
        })

        # Keep only recent utterances
        if len(self.context['previous_utterances']) > 10:
            self.context['previous_utterances'] = self.context['previous_utterances'][-10:]

    def get_context_summary(self):
        """Get summary of current context"""
        return {
            'topic': self.context['current_topic'],
            'conversation_length': len(self.context['previous_utterances']),
            'duration': time.time() - self.context['time_started'],
            'recent_utterances': self.context['previous_utterances'][-3:] if self.context['previous_utterances'] else []
        }

    def infer_user_intent(self, current_input):
        """Infer user intent based on context"""
        # Simple intent inference based on keywords and context
        recent_context = ' '.join([u['user'] for u in self.context['previous_utterances'][-3:]])

        combined_input = f"{recent_context} {current_input}".lower()

        if any(word in combined_input for word in ['stop', 'cancel', 'abort']):
            return 'cancel_task'
        elif any(word in combined_input for word in ['repeat', 'again', 'more']):
            return 'repeat_action'
        elif any(word in combined_input for word in ['different', 'change', 'other']):
            return 'change_approach'
        else:
            return 'new_request'

# Example usage
def example_voice_interaction():
    voice_manager = VoiceInteractionManager()
    context_manager = ConversationalContextManager()

    # Simulate some voice commands
    test_commands = [
        "Hello robot",
        "What time is it?",
        "Navigate to the kitchen",
        "Grasp the cup",
        "Thank you"
    ]

    for command in test_commands:
        print(f"\nUser says: {command}")
        response = voice_manager.get_response(command)
        print(f"Robot response: {response}")

        # Update context
        context_manager.update_context(command, response)

        # Show context summary
        summary = context_manager.get_context_summary()
        print(f"Context: Topic={summary['topic']}, Length={summary['conversation_length']}")

if __name__ == "__main__":
    try:
        example_voice_interaction()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install SpeechRecognition pyttsx3")
```

### Gesture-Based Interaction

```python
# gesture_interaction.py
import cv2
import mediapipe as mp
import numpy as np
from enum import Enum

class GestureType(Enum):
    WAVE = "wave"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINTING = "pointing"
    STOP = "stop"
    COME_HERE = "come_here"
    FOLLOW_ME = "follow_me"
    GRASP = "grasp"
    RELEASE = "release"

class GestureRecognitionManager:
    def __init__(self):
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Gesture recognition state
        self.previous_gestures = []
        self.gesture_buffer_size = 5
        self.confidence_threshold = 0.8

    def process_frame(self, frame):
        """Process a video frame for gesture recognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.hands.process(rgb_frame)

        gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Recognize gesture from hand landmarks
                gesture = self.recognize_gesture(hand_landmarks, results.multi_handedness)
                if gesture:
                    gestures.append(gesture)

                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        # Update gesture history
        self.previous_gestures.extend(gestures)
        if len(self.previous_gestures) > self.gesture_buffer_size:
            self.previous_gestures = self.previous_gestures[-self.gesture_buffer_size:]

        return frame, gestures

    def recognize_gesture(self, hand_landmarks, handedness):
        """Recognize specific gesture from hand landmarks"""
        # Get landmark coordinates
        landmarks = hand_landmarks.landmark

        # Determine if this is left or right hand
        is_right = handedness[0].classification[0].label == 'Right'

        # Calculate distances between key points
        thumb_tip = np.array([landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x,
                             landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                             landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        middle_tip = np.array([landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                              landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        ring_tip = np.array([landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].x,
                            landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y])
        pinky_tip = np.array([landmarks[self.mp_hands.HandLandmark.PINKY_TIP].x,
                             landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y])

        wrist = np.array([landmarks[self.mp_hands.HandLandmark.WRIST].x,
                         landmarks[self.mp_hands.HandLandmark.WRIST].y])

        # Calculate distances
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        middle_ring_dist = np.linalg.norm(middle_tip - ring_tip)
        ring_pinky_dist = np.linalg.norm(ring_tip - pinky_tip)

        # Calculate angles and positions
        index_direction = index_tip - wrist
        middle_direction = middle_tip - wrist
        thumb_direction = thumb_tip - wrist

        # Recognize specific gestures
        gesture = self.classify_gesture(
            thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, wrist,
            thumb_index_dist, index_middle_dist, middle_ring_dist, ring_pinky_dist,
            index_direction, middle_direction, thumb_direction, is_right
        )

        return gesture

    def classify_gesture(self, thumb, index, middle, ring, pinky, wrist,
                        d_thumb_index, d_index_middle, d_middle_ring, d_ring_pinky,
                        index_dir, middle_dir, thumb_dir, is_right):
        """Classify gesture based on finger positions and distances"""

        # WAVE: Hand moving side to side
        # This would be detected by tracking movement over frames
        # For static detection, we'll focus on pose-based gestures

        # THUMBS_UP: Thumb up, other fingers closed
        if (thumb[1] < index[1] and  # Thumb higher than index finger
            d_index_middle < 0.1 and  # Index and middle fingers close together
            d_middle_ring < 0.1 and   # Middle and ring fingers close together
            d_ring_pinky < 0.1):      # Ring and pinky fingers close together
            return GestureType.THUMBS_UP

        # THUMBS_DOWN: Thumb down, other fingers closed
        elif (thumb[1] > index[1] and  # Thumb lower than index finger
              d_index_middle < 0.1 and
              d_middle_ring < 0.1 and
              d_ring_pinky < 0.1):
            return GestureType.THUMBS_DOWN

        # STOP: Open palm facing robot
        elif (abs(index[0] - wrist[0]) > 0.1 and  # Fingers extended
              abs(middle[0] - wrist[0]) > 0.1 and
              abs(ring[0] - wrist[0]) > 0.1 and
              abs(pinky[0] - wrist[0]) > 0.1 and
              index[1] < wrist[1] and  # Hand upright
              middle[1] < wrist[1] and
              ring[1] < wrist[1] and
              pinky[1] < wrist[1]):
            return GestureType.STOP

        # POINTING: Index finger extended, others closed
        elif (np.linalg.norm(index - wrist) > np.linalg.norm(thumb - wrist) and  # Index finger extended
              d_index_middle > 0.15 and  # Index and middle fingers apart
              d_middle_ring < 0.1 and    # Other fingers together
              d_ring_pinky < 0.1):
            return GestureType.POINTING

        # GRASP: All fingers curled as if grasping
        elif (d_thumb_index < 0.05 and  # Thumb and index close
              d_index_middle < 0.05 and  # All fingers close together
              d_middle_ring < 0.05 and
              d_ring_pinky < 0.05):
            return GestureType.GRASP

        # RELEASE: Hand open as if releasing
        elif (d_thumb_index > 0.15 and  # Fingers spread
              d_index_middle > 0.15 and
              d_middle_ring > 0.15 and
              d_ring_pinky > 0.15):
            return GestureType.RELEASE

        # COME HERE: Index finger pointing toward robot
        elif (np.linalg.norm(index - wrist) > 0.1 and  # Index extended
              d_index_middle > 0.1 and   # Index separate
              d_middle_ring < 0.05 and   # Others together
              d_ring_pinky < 0.05 and
              index[0] < wrist[0] and    # Pointing toward robot (assuming right hand)
              is_right):
            return GestureType.COME_HERE

        # FOLLOW ME: Hand moving in guiding motion
        # This would require tracking movement over time
        # For now, we'll use a specific hand shape
        elif (abs(index[0] - middle[0]) < 0.05 and  # Index and middle together
              abs(middle[0] - ring[0]) < 0.05 and   # All fingers together
              abs(ring[0] - pinky[0]) < 0.05 and
              index[1] < wrist[1] and               # Hand upright
              middle[1] < wrist[1] and
              ring[1] < wrist[1] and
              pinky[1] < wrist[1] and
              thumb[0] > wrist[0]):                 # Thumb on outside
            return GestureType.FOLLOW_ME

        return None

    def get_gesture_meaning(self, gesture_type):
        """Get the meaning/action associated with a gesture"""
        meanings = {
            GestureType.WAVE: "Greeting/Acknowledgment",
            GestureType.THUMBS_UP: "Approval/Confirmation",
            GestureType.THUMBS_DOWN: "Disapproval/Denial",
            GestureType.POINTING: "Directing attention/Indicating location",
            GestureType.STOP: "Halt/Wait/Stop action",
            GestureType.COME_HERE: "Approach user",
            GestureType.FOLLOW_ME: "Follow user",
            GestureType.GRASP: "Prepare to grasp object",
            GestureType.RELEASE: "Release object/Stop grasping"
        }
        return meanings.get(gesture_type, "Unknown gesture")

class GestureInteractionController:
    """Controller for gesture-based robot interaction"""
    def __init__(self):
        self.gesture_manager = GestureRecognitionManager()
        self.robot_actions = {
            GestureType.COME_HERE: self.move_towards_user,
            GestureType.FOLLOW_ME: self.follow_user,
            GestureType.STOP: self.stop_robot,
            GestureType.GRASP: self.prepare_grasp,
            GestureType.RELEASE: self.release_object,
            GestureType.THUMBS_UP: self.confirm_action,
            GestureType.THUMBS_DOWN: self.reject_action
        }
        self.current_task = None
        self.user_position = None

    def process_gesture(self, gesture_type):
        """Process a recognized gesture"""
        if gesture_type in self.robot_actions:
            action = self.robot_actions[gesture_type]
            result = action()
            return result
        else:
            print(f"Gesture {gesture_type} not mapped to any action")
            return None

    def move_towards_user(self):
        """Move robot towards user"""
        print("Moving towards user...")
        # In practice, this would use navigation system
        return "Moving towards user"

    def follow_user(self):
        """Follow user to a destination"""
        print("Following user...")
        # In practice, this would start person-following behavior
        return "Following user"

    def stop_robot(self):
        """Stop current robot action"""
        print("Stopping robot...")
        # In practice, this would stop any ongoing motion
        return "Robot stopped"

    def prepare_grasp(self):
        """Prepare for grasping action"""
        print("Preparing to grasp...")
        # In practice, this would ready the manipulation system
        return "Grasp preparation initiated"

    def release_object(self):
        """Release currently held object"""
        print("Releasing object...")
        # In practice, this would open grippers
        return "Object released"

    def confirm_action(self):
        """Confirm current action"""
        print("Action confirmed")
        return "Action confirmed"

    def reject_action(self):
        """Reject current action"""
        print("Action rejected")
        return "Action rejected"

    def run_gesture_interaction(self):
        """Run gesture-based interaction loop"""
        cap = cv2.VideoCapture(0)

        print("Gesture interaction started. Show gestures to control the robot.")
        print("Available gestures: Stop, Come Here, Follow Me, Grasp, Release, Thumbs Up/Down")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame for gestures
            processed_frame, gestures = self.gesture_manager.process_frame(frame)

            # Process recognized gestures
            for gesture in gestures:
                print(f"Recognized gesture: {gesture}")
                meaning = self.gesture_manager.get_gesture_meaning(gesture)
                print(f"Meaning: {meaning}")

                # Execute corresponding action
                result = self.process_gesture(gesture)
                if result:
                    print(f"Action result: {result}")

            # Display frame
            cv2.imshow('Gesture Recognition', processed_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
def example_gesture_interaction():
    controller = GestureInteractionController()

    # Simulate gesture recognition and processing
    print("Gesture interaction examples:")

    # Test some gestures
    test_gestures = [
        GestureType.COME_HERE,
        GestureType.STOP,
        GestureType.GRASP,
        GestureType.THUMBS_UP
    ]

    for gesture in test_gestures:
        print(f"\nProcessing gesture: {gesture}")
        meaning = controller.gesture_manager.get_gesture_meaning(gesture)
        print(f"Meaning: {meaning}")
        result = controller.process_gesture(gesture)
        print(f"Result: {result}")

if __name__ == "__main__":
    try:
        example_gesture_interaction()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install opencv-python mediapipe")
```

### Visual Interaction and Social Cues

```python
# visual_interaction.py
import cv2
import numpy as np
import math
from enum import Enum

class SocialDistance(Enum):
    INTIMATE = (0.0, 0.45)    # 0-1.5 feet - very close
    PERSONAL = (0.45, 1.2)    # 1.5-4 feet - friends, family
    SOCIAL = (1.2, 3.6)       # 4-12 feet - social interactions
    PUBLIC = (3.6, 10.0)      # 12+ feet - public speaking

class VisualInteractionManager:
    def __init__(self):
        self.eye_contact_enabled = True
        self.head_movement_enabled = True
        self.social_distance_preference = SocialDistance.PERSONAL
        self.user_tracking = True
        self.face_detection_enabled = True

        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Robot head parameters
        self.head_position = np.array([0.0, 0.0, 1.7])  # Robot head position (x, y, z)
        self.current_gaze_target = None
        self.head_orientation = np.array([0.0, 0.0, 0.0])  # Pitch, yaw, roll

    def detect_faces(self, frame):
        """Detect faces in the input frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def track_user_attention(self, frame):
        """Track user attention and engagement"""
        faces = self.detect_faces(frame)

        if len(faces) > 0:
            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            # Calculate face center
            face_center = (x + w//2, y + h//2)

            # Draw face rectangle and center
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, face_center, 5, (0, 255, 0), -1)

            # Calculate distance to face (rough estimate based on size)
            face_size = max(w, h)
            estimated_distance = self.estimate_distance_from_size(face_size, frame.shape[1])

            # Check if user is in appropriate social distance
            appropriate_distance = self.check_social_distance(estimated_distance)

            return {
                'face_detected': True,
                'face_center': face_center,
                'distance': estimated_distance,
                'appropriate_distance': appropriate_distance,
                'engagement_level': self.calculate_engagement_level(faces, frame)
            }

        return {
            'face_detected': False,
            'engagement_level': 0.0
        }

    def estimate_distance_from_size(self, face_width_pixels, frame_width_pixels):
        """Estimate distance to face based on face size in frame"""
        # This is a simplified estimation
        # In practice, you'd use calibrated camera parameters
        # and known face size (average face width ~20cm)
        known_face_width = 0.2  # meters
        focal_length = 800  # pixels (typical for webcams)

        estimated_distance = (known_face_width * focal_length) / face_width_pixels
        return estimated_distance

    def check_social_distance(self, distance):
        """Check if distance is appropriate for current interaction"""
        min_dist, max_dist = self.social_distance_preference.value
        return min_dist <= distance <= max_dist

    def calculate_engagement_level(self, faces, frame):
        """Calculate user engagement level based on multiple factors"""
        if len(faces) == 0:
            return 0.0

        # Engagement factors:
        # 1. Number of people (more people = more engagement)
        num_people = len(faces)

        # 2. Average face size (closer = more engaged)
        avg_face_size = np.mean([max(f[2], f[3]) for f in faces])
        size_factor = min(avg_face_size / 200, 1.0)  # Normalize

        # 3. Face positions (central faces = more engaged)
        frame_center = np.array([frame.shape[1]/2, frame.shape[0]/2])
        position_scores = []
        for (x, y, w, h) in faces:
            face_center = np.array([x + w/2, y + h/2])
            distance_to_center = np.linalg.norm(frame_center - face_center)
            max_distance = np.linalg.norm(frame_center)
            position_score = 1 - (distance_to_center / max_distance)
            position_scores.append(position_score)

        avg_position_score = np.mean(position_scores) if position_scores else 0.0

        # Combine factors
        engagement = (0.3 * min(num_people / 3, 1.0) +
                     0.4 * size_factor +
                     0.3 * avg_position_score)

        return min(engagement, 1.0)

    def maintain_eye_contact(self, face_center, frame_shape):
        """Adjust head orientation to maintain eye contact"""
        if face_center is None:
            return

        frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)

        # Calculate the angle difference
        dx = face_center[0] - frame_center[0]
        dy = frame_center[1] - face_center[1]  # Inverted because y increases downward

        # Convert to head movement (simplified)
        max_head_movement = 30  # degrees
        x_angle = max(-max_head_movement, min(max_head_movement, dx * max_head_movement / (frame_shape[1]/2)))
        y_angle = max(-max_head_movement, min(max_head_movement, dy * max_head_movement / (frame_shape[0]/2)))

        # Update head orientation
        self.head_orientation[1] = math.radians(x_angle)  # Yaw
        self.head_orientation[0] = math.radians(y_angle)  # Pitch

    def generate_head_movement(self, engagement_level, base_movement=True):
        """Generate natural head movements based on engagement"""
        movements = []

        if base_movement:
            # Subtle head nods to show attention
            if engagement_level > 0.3:
                movements.append({
                    'type': 'nod',
                    'amplitude': 0.1 * engagement_level,
                    'frequency': 0.5,
                    'duration': 0.5
                })

        # Head tilts when listening
        if engagement_level > 0.6:
            movements.append({
                'type': 'tilt',
                'amplitude': 0.05,
                'frequency': 0.3,
                'direction': 'random'  # Left or right
            })

        return movements

    def display_social_feedback(self, frame, interaction_data):
        """Display social feedback on the frame"""
        if interaction_data['face_detected']:
            face_center = interaction_data['face_center']

            # Draw social distance indicator
            color = (0, 255, 0) if interaction_data['appropriate_distance'] else (0, 0, 255)
            cv2.circle(frame, face_center, 50, color, 2)

            # Display engagement level
            cv2.putText(frame, f'Engagement: {interaction_data["engagement_level"]:.2f}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display distance
            cv2.putText(frame, f'Distance: {interaction_data["distance"]:.2f}m',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display social distance zone
            zone_text = self.get_social_distance_zone(interaction_data['distance'])
            cv2.putText(frame, f'Zone: {zone_text}',
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    def get_social_distance_zone(self, distance):
        """Get the social distance zone for a given distance"""
        for zone in SocialDistance:
            min_dist, max_dist = zone.value
            if min_dist <= distance <= max_dist:
                return zone.name
        return 'UNKNOWN'

class SocialBehaviorController:
    """Controller for social behaviors and responses"""
    def __init__(self):
        self.visual_manager = VisualInteractionManager()
        self.engagement_threshold = 0.5
        self.response_delay = 1.0  # seconds
        self.last_response_time = 0

    def process_social_interaction(self, frame):
        """Process social interaction in the current frame"""
        # Track user attention
        interaction_data = self.visual_manager.track_user_attention(frame)

        # Update eye contact if enabled
        if self.visual_manager.eye_contact_enabled and interaction_data['face_detected']:
            self.visual_manager.maintain_eye_contact(
                interaction_data['face_center'],
                frame.shape
            )

        # Generate social feedback
        frame_with_feedback = self.visual_manager.display_social_feedback(frame, interaction_data)

        # Check for appropriate response timing
        current_time = time.time()
        if (current_time - self.last_response_time > self.response_delay and
            interaction_data['engagement_level'] > self.engagement_threshold):

            # Generate appropriate social response
            response = self.generate_social_response(interaction_data)
            self.last_response_time = current_time

            # In practice, this would trigger robot actions
            print(f"Social response: {response}")

        # Generate head movements based on engagement
        if self.visual_manager.head_movement_enabled:
            head_movements = self.visual_manager.generate_head_movement(
                interaction_data['engagement_level']
            )
            # Apply head movements to robot (simulated here)
            self.apply_head_movements(head_movements)

        return frame_with_feedback, interaction_data

    def generate_social_response(self, interaction_data):
        """Generate appropriate social response based on interaction data"""
        if interaction_data['engagement_level'] > 0.8:
            responses = [
                "Hello! It's great to see you!",
                "I'm happy you're here!",
                "How can I help you today?"
            ]
        elif interaction_data['engagement_level'] > 0.5:
            responses = [
                "Hello there!",
                "Hi! How are you doing?",
                "Good to see you!"
            ]
        else:
            responses = [
                "Hello!",
                "Hi!",
                "Welcome!"
            ]

        import random
        return random.choice(responses)

    def apply_head_movements(self, movements):
        """Apply head movements to robot (simulated)"""
        for movement in movements:
            if movement['type'] == 'nod':
                print(f"Head nod: amplitude={movement['amplitude']}, frequency={movement['frequency']}")
            elif movement['type'] == 'tilt':
                print(f"Head tilt: amplitude={movement['amplitude']}, direction={movement['direction']}")

    def adjust_social_distance(self, preferred_zone):
        """Adjust preferred social distance zone"""
        if preferred_zone in SocialDistance.__members__:
            self.visual_manager.social_distance_preference = SocialDistance[preferred_zone]
            print(f"Social distance adjusted to {preferred_zone} zone")

# Example usage
def example_visual_interaction():
    social_controller = SocialBehaviorController()

    # Simulate video capture
    cap = cv2.VideoCapture(0)

    print("Starting social interaction simulation...")
    print("Show faces to the camera to test engagement detection and social responses")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process social interaction
        processed_frame, interaction_data = social_controller.process_social_interaction(frame)

        # Display the frame
        cv2.imshow('Social Interaction', processed_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import time
    try:
        example_visual_interaction()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install opencv-python")
```

## User Experience Design

### Interaction Flow Design

```python
# interaction_flow.py
import time
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Callable

class InteractionState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class InteractionModality(Enum):
    SPEECH = "speech"
    GESTURE = "gesture"
    TOUCH = "touch"
    VISUAL = "visual"
    MULTIMODAL = "multimodal"

@dataclass
class InteractionEvent:
    """Represents an interaction event"""
    timestamp: float
    modality: InteractionModality
    content: Any
    confidence: float = 1.0
    user_id: str = "default_user"

class InteractionFlowManager:
    """Manages the flow of human-robot interactions"""
    def __init__(self):
        self.current_state = InteractionState.IDLE
        self.event_history = []
        self.context_stack = []
        self.response_callbacks = {}
        self.timeout_settings = {
            'listening': 10.0,  # seconds
            'processing': 30.0,
            'response': 5.0,
            'confirmation': 15.0
        }
        self.last_event_time = time.time()
        self.session_id = self.generate_session_id()

    def generate_session_id(self):
        """Generate a unique session ID"""
        import uuid
        return str(uuid.uuid4())

    def add_event(self, event: InteractionEvent):
        """Add an interaction event to the history"""
        self.event_history.append(event)
        self.last_event_time = event.timestamp

        # Trigger state transition based on event
        self.transition_state_based_on_event(event)

    def transition_state_based_on_event(self, event: InteractionEvent):
        """Transition interaction state based on event"""
        if event.modality == InteractionModality.SPEECH and "hello" in str(event.content).lower():
            self.set_state(InteractionState.LISTENING)
        elif event.modality in [InteractionModality.SPEECH, InteractionModality.GESTURE]:
            if self.current_state == InteractionState.IDLE:
                self.set_state(InteractionState.LISTENING)
            elif self.current_state == InteractionState.LISTENING:
                self.set_state(InteractionState.PROCESSING)

    def set_state(self, new_state: InteractionState):
        """Set the interaction state"""
        old_state = self.current_state
        self.current_state = new_state

        print(f"Interaction state: {old_state.value} -> {new_state.value}")

        # Trigger state-specific actions
        self.on_state_enter(new_state, old_state)

    def on_state_enter(self, new_state: InteractionState, old_state: InteractionState):
        """Actions to perform when entering a new state"""
        if new_state == InteractionState.LISTENING:
            self.on_enter_listening()
        elif new_state == InteractionState.PROCESSING:
            self.on_enter_processing()
        elif new_state == InteractionState.RESPONDING:
            self.on_enter_responding()
        elif new_state == InteractionState.AWAITING_CONFIRMATION:
            self.on_enter_awaiting_confirmation()

    def on_enter_listening(self):
        """Actions when entering listening state"""
        print("Robot is now listening...")

    def on_enter_processing(self):
        """Actions when entering processing state"""
        print("Processing user input...")

    def on_enter_responding(self):
        """Actions when entering responding state"""
        print("Generating response...")

    def on_enter_awaiting_confirmation(self):
        """Actions when entering confirmation state"""
        print("Awaiting user confirmation...")

    def check_timeouts(self):
        """Check for state timeouts"""
        current_time = time.time()

        if self.current_state == InteractionState.LISTENING:
            if current_time - self.last_event_time > self.timeout_settings['listening']:
                self.handle_timeout(InteractionState.LISTENING)

        elif self.current_state == InteractionState.PROCESSING:
            if current_time - self.last_event_time > self.timeout_settings['processing']:
                self.handle_timeout(InteractionState.PROCESSING)

    def handle_timeout(self, state: InteractionState):
        """Handle timeout for a specific state"""
        print(f"Timeout in {state.value} state")

        if state == InteractionState.LISTENING:
            self.set_state(InteractionState.IDLE)
        elif state == InteractionState.PROCESSING:
            self.set_state(InteractionState.ERROR)

    def get_context(self) -> Dict[str, Any]:
        """Get current interaction context"""
        return {
            'session_id': self.session_id,
            'current_state': self.current_state.value,
            'event_count': len(self.event_history),
            'last_event_time': self.last_event_time,
            'context_stack': self.context_stack.copy()
        }

    def push_context(self, context: Dict[str, Any]):
        """Push context onto the stack"""
        self.context_stack.append(context)

    def pop_context(self) -> Dict[str, Any]:
        """Pop context from the stack"""
        if self.context_stack:
            return self.context_stack.pop()
        return {}

    def clear_context(self):
        """Clear all context"""
        self.context_stack.clear()

class TaskOrientedInteraction:
    """Handles task-oriented interactions with proper flow"""
    def __init__(self, flow_manager: InteractionFlowManager):
        self.flow_manager = flow_manager
        self.active_tasks = {}
        self.task_templates = self.load_task_templates()

    def load_task_templates(self):
        """Load predefined task templates"""
        return {
            'navigation': {
                'steps': ['destination_request', 'route_confirmation', 'navigation_start', 'arrival_confirmation'],
                'required_params': ['destination'],
                'success_criteria': ['reached_destination']
            },
            'grasping': {
                'steps': ['object_identification', 'grasp_planning', 'grasp_execution', 'success_verification'],
                'required_params': ['object_description'],
                'success_criteria': ['object_grasped']
            },
            'information': {
                'steps': ['query_understanding', 'information_retrieval', 'response_generation'],
                'required_params': ['query'],
                'success_criteria': ['information_provided']
            }
        }

    def start_task(self, task_type: str, params: Dict[str, Any]):
        """Start a new task"""
        if task_type not in self.task_templates:
            raise ValueError(f"Unknown task type: {task_type}")

        task_id = self.generate_task_id()
        task_template = self.task_templates[task_type]

        task = {
            'id': task_id,
            'type': task_type,
            'template': task_template,
            'params': params,
            'current_step': 0,
            'status': 'active',
            'start_time': time.time(),
            'history': []
        }

        self.active_tasks[task_id] = task
        self.execute_task_step(task_id)
        return task_id

    def generate_task_id(self):
        """Generate a unique task ID"""
        import uuid
        return f"task_{str(uuid.uuid4())[:8]}"

    def execute_task_step(self, task_id: str):
        """Execute the current step of a task"""
        task = self.active_tasks[task_id]
        template = task['template']
        step_name = template['steps'][task['current_step']]

        print(f"Executing task step: {step_name}")

        # Execute step-specific logic
        step_result = self.execute_step_logic(step_name, task['params'])

        # Update task history
        task['history'].append({
            'step': step_name,
            'result': step_result,
            'timestamp': time.time()
        })

        # Check if step was successful
        if step_result['success']:
            task['current_step'] += 1

            # Check if task is complete
            if task['current_step'] >= len(template['steps']):
                self.complete_task(task_id)
            else:
                # Continue to next step
                self.execute_task_step(task_id)
        else:
            # Handle step failure
            self.handle_step_failure(task_id, step_name, step_result)

    def execute_step_logic(self, step_name: str, params: Dict[str, Any]):
        """Execute the logic for a specific step"""
        # This would contain task-specific step implementations
        step_functions = {
            'destination_request': self.step_destination_request,
            'route_confirmation': self.step_route_confirmation,
            'navigation_start': self.step_navigation_start,
            'arrival_confirmation': self.step_arrival_confirmation,
            'object_identification': self.step_object_identification,
            'grasp_planning': self.step_grasp_planning,
            'grasp_execution': self.step_grasp_execution,
            'success_verification': self.step_success_verification,
            'query_understanding': self.step_query_understanding,
            'information_retrieval': self.step_information_retrieval,
            'response_generation': self.step_response_generation
        }

        if step_name in step_functions:
            return step_functions[step_name](params)
        else:
            return {'success': False, 'error': f'Unknown step: {step_name}'}

    def step_destination_request(self, params: Dict[str, Any]):
        """Request destination from user"""
        if 'destination' in params:
            return {'success': True, 'destination': params['destination']}
        else:
            # In practice, this would prompt user for destination
            return {'success': False, 'error': 'No destination provided'}

    def step_route_confirmation(self, params: Dict[str, Any]):
        """Confirm route with user"""
        return {'success': True, 'route_confirmed': True}

    def step_navigation_start(self, params: Dict[str, Any]):
        """Start navigation"""
        return {'success': True, 'navigation_started': True}

    def step_arrival_confirmation(self, params: Dict[str, Any]):
        """Confirm arrival at destination"""
        return {'success': True, 'arrived': True}

    def step_object_identification(self, params: Dict[str, Any]):
        """Identify object to be grasped"""
        if 'object_description' in params:
            return {'success': True, 'object': params['object_description']}
        else:
            return {'success': False, 'error': 'No object description provided'}

    def step_grasp_planning(self, params: Dict[str, Any]):
        """Plan the grasp"""
        return {'success': True, 'grasp_plan': 'calculated'}

    def step_grasp_execution(self, params: Dict[str, Any]):
        """Execute the grasp"""
        return {'success': True, 'grasp_executed': True}

    def step_success_verification(self, params: Dict[str, Any]):
        """Verify grasp success"""
        return {'success': True, 'success_verified': True}

    def step_query_understanding(self, params: Dict[str, Any]):
        """Understand the information query"""
        if 'query' in params:
            return {'success': True, 'understood_query': params['query']}
        else:
            return {'success': False, 'error': 'No query provided'}

    def step_information_retrieval(self, params: Dict[str, Any]):
        """Retrieve requested information"""
        return {'success': True, 'information': 'retrieved'}

    def step_response_generation(self, params: Dict[str, Any]):
        """Generate response to user"""
        return {'success': True, 'response_generated': True}

    def handle_step_failure(self, task_id: str, step_name: str, result: Dict[str, Any]):
        """Handle failure of a task step"""
        print(f"Task step failed: {step_name}, Error: {result.get('error', 'Unknown error')}")

        # Implement recovery logic
        task = self.active_tasks[task_id]
        if task['current_step'] < 3:  # Retry for first few steps
            print("Retrying step...")
            self.execute_task_step(task_id)
        else:
            # Mark task as failed
            self.fail_task(task_id, result['error'])

    def complete_task(self, task_id: str):
        """Complete a task successfully"""
        task = self.active_tasks[task_id]
        task['status'] = 'completed'
        task['end_time'] = time.time()
        task['duration'] = task['end_time'] - task['start_time']

        print(f"Task {task_id} completed successfully in {task['duration']:.2f}s")

        # Clean up
        del self.active_tasks[task_id]

    def fail_task(self, task_id: str, error: str):
        """Fail a task"""
        task = self.active_tasks[task_id]
        task['status'] = 'failed'
        task['error'] = error
        task['end_time'] = time.time()

        print(f"Task {task_id} failed: {error}")

        # Clean up
        del self.active_tasks[task_id]

    def get_active_tasks(self):
        """Get information about active tasks"""
        return {
            task_id: {
                'type': task['type'],
                'status': task['status'],
                'current_step': task['current_step'],
                'progress': task['current_step'] / len(task['template']['steps'])
            }
            for task_id, task in self.active_tasks.items()
        }

# Example usage
def example_interaction_flow():
    flow_manager = InteractionFlowManager()
    task_manager = TaskOrientedInteraction(flow_manager)

    print("Interaction Flow Manager Example")
    print(f"Initial state: {flow_manager.current_state.value}")

    # Simulate some interaction events
    events = [
        InteractionEvent(time.time(), InteractionModality.SPEECH, "Hello robot", 0.9),
        InteractionEvent(time.time(), InteractionModality.SPEECH, "Navigate to kitchen", 0.8),
    ]

    for event in events:
        flow_manager.add_event(event)
        time.sleep(0.1)  # Small delay to simulate real timing

    print(f"Final state: {flow_manager.current_state.value}")
    print(f"Context: {flow_manager.get_context()}")

    # Start a navigation task
    task_params = {
        'destination': 'kitchen',
        'user_preference': 'shortest_path'
    }

    task_id = task_manager.start_task('navigation', task_params)
    print(f"Started task: {task_id}")

    # Simulate task execution
    time.sleep(2)  # Wait for task to complete
    print(f"Active tasks: {task_manager.get_active_tasks()}")

if __name__ == "__main__":
    example_interaction_flow()
```

### Context-Aware Interaction

```python
# context_aware_interaction.py
import datetime
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ContextType(Enum):
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SOCIAL = "social"
    TASK = "task"
    EMOTIONAL = "emotional"
    ENVIRONMENTAL = "environmental"

@dataclass
class TemporalContext:
    """Temporal context information"""
    current_time: datetime.datetime
    day_of_week: int  # 0=Monday, 6=Sunday
    time_of_day: str  # morning, afternoon, evening, night
    season: str

@dataclass
class SpatialContext:
    """Spatial context information"""
    location: str
    room_type: str
    coordinates: tuple  # (x, y, z)
    orientation: tuple  # (roll, pitch, yaw)
    environment_map: Optional[Dict] = None

@dataclass
class SocialContext:
    """Social context information"""
    user_count: int
    user_relationships: Dict[str, str]  # user_id -> relationship
    interaction_history: List[Dict[str, Any]]
    group_dynamics: str  # formal, casual, family, etc.

@dataclass
class TaskContext:
    """Task context information"""
    active_task: Optional[str]
    task_progress: float
    task_priority: int
    task_dependencies: List[str]
    task_deadline: Optional[datetime.datetime]

@dataclass
class EmotionalContext:
    """Emotional context information"""
    user_mood: str
    user_stress_level: float  # 0.0 to 1.0
    interaction_tone: str
    empathy_level: float  # 0.0 to 1.0

@dataclass
class EnvironmentalContext:
    """Environmental context information"""
    lighting: str  # bright, dim, dark
    noise_level: float  # 0.0 to 1.0
    temperature: float  # in Celsius
    occupancy: bool
    privacy_level: str  # public, semi-private, private

class ContextManager:
    """Manages all context information for the robot"""
    def __init__(self):
        self.temporal_context = self.update_temporal_context()
        self.spatial_context = SpatialContext(
            location="unknown",
            room_type="unknown",
            coordinates=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0)
        )
        self.social_context = SocialContext(
            user_count=0,
            user_relationships={},
            interaction_history=[],
            group_dynamics="casual"
        )
        self.task_context = TaskContext(
            active_task=None,
            task_progress=0.0,
            task_priority=0,
            task_dependencies=[],
            task_deadline=None
        )
        self.emotional_context = EmotionalContext(
            user_mood="neutral",
            user_stress_level=0.5,
            interaction_tone="neutral",
            empathy_level=0.5
        )
        self.environmental_context = EnvironmentalContext(
            lighting="normal",
            noise_level=0.5,
            temperature=22.0,
            occupancy=False,
            privacy_level="public"
        )

        self.context_history = []
        self.max_history_length = 100

    def update_temporal_context(self) -> TemporalContext:
        """Update temporal context based on current time"""
        now = datetime.datetime.now()

        # Determine time of day
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Determine season
        month = now.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"

        return TemporalContext(
            current_time=now,
            day_of_week=now.weekday(),
            time_of_day=time_of_day,
            season=season
        )

    def update_spatial_context(self, location: str, room_type: str, coordinates: tuple, orientation: tuple):
        """Update spatial context"""
        self.spatial_context = SpatialContext(
            location=location,
            room_type=room_type,
            coordinates=coordinates,
            orientation=orientation
        )

    def update_social_context(self, user_count: int, user_relationships: Dict[str, str]):
        """Update social context"""
        self.social_context.user_count = user_count
        self.social_context.user_relationships = user_relationships

    def update_task_context(self, active_task: Optional[str], progress: float = 0.0):
        """Update task context"""
        self.task_context.active_task = active_task
        self.task_context.task_progress = progress

    def update_emotional_context(self, user_mood: str, stress_level: float):
        """Update emotional context"""
        self.emotional_context.user_mood = user_mood
        self.emotional_context.user_stress_level = stress_level

    def update_environmental_context(self, lighting: str, noise_level: float, temperature: float):
        """Update environmental context"""
        self.environmental_context.lighting = lighting
        self.environmental_context.noise_level = noise_level
        self.environmental_context.temperature = temperature

    def get_context_profile(self) -> Dict[str, Any]:
        """Get complete context profile"""
        return {
            'temporal': asdict(self.temporal_context),
            'spatial': asdict(self.spatial_context),
            'social': asdict(self.social_context),
            'task': asdict(self.task_context),
            'emotional': asdict(self.emotional_context),
            'environmental': asdict(self.environmental_context),
            'timestamp': datetime.datetime.now().isoformat()
        }

    def get_context_for_interaction(self) -> Dict[str, Any]:
        """Get context relevant for interaction decisions"""
        profile = self.get_context_profile()

        # Extract key context indicators
        interaction_context = {
            'time_of_day': profile['temporal']['time_of_day'],
            'location': profile['spatial']['location'],
            'user_count': profile['social']['user_count'],
            'active_task': profile['task']['active_task'],
            'user_mood': profile['emotional']['user_mood'],
            'environment': {
                'lighting': profile['environmental']['lighting'],
                'noise': profile['environmental']['noise_level'],
                'temperature': profile['environmental']['temperature']
            }
        }

        return interaction_context

    def adapt_behavior_to_context(self) -> Dict[str, Any]:
        """Adapt robot behavior based on current context"""
        context = self.get_context_for_interaction()
        adaptations = {}

        # Adapt based on time of day
        if context['time_of_day'] in ['night', 'early_morning']:
            adaptations['volume'] = 'low'
            adaptations['energy'] = 'calm'
            adaptations['response_time'] = 'patient'
        elif context['time_of_day'] == 'afternoon':
            adaptations['volume'] = 'normal'
            adaptations['energy'] = 'engaged'
            adaptations['response_time'] = 'responsive'

        # Adapt based on user count
        if context['user_count'] > 1:
            adaptations['interaction_mode'] = 'group'
            adaptations['attention'] = 'distributed'
        else:
            adaptations['interaction_mode'] = 'individual'
            adaptations['attention'] = 'focused'

        # Adapt based on user mood
        if context['user_mood'] == 'stressed':
            adaptations['tone'] = 'soothing'
            adaptations['pace'] = 'slow'
            adaptations['helpfulness'] = 'high'
        elif context['user_mood'] == 'happy':
            adaptations['tone'] = 'cheerful'
            adaptations['energy'] = 'positive'

        # Adapt based on environment
        env = context['environment']
        if env['noise'] > 0.7:
            adaptations['volume'] = 'high'
            adaptations['repetition'] = 'increased'
        if env['lighting'] == 'dim':
            adaptations['visual_feedback'] = 'reduced'
            adaptations['verbal_feedback'] = 'increased'

        return adaptations

    def predict_user_needs(self) -> List[str]:
        """Predict user needs based on context"""
        needs = []
        context = self.get_context_for_interaction()

        # Predict based on time and location
        if context['time_of_day'] == 'morning' and context['location'] == 'kitchen':
            needs.append('coffee')
            needs.append('weather information')
            needs.append('schedule reminder')

        if context['time_of_day'] == 'evening' and context['location'] == 'living_room':
            needs.append('entertainment')
            needs.append('relaxation')

        # Predict based on user mood
        if context['user_mood'] == 'tired':
            needs.append('rest')
            needs.append('calming activity')

        if context['user_mood'] == 'excited':
            needs.append('engaging activity')
            needs.append('information sharing')

        # Predict based on social context
        if context['user_count'] > 1:
            needs.append('group activity')
            needs.append('mediation')

        return list(set(needs))  # Remove duplicates

    def generate_contextual_response(self, user_input: str) -> str:
        """Generate response considering current context"""
        context = self.get_context_for_interaction()
        adaptations = self.adapt_behavior_to_context()
        predicted_needs = self.predict_user_needs()

        # Generate contextual response
        response_parts = []

        # Acknowledge context
        if context['time_of_day'] in ['morning', 'afternoon', 'evening']:
            response_parts.append(f"Good {context['time_of_day']}!")

        # Address predicted needs if relevant
        if predicted_needs:
            relevant_needs = [need for need in predicted_needs if need in user_input.lower()]
            if relevant_needs:
                response_parts.append(f"I notice you might need {', '.join(relevant_needs)}. How can I help?")

        # Adapt response based on mood
        if context['user_mood'] == 'stressed':
            response_parts.append("I'm here to help you relax. What would be most helpful right now?")

        # Default response if no specific context applies
        if not response_parts:
            response_parts.append("How can I assist you today?")

        return " ".join(response_parts)

class ContextAwareInteractionManager:
    """Main manager for context-aware interactions"""
    def __init__(self):
        self.context_manager = ContextManager()
        self.interaction_history = []
        self.user_preferences = {}

    def process_user_input(self, user_input: str, user_id: str = "default_user") -> str:
        """Process user input with full context awareness"""
        # Update temporal context
        self.context_manager.update_temporal_context()

        # Analyze input for context clues
        self.analyze_input_context(user_input, user_id)

        # Generate contextual response
        response = self.context_manager.generate_contextual_response(user_input)

        # Record interaction
        self.record_interaction(user_input, response, user_id)

        return response

    def analyze_input_context(self, user_input: str, user_id: str):
        """Analyze user input for contextual information"""
        input_lower = user_input.lower()

        # Analyze emotional content
        emotional_keywords = {
            'happy': ['happy', 'great', 'wonderful', 'excellent'],
            'sad': ['sad', 'upset', 'depressed', 'unhappy'],
            'angry': ['angry', 'frustrated', 'mad', 'annoyed'],
            'stressed': ['stressed', 'overwhelmed', 'tired', 'exhausted']
        }

        for mood, keywords in emotional_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                # Estimate stress level based on intensity words
                stress_keywords = ['very', 'really', 'extremely', 'super']
                stress_level = 0.7 if any(word in input_lower for word in stress_keywords) else 0.5
                self.context_manager.update_emotional_context(mood, stress_level)
                break

        # Analyze spatial references
        location_keywords = {
            'kitchen': ['kitchen', 'cooking', 'food', 'eat'],
            'bedroom': ['bedroom', 'sleep', 'bed', 'rest'],
            'office': ['office', 'work', 'computer', 'meeting'],
            'living_room': ['living room', 'couch', 'tv', 'relax']
        }

        for location, keywords in location_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                self.context_manager.update_spatial_context(location, location, (0, 0, 0), (0, 0, 0))
                break

    def record_interaction(self, user_input: str, response: str, user_id: str):
        """Record interaction in history"""
        interaction = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_id': user_id,
            'user_input': user_input,
            'robot_response': response,
            'context': self.context_manager.get_context_profile()
        }

        self.interaction_history.append(interaction)

        # Keep history size manageable
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-500:]

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences based on interaction history"""
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]

        # Analyze interaction history for preferences
        user_interactions = [ih for ih in self.interaction_history if ih['user_id'] == user_id]

        if not user_interactions:
            return {}

        # Extract common patterns
        common_topics = {}
        for interaction in user_interactions[-20:]:  # Analyze last 20 interactions
            input_text = interaction['user_input'].lower()
            for topic in ['weather', 'news', 'schedule', 'music', 'jokes', 'help']:
                if topic in input_text:
                    common_topics[topic] = common_topics.get(topic, 0) + 1

        preferences = {
            'preferred_topics': sorted(common_topics.items(), key=lambda x: x[1], reverse=True)[:3],
            'interaction_style': 'casual',  # Default, could be personalized
            'preferred_time': 'afternoon'   # Default, could be learned
        }

        self.user_preferences[user_id] = preferences
        return preferences

# Example usage
def example_context_aware_interaction():
    manager = ContextAwareInteractionManager()

    # Simulate some interactions
    test_inputs = [
        "Hello! I'm feeling stressed today.",
        "What's the weather like?",
        "I'm in the kitchen and need to make coffee.",
        "Good evening! How was your day?"
    ]

    for i, user_input in enumerate(test_inputs):
        print(f"\nInteraction {i+1}:")
        print(f"User: {user_input}")

        response = manager.process_user_input(user_input, f"user_{i}")
        print(f"Robot: {response}")

        # Show current context
        context = manager.context_manager.get_context_for_interaction()
        print(f"Context: {context}")

        # Show adaptations
        adaptations = manager.context_manager.adapt_behavior_to_context()
        print(f"Adaptations: {adaptations}")

    # Show predicted needs
    needs = manager.context_manager.predict_user_needs()
    print(f"\nPredicted user needs: {needs}")

if __name__ == "__main__":
    example_context_aware_interaction()
```

## Knowledge Check

1. What are the key principles of natural human-robot interaction?
2. How do different communication modalities (speech, gesture, visual) complement each other?
3. What is the importance of context awareness in human-robot interaction?
4. How can robots adapt their behavior based on social cues and environmental context?

## Summary

This chapter covered natural human-robot interaction, including voice-based and gesture-based interaction systems, visual interaction and social cues, user experience design principles, and context-aware interaction systems. We explored how humanoid robots can interact naturally with humans using multiple modalities and adapt their behavior based on context and social cues.

## Next Steps

In the next module, we'll explore the integration of Large Language Models (LLMs) in robotics, covering conversational AI, speech recognition, natural language understanding, and cognitive planning with LLMs for humanoid robotics applications.