---
sidebar_position: 17
title: "Chapter 17: Integrating LLMs for Conversational AI in Robots"
---

# Chapter 17: Integrating LLMs for Conversational AI in Robots

## Learning Objectives
- Understand the integration of Large Language Models in robotics
- Implement conversational AI principles for robot interaction
- Design architecture for LLM-robot interfaces
- Create ROS 2 action servers for LLM command processing

## Introduction to LLM Integration in Robotics

Large Language Models (LLMs) have revolutionized the field of natural language processing and are increasingly being integrated into robotics systems to enable more natural and sophisticated human-robot interaction. For humanoid robots, LLMs provide the capability to understand complex natural language commands, engage in meaningful conversations, and translate high-level human instructions into executable robotic actions.

### Benefits of LLM Integration

1. **Natural Language Understanding**: Process complex, nuanced human commands
2. **Contextual Awareness**: Maintain conversation context and history
3. **Flexible Interaction**: Handle diverse types of requests and queries
4. **Learning Capability**: Improve over time through interaction
5. **Multimodal Integration**: Combine language with other sensory inputs

### Challenges in LLM-Robot Integration

1. **Latency**: LLM inference can be slow, affecting real-time interaction
2. **Safety**: Ensuring LLM responses result in safe robot behaviors
3. **Grounding**: Connecting language to physical world and robot capabilities
4. **Reliability**: Handling cases where LLM provides incorrect information
5. **Privacy**: Managing sensitive information in conversational data

## Conversational AI Principles

### Architecture for LLM-Robot Integration

```python
# llm_robot_integration.py
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import openai
import time

class InteractionMode(Enum):
    COMMAND = "command"
    CONVERSATION = "conversation"
    INFORMATION = "information"
    TASK = "task"

class SafetyLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

@dataclass
class UserIntent:
    """Represents the user's intent extracted from their input"""
    action: str
    parameters: Dict[str, Any]
    confidence: float
    mode: InteractionMode

@dataclass
class RobotResponse:
    """Represents the robot's response to user input"""
    text: str
    action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    safety_check_passed: bool = True
    execution_required: bool = False

class SafetyChecker:
    """Checks if LLM responses are safe for robot execution"""
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MODERATE):
        self.safety_level = safety_level
        self.banned_actions = [
            'self_harm', 'damage_robot', 'unsafe_movement', 'inappropriate_behavior'
        ]

    def check_response_safety(self, response: RobotResponse) -> bool:
        """Check if a response is safe to execute"""
        if not response.action:
            return True  # No action to check

        # Check for banned actions
        if response.action.lower() in self.banned_actions:
            return False

        # Check for unsafe movements based on safety level
        if self.safety_level == SafetyLevel.STRICT:
            if response.action.lower() in ['move', 'step', 'walk']:
                # Verify movement parameters are safe
                if response.parameters:
                    speed = response.parameters.get('speed', 1.0)
                    if speed > 2.0:  # Too fast
                        return False

        return True

    def check_intent_safety(self, intent: UserIntent) -> bool:
        """Check if an intent is safe to process"""
        # Check for potentially harmful commands
        dangerous_keywords = ['harm', 'damage', 'break', 'destroy', 'hurt']

        for param_value in intent.parameters.values():
            if isinstance(param_value, str):
                if any(keyword in param_value.lower() for keyword in dangerous_keywords):
                    return False

        return True

class ContextManager:
    """Manages conversation context and state"""
    def __init__(self, max_history: int = 10):
        self.conversation_history = []
        self.current_context = {}
        self.max_history = max_history
        self.user_profiles = {}

    def add_interaction(self, user_input: str, robot_response: str, user_id: str = "default"):
        """Add an interaction to the conversation history"""
        interaction = {
            'timestamp': time.time(),
            'user_input': user_input,
            'robot_response': robot_response,
            'user_id': user_id
        }

        self.conversation_history.append(interaction)

        # Keep history size manageable
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_context_for_prompt(self) -> str:
        """Get context to include in LLM prompts"""
        if not self.conversation_history:
            return "This is the beginning of the conversation."

        context_parts = []
        recent_interactions = self.conversation_history[-3:]  # Last 3 interactions

        for interaction in recent_interactions:
            timestamp = time.strftime('%H:%M:%S', time.localtime(interaction['timestamp']))
            context_parts.append(f"[{timestamp}] User: {interaction['user_input']}")
            context_parts.append(f"[{timestamp}] Robot: {interaction['robot_response']}")

        return "\n".join(context_parts)

    def update_user_profile(self, user_id: str, attributes: Dict[str, Any]):
        """Update user profile with new attributes"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}

        self.user_profiles[user_id].update(attributes)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        return self.user_profiles.get(user_id, {})

class LLMPromptBuilder:
    """Builds prompts for LLM interaction"""
    def __init__(self):
        self.system_prompt = self._get_system_prompt()
        self.robot_capabilities = self._get_robot_capabilities()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """
You are an AI assistant integrated into a humanoid robot. Your role is to:
1. Understand user commands and questions
2. Extract actionable intents from natural language
3. Provide helpful, safe responses
4. Recognize when to ask for clarification
5. Only suggest actions that are safe and within the robot's capabilities

Robot capabilities include: navigation, object manipulation, conversation, information retrieval, and basic assistance tasks.

Always prioritize safety and ask for clarification if a request is ambiguous.
"""

    def _get_robot_capabilities(self) -> List[str]:
        """Define robot capabilities for grounding"""
        return [
            "navigation",
            "object_grasping",
            "object_manipulation",
            "conversation",
            "information_retrieval",
            "environmental_sensing",
            "basic_assistance",
            "social_interaction"
        ]

    def build_intent_extraction_prompt(self, user_input: str, context: str = "") -> str:
        """Build prompt for intent extraction"""
        prompt = f"""
{self.system_prompt}

Previous conversation context:
{context}

User input: "{user_input}"

Extract the user's intent from their input. Provide your response in JSON format with the following structure:
{{
    "action": "the main action the user wants (e.g., 'navigate', 'grasp', 'answer_question')",
    "parameters": {{"param_name": "param_value", ...}},
    "confidence": "confidence level (0.0 to 1.0)",
    "mode": "command, conversation, information, or task"
}}

Example responses:
{{
    "action": "navigate",
    "parameters": {{"destination": "kitchen", "speed": "normal"}},
    "confidence": 0.9,
    "mode": "command"
}}

{{
    "action": "answer_question",
    "parameters": {{"question": "What time is it?", "topic": "time"}},
    "confidence": 0.8,
    "mode": "information"
}}

Now analyze the user input:
"""

        return prompt

    def build_response_generation_prompt(self, user_input: str, intent: UserIntent, context: str = "") -> str:
        """Build prompt for generating robot response"""
        prompt = f"""
{self.system_prompt}

Previous conversation context:
{context}

User input: "{user_input}"

Recognized intent: {json.dumps(intent.__dict__)}

Generate an appropriate response for the user. Consider:
1. The user's intent and parameters
2. The conversation context
3. The robot's capabilities
4. Safety considerations
5. Natural, conversational tone

If the intent requires robot action, include both the response text and the action to take.

Provide your response in JSON format:
{{
    "response_text": "the text response to the user",
    "action": "the action for the robot to take (optional)",
    "parameters": {{"param_name": "param_value", ...}},
    "execution_required": true/false
}}

Example:
{{
    "response_text": "I'll navigate to the kitchen for you. Is that correct?",
    "action": "navigate",
    "parameters": {{"destination": "kitchen"}},
    "execution_required": true
}}
"""

        return prompt

class LLMRobotInterface:
    """Main interface for LLM-robot integration"""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.context_manager = ContextManager()
        self.safety_checker = SafetyChecker()
        self.prompt_builder = LLMPromptBuilder()
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        openai.api_key = api_key

        # Robot action handlers
        self.action_handlers = {
            'navigate': self.handle_navigate,
            'grasp': self.handle_grasp,
            'answer_question': self.handle_answer_question,
            'conversation': self.handle_conversation,
            'task': self.handle_task
        }

    async def process_user_input(self, user_input: str, user_id: str = "default") -> RobotResponse:
        """Process user input through LLM and return robot response"""
        try:
            # Get conversation context
            context = self.context_manager.get_context_for_prompt()

            # Extract intent using LLM
            intent = await self.extract_intent(user_input, context)

            # Check intent safety
            if not self.safety_checker.check_intent_safety(intent):
                return RobotResponse(
                    text="I can't process that request for safety reasons.",
                    safety_check_passed=False
                )

            # Generate response using LLM
            response = await self.generate_response(user_input, intent, context)

            # Check response safety
            if not self.safety_checker.check_response_safety(response):
                response.safety_check_passed = False
                response.text = "I can't perform that action for safety reasons."

            # Add to conversation history
            self.context_manager.add_interaction(user_input, response.text, user_id)

            return response

        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            return RobotResponse(
                text="Sorry, I encountered an error processing your request.",
                safety_check_passed=False
            )

    async def extract_intent(self, user_input: str, context: str) -> UserIntent:
        """Extract user intent using LLM"""
        prompt = self.prompt_builder.build_intent_extraction_prompt(user_input, context)

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]

            intent_data = json.loads(json_str)

            return UserIntent(
                action=intent_data.get('action', 'unknown'),
                parameters=intent_data.get('parameters', {}),
                confidence=intent_data.get('confidence', 0.5),
                mode=InteractionMode(intent_data.get('mode', 'conversation'))
            )

        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON")
            return UserIntent(action='unknown', parameters={}, confidence=0.0, mode=InteractionMode.CONVERSATION)
        except Exception as e:
            self.logger.error(f"Error extracting intent: {e}")
            return UserIntent(action='unknown', parameters={}, confidence=0.0, mode=InteractionMode.CONVERSATION)

    async def generate_response(self, user_input: str, intent: UserIntent, context: str) -> RobotResponse:
        """Generate robot response using LLM"""
        prompt = self.prompt_builder.build_response_generation_prompt(user_input, intent, context)

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]

            response_data = json.loads(json_str)

            return RobotResponse(
                text=response_data.get('response_text', ''),
                action=response_data.get('action'),
                parameters=response_data.get('parameters', {}),
                execution_required=response_data.get('execution_required', False)
            )

        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON")
            return RobotResponse(text="I understand.", execution_required=False)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return RobotResponse(text="I'm not sure how to respond to that.", execution_required=False)

    # Action handlers
    async def handle_navigate(self, parameters: Dict[str, Any]) -> str:
        """Handle navigation commands"""
        destination = parameters.get('destination', 'unknown location')
        speed = parameters.get('speed', 'normal')

        # In practice, this would interface with navigation system
        return f"Okay, I'm navigating to the {destination} at {speed} speed."

    async def handle_grasp(self, parameters: Dict[str, Any]) -> str:
        """Handle object grasping commands"""
        object_name = parameters.get('object', 'unknown object')

        # In practice, this would interface with manipulation system
        return f"Okay, I'll grasp the {object_name} for you."

    async def handle_answer_question(self, parameters: Dict[str, Any]) -> str:
        """Handle information requests"""
        question = parameters.get('question', '')

        # In practice, this might interface with knowledge base or internet
        return f"I'd be happy to help with information about '{question}'."

    async def handle_conversation(self, parameters: Dict[str, Any]) -> str:
        """Handle conversational responses"""
        topic = parameters.get('topic', 'general')
        return f"That's interesting! Tell me more about {topic}."

    async def handle_task(self, parameters: Dict[str, Any]) -> str:
        """Handle complex task commands"""
        task_description = parameters.get('description', 'unknown task')
        return f"I'll help you with the {task_description} task."

    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Execute a robot action"""
        if action in self.action_handlers:
            try:
                result = await self.action_handlers[action](parameters)
                self.logger.info(f"Executed action '{action}' with parameters {parameters}")
                return True
            except Exception as e:
                self.logger.error(f"Error executing action '{action}': {e}")
                return False
        else:
            self.logger.warning(f"Unknown action: {action}")
            return False

# Example usage
async def example_llm_robot_integration():
    """Example of LLM-robot integration"""
    # This would require an actual OpenAI API key
    # api_key = "your-openai-api-key"
    # llm_interface = LLMRobotInterface(api_key)

    print("LLM-Robot Integration Example")
    print("Note: This example requires a valid OpenAI API key to run")

    # Simulate the interface for demonstration
    class MockLLMInterface:
        async def process_user_input(self, user_input, user_id="default"):
            # Mock response for demonstration
            responses = {
                "hello": RobotResponse(text="Hello! How can I help you today?", execution_required=False),
                "navigate to kitchen": RobotResponse(
                    text="Okay, I'll navigate to the kitchen for you.",
                    action="navigate",
                    parameters={"destination": "kitchen"},
                    execution_required=True
                ),
                "grasp the cup": RobotResponse(
                    text="I'll grasp the cup for you.",
                    action="grasp",
                    parameters={"object": "cup"},
                    execution_required=True
                )
            }

            key = user_input.lower()
            return responses.get(key, RobotResponse(text="I'm not sure how to help with that.", execution_required=False))

    llm_interface = MockLLMInterface()

    # Test inputs
    test_inputs = [
        "Hello robot",
        "Navigate to kitchen",
        "Grasp the cup",
        "What's the weather like?"
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")

        response = await llm_interface.process_user_input(user_input)
        print(f"Robot: {response.text}")

        if response.execution_required and response.action:
            print(f"Action to execute: {response.action} with parameters {response.parameters}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_llm_robot_integration())
```

## Architecture for LLM-Robot Interfaces

### ROS 2 Integration Architecture

```python
# ros2_llm_integration.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json

# Define custom action message for LLM commands
# This would typically be in a custom action file
class LLMCommand:
    class Goal:
        def __init__(self):
            self.command = ""
            self.parameters = ""

    class Result:
        def __init__(self):
            self.response = ""
            self.action_taken = ""
            self.success = False

    class Feedback:
        def __init__(self):
            self.status = ""

class LLMRobotNode(Node):
    def __init__(self):
        super().__init__('llm_robot_node')

        # Initialize LLM interface
        self.llm_interface = None  # Will be initialized after API key is available

        # Publishers and subscribers
        self.speech_pub = self.create_publisher(String, 'speech_output', 10)
        self.navigation_pub = self.create_publisher(PoseStamped, 'move_base_simple/goal', 10)
        self.vision_sub = self.create_subscription(Image, 'camera/image_raw', self.vision_callback, 10)

        # Action server for LLM commands
        self.llm_action_server = ActionServer(
            self,
            LLMCommand,
            'llm_command',
            execute_callback=self.execute_llm_command,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Service clients for robot capabilities
        self.navigation_client = None  # Will be initialized as needed
        self.manipulation_client = None  # Will be initialized as needed

        # Threading for async LLM operations
        self.llm_executor = ThreadPoolExecutor(max_workers=2)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.run_async_loop, args=(self.loop,))
        self.loop_thread.start()

        self.get_logger().info('LLM Robot Node initialized')

    def run_async_loop(self, loop):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def goal_callback(self, goal_request):
        """Accept or reject LLM command goals"""
        self.get_logger().info(f'Received LLM command goal: {goal_request.command}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject LLM command cancel requests"""
        self.get_logger().info('Received LLM command cancel request')
        return CancelResponse.ACCEPT

    def vision_callback(self, msg):
        """Handle vision data for multimodal interaction"""
        # Process vision data for context
        pass

    async def process_llm_command_async(self, command: str, user_id: str = "default"):
        """Process LLM command asynchronously"""
        if not self.llm_interface:
            return "LLM interface not initialized. Please set API key."

        # Process through LLM interface
        response = await self.llm_interface.process_user_input(command, user_id)

        # Publish speech response
        speech_msg = String()
        speech_msg.data = response.text
        self.speech_pub.publish(speech_msg)

        # Execute action if required
        if response.execution_required and response.action:
            success = await self.execute_robot_action(response.action, response.parameters or {})
            return f"{response.text} Action execution: {'Success' if success else 'Failed'}"

        return response.text

    async def execute_robot_action(self, action: str, parameters: dict):
        """Execute robot action based on LLM command"""
        try:
            if action == 'navigate':
                return await self.execute_navigation(parameters)
            elif action == 'grasp':
                return await self.execute_grasp(parameters)
            elif action == 'speak':
                return await self.execute_speech(parameters)
            else:
                self.get_logger().warning(f'Unknown action: {action}')
                return False
        except Exception as e:
            self.get_logger().error(f'Error executing action {action}: {e}')
            return False

    async def execute_navigation(self, parameters: dict):
        """Execute navigation action"""
        destination = parameters.get('destination', 'unknown')

        # Create navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'

        # Set destination coordinates (simplified)
        if destination.lower() == 'kitchen':
            goal_msg.pose.position.x = 2.0
            goal_msg.pose.position.y = 1.0
        elif destination.lower() == 'living_room':
            goal_msg.pose.position.x = -1.0
            goal_msg.pose.position.y = 0.5
        else:
            self.get_logger().warning(f'Unknown destination: {destination}')
            return False

        goal_msg.pose.orientation.w = 1.0  # Default orientation

        self.navigation_pub.publish(goal_msg.pose)
        return True

    async def execute_grasp(self, parameters: dict):
        """Execute grasp action"""
        object_name = parameters.get('object', 'unknown')
        self.get_logger().info(f'Attempting to grasp {object_name}')

        # In practice, this would interface with manipulation system
        # For now, return success
        return True

    async def execute_speech(self, parameters: dict):
        """Execute speech action"""
        text = parameters.get('text', '')
        speech_msg = String()
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)
        return True

    async def execute_llm_command(self, goal_handle):
        """Execute the LLM command goal"""
        self.get_logger().info(f'Executing LLM command: {goal_handle.request.command}')

        # Parse command and parameters
        command = goal_handle.request.command
        params_str = goal_handle.request.parameters

        try:
            parameters = json.loads(params_str) if params_str else {}
        except json.JSONDecodeError:
            parameters = {}
            self.get_logger().warning('Could not parse parameters as JSON')

        # Process command asynchronously
        response_text = await self.process_llm_command_async(command)

        # Create result
        result = LLMCommand.Result()
        result.response = response_text
        result.success = True

        # Set result
        goal_handle.succeed()
        return result

    def set_llm_interface(self, llm_interface):
        """Set the LLM interface"""
        self.llm_interface = llm_interface

def main(args=None):
    rclpy.init(args=args)

    node = LLMRobotNode()

    # Example of how to set up the LLM interface
    # This would typically be done after getting API keys from parameters or config
    """
    api_key = node.declare_parameter('openai_api_key', '').value
    if api_key:
        llm_interface = LLMRobotInterface(api_key)
        node.set_llm_interface(llm_interface)
    """

    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Multimodal Integration

```python
# multimodal_integration.py
import numpy as np
import cv2
from PIL import Image
import base64
import io
import requests
import json
from typing import Dict, Any, List, Optional

class MultimodalLLMInterface:
    """Interface for multimodal LLM integration (text + vision)"""
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_pil_image(self, pil_image: Image.Image) -> str:
        """Encode PIL image to base64"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def process_vision_command(self, image: Image.Image, text_command: str) -> str:
        """Process a command that includes visual context"""
        base64_image = self.encode_pil_image(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on this image, please help me: {text_command}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")

    def describe_scene(self, image: Image.Image) -> str:
        """Generate a description of the current scene"""
        base64_image = self.encode_pil_image(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed description of this scene, including objects, people, and the general environment."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")

    def identify_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Identify and locate objects in the image"""
        base64_image = self.encode_pil_image(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Identify the objects in this image and provide their locations.
                            Respond in JSON format with the following structure:
                            {
                                "objects": [
                                    {
                                        "name": "object name",
                                        "category": "object category",
                                        "position": {"x": number, "y": number},
                                        "confidence": 0.0-1.0
                                    }
                                ]
                            }"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']

            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str).get('objects', [])

        return []

    def answer_vision_question(self, image: Image.Image, question: str) -> str:
        """Answer a specific question about the image"""
        base64_image = self.encode_pil_image(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")

class VisionEnhancedRobotInterface(LLMRobotInterface):
    """Robot interface enhanced with vision capabilities"""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        self.vision_interface = MultimodalLLMInterface(api_key, "gpt-4-vision-preview")
        self.last_seen_objects = []

    async def process_vision_enhanced_input(self, user_input: str, camera_image: Optional[Image.Image] = None) -> RobotResponse:
        """Process input with vision enhancement"""
        if camera_image is None:
            # Process without vision
            return await self.process_user_input(user_input)

        # Check if the user's request involves vision
        vision_keywords = ['see', 'look', 'there', 'what', 'where', 'find', 'locate', 'show', 'object', 'thing']
        needs_vision = any(keyword in user_input.lower() for keyword in vision_keywords)

        if needs_vision:
            try:
                # Process with vision context
                vision_response = self.vision_interface.process_vision_command(camera_image, user_input)

                # Extract intent from vision-enhanced response
                context = self.context_manager.get_context_for_prompt()
                intent_prompt = self.prompt_builder.build_intent_extraction_prompt(
                    f"{user_input} (Image context: {vision_response})", context
                )

                # For simplicity, we'll return a response that includes vision info
                return RobotResponse(
                    text=f"Based on what I see: {vision_response}",
                    execution_required=False
                )
            except Exception as e:
                self.logger.error(f"Vision processing error: {e}")
                # Fall back to text-only processing
                return await self.process_user_input(user_input)
        else:
            # Process without vision
            return await self.process_user_input(user_input)

    def update_seen_objects(self, image: Image.Image):
        """Update the list of seen objects"""
        try:
            objects = self.vision_interface.identify_objects(image)
            self.last_seen_objects = objects
            self.logger.info(f"Identified {len(objects)} objects in scene")
        except Exception as e:
            self.logger.error(f"Error identifying objects: {e}")

    def get_object_location(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get the location of a specific object"""
        for obj in self.last_seen_objects:
            if object_name.lower() in obj['name'].lower():
                return obj
        return None

# Example usage
def example_multimodal_integration():
    """Example of multimodal integration"""
    print("Multimodal LLM Integration Example")

    # This would require actual API keys to run
    # api_key = "your-openai-api-key"
    # vision_interface = MultimodalLLMInterface(api_key)

    # For demonstration, we'll show the structure
    print("Multimodal interface would connect to vision models like GPT-4 Vision")
    print("Key capabilities:")
    print("1. Scene description and understanding")
    print("2. Object identification and localization")
    print("3. Question answering about visual content")
    print("4. Vision-guided action planning")

if __name__ == "__main__":
    example_multimodal_integration()
```

## Cognitive Planning with LLMs

### Planning and Execution Framework

```python
# cognitive_planning.py
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class PlanStatus(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

class ActionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class ActionStep:
    """A single step in a plan"""
    id: str
    action: str
    parameters: Dict[str, Any]
    description: str
    dependencies: List[str]  # IDs of actions this depends on
    status: ActionStatus = ActionStatus.PENDING
    result: Optional[Any] = None

@dataclass
class CognitivePlan:
    """A complete cognitive plan"""
    id: str
    goal: str
    steps: List[ActionStep]
    status: PlanStatus = PlanStatus.PLANNING
    current_step: int = 0
    created_at: float = 0.0
    completed_at: Optional[float] = None

class PlanExecutor:
    """Executes cognitive plans with LLM assistance"""
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface
        self.active_plans = {}
        self.action_handlers = {
            'navigate': self.execute_navigate,
            'grasp': self.execute_grasp,
            'speak': self.execute_speak,
            'perceive': self.execute_perceive,
            'wait': self.execute_wait,
            'query': self.execute_query
        }

    async def create_plan(self, goal: str, context: str = "") -> CognitivePlan:
        """Create a plan for achieving a goal using LLM"""
        plan_prompt = f"""
Create a detailed plan to achieve the following goal: "{goal}"

Context: {context}

The plan should be broken down into specific, executable actions. Each action should be one of: navigate, grasp, speak, perceive, wait, query.

Provide the plan in JSON format:
{{
    "steps": [
        {{
            "id": "unique_id",
            "action": "action_type",
            "parameters": {{"param": "value"}},
            "description": "what this step does",
            "dependencies": ["id_of_step_this_depends_on"]
        }}
    ]
}}

Example plan for "Bring me a cup of water from the kitchen":
{{
    "steps": [
        {{
            "id": "1",
            "action": "navigate",
            "parameters": {{"destination": "kitchen"}},
            "description": "Go to the kitchen",
            "dependencies": []
        }},
        {{
            "id": "2",
            "action": "perceive",
            "parameters": {{"target": "cup"}},
            "description": "Look for a cup",
            "dependencies": ["1"]
        }},
        {{
            "id": "3",
            "action": "grasp",
            "parameters": {{"object": "cup"}},
            "description": "Pick up the cup",
            "dependencies": ["2"]
        }},
        {{
            "id": "4",
            "action": "navigate",
            "parameters": {{"destination": "user"}},
            "description": "Return to user",
            "dependencies": ["3"]
        }}
    ]
}}
"""

        try:
            response = await self.llm_interface.process_user_input(plan_prompt)

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                plan_data = json.loads(json_str)

                steps = []
                for step_data in plan_data.get('steps', []):
                    step = ActionStep(
                        id=step_data['id'],
                        action=step_data['action'],
                        parameters=step_data.get('parameters', {}),
                        description=step_data['description'],
                        dependencies=step_data.get('dependencies', [])
                    )
                    steps.append(step)

                plan = CognitivePlan(
                    id=f"plan_{len(self.active_plans)}",
                    goal=goal,
                    steps=steps,
                    created_at=time.time()
                )

                self.active_plans[plan.id] = plan
                return plan

        except Exception as e:
            print(f"Error creating plan: {e}")

        # Return a default plan if LLM fails
        return CognitivePlan(
            id=f"plan_{len(self.active_plans)}",
            goal=goal,
            steps=[],
            status=PlanStatus.FAILED
        )

    async def execute_plan(self, plan_id: str) -> PlanStatus:
        """Execute a cognitive plan"""
        if plan_id not in self.active_plans:
            return PlanStatus.FAILED

        plan = self.active_plans[plan_id]
        plan.status = PlanStatus.EXECUTING

        for i, step in enumerate(plan.steps):
            if step.status == ActionStatus.PENDING:
                # Check dependencies
                if not self.check_dependencies_satisfied(step, plan):
                    continue

                # Execute the step
                step.status = ActionStatus.IN_PROGRESS
                success = await self.execute_action_step(step)

                if success:
                    step.status = ActionStatus.SUCCESS
                    plan.current_step = i + 1
                else:
                    step.status = ActionStatus.FAILED
                    plan.status = PlanStatus.FAILED
                    return PlanStatus.FAILED

        # Check if all steps are completed
        if all(step.status == ActionStatus.SUCCESS for step in plan.steps):
            plan.status = PlanStatus.COMPLETED
            plan.completed_at = time.time()
            return PlanStatus.COMPLETED

        return plan.status

    def check_dependencies_satisfied(self, step: ActionStep, plan: CognitivePlan) -> bool:
        """Check if all dependencies for a step are satisfied"""
        for dep_id in step.dependencies:
            dep_step = next((s for s in plan.steps if s.id == dep_id), None)
            if dep_step is None or dep_step.status != ActionStatus.SUCCESS:
                return False
        return True

    async def execute_action_step(self, step: ActionStep) -> bool:
        """Execute a single action step"""
        if step.action in self.action_handlers:
            try:
                result = await self.action_handlers[step.action](step.parameters)
                step.result = result
                return True
            except Exception as e:
                print(f"Error executing action {step.action}: {e}")
                return False
        else:
            print(f"Unknown action: {step.action}")
            return False

    # Action execution methods
    async def execute_navigate(self, parameters: Dict[str, Any]) -> bool:
        """Execute navigation action"""
        destination = parameters.get('destination', 'unknown')
        print(f"Navigating to {destination}")
        # In practice, this would interface with navigation system
        return True

    async def execute_grasp(self, parameters: Dict[str, Any]) -> bool:
        """Execute grasping action"""
        object_name = parameters.get('object', 'unknown')
        print(f"Grasping {object_name}")
        # In practice, this would interface with manipulation system
        return True

    async def execute_speak(self, parameters: Dict[str, Any]) -> bool:
        """Execute speech action"""
        text = parameters.get('text', '')
        print(f"Speaking: {text}")
        # In practice, this would interface with TTS system
        return True

    async def execute_perceive(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute perception action"""
        target = parameters.get('target', 'environment')
        print(f"Perceiving {target}")
        # In practice, this would interface with perception system
        return {"success": True, "objects": [target]}

    async def execute_wait(self, parameters: Dict[str, Any]) -> bool:
        """Execute wait action"""
        duration = parameters.get('duration', 1.0)
        print(f"Waiting for {duration} seconds")
        await asyncio.sleep(duration)
        return True

    async def execute_query(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query action"""
        query = parameters.get('query', '')
        print(f"Querying: {query}")
        # In practice, this would interface with knowledge system
        return {"result": "query_result"}

class CognitiveRobotInterface(LLMRobotInterface):
    """Robot interface with cognitive planning capabilities"""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        self.plan_executor = PlanExecutor(self)

    async def process_complex_task(self, user_input: str) -> RobotResponse:
        """Process a complex task that requires planning"""
        # Create a plan for the task
        plan = await self.plan_executor.create_plan(user_input)

        if plan.status == PlanStatus.FAILED:
            return RobotResponse(
                text="I'm sorry, I couldn't create a plan for that task.",
                execution_required=False
            )

        # Execute the plan
        status = await self.plan_executor.execute_plan(plan.id)

        if status == PlanStatus.COMPLETED:
            return RobotResponse(
                text=f"I've completed the task: {user_input}",
                execution_required=False
            )
        else:
            return RobotResponse(
                text=f"I couldn't complete the task: {user_input}. Something went wrong.",
                execution_required=False
            )

    async def process_user_input(self, user_input: str, user_id: str = "default") -> RobotResponse:
        """Enhanced input processing with planning capability"""
        # Check if this is a complex task that needs planning
        complex_task_indicators = [
            'bring me', 'get me', 'go to', 'find', 'help me',
            'make', 'prepare', 'do', 'complete', 'perform'
        ]

        is_complex_task = any(indicator in user_input.lower() for indicator in complex_task_indicators)

        if is_complex_task:
            return await self.process_complex_task(user_input)
        else:
            # Use regular processing for simple commands
            return await super().process_user_input(user_input, user_id)

# Example usage
async def example_cognitive_planning():
    """Example of cognitive planning"""
    print("Cognitive Planning Example")

    # This would require an API key to run
    # api_key = "your-openai-api-key"
    # cognitive_interface = CognitiveRobotInterface(api_key)

    # For demonstration, show the concept
    print("Cognitive planning involves:")
    print("1. Breaking down complex goals into action steps")
    print("2. Managing dependencies between actions")
    print("3. Executing plans with error handling")
    print("4. Adapting plans based on execution results")

    # Example plan for a complex task
    print("\nExample plan for 'Bring me a cup of water from the kitchen':")
    print("Step 1: Navigate to kitchen")
    print("Step 2: Perceive and locate cup")
    print("Step 3: Grasp the cup")
    print("Step 4: Navigate to user")
    print("Step 5: Deliver cup to user")

if __name__ == "__main__":
    import time
    asyncio.run(example_cognitive_planning())
```

## Knowledge Check

1. What are the main benefits of integrating LLMs into robotics systems?
2. How does the architecture for LLM-robot interfaces handle safety considerations?
3. What are the key components of a cognitive planning system for robots?
4. How does multimodal integration enhance LLM capabilities in robotics?

## Summary

This chapter explored the integration of Large Language Models in robotics for conversational AI. We covered the architecture for LLM-robot interfaces, including safety considerations, contextual awareness, and multimodal integration. We also examined cognitive planning systems that use LLMs to break down complex tasks into executable robot actions. The chapter provided practical examples of how to implement these systems with proper safety checks and execution frameworks.

## Next Steps

In the next chapter, we'll explore speech recognition and natural language understanding in robotics, covering voice command processing, dialogue management, and multi-modal interaction techniques for humanoid robots.