---
sidebar_position: 19
title: "Chapter 19: Cognitive Planning with LLMs"
---

# Chapter 19: Cognitive Planning with LLMs

## Learning Objectives
- Translate natural language commands into executable robotic actions
- Implement cognitive planning systems using LLMs
- Design planning and execution frameworks for humanoid robots
- Create robust task decomposition and execution systems

## Introduction to Cognitive Planning

Cognitive planning represents the ability of robots to understand high-level human commands expressed in natural language and decompose them into executable action sequences. Large Language Models (LLMs) excel at this task by bridging the gap between human intention and robot execution, enabling more intuitive and flexible human-robot interaction.

### The Cognitive Planning Challenge

Traditional robotics requires precise, low-level commands that are difficult for humans to formulate. Cognitive planning addresses this by:

1. **Understanding Intent**: Extracting the user's true goal from natural language
2. **World Modeling**: Understanding the current state of the environment
3. **Action Decomposition**: Breaking down complex tasks into primitive actions
4. **Plan Execution**: Executing the plan while monitoring for failures
5. **Adaptive Reasoning**: Adjusting plans based on changing conditions

### Planning Hierarchy in Cognitive Systems

```
High-Level Goal: "Bring me a cup of coffee from the kitchen"
    ↓
Task Decomposition: [Navigate to kitchen, Find coffee, Grasp cup, Pour coffee, Navigate to user]
    ↓
Action Sequences: [Move forward, Turn right, Detect object, Grasp object, ...]
    ↓
Primitive Actions: [Motor commands, sensor activations, ...]
```

## Natural Language to Action Mapping

### Intent Recognition and Action Extraction

```python
# cognitive_planning.py
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

class ActionCategory(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    SYSTEM = "system"

@dataclass
class PrimitiveAction:
    """Lowest level action that can be executed"""
    action_type: str
    parameters: Dict[str, Any]
    description: str
    required_sensors: List[str]
    required_actuators: List[str]

@dataclass
class TaskStep:
    """Intermediate task step composed of primitive actions"""
    id: str
    name: str
    category: ActionCategory
    dependencies: List[str]  # IDs of steps this depends on
    primitive_actions: List[PrimitiveAction]
    estimated_duration: float
    success_criteria: List[str]

@dataclass
class CognitivePlan:
    """Complete cognitive plan"""
    id: str
    goal: str
    steps: List[TaskStep]
    current_step: int = 0
    status: str = "planning"
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class ActionMapper:
    """Maps natural language to robot actions"""
    def __init__(self):
        self.action_database = self.initialize_action_database()
        self.location_map = self.initialize_location_map()
        self.object_map = self.initialize_object_map()

    def initialize_action_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the action database with mappings"""
        return {
            # Navigation actions
            "navigate_to": {
                "category": ActionCategory.NAVIGATION,
                "keywords": ["go to", "move to", "navigate to", "walk to", "head to", "go toward"],
                "required_params": ["destination"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="move_base",
                        parameters={"target_pose": None, "speed": "normal"},
                        description="Move robot to specified location",
                        required_sensors=["odometry", "lidar"],
                        required_actuators=["base_motors"]
                    )
                ]
            },
            "navigate_from_to": {
                "category": ActionCategory.NAVIGATION,
                "keywords": ["go from", "move from", "navigate from"],
                "required_params": ["start_location", "end_location"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="move_base",
                        parameters={"target_pose": None, "speed": "normal"},
                        description="Move robot from start to end location",
                        required_sensors=["odometry", "lidar"],
                        required_actuators=["base_motors"]
                    )
                ]
            },
            "explore_area": {
                "category": ActionCategory.NAVIGATION,
                "keywords": ["explore", "patrol", "survey", "check"],
                "required_params": ["area"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="explore",
                        parameters={"area_bounds": None, "coverage_percent": 0.9},
                        description="Explore specified area systematically",
                        required_sensors=["lidar", "camera"],
                        required_actuators=["base_motors"]
                    )
                ]
            },

            # Manipulation actions
            "grasp_object": {
                "category": ActionCategory.MANIPULATION,
                "keywords": ["grasp", "grab", "pick up", "take", "hold", "catch"],
                "required_params": ["object"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="move_to_object",
                        parameters={"object_name": None, "approach_distance": 0.1},
                        description="Move end-effector to object",
                        required_sensors=["camera", "arm_encoders"],
                        required_actuators=["arm_joints", "gripper"]
                    ),
                    PrimitiveAction(
                        action_type="grasp",
                        parameters={"grip_strength": 0.5},
                        description="Grasp the object",
                        required_sensors=["force_sensors"],
                        required_actuators=["gripper"]
                    )
                ]
            },
            "release_object": {
                "category": ActionCategory.MANIPULATION,
                "keywords": ["release", "let go", "drop", "put down", "place"],
                "required_params": ["object"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="release",
                        parameters={"object_name": None},
                        description="Release the held object",
                        required_sensors=["force_sensors"],
                        required_actuators=["gripper"]
                    )
                ]
            },
            "manipulate_object": {
                "category": ActionCategory.MANIPULATION,
                "keywords": ["move", "push", "pull", "lift", "carry", "transport"],
                "required_params": ["object", "destination"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="grasp",
                        parameters={"object_name": None},
                        description="Grasp the object",
                        required_sensors=["camera", "force_sensors"],
                        required_actuators=["gripper", "arm_joints"]
                    ),
                    PrimitiveAction(
                        action_type="move_to",
                        parameters={"target_pose": None},
                        description="Move to destination while holding object",
                        required_sensors=["odometry", "lidar"],
                        required_actuators=["base_motors", "arm_joints"]
                    ),
                    PrimitiveAction(
                        action_type="release",
                        parameters={"object_name": None},
                        description="Release the object at destination",
                        required_sensors=["force_sensors"],
                        required_actuators=["gripper"]
                    )
                ]
            },

            # Perception actions
            "detect_object": {
                "category": ActionCategory.PERCEPTION,
                "keywords": ["find", "locate", "detect", "identify", "spot", "see"],
                "required_params": ["object"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="object_detection",
                        parameters={"target_object": None, "confidence_threshold": 0.7},
                        description="Detect and locate specified object",
                        required_sensors=["camera", "depth_camera"],
                        required_actuators=[]
                    )
                ]
            },
            "inspect_area": {
                "category": ActionCategory.PERCEPTION,
                "keywords": ["inspect", "check", "examine", "scan", "look_at"],
                "required_params": ["location"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="panorama_scan",
                        parameters={"center_pose": None, "angle_span": 360},
                        description="Scan area for inspection",
                        required_sensors=["camera", "lidar"],
                        required_actuators=["pan_tilt_unit"]
                    )
                ]
            },

            # Communication actions
            "speak": {
                "category": ActionCategory.COMMUNICATION,
                "keywords": ["say", "speak", "tell", "announce", "reply", "respond"],
                "required_params": ["message"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="text_to_speech",
                        parameters={"text": None, "voice": "default"},
                        description="Convert text to speech",
                        required_sensors=[],
                        required_actuators=["speakers"]
                    )
                ]
            },
            "listen": {
                "category": ActionCategory.COMMUNICATION,
                "keywords": ["listen", "hear", "pay attention", "wait for"],
                "required_params": ["duration"],
                "primitive_actions": [
                    PrimitiveAction(
                        action_type="start_listening",
                        parameters={"timeout": 10.0, "keywords": []},
                        description="Listen for user input",
                        required_sensors=["microphone"],
                        required_actuators=[]
                    )
                ]
            }
        }

    def initialize_location_map(self) -> Dict[str, Any]:
        """Initialize location mappings"""
        return {
            "kitchen": {"x": 2.0, "y": 1.0, "z": 0.0, "frame": "map"},
            "living_room": {"x": -1.0, "y": 0.5, "z": 0.0, "frame": "map"},
            "bedroom": {"x": 0.0, "y": -2.0, "z": 0.0, "frame": "map"},
            "office": {"x": 1.5, "y": -1.0, "z": 0.0, "frame": "map"},
            "entrance": {"x": 0.0, "y": 0.0, "z": 0.0, "frame": "map"}
        }

    def initialize_object_map(self) -> Dict[str, Any]:
        """Initialize object mappings"""
        return {
            "cup": {"type": "graspable", "size": "small", "grasp_method": "top_grasp"},
            "bottle": {"type": "graspable", "size": "medium", "grasp_method": "side_grasp"},
            "book": {"type": "graspable", "size": "medium", "grasp_method": "edge_grasp"},
            "phone": {"type": "graspable", "size": "small", "grasp_method": "pinch_grasp"},
            "chair": {"type": "large", "grasp_method": "not_graspable"},
            "table": {"type": "large", "grasp_method": "not_graspable"}
        }

    def extract_intent_and_parameters(self, natural_language: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extract intent and parameters from natural language"""
        # Convert to lowercase for easier matching
        text_lower = natural_language.lower()

        # Find the best matching action
        best_match = None
        best_score = 0

        for action_name, action_def in self.action_database.items():
            for keyword in action_def["keywords"]:
                # Calculate similarity score
                score = self.calculate_similarity(keyword, text_lower)
                if score > best_score:
                    best_score = score
                    best_match = action_name

        if not best_match:
            return None

        # Extract parameters
        action_def = self.action_database[best_match]
        parameters = self.extract_parameters(text_lower, action_def["required_params"])

        # Validate required parameters
        for param in action_def["required_params"]:
            if param not in parameters or parameters[param] is None:
                # Try to infer from context or ask for clarification
                if param == "destination" and "to" in text_lower:
                    # Extract location after "to"
                    match = re.search(r'to\s+(\w+)', text_lower)
                    if match:
                        location = match.group(1)
                        if location in self.location_map:
                            parameters["destination"] = location

        return best_match, parameters

    def calculate_similarity(self, keyword: str, text: str) -> float:
        """Calculate similarity between keyword and text"""
        if keyword in text:
            return 1.0
        elif any(word in text for word in keyword.split()):
            return 0.7
        else:
            return 0.0

    def extract_parameters(self, text: str, required_params: List[str]) -> Dict[str, Any]:
        """Extract parameters from text"""
        parameters = {}

        for param in required_params:
            if param == "destination":
                # Look for location keywords
                for location, coords in self.location_map.items():
                    if location in text:
                        parameters["destination"] = location
                        break

            elif param == "object":
                # Look for object keywords
                for obj_name, obj_info in self.object_map.items():
                    if obj_name in text:
                        parameters["object"] = obj_name
                        break

            elif param == "message":
                # Extract message content (everything after command)
                command_end = -1
                for keyword in ["say", "speak", "tell", "announce"]:
                    if keyword in text:
                        command_end = text.find(keyword) + len(keyword)
                        break

                if command_end > 0:
                    message = text[command_end:].strip()
                    if message.startswith("to "):
                        message = message[3:].strip()
                    parameters["message"] = message

            elif param == "duration":
                # Look for time expressions
                time_match = re.search(r'(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)', text)
                if time_match:
                    value = float(time_match.group(1))
                    unit = time_match.group(2)
                    if 'minute' in unit or 'min' in unit:
                        value *= 60
                    elif 'hour' in unit or 'hr' in unit:
                        value *= 3600
                    parameters["duration"] = value

        return parameters

    def create_primitive_actions(self, action_name: str, parameters: Dict[str, Any]) -> List[PrimitiveAction]:
        """Create primitive actions for an action with given parameters"""
        if action_name not in self.action_database:
            return []

        action_def = self.action_database[action_name]
        primitive_actions = []

        for base_action in action_def["primitive_actions"]:
            # Create a copy and update parameters
            action_copy = PrimitiveAction(
                action_type=base_action.action_type,
                parameters=base_action.parameters.copy(),
                description=base_action.description,
                required_sensors=base_action.required_sensors.copy(),
                required_actuators=base_action.required_actuators.copy()
            )

            # Update parameters with specific values
            for param_key, param_value in parameters.items():
                if param_key in action_copy.parameters:
                    action_copy.parameters[param_key] = param_value

            # Special handling for location parameters
            if "destination" in parameters and action_copy.action_type == "move_base":
                if parameters["destination"] in self.location_map:
                    location_data = self.location_map[parameters["destination"]]
                    action_copy.parameters["target_pose"] = location_data

            primitive_actions.append(action_copy)

        return primitive_actions

class LLMActionTranslator:
    """Translates natural language to actions using LLM"""
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.action_mapper = ActionMapper()

    async def translate_command(self, user_command: str) -> Optional[CognitivePlan]:
        """Translate user command to cognitive plan using LLM"""
        prompt = f"""
You are a robot command translator. Your job is to convert natural language commands into executable robot actions.

Given the command: "{user_command}"

Please provide the response in the following JSON format:

{{
    "goal": "the user's ultimate goal",
    "steps": [
        {{
            "id": "step_unique_id",
            "name": "descriptive name of the step",
            "category": "NAVIGATION|MANIPULATION|PERCEPTION|COMMUNICATION|SYSTEM",
            "dependencies": ["id_of_previous_step_if_any"],
            "actions": [
                {{
                    "action_type": "specific_robot_action",
                    "parameters": {{"param_name": "param_value"}},
                    "description": "what this action does"
                }}
            ],
            "estimated_duration": 5.0,
            "success_criteria": ["list", "of", "success", "criteria"]
        }}
    ]
}}

Example for "Go to the kitchen and bring me a cup":
{{
    "goal": "Deliver a cup to the user",
    "steps": [
        {{
            "id": "nav_to_kitchen",
            "name": "Navigate to kitchen",
            "category": "NAVIGATION",
            "dependencies": [],
            "actions": [
                {{
                    "action_type": "move_base",
                    "parameters": {{"target_pose": {{"x": 2.0, "y": 1.0, "z": 0.0}}}},
                    "description": "Move robot to kitchen location"
                }}
            ],
            "estimated_duration": 30.0,
            "success_criteria": ["robot_reached_kitchen", "navigation_successful"]
        }},
        {{
            "id": "find_cup",
            "name": "Find cup in kitchen",
            "category": "PERCEPTION",
            "dependencies": ["nav_to_kitchen"],
            "actions": [
                {{
                    "action_type": "object_detection",
                    "parameters": {{"target_object": "cup", "confidence_threshold": 0.7}},
                    "description": "Detect cup in kitchen environment"
                }}
            ],
            "estimated_duration": 10.0,
            "success_criteria": ["cup_detected", "object_confirmed"]
        }}
    ]
}}

Now translate the given command:
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                plan_data = json.loads(json_str)

                # Create cognitive plan
                steps = []
                for step_data in plan_data.get('steps', []):
                    actions = []
                    for action_data in step_data.get('actions', []):
                        action = PrimitiveAction(
                            action_type=action_data['action_type'],
                            parameters=action_data.get('parameters', {}),
                            description=action_data['description'],
                            required_sensors=[],  # Would be determined by action type
                            required_actuators=[]  # Would be determined by action type
                        )
                        actions.append(action)

                    step = TaskStep(
                        id=step_data['id'],
                        name=step_data['name'],
                        category=ActionCategory(step_data['category']),
                        dependencies=step_data.get('dependencies', []),
                        primitive_actions=actions,
                        estimated_duration=step_data.get('estimated_duration', 10.0),
                        success_criteria=step_data.get('success_criteria', [])
                    )
                    steps.append(step)

                plan = CognitivePlan(
                    id=f"plan_{int(time.time())}",
                    goal=plan_data.get('goal', user_command),
                    steps=steps,
                    created_at=time.time()
                )

                return plan

        except Exception as e:
            print(f"Error translating command: {e}")
            return None

    def validate_plan(self, plan: CognitivePlan) -> Tuple[bool, List[str]]:
        """Validate that a plan is executable"""
        errors = []

        # Check that all required parameters are present
        for step in plan.steps:
            for action in step.primitive_actions:
                if action.action_type == "move_base":
                    if "target_pose" not in action.parameters:
                        errors.append(f"Step {step.id}: move_base action missing target_pose parameter")

                elif action.action_type == "object_detection":
                    if "target_object" not in action.parameters:
                        errors.append(f"Step {step.id}: object_detection action missing target_object parameter")

        # Check dependencies are valid
        step_ids = [step.id for step in plan.steps]
        for step in plan.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    errors.append(f"Step {step.id}: dependency {dep_id} not found")

        return len(errors) == 0, errors

# Example usage
async def example_action_translation():
    # This would require an actual LLM client
    # For demonstration, we'll show the structure
    print("LLM Action Translation Example")

    # Initialize with mock LLM client
    class MockLLMClient:
        async def chat(self):
            class Completions:
                async def create(self, **kwargs):
                    # Mock response
                    class Choice:
                        class Message:
                            content = '''{
    "goal": "Navigate to kitchen and bring cup",
    "steps": [
        {
            "id": "nav_to_kitchen",
            "name": "Navigate to kitchen",
            "category": "NAVIGATION",
            "dependencies": [],
            "actions": [
                {
                    "action_type": "move_base",
                    "parameters": {"target_pose": {"x": 2.0, "y": 1.0, "z": 0.0}},
                    "description": "Move to kitchen location"
                }
            ],
            "estimated_duration": 30.0,
            "success_criteria": ["navigation_completed"]
        }
    ]
}'''
                    class Response:
                        choices = [Choice()]
                    return Response()

            self.completions = Completions()

    mock_client = MockLLMClient()
    translator = LLMActionTranslator(mock_client)

    test_commands = [
        "Go to the kitchen and bring me a cup",
        "Find my keys and bring them to me",
        "Navigate to the living room and wait for me there"
    ]

    for command in test_commands:
        print(f"\nCommand: {command}")
        plan = await translator.translate_command(command)
        if plan:
            print(f"Goal: {plan.goal}")
            print(f"Steps: {len(plan.steps)}")
            for step in plan.steps:
                print(f"  - {step.name} ({step.category.value}): {len(step.primitive_actions)} actions")
        else:
            print("Could not translate command")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_action_translation())
```

## Planning and Execution Framework

### Hierarchical Task Decomposition

```python
# hierarchical_planning.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PlanStatus(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

@dataclass
class TaskNode:
    """Node in the hierarchical task tree"""
    id: str
    name: str
    description: str
    task_type: str  # "composite" or "primitive"
    children: List['TaskNode'] = None
    parent: Optional['TaskNode'] = None
    parameters: Dict[str, Any] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    dependencies: List[str] = None
    estimated_duration: float = 0.0
    actual_duration: Optional[float] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []

    def add_child(self, child: 'TaskNode'):
        """Add a child task node"""
        child.parent = self
        self.children.append(child)

    def remove_child(self, child_id: str) -> bool:
        """Remove a child task node by ID"""
        for i, child in enumerate(self.children):
            if child.id == child_id:
                self.children.pop(i)
                child.parent = None
                return True
        return False

    def get_ancestors(self) -> List['TaskNode']:
        """Get all ancestor nodes"""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> List['TaskNode']:
        """Get all descendant nodes"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

class HierarchicalPlanner:
    """Hierarchical task planner using LLM for decomposition"""
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.root_task = None
        self.task_graph = {}  # id -> TaskNode mapping
        self.execution_context = {}

    async def create_plan(self, goal: str, context: Dict[str, Any] = None) -> Optional[TaskNode]:
        """Create hierarchical plan using LLM"""
        prompt = f"""
Create a hierarchical plan to achieve: "{goal}"

Context information: {json.dumps(context, indent=2) if context else 'None'}

The plan should be structured as a tree with composite tasks and primitive actions.

Return the plan in JSON format:
{{
    "root_task": {{
        "id": "unique_id",
        "name": "task_name",
        "description": "what the task accomplishes",
        "task_type": "composite",  # or "primitive"
        "children": [
            {{
                "id": "child_id",
                "name": "child_task_name",
                "description": "child task description",
                "task_type": "primitive",  # leaf node
                "parameters": {{"param": "value"}},
                "estimated_duration": 5.0,
                "dependencies": ["dependency_task_id"]
            }}
        ],
        "parameters": {{"param": "value"}},
        "estimated_duration": 10.0,
        "dependencies": []
    }}
}}

Example for "Bring coffee from kitchen":
{{
    "root_task": {{
        "id": "bring_coffee",
        "name": "Bring Coffee",
        "description": "Bring coffee from kitchen to user",
        "task_type": "composite",
        "children": [
            {{
                "id": "navigate_to_kitchen",
                "name": "Navigate to Kitchen",
                "description": "Move robot to kitchen location",
                "task_type": "composite",
                "children": [
                    {{
                        "id": "plan_path_to_kitchen",
                        "name": "Plan Path to Kitchen",
                        "description": "Calculate navigation path",
                        "task_type": "primitive",
                        "parameters": {{"destination": "kitchen"}},
                        "estimated_duration": 2.0
                    }},
                    {{
                        "id": "execute_navigation",
                        "name": "Execute Navigation",
                        "description": "Move robot along calculated path",
                        "task_type": "primitive",
                        "parameters": {{"path": "calculated_path"}},
                        "estimated_duration": 25.0
                    }}
                ],
                "parameters": {{"destination": "kitchen"}},
                "estimated_duration": 30.0
            }},
            {{
                "id": "fetch_coffee",
                "name": "Fetch Coffee",
                "description": "Locate and grasp coffee",
                "task_type": "composite",
                "children": [
                    {{
                        "id": "detect_coffee",
                        "name": "Detect Coffee",
                        "description": "Find coffee in kitchen",
                        "task_type": "primitive",
                        "parameters": {{"target_object": "coffee"}},
                        "estimated_duration": 10.0
                    }},
                    {{
                        "id": "grasp_coffee",
                        "name": "Grasp Coffee",
                        "description": "Pick up coffee cup",
                        "task_type": "primitive",
                        "parameters": {{"object": "coffee"}},
                        "estimated_duration": 5.0
                    }}
                ],
                "parameters": {{"object": "coffee"}},
                "estimated_duration": 20.0
            }}
        ],
        "parameters": {{"object": "coffee", "destination": "kitchen"}},
        "estimated_duration": 60.0
    }}
}}

Now create the plan for: {goal}
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                plan_data = json.loads(json_str)

                # Build task tree
                root_task = self._build_task_tree(plan_data['root_task'])
                self.root_task = root_task
                self._index_tasks(root_task)

                return root_task

        except Exception as e:
            print(f"Error creating plan: {e}")
            return None

    def _build_task_tree(self, task_data: Dict[str, Any], parent: TaskNode = None) -> TaskNode:
        """Recursively build task tree from data"""
        task = TaskNode(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data['description'],
            task_type=task_data['task_type'],
            parameters=task_data.get('parameters', {}),
            estimated_duration=task_data.get('estimated_duration', 0.0),
            dependencies=task_data.get('dependencies', [])
        )

        task.parent = parent

        # Build children
        for child_data in task_data.get('children', []):
            child_task = self._build_task_tree(child_data, task)
            task.children.append(child_task)

        return task

    def _index_tasks(self, root_task: TaskNode):
        """Index all tasks by ID for quick lookup"""
        def index_recursive(node):
            self.task_graph[node.id] = node
            for child in node.children:
                index_recursive(child)

        index_recursive(root_task)

    def get_task_by_id(self, task_id: str) -> Optional[TaskNode]:
        """Get task by ID"""
        return self.task_graph.get(task_id)

    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        ready_tasks = []

        for task_id, task in self.task_graph.items():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                all_deps_met = True
                for dep_id in task.dependencies:
                    dep_task = self.task_graph.get(dep_id)
                    if dep_task and dep_task.status != TaskStatus.COMPLETED:
                        all_deps_met = False
                        break

                if all_deps_met:
                    ready_tasks.append(task)

        return ready_tasks

    def validate_plan(self, root_task: TaskNode) -> Tuple[bool, List[str]]:
        """Validate the plan for consistency"""
        errors = []

        def validate_recursive(node):
            # Check for circular dependencies
            ancestors = node.get_ancestors()
            ancestor_ids = [a.id for a in ancestors]

            for dep_id in node.dependencies:
                if dep_id in ancestor_ids:
                    errors.append(f"Circular dependency: {node.id} depends on its ancestor {dep_id}")

            # Validate children
            for child in node.children:
                validate_recursive(child)

        validate_recursive(root_task)

        # Check for duplicate IDs
        all_ids = [task.id for task in self.task_graph.values()]
        if len(all_ids) != len(set(all_ids)):
            errors.append("Duplicate task IDs found")

        return len(errors) == 0, errors

class PlanExecutor:
    """Executes hierarchical plans with monitoring and recovery"""
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.planner = None
        self.execution_queue = queue.Queue()
        self.execution_thread = None
        self.execution_active = False
        self.execution_results = {}
        self.failure_recovery_strategies = self._initialize_recovery_strategies()

    def _initialize_recovery_strategies(self):
        """Initialize failure recovery strategies"""
        return {
            'navigation_failure': self._handle_navigation_failure,
            'grasp_failure': self._handle_grasp_failure,
            'perception_failure': self._handle_perception_failure,
            'timeout_failure': self._handle_timeout_failure
        }

    async def execute_plan(self, plan: TaskNode, timeout: float = 300.0) -> PlanStatus:
        """Execute the hierarchical plan"""
        if not plan:
            return PlanStatus.FAILED

        self.planner = HierarchicalPlanner(None)  # We'll reuse the indexing
        self.planner.root_task = plan
        self.planner._index_tasks(plan)

        plan.status = PlanStatus.EXECUTING
        plan.started_at = time.time()

        # Start execution thread
        self.execution_active = True
        self.execution_thread = threading.Thread(target=self._execution_worker, daemon=True)
        self.execution_thread.start()

        # Monitor execution
        start_time = time.time()
        while plan.status == PlanStatus.EXECUTING:
            if time.time() - start_time > timeout:
                await self.abort_execution(plan)
                return PlanStatus.ABORTED

            # Check for plan completion
            if self._is_plan_complete(plan):
                plan.status = PlanStatus.COMPLETED
                plan.completed_at = time.time()
                break

            # Check for plan failure
            if self._is_plan_failed(plan):
                plan.status = PlanStatus.FAILED
                break

            await asyncio.sleep(0.1)  # Check every 100ms

        self.execution_active = False
        return plan.status

    def _execution_worker(self):
        """Worker thread for executing tasks"""
        while self.execution_active:
            # Get ready tasks
            ready_tasks = self.planner.get_ready_tasks()

            for task in ready_tasks:
                if not self.execution_active:
                    break

                # Execute the task
                success = self._execute_task(task)

                if success:
                    task.status = TaskStatus.COMPLETED
                    task.actual_duration = time.time() - (getattr(task, 'start_time', time.time()))
                else:
                    task.status = TaskStatus.FAILED
                    # Try recovery
                    recovered = self._attempt_recovery(task)
                    if not recovered:
                        # Propagate failure up the tree
                        self._propagate_failure(task)

            # Sleep briefly to prevent busy waiting
            time.sleep(0.05)

    def _execute_task(self, task: TaskNode) -> bool:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()

        if task.task_type == 'primitive':
            # Execute primitive action
            return self._execute_primitive_action(task)
        elif task.task_type == 'composite':
            # For composite tasks, ensure all children are completed
            return self._execute_composite_task(task)
        else:
            print(f"Unknown task type: {task.task_type}")
            return False

    def _execute_primitive_action(self, task: TaskNode) -> bool:
        """Execute a primitive action"""
        try:
            action_type = task.name.lower().replace(' ', '_')

            # Map action names to robot interface methods
            action_methods = {
                'navigate_to_kitchen': lambda: self.robot_interface.navigate_to_location('kitchen'),
                'navigate_to_bedroom': lambda: self.robot_interface.navigate_to_location('bedroom'),
                'grasp_object': lambda: self.robot_interface.grasp_object(task.parameters.get('object')),
                'detect_object': lambda: self.robot_interface.detect_object(task.parameters.get('target_object')),
                'move_base': lambda: self.robot_interface.move_to_pose(task.parameters.get('target_pose')),
                'object_detection': lambda: self.robot_interface.detect_object(task.parameters.get('target_object')),
                'speak': lambda: self.robot_interface.speak(task.parameters.get('text', '')),
            }

            if action_type in action_methods:
                success = action_methods[action_type]()
                return success
            else:
                print(f"Unknown primitive action: {action_type}")
                return False

        except Exception as e:
            print(f"Error executing task {task.id}: {e}")
            return False

    def _execute_composite_task(self, task: TaskNode) -> bool:
        """Execute a composite task (ensure children are completed)"""
        # Composite tasks are considered complete when all children are complete
        for child in task.children:
            if child.status != TaskStatus.COMPLETED:
                return False
        return True

    def _attempt_recovery(self, failed_task: TaskNode) -> bool:
        """Attempt to recover from task failure"""
        # Determine failure type
        failure_type = self._classify_failure(failed_task)

        if failure_type in self.failure_recovery_strategies:
            return self.failure_recovery_strategies[failure_type](failed_task)
        else:
            print(f"No recovery strategy for failure type: {failure_type}")
            return False

    def _classify_failure(self, task: TaskNode) -> str:
        """Classify the type of failure"""
        # This would be more sophisticated in practice
        # For now, use simple heuristics
        if 'navigate' in task.name.lower():
            return 'navigation_failure'
        elif 'grasp' in task.name.lower() or 'pick' in task.name.lower():
            return 'grasp_failure'
        elif 'detect' in task.name.lower() or 'find' in task.name.lower():
            return 'perception_failure'
        else:
            return 'general_failure'

    def _handle_navigation_failure(self, task: TaskNode) -> bool:
        """Handle navigation failure"""
        print(f"Handling navigation failure for task: {task.id}")
        # Try alternative path
        # Retry with different parameters
        # Report failure to user
        return False  # For now, don't recover

    def _handle_grasp_failure(self, task: TaskNode) -> bool:
        """Handle grasp failure"""
        print(f"Handling grasp failure for task: {task.id}")
        # Try different grasp approach
        # Check if object is accessible
        # Report failure
        return False

    def _handle_perception_failure(self, task: TaskNode) -> bool:
        """Handle perception failure"""
        print(f"Handling perception failure for task: {task.id}")
        # Try different viewing angle
        # Adjust lighting conditions
        # Report failure
        return False

    def _handle_timeout_failure(self, task: TaskNode) -> bool:
        """Handle timeout failure"""
        print(f"Handling timeout failure for task: {task.id}")
        # Report timeout to user
        return False

    def _propagate_failure(self, failed_task: TaskNode):
        """Propagate failure up the task hierarchy"""
        # Mark all dependent tasks as failed
        for task_id, task in self.planner.task_graph.items():
            if failed_task.id in task.dependencies:
                task.status = TaskStatus.FAILED

    def _is_plan_complete(self, plan: TaskNode) -> bool:
        """Check if the plan is complete"""
        def check_complete(node):
            if node.status != TaskStatus.COMPLETED:
                return False
            for child in node.children:
                if not check_complete(child):
                    return False
            return True

        return check_complete(plan)

    def _is_plan_failed(self, plan: TaskNode) -> bool:
        """Check if the plan has failed"""
        def check_failed(node):
            if node.status == TaskStatus.FAILED:
                return True
            for child in node.children:
                if check_failed(child):
                    return True
            return False

        return check_failed(plan)

    async def abort_execution(self, plan: TaskNode):
        """Abort plan execution"""
        self.execution_active = False
        if self.execution_thread:
            self.execution_thread.join(timeout=1.0)

        # Mark all running tasks as cancelled
        for task_id, task in self.planner.task_graph.items():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED

        plan.status = PlanStatus.ABORTED

# Example usage
async def example_hierarchical_planning():
    print("Hierarchical Planning Example")

    # This would require a real LLM client and robot interface
    # For demonstration, we'll show the structure

    class MockLLMClient:
        async def chat(self):
            class Completions:
                async def create(self, **kwargs):
                    # Mock response
                    class Choice:
                        class Message:
                            content = '''{
    "root_task": {
        "id": "bring_coffee",
        "name": "Bring Coffee",
        "description": "Bring coffee from kitchen to user",
        "task_type": "composite",
        "children": [
            {
                "id": "navigate_to_kitchen",
                "name": "Navigate to Kitchen",
                "description": "Move robot to kitchen location",
                "task_type": "composite",
                "children": [
                    {
                        "id": "plan_path_to_kitchen",
                        "name": "Plan Path to Kitchen",
                        "description": "Calculate navigation path",
                        "task_type": "primitive",
                        "parameters": {"destination": "kitchen"},
                        "estimated_duration": 2.0
                    },
                    {
                        "id": "execute_navigation",
                        "name": "Execute Navigation",
                        "description": "Move robot along calculated path",
                        "task_type": "primitive",
                        "parameters": {"path": "calculated_path"},
                        "estimated_duration": 25.0
                    }
                ],
                "parameters": {"destination": "kitchen"},
                "estimated_duration": 30.0
            }
        ],
        "parameters": {"object": "coffee", "destination": "kitchen"},
        "estimated_duration": 60.0
    }
}'''
                    class Response:
                        choices = [Choice()]
                    return Response()

            self.completions = Completions()

    class MockRobotInterface:
        def navigate_to_location(self, location):
            print(f"Navigating to {location}")
            time.sleep(1)  # Simulate navigation
            return True

        def grasp_object(self, obj):
            print(f"Grasping {obj}")
            time.sleep(0.5)
            return True

        def detect_object(self, obj):
            print(f"Detecting {obj}")
            time.sleep(0.3)
            return True

        def move_to_pose(self, pose):
            print(f"Moving to pose: {pose}")
            time.sleep(1)
            return True

        def speak(self, text):
            print(f"Speaking: {text}")
            return True

    # Initialize components
    llm_client = MockLLMClient()
    robot_interface = MockRobotInterface()
    planner = HierarchicalPlanner(llm_client)
    executor = PlanExecutor(robot_interface)

    # Create a plan
    goal = "Go to the kitchen and bring me a cup of coffee"
    plan = await planner.create_plan(goal)

    if plan:
        print(f"Created plan: {plan.name}")
        print(f"Root task: {plan.description}")
        print(f"Children: {len(plan.children)}")

        # Execute the plan
        status = await executor.execute_plan(plan, timeout=120.0)
        print(f"Plan execution status: {status.value}")

if __name__ == "__main__":
    import asyncio
    import json
    asyncio.run(example_hierarchical_planning())
```

## Task Decomposition and Execution

### Advanced Task Decomposition with LLMs

```python
# advanced_task_decomposition.py
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import re

class TaskDecompositionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLELIZABLE = "parallelizable"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"

class ExecutionPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class DecomposedTask:
    """A task decomposed by LLM"""
    id: str
    name: str
    description: str
    subtasks: List['DecomposedTask']
    action_sequence: List[Dict[str, Any]]  # Low-level actions
    prerequisites: List[str]  # Task IDs that must complete first
    estimated_duration: float
    priority: ExecutionPriority
    success_criteria: List[str]
    failure_modes: List[str]
    recovery_strategies: List[str]

    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.action_sequence is None:
            self.action_sequence = []
        if self.prerequisites is None:
            self.prerequisites = []
        if self.success_criteria is None:
            self.success_criteria = []
        if self.failure_modes is None:
            self.failure_modes = []
        if self.recovery_strategies is None:
            self.recovery_strategies = []

class TaskDecomposer:
    """Advanced task decomposer using LLMs"""
    def __init__(self, llm_client, strategy: TaskDecompositionStrategy = TaskDecompositionStrategy.HIERARCHICAL):
        self.llm_client = llm_client
        self.strategy = strategy
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base for task decomposition"""
        return {
            "navigation_tasks": {
                "common_sequences": [
                    ["localize", "plan_path", "execute_path", "reach_goal"]
                ],
                "prerequisites": {
                    "localize": [],
                    "plan_path": ["localize"],
                    "execute_path": ["plan_path"],
                    "reach_goal": ["execute_path"]
                },
                "failure_modes": ["obstacle_detected", "lost_localization", "goal_unreachable"]
            },
            "manipulation_tasks": {
                "common_sequences": [
                    ["detect_object", "plan_grasp", "execute_grasp", "verify_grasp"]
                ],
                "prerequisites": {
                    "detect_object": [],
                    "plan_grasp": ["detect_object"],
                    "execute_grasp": ["plan_grasp"],
                    "verify_grasp": ["execute_grasp"]
                },
                "failure_modes": ["object_not_found", "grasp_failed", "object_dropped"]
            },
            "perception_tasks": {
                "common_sequences": [
                    ["acquire_sensor_data", "process_data", "interpret_results", "report_findings"]
                ],
                "prerequisites": {
                    "acquire_sensor_data": [],
                    "process_data": ["acquire_sensor_data"],
                    "interpret_results": ["process_data"],
                    "report_findings": ["interpret_results"]
                },
                "failure_modes": ["sensor_failure", "data_corruption", "misidentification"]
            }
        }

    async def decompose_task(self, goal: str, context: Dict[str, Any] = None) -> Optional[DecomposedTask]:
        """Decompose a high-level goal into executable tasks"""
        prompt = self._create_decomposition_prompt(goal, context)

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                task_data = json.loads(json_str)

                # Build task structure recursively
                return self._build_task_from_data(task_data)

        except Exception as e:
            print(f"Error decomposing task: {e}")
            return None

    def _create_decomposition_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        """Create prompt for task decomposition"""
        return f"""
Decompose the following goal into executable subtasks: "{goal}"

Context: {json.dumps(context, indent=2) if context else 'No additional context'}

Decompose the task using the strategy: {self.strategy.value}

Provide the response in JSON format:

{{
    "id": "unique_task_id",
    "name": "task_name",
    "description": "what this task accomplishes",
    "subtasks": [
        {{
            "id": "subtask_id",
            "name": "subtask_name",
            "description": "subtask description",
            "subtasks": [...],  # Further decomposition if needed
            "action_sequence": [
                {{
                    "action": "specific_action_name",
                    "parameters": {{"param": "value"}},
                    "description": "what this action does"
                }}
            ],
            "prerequisites": ["prerequisite_task_id"],
            "estimated_duration": 5.0,
            "priority": "NORMAL",  # CRITICAL, HIGH, NORMAL, LOW
            "success_criteria": ["list", "of", "success", "conditions"],
            "failure_modes": ["potential", "failure", "scenarios"],
            "recovery_strategies": ["ways", "to", "recover", "from", "failures"]
        }}
    ],
    "action_sequence": [...],  # Combined actions from all subtasks
    "prerequisites": [...],  # Overall prerequisites
    "estimated_duration": 10.0,  # Total estimated duration
    "priority": "NORMAL",
    "success_criteria": [...],
    "failure_modes": [...],
    "recovery_strategies": [...]
}}

Example for "Navigate to kitchen and find the red cup":
{{
    "id": "find_red_cup",
    "name": "Find Red Cup",
    "description": "Navigate to kitchen and locate red cup",
    "subtasks": [
        {{
            "id": "navigate_to_kitchen",
            "name": "Navigate to Kitchen",
            "description": "Move robot to kitchen location",
            "subtasks": [],
            "action_sequence": [
                {{
                    "action": "localize_robot",
                    "parameters": {{}},
                    "description": "Determine current robot location"
                }},
                {{
                    "action": "plan_path_to_kitchen",
                    "parameters": {{"destination": "kitchen"}},
                    "description": "Calculate path to kitchen"
                }},
                {{
                    "action": "execute_navigation",
                    "parameters": {{"path": "calculated_path", "speed": "normal"}},
                    "description": "Navigate to kitchen"
                }}
            ],
            "prerequisites": [],
            "estimated_duration": 30.0,
            "priority": "HIGH",
            "success_criteria": ["robot_reached_kitchen", "navigation_successful"],
            "failure_modes": ["path_blocked", "localization_lost"],
            "recovery_strategies": ["replan_path", "request_assistance"]
        }},
        {{
            "id": "detect_red_cup",
            "name": "Detect Red Cup",
            "description": "Find red cup in kitchen environment",
            "subtasks": [],
            "action_sequence": [
                {{
                    "action": "activate_camera",
                    "parameters": {{"resolution": "high"}},
                    "description": "Turn on camera for object detection"
                }},
                {{
                    "action": "detect_objects",
                    "parameters": {{"target_color": "red", "target_shape": "cup"}},
                    "description": "Detect red cup-shaped objects"
                }},
                {{
                    "action": "verify_detection",
                    "parameters": {{"confidence_threshold": 0.8}},
                    "description": "Verify object detection"
                }}
            ],
            "prerequisites": ["navigate_to_kitchen"],
            "estimated_duration": 15.0,
            "priority": "NORMAL",
            "success_criteria": ["red_cup_detected", "confidence_high"],
            "failure_modes": ["no_red_cup_found", "detection_uncertain"],
            "recovery_strategies": ["expand_search_area", "adjust_detection_params"]
        }}
    ],
    "action_sequence": [
        // Combined from all subtasks
    ],
    "prerequisites": [],
    "estimated_duration": 45.0,
    "priority": "NORMAL",
    "success_criteria": ["red_cup_located", "position_known"],
    "failure_modes": ["kitchen_access_denied", "object_not_found"],
    "recovery_strategies": ["try_different_path", "search_other_areas"]
}}

Now decompose the goal: {goal}
"""

    def _build_task_from_data(self, task_data: Dict[str, Any]) -> DecomposedTask:
        """Build task structure from JSON data"""
        subtasks = []
        for subtask_data in task_data.get('subtasks', []):
            subtask = self._build_task_from_data(subtask_data)
            subtasks.append(subtask)

        # Convert priority string to enum
        priority_str = task_data.get('priority', 'NORMAL')
        priority_enum = ExecutionPriority[priority_str.upper()]

        task = DecomposedTask(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data['description'],
            subtasks=subtasks,
            action_sequence=task_data.get('action_sequence', []),
            prerequisites=task_data.get('prerequisites', []),
            estimated_duration=task_data.get('estimated_duration', 10.0),
            priority=priority_enum,
            success_criteria=task_data.get('success_criteria', []),
            failure_modes=task_data.get('failure_modes', []),
            recovery_strategies=task_data.get('recovery_strategies', [])
        )

        return task

    def analyze_task_dependencies(self, task: DecomposedTask) -> Dict[str, List[str]]:
        """Analyze dependencies between subtasks"""
        dependencies = {}

        def analyze_recursive(current_task: DecomposedTask):
            deps = current_task.prerequisites
            dependencies[current_task.id] = deps

            for subtask in current_task.subtasks:
                analyze_recursive(subtask)

        analyze_recursive(task)
        return dependencies

    def optimize_execution_order(self, task: DecomposedTask) -> List[str]:
        """Optimize execution order based on dependencies and priorities"""
        # Topological sort with priority consideration
        all_task_ids = []

        def collect_ids_recursive(current_task: DecomposedTask):
            all_task_ids.append(current_task.id)
            for subtask in current_task.subtasks:
                collect_ids_recursive(subtask)

        collect_ids_recursive(task)

        # Build dependency graph
        dependencies = self.analyze_task_dependencies(task)

        # Topological sort
        from collections import defaultdict, deque

        graph = defaultdict(list)
        in_degree = {task_id: 0 for task_id in all_task_ids}

        for task_id, prereqs in dependencies.items():
            for prereq in prereqs:
                graph[prereq].append(task_id)
                in_degree[task_id] += 1

        # Kahn's algorithm for topological sort
        queue = deque([task_id for task_id in all_task_ids if in_degree[task_id] == 0])
        sorted_order = []

        while queue:
            current = queue.popleft()
            sorted_order.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If there are cycles, return original order
        if len(sorted_order) != len(all_task_ids):
            return all_task_ids  # Cycle detected, return original

        return sorted_order

    def generate_execution_plan(self, task: DecomposedTask) -> Dict[str, Any]:
        """Generate detailed execution plan"""
        execution_plan = {
            "task_id": task.id,
            "task_name": task.name,
            "execution_order": self.optimize_execution_order(task),
            "timeline": {},
            "resource_requirements": {},
            "monitoring_points": [],
            "recovery_procedures": {}
        }

        # Calculate timeline
        current_time = 0.0
        for task_id in execution_plan["execution_order"]:
            task_obj = self._find_task_by_id(task, task_id)
            if task_obj:
                execution_plan["timeline"][task_id] = {
                    "start_time": current_time,
                    "end_time": current_time + task_obj.estimated_duration,
                    "duration": task_obj.estimated_duration
                }
                current_time += task_obj.estimated_duration

        # Collect resource requirements
        all_tasks = self._collect_all_tasks(task)
        for task_obj in all_tasks:
            for action in task_obj.action_sequence:
                action_type = action.get('action', 'unknown')
                # In practice, this would map to actual robot resources

        # Set monitoring points at critical junctures
        execution_plan["monitoring_points"] = self._identify_monitoring_points(task)

        # Map recovery procedures
        for task_obj in all_tasks:
            if task_obj.failure_modes:
                execution_plan["recovery_procedures"][task_obj.id] = {
                    "failure_modes": task_obj.failure_modes,
                    "strategies": task_obj.recovery_strategies
                }

        return execution_plan

    def _find_task_by_id(self, root_task: DecomposedTask, task_id: str) -> Optional[DecomposedTask]:
        """Find task by ID in the task hierarchy"""
        if root_task.id == task_id:
            return root_task

        for subtask in root_task.subtasks:
            found = self._find_task_by_id(subtask, task_id)
            if found:
                return found

        return None

    def _collect_all_tasks(self, root_task: DecomposedTask) -> List[DecomposedTask]:
        """Collect all tasks in the hierarchy"""
        tasks = [root_task]

        for subtask in root_task.subtasks:
            tasks.extend(self._collect_all_tasks(subtask))

        return tasks

    def _identify_monitoring_points(self, task: DecomposedTask) -> List[str]:
        """Identify critical monitoring points"""
        monitoring_points = []

        # Add monitoring for critical tasks
        all_tasks = self._collect_all_tasks(task)
        for task_obj in all_tasks:
            if task_obj.priority == ExecutionPriority.CRITICAL:
                monitoring_points.append(task_obj.id)
            elif task_obj.failure_modes:  # Tasks with failure modes
                monitoring_points.append(task_obj.id)

        return monitoring_points

class TaskExecutionMonitor:
    """Monitors task execution and handles failures"""
    def __init__(self):
        self.execution_log = []
        self.current_task = None
        self.task_progress = {}
        self.failure_count = 0
        self.success_count = 0

    def start_task_execution(self, task: DecomposedTask):
        """Start monitoring task execution"""
        self.current_task = task
        self.task_progress[task.id] = {
            "start_time": time.time(),
            "status": "running",
            "subtask_progress": {}
        }

    def update_task_progress(self, task_id: str, progress: float, status: str = "running"):
        """Update progress for a specific task"""
        if task_id in self.task_progress:
            self.task_progress[task_id]["progress"] = progress
            self.task_progress[task_id]["status"] = status
            self.task_progress[task_id]["last_update"] = time.time()

    def log_execution_event(self, event_type: str, task_id: str, details: Dict[str, Any]):
        """Log execution events"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "task_id": task_id,
            "details": details
        }
        self.execution_log.append(event)

    def check_for_failures(self, task: DecomposedTask) -> List[Dict[str, Any]]:
        """Check for task failures"""
        failures = []

        # Check timeout
        if task.id in self.task_progress:
            start_time = self.task_progress[task.id].get("start_time", time.time())
            current_time = time.time()
            estimated_duration = task.estimated_duration * 1.5  # 150% of estimated time

            if current_time - start_time > estimated_duration:
                failures.append({
                    "task_id": task.id,
                    "failure_type": "timeout",
                    "severity": "high",
                    "suggested_action": "abort_or_retry"
                })

        return failures

    def generate_execution_report(self) -> Dict[str, Any]:
        """Generate execution report"""
        total_tasks = len(self.task_progress)
        completed_tasks = sum(1 for progress in self.task_progress.values()
                            if progress.get("status") == "completed")

        report = {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "execution_time": time.time() - min([p["start_time"] for p in self.task_progress.values()] or [time.time()]),
            "detailed_progress": self.task_progress.copy(),
            "events": self.execution_log[-20:]  # Last 20 events
        }

        return report

# Example usage
async def example_task_decomposition():
    print("Advanced Task Decomposition Example")

    # Mock LLM client for demonstration
    class MockLLMClient:
        async def chat(self):
            class Completions:
                async def create(self, **kwargs):
                    # Mock response for "Clean the living room"
                    class Choice:
                        class Message:
                            content = '''{
    "id": "clean_living_room",
    "name": "Clean Living Room",
    "description": "Clean the living room by organizing and tidying up",
    "subtasks": [
        {
            "id": "organize_table",
            "name": "Organize Table",
            "description": "Put items back in their proper places on the table",
            "subtasks": [],
            "action_sequence": [
                {
                    "action": "detect_items_on_table",
                    "parameters": {"table_location": "center"},
                    "description": "Identify items on the table"
                },
                {
                    "action": "categorize_items",
                    "parameters": {"items": ["books", "cups", "papers"]},
                    "description": "Sort items by category"
                },
                {
                    "action": "place_items_in_order",
                    "parameters": {"organized_placement": true},
                    "description": "Place items in organized fashion"
                }
            ],
            "prerequisites": [],
            "estimated_duration": 15.0,
            "priority": "HIGH",
            "success_criteria": ["table_organized", "items_in_places"],
            "failure_modes": ["unknown_item", "space_insufficient"],
            "recovery_strategies": ["ask_for_help", "skip_unknown"]
        }
    ],
    "action_sequence": [
        {
            "action": "detect_items_on_table",
            "parameters": {"table_location": "center"},
            "description": "Identify items on the table"
        }
    ],
    "prerequisites": [],
    "estimated_duration": 15.0,
    "priority": "NORMAL",
    "success_criteria": ["living_room_cleaned"],
    "failure_modes": ["object_unmovable", "space_full"],
    "recovery_strategies": ["request_assistance", "alternative_storage"]
}'''
                    class Response:
                        choices = [Choice()]
                    return Response()

            self.completions = Completions()

    # Initialize components
    llm_client = MockLLMClient()
    decomposer = TaskDecomposer(llm_client, TaskDecompositionStrategy.HIERARCHICAL)
    monitor = TaskExecutionMonitor()

    # Define a complex goal
    goal = "Clean the living room by organizing the center table"

    print(f"Decomposing goal: {goal}")

    # Decompose the task
    task = await decomposer.decompose_task(goal)

    if task:
        print(f"Task: {task.name}")
        print(f"Description: {task.description}")
        print(f"Subtasks: {len(task.subtasks)}")
        print(f"Estimated duration: {task.estimated_duration}s")
        print(f"Priority: {task.priority.value}")

        # Analyze dependencies
        dependencies = decomposer.analyze_task_dependencies(task)
        print(f"Dependencies: {dependencies}")

        # Generate execution plan
        execution_plan = decomposer.generate_execution_plan(task)
        print(f"Execution order: {execution_plan['execution_order']}")
        print(f"Timeline: {execution_plan['timeline']}")

        # Monitor execution (simulated)
        monitor.start_task_execution(task)

        # Simulate task execution
        for task_id in execution_plan['execution_order']:
            print(f"Executing: {task_id}")
            monitor.update_task_progress(task_id, 1.0, "completed")
            time.sleep(0.1)  # Simulate execution

        # Generate report
        report = monitor.generate_execution_report()
        print(f"Execution report: {report}")
    else:
        print("Failed to decompose task")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_task_decomposition())
```

## Knowledge Check

1. How do LLMs bridge the gap between natural language commands and executable robot actions?
2. What are the key components of a cognitive planning system?
3. How does hierarchical task decomposition improve complex task execution?
4. What strategies can be used for failure recovery during plan execution?

## Summary

This chapter covered cognitive planning with LLMs, focusing on translating natural language commands into executable robotic actions. We explored intent recognition and action extraction, developed planning and execution frameworks, and implemented advanced task decomposition systems. The chapter provided practical examples of how to create hierarchical plans, manage task dependencies, and execute complex tasks while monitoring for failures and adapting to changing conditions.

## Next Steps

In the final chapter, we'll explore the complete implementation of the autonomous humanoid system, integrating all the components we've developed into a cohesive whole, and discuss the capstone project of building an autonomous humanoid robot that can process voice commands and execute complex tasks using LLMs and physical AI.