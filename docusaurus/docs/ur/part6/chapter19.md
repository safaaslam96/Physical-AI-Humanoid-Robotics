---
title: "چیپٹر 19: بڑے زبانی ماڈلز کے ساتھ شعوری منصوبہ بندی"
sidebar_label: "چیپٹر 19: بڑے زبانی ماڈلز کے ساتھ شعوری منصوبہ بندی"
---

# چیپٹر 19: بڑے زبانی ماڈلز کے ساتھ شعوری منصوبہ بندی

## سیکھنے کے اہداف
- سمجھیں کہ بڑے زبانی ماڈلز کو روبوٹکس میں شعوری منصوبہ بندی کے لیے کیسے استعمال کیا جاتا ہے
- ہیروارکیکل منصوبہ بندی کے نظام کو LLMs کے ساتھ نافذ کریں
- علامتی منصوبہ بندی اور نیورل نیٹ ورکس کو جوڑنے والے دلائل کے ڈھانچے کو ڈیزائن کریں
- حقیقی دنیا کی روبوٹک ایپلی کیشنز کے لیے LLM-بیسڈ منصوبہ بندی کا جائزہ لیں اور بہتر کریں

## تعارف

شعوری منصوبہ بندی روبوٹکس میں مصنوعی ذہانت کے اعلیٰ ترین درجے کی نمائندگی کرتی ہے، جس سے ہیومنوڈ روبوٹس کو پیچیدہ، متعدد اقدامات والے کاموں کو سمجھنے اور پیچیدہ ایکشن سیکوئنس جنریٹ کرنے کے قابل بنایا جا سکتا ہے۔ بڑے زبانی ماڈلز (LLMs) جامع سطح کی سوچ، متعدد اقدامات والے کاموں کو تقسیم کرنے، متعدد منصوبہ بندی کے متبادل حل سوچنے، اور تبدیل ہوتے حالات کے مطابق اپنے منصوبے کو ایڈجسٹ کرنے کی بے مثال صلاحیتیں فراہم کرتے ہیں۔ یہ چیپٹر LLMs کو روبوٹک منصوبہ بندی کے نظام میں انضمام کا جائزہ لیتا ہے، ایسے شعوری ڈھانچے تخلیق کرتا ہے جو حقیقی دنیا کے ماحول میں پیچیدگی اور عدم یقینی کو سنبھال سکتے ہیں۔

## LLM-بیسڈ شعوری منصوبہ بندی کی بنیادیں

### منصوبہ بندی کے بطور زبان جنریشن

LLM-بیسڈ منصوبہ بندی منصوبہ بندی کے مسئلے کو ایک زبان جنریشن ٹاسک کے طور پر سمجھتی ہے، جہاں ماڈل اہداف کو پورا کرنے کے لیے ایکشن سیکوئنس جنریٹ کرتا ہے:

```python
# LLM-بیسڈ منصوبہ بندی کا نظام
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from typing import List, Dict, Any

class LLMPlanner:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # اگر پیڈنگ ٹوکن موجود نہ ہو تو اسے شامل کریں
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # منصوبہ بندی کے لیے مخصوص ویکیبیلری اور ٹیمپلیٹس
        self.planning_templates = {
            'task_decomposition': (
                "کام '{task}' کو اقدامات کی ترتیب میں تقسیم کریں۔ "
                "اقدامات ہونے چاہئیں: 1) ترتیب سے انجام دیئے جا سکیں، "
                "2) منطقی طور پر جڑے ہوئے، 3) کل ہدف کو پورا کریں۔ "
                "اقدامات:"
            ),
            'action_generation': (
                "موجودہ حالت: {state} اور ہدف: {goal} کو دیکھتے ہوئے، "
                "اگلا ایکشن جنریٹ کریں۔ اس بات کو مدنظر رکھیں: {constraints}. "
                "ایکشن:"
            ),
            'plan_refinement': (
                "منصوبہ: {plan} کو اس بات کو مدنظر رکھتے ہوئے بہتر کریں: {feedback}. "
                "ایک بہتر منصوبہ واپس کریں:"
            )
        }

    def decompose_task(self, high_level_task: str, context: Dict[str, Any]) -> List[str]:
        """
        LLM کا استعمال کرتے ہوئے اعلیٰ سطح کے کام کو ذیلی کاموں میں تقسیم کریں
        """
        prompt = self.planning_templates['task_decomposition'].format(
            task=high_level_task
        )

        # سیاق و سباق کی معلومات شامل کریں
        context_str = self.format_context(context)
        full_prompt = f"{prompt}\nسیاق و سباق: {context_str}\n"

        # ذیلی کام جنریٹ کریں
        inputs = self.tokenizer.encode(full_prompt, return_tensors='pt', truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ذیلی کاموں کو جواب سے نکالیں
        subtasks = self.parse_subtasks(response, full_prompt)
        return subtasks

    def generate_action(self, current_state: Dict[str, Any], goal: str,
                       constraints: List[str] = None) -> str:
        """
        موجودہ حالت اور ہدف کے مطابق اگلا ایکشن جنریٹ کریں
        """
        state_str = json.dumps(current_state, indent=2)
        constraints_str = ", ".join(constraints) if constraints else "none"

        prompt = self.planning_templates['action_generation'].format(
            state=state_str,
            goal=goal,
            constraints=constraints_str
        )

        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        action = self.extract_action(response, prompt)

        return action

    def refine_plan(self, current_plan: List[str], feedback: Dict[str, Any]) -> List[str]:
        """
        فیڈ بیک کی بنیاد پر موجودہ منصوبہ کو بہتر کریں
        """
        plan_str = "\n".join(current_plan)
        feedback_str = json.dumps(feedback, indent=2)

        prompt = self.planning_templates['plan_refinement'].format(
            plan=plan_str,
            feedback=feedback_str
        )

        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                num_return_sequences=1,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refined_plan = self.parse_plan(response, prompt)

        return refined_plan

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        منصوبہ بندی کے لیے سیاق و سباق کی شکل دیں
        """
        formatted_parts = []

        if 'environment' in context:
            formatted_parts.append(f"ماحول: {json.dumps(context['environment'])}")

        if 'robot_capabilities' in context:
            formatted_parts.append(f"صلاحیتیں: {context['robot_capabilities']}")

        if 'constraints' in context:
            formatted_parts.append(f"پابندیاں: {context['constraints']}")

        if 'previous_attempts' in context:
            formatted_parts.append(f"پچھلے اقدامات: {context['previous_attempts']}")

        return " | ".join(formatted_parts)

    def parse_subtasks(self, response: str, original_prompt: str) -> List[str]:
        """
        LLM کے جواب سے ذیلی کام نکالیں
        """
        # اصل پرامپٹ کو جواب سے ہٹا دیں
        response_clean = response[len(original_prompt):].strip()

        # ذیلی کاموں کے عام اشاروں کے مطابق تقسیم کریں
        import re
        subtasks = re.split(r'\d+\.\s*|\n\s*\n\s*|\*\s*', response_clean)

        # صاف کریں اور ذیلی کاموں کو فلٹر کریں
        subtasks = [task.strip() for task in subtasks if task.strip()]

        return subtasks[:10]  # 10 ذیلی کاموں تک محدود

    def extract_action(self, response: str, original_prompt: str) -> str:
        """
        LLM کے جواب سے ایکشن نکالیں
        """
        response_clean = response[len(original_prompt):].strip()
        # پہلا مکمل جملہ ایکشن کے طور پر لیں
        sentences = response_clean.split('.')
        return sentences[0].strip() if sentences else response_clean

    def parse_plan(self, response: str, original_prompt: str) -> List[str]:
        """
        LLM کے جواب سے بہتر منصوبہ نکالیں
        """
        response_clean = response[len(original_prompt):].strip()
        # منصوبہ کے اشاروں کے مطابق تقسیم کریں
        import re
        plan_steps = re.split(r'\d+\.\s*|\n\s*\n\s*|\*\s*', response_clean)
        plan_steps = [step.strip() for step in plan_steps if step.strip()]
        return plan_steps[:20]  # 20 اقدامات تک محدود
```

### ہیروارکیکل منصوبہ بندی کا آرکیٹیکچر

ہیروارکیکل منصوبہ بندی پیچیدہ کاموں کو قابلِ انتظام سطحوں میں مرتب کرتی ہے:

```python
# ہیروارکیکل منصوبہ بندی کا نظام
class HierarchicalLLMPlanner:
    def __init__(self):
        self.high_level_planner = LLMPlanner()
        self.mid_level_planner = LLMPlanner()
        self.low_level_planner = LLMPlanner()
        self.plan_cache = {}
        self.execution_monitor = ExecutionMonitor()

    def generate_hierarchical_plan(self, high_level_goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        متعدد سطحوں کی انتہا کے ساتھ ہیروارکیکل منصوبہ جنریٹ کریں
        """
        plan = {
            'high_level': [],
            'mid_level': {},
            'low_level': {},
            'dependencies': {}
        }

        # اعلیٰ سطح کا منصوبہ جنریٹ کریں
        high_level_tasks = self.high_level_planner.decompose_task(
            high_level_goal, context
        )
        plan['high_level'] = high_level_tasks

        # ہر اعلیٰ سطح کے کام کے لیے درمیانی سطح کے منصوبے جنریٹ کریں
        for i, task in enumerate(high_level_tasks):
            mid_level_context = self.update_context_for_task(context, task, i)
            mid_level_tasks = self.mid_level_planner.decompose_task(
                task, mid_level_context
            )
            plan['mid_level'][f'task_{i}'] = mid_level_tasks

            # ہر درمیانی سطح کے کام کے لیے کم سطح کے منصوبے جنریٹ کریں
            for j, mid_task in enumerate(mid_level_tasks):
                low_level_context = self.update_context_for_task(
                    mid_level_context, mid_task, j
                )
                low_level_actions = self.generate_low_level_actions(
                    mid_task, low_level_context
                )
                plan['low_level'][f'task_{i}_sub_{j}'] = low_level_actions

        # کاموں کے درمیان انحصاریتیں طے کریں
        plan['dependencies'] = self.calculate_dependencies(plan)

        return plan

    def update_context_for_task(self, context: Dict[str, Any], task: str, index: int) -> Dict[str, Any]:
        """
        مخصوص کام کے لیے سیاق و سباق کو اپ ڈیٹ کریں
        """
        updated_context = context.copy()
        updated_context['current_task'] = task
        updated_context['task_index'] = index
        updated_context['task_progress'] = f"{index}/{len(context.get('high_level_tasks', []))}"

        return updated_context

    def generate_low_level_actions(self, mid_task: str, context: Dict[str, Any]) -> List[str]:
        """
        قابلِ انجام کم سطح کے ایکشن جنریٹ کریں
        """
        # کم سطح کی منصوبہ بندی کے لیے، مخصوص ٹیمپلیٹس اور پابندیوں کا استعمال کریں
        state = context.get('current_state', {})
        goal = mid_task

        # بنیادی ایکشن کی ترتیب جنریٹ کریں
        actions = []
        current_state = state.copy()

        # متعدد اقدامات کی ترتیب کا شبیہہ بنائیں
        for step in range(5):  # ہر درمیانی سطح کے کام کے لیے 5 اقدامات تک محدود
            action = self.low_level_planner.generate_action(
                current_state, goal,
                constraints=self.get_execution_constraints(context)
            )

            if self.is_valid_action(action, current_state):
                actions.append(action)
                # ایکشن کے مطابق حالت کو اپ ڈیٹ کریں
                current_state = self.apply_action_to_state(action, current_state)

                # یہ دیکھیں کہ ہدف حاصل کر لیا گیا ہے یا نہیں
                if self.is_goal_achieved(mid_task, current_state):
                    break
            else:
                break  # غلط ایکشن، منصوبہ بندی بند کریں
        return actions

    def get_execution_constraints(self, context: Dict[str, Any]) -> List[str]:
        """
        موجودہ سیاق و سباق کے لیے انجام دہی کی پابندیاں حاصل کریں
        """
        constraints = [
            "ایکشن روبوٹ کے ذریعہ انجام دیئے جا سکتے ہیں",
            "حفاظت کی پابندیوں کو مدنظر رکھیں",
            "جسمانی حدود کا احترام کریں"
        ]

        if 'environment' in context:
            env_constraints = context['environment'].get('constraints', [])
            constraints.extend(env_constraints)

        if 'robot_state' in context:
            robot_constraints = context['robot_state'].get('limitations', [])
            constraints.extend(robot_constraints)

        return constraints

    def is_valid_action(self, action: str, state: Dict[str, Any]) -> bool:
        """
        یہ دیکھیں کہ موجودہ حالت کے مطابق ایکشن درست ہے یا نہیں
        """
        # سادہ توثیق - عملی طور پر، یہ زیادہ تفصیلی ہوگی
        invalid_keywords = ['impossible', 'cannot', 'not possible']
        action_lower = action.lower()

        for keyword in invalid_keywords:
            if keyword in action_lower:
                return False

        return True

    def apply_action_to_state(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        حالت کو ایکشن کے مطابق اپ ڈیٹ کریں
        """
        # یہ حقیقی روبوٹ حالت کے ساتھ انضمام کرے گا
        new_state = state.copy()

        # ایکشن کی قسم کے مطابق حالت کو اپ ڈیٹ کریں
        if 'move' in action.lower() or 'go' in action.lower():
            new_state['position'] = self.calculate_new_position(action, state)
        elif 'pick' in action.lower() or 'grasp' in action.lower():
            new_state['holding'] = self.extract_object(action)

        return new_state

    def is_goal_achieved(self, goal: str, state: Dict[str, Any]) -> bool:
        """
        یہ دیکھیں کہ موجودہ حالت کے مطابق ہدف حاصل کر لیا گیا ہے یا نہیں
        """
        # سادہ ہدف کی جانچ - عملی طور پر، یہ زیادہ تفصیلی ہوگی
        goal_lower = goal.lower()
        state_str = str(state).lower()

        # دیکھیں کہ کیا ہدف سے متعلق الفاظ حالت میں ہیں
        return any(term in state_str for term in goal_lower.split())

    def calculate_dependencies(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        منصوبہ کے اجزاء کے درمیان انحصاریتیں حساب کریں
        """
        dependencies = {}

        # اعلیٰ سطح کی انحصاریتیں
        for i in range(1, len(plan['high_level'])):
            dependencies[f'task_{i}'] = [f'task_{i-1}']

        # ہر اعلیٰ سطح کے کام کے اندر درمیانی سطح کی انحصاریتیں
        for task_key, mid_tasks in plan['mid_level'].items():
            for j in range(1, len(mid_tasks)):
                dependencies[f'{task_key}_sub_{j}'] = [f'{task_key}_sub_{j-1}']

        return dependencies

    def execute_plan_with_monitoring(self, plan: Dict[str, Any], robot_interface) -> Dict[str, Any]:
        """
        نگرانی اور مطابقت کے ساتھ ہیروارکیکل منصوبہ انجام دیں
        """
        execution_result = {
            'success': True,
            'completed_tasks': [],
            'failed_tasks': [],
            'adaptations': []
        }

        for i, high_task in enumerate(plan['high_level']):
            task_key = f'task_{i}'

            # اس اعلیٰ سطح کے کام کے لیے درمیانی سطح کے کام انجام دیں
            mid_tasks = plan['mid_level'].get(task_key, [])

            for j, mid_task in enumerate(mid_tasks):
                sub_task_key = f'{task_key}_sub_{j}'
                low_level_actions = plan['low_level'].get(sub_task_key, [])

                # کم سطح کے ایکشن انجام دیں
                task_success = self.execute_low_level_actions(
                    low_level_actions, robot_interface
                )

                if task_success:
                    execution_result['completed_tasks'].append(sub_task_key)
                else:
                    execution_result['failed_tasks'].append(sub_task_key)

                    # منصوبہ کو مطابقت دینے کی کوشش کریں
                    adaptation = self.adapt_plan_for_failure(
                        sub_task_key, plan, robot_interface
                    )
                    execution_result['adaptations'].append(adaptation)

                    if not adaptation['success']:
                        execution_result['success'] = False
                        break

        return execution_result

    def execute_low_level_actions(self, actions: List[str], robot_interface) -> bool:
        """
        کم سطح کے ایکشن کی ترتیب انجام دیں
        """
        for action in actions:
            try:
                # روبوٹ انٹرفیس کے ذریعہ ایکشن انجام دیں
                success = robot_interface.execute_action(action)
                if not success:
                    return False
            except Exception as e:
                print(f"ایکشن '{action}' انجام دینے میں خرابی: {e}")
                return False

        return True

    def adapt_plan_for_failure(self, failed_task: str, plan: Dict[str, Any],
                              robot_interface) -> Dict[str, Any]:
        """
        کام ناکام ہونے پر منصوبہ کو مطابقت دیں
        """
        adaptation = {
            'failed_task': failed_task,
            'new_plan': None,
            'success': False,
            'reason': None
        }

        # موجودہ حالت اور سیاق و سباق حاصل کریں
        current_state = robot_interface.get_current_state()

        # ناکام ہونے والے کام کے لیے متبادل منصوبہ جنریٹ کریں
        try:
            alternative_plan = self.low_level_planner.refine_plan(
                plan['low_level'].get(failed_task, []),
                {
                    'failure_reason': 'execution_failed',
                    'current_state': current_state,
                    'constraints': self.get_execution_constraints(current_state)
                }
            )

            # متبادل منصوبہ انجام دیں
            success = self.execute_low_level_actions(alternative_plan, robot_interface)

            adaptation['new_plan'] = alternative_plan
            adaptation['success'] = success
            adaptation['reason'] = 'plan_refined' if success else 'refinement_failed'

        except Exception as e:
            adaptation['reason'] = f'adaptation_error: {str(e)}'

        return adaptation
```

## دلائل کے ڈھانچے اور انضمام

### علامتی-نیورل انضمام

علامتی منطق کو نیورل نیٹ ورک کی صلاحیتوں کے ساتھ جوڑنا:

```python
# علامتی-نیورل دلائل کا انضمام
import networkx as nx
from typing import Tuple, Optional

class SymbolicNeuralPlanner:
    def __init__(self):
        self.symbolic_planner = SymbolicPlanner()
        self.neural_planner = LLMPlanner()
        self.knowledge_graph = self.build_knowledge_graph()
        self.reasoning_engine = ReasoningEngine()

    def build_knowledge_graph(self) -> nx.DiGraph:
        """
        دلائل کے لیے علم کا گراف بنائیں
        """
        G = nx.DiGraph()

        # اشیاء کے تعلقات شامل کریں
        objects = ['cup', 'bottle', 'book', 'phone', 'table', 'chair', 'kitchen', 'living_room']

        for obj in objects:
            G.add_node(obj, type='object')

        # جگہ کے تعلقات
        spatial_relations = [
            ('cup', 'table', 'on'),
            ('book', 'table', 'on'),
            ('phone', 'table', 'on'),
            ('table', 'kitchen', 'in'),
            ('chair', 'kitchen', 'in'),
            ('kitchen', 'house', 'in')
        ]

        for source, target, relation in spatial_relations:
            G.add_edge(source, target, relation=relation)

        # صلاحیت کے تعلقات
        capabilities = [
            ('robot', 'navigation', 'can_perform'),
            ('robot', 'manipulation', 'can_perform'),
            ('robot', 'communication', 'can_perform')
        ]

        for source, target, relation in capabilities:
            G.add_edge(source, target, relation=relation)

        return G

    def integrate_symbolic_neural_reasoning(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        منصوبہ بندی کے لیے علامتی اور نیورل دلائل کو جوڑیں
        """
        # منطقی ڈھانچہ کے لیے علامتی منصوبہ بند استعمال کریں
        symbolic_plan = self.symbolic_planner.generate_plan(goal, context)

        # تخلیقی مسئلہ حل کرنے کے لیے نیورل منصوبہ بند استعمال کریں
        neural_insights = self.neural_planner.decompose_task(goal, context)

        # دونوں طریقوں کو جوڑیں
        integrated_plan = self.combine_plans(symbolic_plan, neural_insights, context)

        # علم کے گراف کے ساتھ منصوبہ کی توثیق کریں
        validated_plan = self.validate_plan_with_knowledge_graph(integrated_plan)

        return validated_plan

    def combine_plans(self, symbolic_plan: List[str], neural_insights: List[str],
                     context: Dict[str, Any]) -> List[str]:
        """
        علامتی اور نیورل منصوبہ بندی کے نتائج کو جوڑیں
        """
        combined_plan = []

        # علامتی ڈھانچہ سے شروع کریں
        combined_plan.extend(symbolic_plan)

        # تخلیقی بصیرت کو مناسب طور پر ضم کریں
        for insight in neural_insights:
            if self.is_insight_relevant(insight, combined_plan, context):
                # مناسب پوزیشن پر بصیرت داخل کریں
                position = self.find_appropriate_position(insight, combined_plan)
                combined_plan.insert(position, insight)

        return combined_plan

    def is_insight_relevant(self, insight: str, current_plan: List[str],
                           context: Dict[str, Any]) -> bool:
        """
        یہ دیکھیں کہ نیورل بصیرت موجودہ منصوبہ کے لیے متعلقہ ہے یا نہیں
        """
        # ہدف کے ساتھ معنی کی مماثلت چیک کریں
        goal = context.get('goal', '')
        similarity = self.calculate_semantic_similarity(insight, goal)

        # دیکھیں کہ بصیرت موجودہ چیلنجوں کو حل کرتی ہے یا نہیں
        challenges = context.get('challenges', [])
        addresses_challenge = any(
            self.calculate_semantic_similarity(insight, challenge) > 0.3
            for challenge in challenges
        )

        return similarity > 0.2 or addresses_challenge

    def find_appropriate_position(self, insight: str, plan: List[str]) -> int:
        """
        بصیرت کو داخل کرنے کے لیے مناسب پوزیشن تلاش کریں
        """
        # سادہ طریقہ: متعلقہ تصور کے بعد داخل کریں
        insight_lower = insight.lower()

        for i, step in enumerate(plan):
            if self.calculate_semantic_similarity(insight, step) > 0.5:
                return i + 1  # متعلقہ قدم کے بعد داخل کریں

        # اگر کوئی متعلقہ قدم نہیں ملا تو آخر میں شامل کریں
        return len(plan)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        متن کے درمیان معنی کی مماثلت کا حساب لگائیں
        """
        # سادہ لفظ کا احاطہ جامعیت کے لیے
        # عملی طور پر، ایمبیڈنگز یا زیادہ تفصیلی پیمائش استعمال کریں
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def validate_plan_with_knowledge_graph(self, plan: List[str]) -> List[str]:
        """
        علم کے گراف کی پابندیوں کے ساتھ منصوبہ کی توثیق کریں
        """
        validated_plan = []

        for step in plan:
            if self.is_step_valid_in_knowledge_graph(step):
                validated_plan.append(step)
            else:
                # مطابقت رکھنے والے متبادل کو تلاش کریں
                alternative = self.find_valid_alternative(step)
                if alternative:
                    validated_plan.append(alternative)

        return validated_plan

    def is_step_valid_in_knowledge_graph(self, step: str) -> bool:
        """
        دیکھیں کہ کیا قدم علم کے گراف کے مطابق درست ہے
        """
        # قدم سے اشیاء اور تعلقات نکالیں
        entities = self.extract_entities_from_step(step)

        # دیکھیں کہ کیا اشیاء علم کے گراف میں موجود ہیں
        for entity in entities:
            if not self.knowledge_graph.has_node(entity):
                return False

        # دیکھیں کہ کیا تعلقات درست ہیں
        relations = self.extract_relations_from_step(step)
        for rel in relations:
            if not self.is_valid_relation(rel):
                return False

        return True

    def extract_entities_from_step(self, step: str) -> List[str]:
        """
        منصوبہ بندی کے قدم سے اشیاء نکالیں
        """
        # سادہ نکالنا - عملی طور پر، NER استعمال کریں
        words = step.lower().split()
        entities = [word for word in words if word in self.knowledge_graph.nodes()]
        return entities

    def extract_relations_from_step(self, step: str) -> List[Tuple[str, str, str]]:
        """
        منصوبہ بندی کے قدم سے تعلقات نکالیں
        """
        # تعلقات کو "robot moves to kitchen" کی طرح حل کریں
        entities = self.extract_entities_from_step(step)
        relations = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if self.knowledge_graph.has_edge(entities[i], entities[j]):
                    relation = self.knowledge_graph[entities[i]][entities[j]].get('relation')
                    relations.append((entities[i], relation, entities[j]))

        return relations

    def is_valid_relation(self, relation: Tuple[str, str, str]) -> bool:
        """
        دیکھیں کہ کیا تعلق علم کے گراف میں درست ہے
        """
        source, rel_type, target = relation
        if self.knowledge_graph.has_edge(source, target):
            return self.knowledge_graph[source][target].get('relation') == rel_type
        return False

    def find_valid_alternative(self, step: str) -> Optional[str]:
        """
        غلط قدم کے لیے درست متبادل تلاش کریں
        """
        # نیورل منصوبہ بند کو متبادل تجویز کرنے کے لیے استعمال کریں
        context = {'invalid_step': step, 'constraints': 'follow_knowledge_graph_rules'}
        alternatives = self.neural_planner.decompose_task(
            f"find alternative to: {step}", context
        )

        for alt in alternatives:
            if self.is_step_valid_in_knowledge_graph(alt):
                return alt

        return None
```

### متعدد ایجنٹ کوآرڈینیشن منصوبہ بندی

LLM کوآرڈینیشن کے ساتھ متعدد روبوٹ نظام کے لیے منصوبہ بندی:

```python
# LLM کوآرڈینیشن کے ساتھ متعدد ایجنٹ منصوبہ بندی
class MultiAgentLLMPlanner:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agent_planners = [LLMPlanner() for _ in range(num_agents)]
        self.coordinator = LLMPlanner()
        self.communication_protocol = CommunicationProtocol()

    def generate_multi_agent_plan(self, global_goal: str, agent_capabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        متعدد ایجنٹس کے لیے مربوط منصوبہ جنریٹ کریں
        """
        plan = {
            'global_goal': global_goal,
            'agent_plans': {},
            'coordination_protocol': {},
            'communication_schedule': {}
        }

        # ایجنٹس کے درمیان عالمی ہدف کو تقسیم کریں
        task_allocation = self.allocate_tasks(global_goal, agent_capabilities)

        # ہر ایجنٹ کے لیے انفرادی منصوبے جنریٹ کریں
        for agent_id, (tasks, capabilities) in enumerate(zip(task_allocation, agent_capabilities)):
            agent_context = {
                'agent_id': agent_id,
                'capabilities': capabilities,
                'assigned_tasks': tasks,
                'global_goal': global_goal
            }

            agent_plan = self.agent_planners[agent_id].decompose_task(
                " and ".join(tasks), agent_context
            )

            plan['agent_plans'][f'agent_{agent_id}'] = agent_plan

        # کوآرڈینیشن پروٹوکول جنریٹ کریں
        plan['coordination_protocol'] = self.generate_coordination_protocol(
            plan['agent_plans'], global_goal
        )

        # مواصلات کا شیڈول جنریٹ کریں
        plan['communication_schedule'] = self.generate_communication_schedule(
            plan['agent_plans']
        )

        return plan

    def allocate_tasks(self, global_goal: str, agent_capabilities: List[Dict[str, Any]]) -> List[List[str]]:
        """
        صلاحیتوں کی بنیاد پر ایجنٹس کے درمیان کام تقسیم کریں
        """
        # کوآرڈینیٹر کو کام کی تقسیم کی تجویز کے لیے استعمال کریں
        capabilities_str = json.dumps(agent_capabilities, indent=2)

        allocation_prompt = (
            f"اہداف کی تقسیم کریں: '{global_goal}' "
            f"ایجنٹس کی صلاحیتوں کے ساتھ: {capabilities_str}. "
            f"JSON کی فہرست کی شکل میں تقسیم واپس کریں:"
        )

        inputs = self.coordinator.tokenizer.encode(allocation_prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.coordinator.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.coordinator.tokenizer.eos_token_id
            )

        response = self.coordinator.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # جواب سے تقسیم نکالیں
            allocation = self.parse_task_allocation(response, allocation_prompt)
        except:
            # بیک اپ: سادہ راؤنڈ رابن تقسیم
            tasks = [f"subtask_{i}" for i in range(5)]  # مثال کے کام
            allocation = [tasks[i::self.num_agents] for i in range(self.num_agents)]

        return allocation

    def parse_task_allocation(self, response: str, original_prompt: str) -> List[List[str]]:
        """
        LLM کے جواب سے کام کی تقسیم نکالیں
        """
        response_clean = response[len(original_prompt):].strip()

        # JSON ڈھانچہ تلاش کریں
        import re
        json_match = re.search(r'\[.*\]', response_clean, re.DOTALL)

        if json_match:
            try:
                allocation = json.loads(json_match.group())
                if isinstance(allocation, list) and all(isinstance(item, list) for item in allocation):
                    return allocation
            except:
                pass

        # بیک اپ: سادہ تقسیم
        return [['task_1'], ['task_2'], ['task_3']] if self.num_agents >= 3 else [['task_1']]

    def generate_coordination_protocol(self, agent_plans: Dict[str, List[str]],
                                    global_goal: str) -> Dict[str, Any]:
        """
        متعدد ایجنٹ انجام دہی کے لیے کوآرڈینیشن پروٹوکول جنریٹ کریں
        """
        plans_str = json.dumps(agent_plans, indent=2)

        protocol_prompt = (
            f"متعدد ایجنٹ سسٹم کے لیے کوآرڈینیشن پروٹوکول جنریٹ کریں: {plans_str} "
            f"عالمی ہدف کو پورا کرنے کے لیے: '{global_goal}'. "
            f"ہم آہنگی کے اشاروں، تنازعہ کے حل، اور مواصلات کی ضروریات کو شامل کریں."
        )

        inputs = self.coordinator.tokenizer.encode(protocol_prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.coordinator.model.generate(
                inputs,
                max_length=inputs.shape[1] + 300,
                num_return_sequences=1,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.coordinator.tokenizer.eos_token_id
            )

        response = self.coordinator.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            protocol = self.parse_coordination_protocol(response, protocol_prompt)
        except:
            protocol = {
                'synchronization_points': [],
                'conflict_resolution': 'first_come_first_serve',
                'communication_frequency': 'every_30_seconds'
            }

        return protocol

    def parse_coordination_protocol(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """
        LLM کے جواب سے کوآرڈینیشن پروٹوکول نکالیں
        """
        response_clean = response[len(original_prompt):].strip()

        # کلیدی اجزاء نکالیں
        protocol = {
            'synchronization_points': self.extract_synchronization_points(response_clean),
            'conflict_resolution': self.extract_conflict_resolution(response_clean),
            'communication_requirements': self.extract_communication_reqs(response_clean)
        }

        return protocol

    def extract_synchronization_points(self, text: str) -> List[str]:
        """
        ہم آہنگی کے اشاروں کو نکالیں
        """
        import re
        sync_points = re.findall(r'synchronization point.*?:(.*?)(?:\n|$)', text, re.IGNORECASE)
        return [point.strip() for point in sync_points if point.strip()]

    def extract_conflict_resolution(self, text: str) -> str:
        """
        تنازعہ کے حل کی حکمت عملی نکالیں
        """
        if 'priority' in text.lower():
            return 'priority_based'
        elif 'negotiation' in text.lower():
            return 'negotiation_based'
        else:
            return 'first_come_first_serve'

    def extract_communication_reqs(self, text: str) -> List[str]:
        """
        مواصلات کی ضروریات نکالیں
        """
        import re
        reqs = re.findall(r'communication.*?:(.*?)(?:\n|$)', text, re.IGNORECASE)
        return [req.strip() for req in reqs if req.strip()]

    def generate_communication_schedule(self, agent_plans: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        ایجنٹس کے لیے مواصلات کا شیڈول جنریٹ کریں
        """
        schedule = {
            'broadcast_times': [],
            'pairwise_communication': {},
            'status_updates': 'every_30_seconds'
        }

        # باقاعدہ اسٹیٹس اپ ڈیٹس کا شیڈول بنائیں
        for agent_id in agent_plans.keys():
            schedule['pairwise_communication'][agent_id] = {
                'frequency': 'every_60_seconds',
                'topics': ['status', 'progress', 'obstacles']
            }

        return schedule

    def execute_coordinated_plan(self, plan: Dict[str, Any], agent_interfaces: List[Any]) -> Dict[str, Any]:
        """
        مربوط متعدد ایجنٹ منصوبہ انجام دیں
        """
        execution_result = {
            'global_success': True,
            'agent_results': {},
            'coordination_issues': [],
            'communication_logs': []
        }

        # مواصلات کے ساتھ ہم آہنگی کے ساتھ انجام دیں
        for round_num in range(10):  # زیادہ سے زیادہ 10 انجام دہی کے دور
            round_results = {}

            # ایجنٹ منصوبے انجام دیں
            for agent_id, agent_interface in enumerate(agent_interfaces):
                agent_plan = plan['agent_plans'].get(f'agent_{agent_id}', [])

                if round_num < len(agent_plan):
                    action = agent_plan[round_num]
                    try:
                        success = agent_interface.execute_action(action)
                        round_results[f'agent_{agent_id}'] = {
                            'action': action,
                            'success': success,
                            'timestamp': time.time()
                        }
                    except Exception as e:
                        round_results[f'agent_{agent_id}'] = {
                            'action': action,
                            'success': False,
                            'error': str(e),
                            'timestamp': time.time()
                        }

            # نتائج کو مواصل کریں
            communication_log = self.communicate_results(round_results, agent_interfaces)
            execution_result['communication_logs'].append(communication_log)

            # کوآرڈینیشن کے مسائل کو چیک کریں
            issues = self.detect_coordination_issues(round_results)
            if issues:
                execution_result['coordination_issues'].extend(issues)

            # عالمی کامیابی چیک کریں
            if self.is_global_goal_achieved(plan['global_goal'], round_results):
                break

        return execution_result

    def communicate_results(self, round_results: Dict[str, Any], agent_interfaces: List[Any]) -> List[str]:
        """
        ایجنٹس کے درمیان نتائج کو مواصل کریں
        """
        communication_log = []

        for agent_id, result in round_results.items():
            # دیگر ایجنٹس کو براڈکاسٹ کریں
            for other_agent_id, other_interface in enumerate(agent_interfaces):
                if f'agent_{other_agent_id}' != agent_id:
                    try:
                        other_interface.receive_status_update(result)
                        communication_log.append(
                            f"{agent_id} -> agent_{other_agent_id}: {result['action']}"
                        )
                    except:
                        communication_log.append(
                            f"agent_{other_agent_id} کے ساتھ مواصلہ کرنے میں ناکامی"
                        )

        return communication_log

    def detect_coordination_issues(self, round_results: Dict[str, Any]) -> List[str]:
        """
        انجام دہی میں کوآرڈینیشن کے مسائل کا پتہ لگائیں
        """
        issues = []

        # تنازعات کے لیے چیک کریں
        completed_actions = [
            result['action'] for result in round_results.values()
            if result.get('success', False)
        ]

        # سادہ تنازعات کا پتہ لگانا
        conflicting_actions = [
            'move_to_same_location',
            'use_same_resource',
            'conflicting_navigation'
        ]

        for action in completed_actions:
            if any(conflict in action.lower() for conflict in conflicting_actions):
                issues.append(f"ایکشن میں ممکنہ تنازع: {action}")

        return issues

    def is_global_goal_achieved(self, global_goal: str, round_results: Dict[str, Any]) -> bool:
        """
        یہ دیکھیں کہ عالمی ہدف حاصل کر لیا گیا ہے یا نہیں
        """
        # سادہ چیک - عملی طور پر، یہ زیادہ تفصیلی ہوگی
        success_count = sum(
            1 for result in round_results.values()
            if result.get('success', False)
        )

        return success_count == len(round_results)  # تمام ایجنٹس نے اس دور میں کامیابی حاصل کی
```

## منصوبہ بندی اور سیکھنے میں اصلاح

### آن لائن منصوبہ بندی کی اصلاح

تبدیل ہوتی حالات کے مطابق منصوبوں کو اصلاح کرنا:

```python
# آن لائن منصوبہ بندی کی اصلاح کا نظام
class AdaptiveLLMPlanner:
    def __init__(self):
        self.base_planner = LLMPlanner()
        self.adaptation_memory = AdaptationMemory()
        self.uncertainty_handler = UncertaintyHandler()
        self.learning_component = PlanLearningComponent()

    def generate_adaptive_plan(self, initial_goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        تبدیل ہوتی حالات کے لیے اصلاح کے قابل منصوبہ جنریٹ کریں
        """
        # ابتدائی منصوبہ جنریٹ کریں
        initial_plan = self.base_planner.decompose_task(initial_goal, context)

        # اصلاح کے اشارے شامل کریں
        adaptive_plan = self.add_adaptation_points(initial_plan, context)

        # عدم یقینی کے حل کو شامل کریں
        adaptive_plan = self.include_uncertainty_handling(adaptive_plan, context)

        return {
            'initial_plan': initial_plan,
            'adaptive_plan': adaptive_plan,
            'adaptation_criteria': self.define_adaptation_criteria(context),
            'recovery_strategies': self.generate_recovery_strategies(context)
        }

    def add_adaptation_points(self, plan: List[str], context: Dict[str, Any]) -> List[str]:
        """
        منصوبہ میں اصلاح کے اشارے شامل کریں جہاں تبدیلیوں کی ضرورت ہو سکتی ہے
        """
        adaptive_plan = []

        for i, step in enumerate(plan):
            adaptive_plan.append(step)

            # اگر قدم غیر یقینی یا خطرناک ہے تو اصلاح کا اشارہ شامل کریں
            if self.is_step_uncertain(step, context):
                adaptation_marker = f"ADAPTATION_POINT_{i}: شرائط تبدیل ہونے کی صورت میں متبادل کا جائزہ لیں"
                adaptive_plan.append(adaptation_marker)

        return adaptive_plan

    def is_step_uncertain(self, step: str, context: Dict[str, Any]) -> bool:
        """
        یہ چیک کریں کہ قدم میں زیادہ عدم یقینی ہے یا نہیں
        """
        uncertainty_indicators = [
            'navigate', 'manipulate', 'interact', 'unknown', 'unfamiliar'
        ]

        step_lower = step.lower()
        return any(indicator in step_lower for indicator in uncertainty_indicators)

    def include_uncertainty_handling(self, plan: List[str], context: Dict[str, Any]) -> List[str]:
        """
        منصوبہ میں عدم یقینی کے حل کو شامل کریں
        """
        plan_with_uncertainty = []

        for step in plan:
            plan_with_uncertainty.append(step)

            # غیر یقینی اقدامات کے بعد عدم یقینی کی چیک شامل کریں
            if self.is_step_uncertain(step, context):
                uncertainty_check = f"UNCERTAINTY_CHECK: آگے بڑھنے سے پہلے حالات کی تصدیق کریں"
                plan_with_uncertainty.append(uncertainty_check)

        return plan_with_uncertainty

    def define_adaptation_criteria(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        منصوبہ کی اصلاح کے لیے معیار طے کریں
        """
        return {
            'environment_changes': ['object_moved', 'obstacle_detected', 'location_changed'],
            'execution_failures': ['action_failed', 'timeout', 'safety_violation'],
            'new_information': ['user_request', 'emergency', 'priority_change'],
            'resource_changes': ['battery_low', 'capability_lost', 'tool_unavailable']
        }

    def generate_recovery_strategies(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        مختلف ناکامی کی اقسام کے لیے بحالی کی حکمت عملیاں جنریٹ کریں
        """
        return {
            'navigation_failure': [
                'try_alternative_path',
                'request_human_assistance',
                'return_to_known_location'
            ],
            'manipulation_failure': [
                'adjust_grasp_approach',
                'request_object_repositioning',
                'use_alternative_manipulation_method'
            ],
            'communication_failure': [
                'retry_communication',
                'use_alternative_communication_method',
                'proceed_with_assumptions'
            ]
        }

    def adapt_plan_during_execution(self, current_plan: List[str],
                                  execution_state: Dict[str, Any]) -> List[str]:
        """
        موجودہ حالت کے مطابق انجام دہی کے دوران منصوبہ کو اصلاح کریں
        """
        # یہ چیک کریں کہ کیا اصلاح کی ضرورت ہے
        adaptation_needed = self.should_adapt_plan(current_plan, execution_state)

        if adaptation_needed:
            # اصلاح کی وجہ حاصل کریں
            adaptation_reason = self.get_adaptation_reason(execution_state)

            # اصلاح لاگو کریں
            adapted_plan = self.apply_adaptation(
                current_plan, adaptation_reason, execution_state
            )

            # سیکھنے کے لیے اصلاح کو محفوظ کریں
            self.adaptation_memory.store_adaptation(
                current_plan, adapted_plan, adaptation_reason, execution_state
            )

            return adapted_plan

        return current_plan

    def should_adapt_plan(self, current_plan: List[str],
                         execution_state: Dict[str, Any]) -> bool:
        """
        یہ فیصلہ کریں کہ کیا منصوبہ بندی کی اصلاح کی ضرورت ہے
        """
        # ماحول کی تبدیلیوں کے لیے چیک کریں
        if execution_state.get('environment_changed', False):
            return True

        # انجام دہی کی ناکامیوں کے لیے چیک کریں
        if execution_state.get('last_action_failed', False):
            return True

        # نئی معلومات کے لیے چیک کریں
        if execution_state.get('new_information', False):
            return True

        # وسائل کی تبدیلیوں کے لیے چیک کریں
        if execution_state.get('resource_changed', False):
            return True

        return False

    def get_adaptation_reason(self, execution_state: Dict[str, Any]) -> str:
        """
        منصوبہ بندی کی اصلاح کی وجہ حاصل کریں
        """
        if execution_state.get('environment_changed'):
            return 'environment_change'
        elif execution_state.get('last_action_failed'):
            return 'execution_failure'
        elif execution_state.get('new_information'):
            return 'new_information'
        elif execution_state.get('resource_changed'):
            return 'resource_change'
        else:
            return 'unknown'

    def apply_adaptation(self, current_plan: List[str], reason: str,
                        execution_state: Dict[str, Any]) -> List[str]:
        """
        موجودہ منصوبہ میں اصلاح لاگو کریں
        """
        if reason == 'environment_change':
            return self.adapt_for_environment_change(current_plan, execution_state)
        elif reason == 'execution_failure':
            return self.adapt_for_execution_failure(current_plan, execution_state)
        elif reason == 'new_information':
            return self.adapt_for_new_information(current_plan, execution_state)
        elif reason == 'resource_change':
            return self.adapt_for_resource_change(current_plan, execution_state)
        else:
            return current_plan  # کوئی اصلاح نہیں

    def adapt_for_environment_change(self, plan: List[str],
                                   execution_state: Dict[str, Any]) -> List[str]:
        """
        ماحولیاتی تبدیلیوں کے لیے منصوبہ کو اصلاح کریں
        """
        new_environment = execution_state.get('new_environment', {})
        current_state = execution_state.get('current_state', {})

        # LLM کو تجاویز کے لیے استعمال کریں
        adaptation_prompt = (
            f"منصوبہ کو اس کے مطابق اصلاح کریں: {plan} نئے ماحول کے لیے: {new_environment} "
            f"موجودہ حالت سے: {current_state}. رکاوٹوں، نئی جگہوں، "
            f"اور تبدیل شدہ چیزوں کے مقامات کو مدنظر رکھیں. اصلاح شدہ منصوبہ واپس کریں:"
        )

        adapted_plan = self.base_planner.decompose_task(adaptation_prompt, {})
        return adapted_plan

    def adapt_for_execution_failure(self, plan: List[str],
                                  execution_state: Dict[str, Any]) -> List[str]:
        """
        انجام دہی کی ناکامی کے لیے منصوبہ کو اصلاح کریں
        """
        failed_action = execution_state.get('failed_action', '')
        failure_reason = execution_state.get('failure_reason', '')

        # بحالی کی حکمت عملی حاصل کریں
        recovery_strategies = self.generate_recovery_strategies({})
        strategy_type = self.classify_failure_type(failure_reason)
        recovery_options = recovery_strategies.get(strategy_type, [])

        # بحالی کو شامل کرنے کے لیے LLM استعمال کریں
        adaptation_prompt = (
            f"منصوبہ ناکام ہو گیا ایکشن میں: '{failed_action}' کی وجہ سے: '{failure_reason}'. "
            f"بحالی کے اختیارات: {recovery_options}. "
            f"منصوبہ کو اس کے مطابق اصلاح کریں: {plan}. اصلاح شدہ منصوبہ واپس کریں:"
        )

        adapted_plan = self.base_planner.decompose_task(adaptation_prompt, {})
        return adapted_plan

    def classify_failure_type(self, failure_reason: str) -> str:
        """
        مناسب بحالی کے لیے ناکامی کی قسم کی درجہ بندی کریں
        """
        failure_lower = failure_reason.lower()

        if any(phrase in failure_lower for phrase in ['navigate', 'path', 'move']):
            return 'navigation_failure'
        elif any(phrase in failure_lower for phrase in ['grasp', 'manipulate', 'hold']):
            return 'manipulation_failure'
        elif any(phrase in failure_lower for phrase in ['communicate', 'speak', 'hear']):
            return 'communication_failure'
        else:
            return 'navigation_failure'  # ڈیفالٹ

    def learn_from_adaptations(self) -> Dict[str, Any]:
        """
        گزشتہ اصلاحات سے سیکھیں تاکہ مستقبل کی منصوبہ بندی کو بہتر کیا جا سکے
        """
        return self.learning_component.analyze_adaptations(self.adaptation_memory.get_memory())
```

### سیکھنے-بہتر منصوبہ بندی

سیکھنے کو منصوبہ بندی میں شامل کرنا:

```python
# سیکھنے-بہتر منصوبہ بندی کا نظام
class LearningEnhancedPlanner:
    def __init__(self):
        self.base_planner = LLMPlanner()
        self.experience_memory = ExperienceMemory()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()

    def plan_with_learning(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        سیکھی گئی تجربوں کا استعمال کرتے ہوئے منصوبہ جنریٹ کریں
        """
        # مماثل گزشتہ تجربے تلاش کریں
        similar_experiences = self.experience_memory.find_similar_experiences(
            goal, context
        )

        # مماثل منصوبوں کی کارکردگی کا جائزہ لیں
        performance_insights = self.performance_analyzer.analyze_performance(
            similar_experiences
        )

        # سیکھی گئی بصیرت کے ساتھ منصوبہ جنریٹ کریں
        plan = self.base_planner.decompose_task(goal, {
            **context,
            'learned_insights': performance_insights,
            'successful_patterns': self.extract_successful_patterns(similar_experiences)
        })

        return {
            'plan': plan,
            'learned_insights': performance_insights,
            'similar_experiences': similar_experiences[:3],  # اوپر 3
            'confidence': self.calculate_plan_confidence(similar_experiences)
        }

    def extract_successful_patterns(self, experiences: List[Dict[str, Any]]) -> List[str]:
        """
        کامیاب تجربوں سے نمونے نکالیں
        """
        successful_experiences = [
            exp for exp in experiences
            if exp.get('success', False)
        ]

        patterns = []
        for exp in successful_experiences:
            plan = exp.get('plan', [])
            # کامیاب ترتیبات نکالیں
            if len(plan) >= 2:
                patterns.extend([
                    f"{plan[i]} -> {plan[i+1]}"
                    for i in range(len(plan)-1)
                ])

        # سب سے زیادہ عام نمونے واپس کریں
        from collections import Counter
        pattern_counts = Counter(patterns)
        return [pattern for pattern, count in pattern_counts.most_common(5)]

    def calculate_plan_confidence(self, experiences: List[Dict[str, Any]]) -> float:
        """
        مماثل تجربوں کی بنیاد پر منصوبہ کی اعتماد کی شرح کا حساب لگائیں
        """
        if not experiences:
            return 0.5  # ڈیفالٹ اعتماد

        success_rate = sum(1 for exp in experiences if exp.get('success', False)) / len(experiences)
        return min(1.0, success_rate + 0.1)  # LLM کی صلاحیت کے لیے چھوٹا بونس شامل کریں

    def update_experience_memory(self, goal: str, plan: List[str],
                               execution_result: Dict[str, Any]) -> None:
        """
        نیا انجام دہی کا نتیجہ کے ساتھ تجربے کی میموری کو اپ ڈیٹ کریں
        """
        experience = {
            'goal': goal,
            'plan': plan,
            'context': execution_result.get('context', {}),
            'result': execution_result,
            'success': execution_result.get('success', False),
            'execution_time': execution_result.get('execution_time', 0),
            'failures': execution_result.get('failures', []),
            'adaptations': execution_result.get('adaptations', []),
            'timestamp': time.time()
        }

        self.experience_memory.store_experience(experience)

    def suggest_plan_improvements(self, plan: List[str],
                                execution_result: Dict[str, Any]) -> List[str]:
        """
        انجام دہی کے نتائج کی بنیاد پر منصوبہ کی بہتری کی تجاویز دیں
        """
        suggestions = []

        # ناکامیوں کا تجزیہ کریں
        failures = execution_result.get('failures', [])
        for failure in failures:
            if 'navigation' in failure.get('action', ''):
                suggestions.append("متبادل نیویگیشن حکمت عملیوں کا جائزہ لیں")
            elif 'manipulation' in failure.get('action', ''):
                suggestions.append("متبادل مینیپولیشن کے طریقوں کا جائزہ لیں")

        # انجام دہی کے وقت کا تجزیہ کریں
        execution_time = execution_result.get('execution_time', 0)
        if execution_time > 300:  # 5 منٹ سے زیادہ
            suggestions.append("کارکردگی کے لیے منصوبہ کی بہتری پر غور کریں")

        # LLM کو بہتری کی تجاویز کے لیے استعمال کریں
        improvement_prompt = (
            f"اس منصوبہ کو بہتر کریں: {plan} انجام دہی کے نتائج کی بنیاد پر: {execution_result}. "
            f"مخصوص تجاویز فراہم کریں:"
        )

        llm_suggestions = self.base_planner.decompose_task(improvement_prompt, {})
        suggestions.extend(llm_suggestions)

        return suggestions
```

## کارکردگی کا جائزہ اور بہتری

### منصوبہ بندی کی کارکردگی کے معیارات

LLM-بیسڈ منصوبہ بندی کے نظام کی مؤثرتا کا جائزہ لینا:

```python
# منصوبہ بندی کی کارکردگی کا جائزہ
class PlanningEvaluator:
    def __init__(self):
        self.metrics = {
            'plan_success_rate': 0.0,
            'plan_efficiency': 0.0,
            'adaptation_frequency': 0.0,
            'reasoning_accuracy': 0.0,
            'user_satisfaction': 0.0
        }
        self.evaluation_history = []

    def evaluate_planning_performance(self, planner, test_scenarios):
        """
        متعدد منظرناموں کے خلاف منصوبہ بندی کی کارکردگی کا جائزہ لیں
        """
        evaluation_results = []

        for scenario in test_scenarios:
            result = self.evaluate_single_scenario(planner, scenario)
            evaluation_results.append(result)

        # مجموعی معیارات کا حساب لگائیں
        aggregate_metrics = self.calculate_aggregate_metrics(evaluation_results)

        # جائزہ محفوظ کریں
        evaluation_record = {
            'timestamp': time.time(),
            'scenarios': len(test_scenarios),
            'individual_results': evaluation_results,
            'aggregate_metrics': aggregate_metrics
        }

        self.evaluation_history.append(evaluation_record)

        return aggregate_metrics

    def evaluate_single_scenario(self, planner, scenario):
        """
        ایک منظر نامے کے لیے منصوبہ بند کا جائزہ لیں
        """
        start_time = time.time()

        # منصوبہ جنریٹ کریں
        if hasattr(planner, 'generate_hierarchical_plan'):
            plan = planner.generate_hierarchical_plan(
                scenario['goal'], scenario['context']
            )
        else:
            plan = planner.decompose_task(scenario['goal'], scenario['context'])

        plan_generation_time = time.time() - start_time

        # انجام دہی کا شبیہہ بنائیں
        execution_result = self.simulate_execution(plan, scenario)

        # منظر نامے کے مخصوص معیارات کا حساب لگائیں
        scenario_metrics = {
            'success': execution_result['success'],
            'plan_length': len(plan) if isinstance(plan, list) else len(plan.get('high_level', [])),
            'generation_time': plan_generation_time,
            'execution_time': execution_result.get('execution_time', 0),
            'adaptations_needed': len(execution_result.get('adaptations', [])),
            'failures': len(execution_result.get('failures', []))
        }

        return scenario_metrics

    def simulate_execution(self, plan, scenario):
        """
        مؤثرتا کا جائزہ لینے کے لیے منصوبہ انجام دہی کا شبیہہ بنائیں
        """
        import random

        # حقیقیت کے لیے کچھ بے ترتیبی کے ساتھ انجام دہی کا شبیہہ بنائیں
        success_probability = 0.8  # بنیادی کامیابی کی شرح

        # منصوبہ کی پیچیدگی کے مطابق ایڈجسٹ کریں
        plan_complexity = len(plan) if isinstance(plan, list) else 10  # تخمینہ
        adjusted_success = max(0.1, success_probability - (plan_complexity * 0.02))

        # انجام دہی کا شبیہہ بنائیں
        execution_success = random.random() < adjusted_success
        execution_time = random.uniform(30, 300)  # 30-300 سیکنڈ

        # کچھ اصلاحات کا شبیہہ بنائیں
        adaptations_needed = 0 if execution_success else random.randint(1, 3)
        failures = 0 if execution_success else random.randint(1, 2)

        return {
            'success': execution_success,
            'execution_time': execution_time,
            'adaptations': ['adaptation_' + str(i) for i in range(adaptations_needed)],
            'failures': ['failure_' + str(i) for i in range(failures)]
        }

    def calculate_aggregate_metrics(self, evaluation_results):
        """
        جائزہ کے نتائج سے مجموعی معیارات کا حساب لگائیں
        """
        if not evaluation_results:
            return {}

        total_scenarios = len(evaluation_results)
        successful_scenarios = sum(1 for r in evaluation_results if r['success'])

        aggregate_metrics = {
            'plan_success_rate': successful_scenarios / total_scenarios if total_scenarios > 0 else 0,
            'average_plan_length': sum(r['plan_length'] for r in evaluation_results) / total_scenarios if total_scenarios > 0 else 0,
            'average_generation_time': sum(r['generation_time'] for r in evaluation_results) / total_scenarios if total_scenarios > 0 else 0,
            'average_execution_time': sum(r['execution_time'] for r in evaluation_results) / total_scenarios if total_scenarios > 0 else 0,
            'average_adaptations': sum(r['adaptations_needed'] for r in evaluation_results) / total_scenarios if total_scenarios > 0 else 0,
            'average_failures': sum(r['failures'] for r in evaluation_results) / total_scenarios if total_scenarios > 0 else 0
        }

        return aggregate_metrics

    def generate_performance_report(self):
        """
        جامع کارکردگی کے جائزہ کی رپورٹ تیار کریں
        """
        if not self.evaluation_history:
            return "رپورٹ کے لیے کوئی جائزہ دستیاب نہیں."

        latest_evaluation = self.evaluation_history[-1]
        report = {
            'summary': self.generate_summary(latest_evaluation),
            'trends': self.analyze_trends(),
            'recommendations': self.generate_recommendations(latest_evaluation)
        }

        return report

    def generate_summary(self, evaluation):
        """
        تازہ ترین جائزہ کا خلاصہ تیار کریں
        """
        metrics = evaluation['aggregate_metrics']

        summary = f"""
        منصوبہ بندی کی کارکردگی کا خلاصہ:
        - کامیابی کی شرح: {metrics.get('plan_success_rate', 0):.2%}
        - اوسط منصوبہ کی لمبائی: {metrics.get('average_plan_length', 0):.1f} اقدامات
        - اوسط جنریشن ٹائم: {metrics.get('average_generation_time', 0):.2f}سیکنڈ
        - اوسط انجام دہی ٹائم: {metrics.get('average_execution_time', 0):.2f}سیکنڈ
        - اوسط اصلاحات: {metrics.get('average_adaptations', 0):.1f} فی منصوبہ
        - اوسط ناکامیاں: {metrics.get('average_failures', 0):.1f} فی منصوبہ
        """

        return summary

    def analyze_trends(self):
        """
        وقت کے ساتھ کارکردگی کے رجحانات کا تجزیہ کریں
        """
        if len(self.evaluation_history) < 2:
            return "رجحانات کے تجزیہ کے لیے ڈیٹا ناکافی ہے."

        # پہلے اور آخری جائزے کا موازنہ کریں
        first_metrics = self.evaluation_history[0]['aggregate_metrics']
        last_metrics = self.evaluation_history[-1]['aggregate_metrics']

        trends = {}
        for metric in first_metrics.keys():
            if metric in last_metrics:
                change = last_metrics[metric] - first_metrics[metric]
                direction = "بہتر ہو رہا ہے" if change > 0 else "گرتا ہوا" if change < 0 else "مستحکم"
                trends[metric] = {
                    'direction': direction,
                    'change': change,
                    'first_value': first_metrics[metric],
                    'last_value': last_metrics[metric]
                }

        return trends

    def generate_recommendations(self, evaluation):
        """
        جائزہ کے نتائج کی بنیاد پر تجاویز تیار کریں
        """
        metrics = evaluation['aggregate_metrics']
        recommendations = []

        if metrics.get('plan_success_rate', 0) < 0.7:
            recommendations.append(
                "کامیابی کی شرح 70% سے کم ہے. منصوبہ کی توثیق کو بہتر کریں "
                "یا زیادہ مضبوط ناکامی کے حل کو شامل کریں."
            )

        if metrics.get('average_generation_time', float('inf')) > 5.0:
            recommendations.append(
                "منصوبہ جنریشن ٹائم 5 سیکنڈ سے زیادہ ہے. LLM انفرسٹ کو بہتر کریں "
                "یا عام منظرناموں کے لیے منصوبہ کیش کا استعمال کریں."
            )

        if metrics.get('average_adaptations', 0) > 2.0:
            recommendations.append(
                "زیادہ اصلاح کی فریکوئنسی ظاہر کرتی ہے کہ منصوبے شاید بہت سخت ہیں. "
                "شروع سے ہی زیادہ اصلاح کے قابل منصوبے تیار کرنے پر غور کریں."
            )

        return recommendations

    def benchmark_against_classical_planners(self, llm_planner, classical_planners, scenarios):
        """
        کلاسیکل منصوبہ بندی کے نقطہ نظر کے مقابلہ میں LLM منصوبہ بندی کا جائزہ لیں
        """
        results = {
            'llm_planner': self.evaluate_planning_performance(llm_planner, scenarios),
            'classical_planners': {}
        }

        for name, planner in classical_planners.items():
            results['classical_planners'][name] = self.evaluate_planning_performance(planner, scenarios)

        return results
```

## عملی مشق: شعوری منصوبہ بندی کا نظام نافذ کریں

### مشق کے اہداف
- روبوٹکس کے لیے LLM-بیسڈ منصوبہ بندی کا مکمل نظام نافذ کریں
- ہم آہنگی کی اصلاح کی صلاحیتوں کے ساتھ ہیروارکیکل منصوبہ بندی کو نافذ کریں
- شبیہ ساز ماحول میں منصوبہ بندی کی کارکردگی کا جائزہ لیں
- منصوبہ بندی کو بہتر کریں اور اصلاح کریں

### اقدام بہ اقدام ہدایات

1. **LLM منصوبہ بندی کے انفراسٹرکچر کو سیٹ اپ** کریں ماڈل لوڈنگ اور بنیادی صلاحیتوں کے ساتھ
2. **ہیروارکیکل منصوبہ بندی کو نافذ** کریں متعدد انتہا کے ساتھ
3. **اصلاح کے میکنزم کو شامل** کریں منصوبہ کی ناکامیوں کو ہینڈل کرنے کے لیے
4. **روبوٹ شبیہ ساز کے ساتھ انضمام** کریں انجام دہی کی جانچ کے لیے
5. **کارکردگی کا جائزہ** متعدد منظرناموں کے خلاف
6. **منصوبہ بندی کے پیرامیٹرز کو بہتر** کریں جائزہ کے نتائج کی بنیاد پر

### متوقع نتائج
- کام کرتا ہوا LLM-بیسڈ شعوری منصوبہ بندی کا نظام
- ہیروارکیکل منصوبہ بندی کے تصورات کی سمجھ
- اصلاح اور سیکھنے کی منصوبہ بندی کا تجربہ
- کارکردگی کے جائزہ کی صلاحیتیں

## علم کی جانچ

1. LLM-بیسڈ منصوبہ بندی کلاسیکل علامتی منصوبہ بندی کے نقطہ نظر سے کیسے مختلف ہے؟
2. روبوٹکس کے نظام میں ہیروارکیکل منصوبہ بندی کے تصور کی وضاحت کریں.
3. روبوٹس کے لیے اصلاح منصوبہ بندی کو نافذ کرنے میں کیا کلیدی چیلنج ہیں؟
4. تجربوں سے سیکھنے کی بہتر منصوبہ بندی کی کارکردگی کیسے کر سکتی ہے؟

## خلاصہ

اس چیپٹر نے ہیومنوڈ روبوٹس کے لیے بڑے زبانی ماڈلز کو شعوری منصوبہ بندی کے نظام میں انضمام کا جائزہ لیا۔ ہم نے ہیروارکیکل منصوبہ بندی کے آرکیٹیکچر، علامتی-نیورل انضمام، متعدد ایجنٹ کوآرڈینیشن، اور اصلاح منصوبہ بندی کے میکنزم کا مطالعہ کیا۔ LLM-بیسڈ منصوبہ بندی روبوٹس کو پیچیدہ، متعدد اقدامات والے کاموں کو قدرتی زبان کے انٹرفیس کے ساتھ ہینڈل کرنے کے قابل بناتی ہے جبکہ تبدیل ہوتی حالات کے مطابق اور تجربے سے سیکھنے کے ساتھ اصلاح کرتی ہے۔ حقیقی دنیا کے ماحول میں کام کرنے کے قابل مضبوط منصوبہ بندی کے نظام کو بنانے کے لیے بالائی سطح کی سوچ کی صلاحیتوں اور عملی انجام دہی کی پابندیوں کا اتحاد۔

## اگلے اقدامات

چیپٹر 20 میں، ہم خود کار ہیومنوڈ کیپسٹون پروجیکٹ پر غور کریں گے، جو ہیومنوڈ روبوٹ کے مکمل نظام کو ڈیزائن اور نافذ کرنے کے لیے پچھلے چیپٹرز کے تمام تصورات کو اکٹھا کرتا ہے جو پیچیدہ کام کے انجام دہی اور انسانی بات چیت کے قابل ہو۔

