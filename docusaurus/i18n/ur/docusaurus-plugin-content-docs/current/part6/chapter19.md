---
title: "باب 19: LLMs کے ساتھ کوگنیٹو پلاننگ"
sidebar_label: "باب 19: کوگنیٹو پلاننگ"
---

# باب 19: LLMs کے ساتھ کوگنیٹو پلاننگ

## سیکھنے کے اہداف
- یہ سمجھنا کہ بڑے زبانی ماڈلز (LLMs) کو روبوٹکس میں کوگنیٹو پلاننگ کے لیے کیسے استعمال کیا جا سکتا ہے
- ہیومنوائڈ روبوٹکس کے لیے LLMs کا استعمال کرتے ہوئے ہیئرآرکیکل پلاننگ سسٹمزم نافذ کرنا
- انٹیگریٹڈ ریزننگ فریم ورکس ڈیزائن کرنا جو علامتی پلاننگ اور نیورل نیٹ ورکس کو جوڑتے ہیں
- حقیقی دنیا کی روبوٹک ایپلیکیشنزم کے لیے LLM-مبنی پلاننگ کا جائزہ لینا اور بہتر بنانا

## تعارف

کوگنیٹو پلاننگ روبوٹکس میں مصنوعی ذہانت کا انتہائی درجہ ہے، جو ہیومنوائڈ روبوٹس کو پیچیدہ، متعدد اسٹیپس والے کاموں کے بارے میں سوچنے اور مؤثر ایکشن سیکوئنسز جنریٹ کرنے کے قابل بناتا ہے۔ بڑے زبانی ماڈلز (LLMs) ہائی-لیول ریزننگ کے لیے بے مثال صلاحیتیں فراہم کرتے ہیں، جس سے روبوٹس کو پیچیدہ اہداف کو قابلِ انتظام ذیلی کاموں میں تقسیم کرنا، متعدد پلاننگ متبادل امکانات پر غور کرنا، اور متغیر حالات کے مطابق اپنے منصوبے ایڈجسٹ کرنا ممکن ہو جاتا ہے۔ یہ باب ہیومنوائڈ روبوٹک سسٹمزم میں LLMs کو ضم کرنے کو تلاش کرتا ہے، ایسے کوگنیٹو آرکیٹیکچرزم کو تخلیق کرتا ہے جو حقیقی دنیا کے ماحول کی پیچیدگی اور عدم یقینی کو سنبھال سکتے ہیں۔

## LLM-مبنی کوگنیٹو پلاننگ کی بنیادیں

### لینگویج جنریشن کے طور پر پلاننگ

LLM-مبنی پلاننگ پلاننگ کے مسئلے کو لینگویج جنریشن ٹاسک کے طور پر سمجھتا ہے، جہاں ماڈل اہداف کو حاصل کرنے کے لیے ایکشن سیکوئنسز جنریٹ کرتا ہے:

```python
# LLM-مبنی پلاننگ سسٹم
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

        # پلاننگ-مخصوص ووکیبیولری اور ٹیمپلیٹس
        self.planning_templates = {
            'task_decomposition': (
                "کام '{task}' کو ذیلی کاموں کی سیکوئنس میں تقسیم کریں۔ "
                "ذیلی کاموں کا ہونا چاہیے: 1) متوالیہ ان ایکسیکیوٹ ہونے والے، "
                "2) منطقی طور پر منسلک، 3) کل ہدف حاصل کرنا۔ "
                "ذیلی کام:"
            ),
            'action_generation': (
                "موجودہ حالت: {state} اور ہدف: {goal} کو دیکھتے ہوئے، "
                "اگلا ایکشن جنریٹ کریں۔ یہ بات مدنظر رکھیں: {constraints}۔ "
                "ایکشن:"
            ),
            'plan_refinement': (
                "منصوبہ: {plan} کو مدنظر رکھتے ہوئے: {feedback} کے تحت بہتر بنائیں۔ "
                "ایک بہتر منصوبہ لوٹائیں:"
            )
        }

    def decompose_task(self, high_level_task: str, context: Dict[str, Any]) -> List[str]:
        """
        LLM کا استعمال کرتے ہوئے ہائی لیول ٹاسک کو ذیلی کاموں میں تقسیم کریں
        """
        prompt = self.planning_templates['task_decomposition'].format(
            task=high_level_task
        )

        # سیاق و سباق کی معلومات شامل کریں
        context_str = self.format_context(context)
        full_prompt = f"{prompt}\nContext: {context_str}\n"

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

        # ریسپانس سے ذیلی کاموں کو نکالیں
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
        فیڈ بیک کے مطابق موجودہ منصوبے کو بہتر بنائیں
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
        پلاننگ کے لیے سیاق و سباق کی معلومات کو فارمیٹ کریں
        """
        formatted_parts = []

        if 'environment' in context:
            formatted_parts.append(f"Environment: {json.dumps(context['environment'])}")

        if 'robot_capabilities' in context:
            formatted_parts.append(f"Capabilities: {context['robot_capabilities']}")

        if 'constraints' in context:
            formatted_parts.append(f"Constraints: {context['constraints']}")

        if 'previous_attempts' in context:
            formatted_parts.append(f"Previous attempts: {context['previous_attempts']}")

        return " | ".join(formatted_parts)

    def parse_subtasks(self, response: str, original_prompt: str) -> List[str]:
        """
        LLM ریسپانس سے ذیلی کاموں کو پارس کریں
        """
        # ریسپانس سے اصل پرامپٹ کو ہٹا دیں
        response_clean = response[len(original_prompt):].strip()

        # عام ذیلی کام اشاروں کے مطابق تقسیم کریں
        import re
        subtasks = re.split(r'\d+\.\s*|\n\s*\n\s*|\*\s*', response_clean)

        # ذیلی کاموں کو صاف اور فلٹر کریں
        subtasks = [task.strip() for task in subtasks if task.strip()]

        return subtasks[:10]  # 10 ذیلی کاموں تک محدود کریں

    def extract_action(self, response: str, original_prompt: str) -> str:
        """
        LLM ریسپانس سے ایکشن نکالیں
        """
        response_clean = response[len(original_prompt):].strip()
        # پہلا مکمل جملہ ایکشن کے طور پر لیں
        sentences = response_clean.split('.')
        return sentences[0].strip() if sentences else response_clean

    def parse_plan(self, response: str, original_prompt: str) -> List[str]:
        """
        LLM ریسپانس سے بہتر منصوبہ پارس کریں
        """
        response_clean = response[len(original_prompt):].strip()
        # منصوبہ اشاروں کے مطابق تقسیم کریں
        import re
        plan_steps = re.split(r'\d+\.\s*|\n\s*\n\s*|\*\s*', response_clean)
        plan_steps = [step.strip() for step in plan_steps if step.strip()]
        return plan_steps[:20]  # 20 اسٹیپس تک محدود کریں
```

### ہیئرآرکیکل پلاننگ آرکیٹیکچر

ہیئرآرکیکل پلاننگ پیچیدہ کاموں کو قابلِ انتظام سطحوں میں سٹرکچر کرتا ہے:

```python
# ہیئرآرکیکل پلاننگ سسٹم
class HierarchicalLLMPlanner:
    def __init__(self):
        self.high_level_planner = LLMPlanner()
        self.mid_level_planner = LLMPlanner()
        self.low_level_planner = LLMPlanner()
        self.plan_cache = {}
        self.execution_monitor = ExecutionMonitor()

    def generate_hierarchical_plan(self, high_level_goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        متعدد ایبسٹریکشن سطحوں کے ساتھ ہیئرآرکیکل منصوبہ جنریٹ کریں
        """
        plan = {
            'high_level': [],
            'mid_level': {},
            'low_level': {},
            'dependencies': {}
        }

        # ہائی لیول منصوبہ جنریٹ کریں
        high_level_tasks = self.high_level_planner.decompose_task(
            high_level_goal, context
        )
        plan['high_level'] = high_level_tasks

        # ہر ہائی لیول ٹاسک کے لیے مڈ لیول منصوبے جنریٹ کریں
        for i, task in enumerate(high_level_tasks):
            mid_level_context = self.update_context_for_task(context, task, i)
            mid_level_tasks = self.mid_level_planner.decompose_task(
                task, mid_level_context
            )
            plan['mid_level'][f'task_{i}'] = mid_level_tasks

            # ہر مڈ لیول ٹاسک کے لیے لو لیول منصوبے جنریٹ کریں
            for j, mid_task in enumerate(mid_level_tasks):
                low_level_context = self.update_context_for_task(
                    mid_level_context, mid_task, j
                )
                low_level_actions = self.generate_low_level_actions(
                    mid_task, low_level_context
                )
                plan['low_level'][f'task_{i}_sub_{j}'] = low_level_actions

        # کاموں کے درمیان انحصاریت کا تعین کریں
        plan['dependencies'] = self.calculate_dependencies(plan)

        return plan

    def update_context_for_task(self, context: Dict[str, Any], task: str, index: int) -> Dict[str, Any]:
        """
        مخصوص ٹاسک کے لیے سیاق و سباق کو اپ ڈیٹ کریں
        """
        updated_context = context.copy()
        updated_context['current_task'] = task
        updated_context['task_index'] = index
        updated_context['task_progress'] = f"{index}/{len(context.get('high_level_tasks', []))}"

        return updated_context

    def generate_low_level_actions(self, mid_task: str, context: Dict[str, Any]) -> List[str]:
        """
        قابلِ ایکسیکیوشن لو لیول ایکشنز جنریٹ کریں
        """
        # لو لیول پلاننگ کے لیے، مزید مخصوص ٹیمپلیٹس اور کنٹرینٹس کا استعمال کریں
        state = context.get('current_state', {})
        goal = mid_task

        # پرائمری ایکشنز کی سیکوئنس جنریٹ کریں
        actions = []
        current_state = state.copy()

        # متعدد اسٹیپس کی پلاننگ کا شبیہہ
        for step in range(5):  # ہر مڈ لیول ٹاسک کے لیے 5 اسٹیپس تک محدود کریں
            action = self.low_level_planner.generate_action(
                current_state, goal,
                constraints=self.get_execution_constraints(context)
            )

            if self.is_valid_action(action, current_state):
                actions.append(action)
                # ایکشن کے مطابق حالت کو اپ ڈیٹ کریں
                current_state = self.apply_action_to_state(action, current_state)

                # چیک کریں کہ ہدف حاصل ہو گیا ہے یا نہیں
                if self.is_goal_achieved(mid_task, current_state):
                    break
            else:
                break  # غلط ایکشن، پلاننگ بند کریں

        return actions

    def get_execution_constraints(self, context: Dict[str, Any]) -> List[str]:
        """
        موجودہ سیاق و سباق کے لیے ایکسیکیوشن کنٹرینٹس حاصل کریں
        """
        constraints = [
            "ایکشنز روبوٹ کے ذریعے ایکسیکیوٹ ہو سکتے ہیں",
            "سیفٹی کنٹرینٹس پر غور کریں",
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
        چیک کریں کہ ایکشن موجودہ حالت کے تحت درست ہے
        """
        # سادہ توثیق - عمل میں، یہ زیادہ جامع ہوگی
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
        # یہ عمل میں اصل روبوٹ حالت کے ساتھ ضم ہوگا
        new_state = state.copy()

        # ایکشن کی قسم کے مطابق حالت کو اپ ڈیٹ کریں
        if 'move' in action.lower() or 'go' in action.lower():
            new_state['position'] = self.calculate_new_position(action, state)
        elif 'pick' in action.lower() or 'grasp' in action.lower():
            new_state['holding'] = self.extract_object(action)

        return new_state

    def is_goal_achieved(self, goal: str, state: Dict[str, Any]) -> bool:
        """
        چیک کریں کہ ہدف موجودہ حالت کے مطابق حاصل ہو گیا ہے
        """
        # سادہ ہدف چیکنگ - عمل میں، یہ زیادہ جامع ہوگی
        goal_lower = goal.lower()
        state_str = str(state).lower()

        # چیک کریں کہ ہدف سے متعلقہ الفاظ حالت میں موجود ہیں
        return any(term in state_str for term in goal_lower.split())

    def calculate_dependencies(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        منصوبہ کے اجزاء کے درمیان انحصاریت کا حساب لگائیں
        """
        dependencies = {}

        # ہائی لیول انحصاریت
        for i in range(1, len(plan['high_level'])):
            dependencies[f'task_{i}'] = [f'task_{i-1}']

        # ہر ہائی لیول ٹاسک کے اندر مڈ لیول انحصاریت
        for task_key, mid_tasks in plan['mid_level'].items():
            for j in range(1, len(mid_tasks)):
                dependencies[f'{task_key}_sub_{j}'] = [f'{task_key}_sub_{j-1}']

        return dependencies

    def execute_plan_with_monitoring(self, plan: Dict[str, Any], robot_interface) -> Dict[str, Any]:
        """
        مانیٹرنگ اور اڈاپٹیشن کے ساتھ ہیئرآرکیکل منصوبہ ایکسیکیوٹ کریں
        """
        execution_result = {
            'success': True,
            'completed_tasks': [],
            'failed_tasks': [],
            'adaptations': []
        }

        for i, high_task in enumerate(plan['high_level']):
            task_key = f'task_{i}'

            # اس ہائی لیول ٹاسک کے لیے مڈ لیول ٹاسکس ایکسیکیوٹ کریں
            mid_tasks = plan['mid_level'].get(task_key, [])

            for j, mid_task in enumerate(mid_tasks):
                sub_task_key = f'{task_key}_sub_{j}'
                low_level_actions = plan['low_level'].get(sub_task_key, [])

                # لو لیول ایکشنز ایکسیکیوٹ کریں
                task_success = self.execute_low_level_actions(
                    low_level_actions, robot_interface
                )

                if task_success:
                    execution_result['completed_tasks'].append(sub_task_key)
                else:
                    execution_result['failed_tasks'].append(sub_task_key)

                    # منصوبے کو اڈاپٹ کرنے کی کوشش کریں
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
        لو لیول ایکشنز کی سیکوئنس ایکسیکیوٹ کریں
        """
        for action in actions:
            try:
                # روبوٹ انٹرفیس کے ذریعے ایکشن ایکسیکیوٹ کریں
                success = robot_interface.execute_action(action)
                if not success:
                    return False
            except Exception as e:
                print(f"ایکشن '{action}' کو ایکسیکیوٹ کرنے میں خامی: {e}")
                return False

        return True

    def adapt_plan_for_failure(self, failed_task: str, plan: Dict[str, Any],
                              robot_interface) -> Dict[str, Any]:
        """
        جب کوئی ٹاسک ناکام ہو تو منصوبے کو اڈاپٹ کریں
        """
        adaptation = {
            'failed_task': failed_task,
            'new_plan': None,
            'success': False,
            'reason': None
        }

        # موجودہ حالت اور سیاق و سباق حاصل کریں
        current_state = robot_interface.get_current_state()

        # ناکام ٹاسک کے لیے متبادل منصوبہ جنریٹ کریں
        try:
            alternative_plan = self.low_level_planner.refine_plan(
                plan['low_level'].get(failed_task, []),
                {
                    'failure_reason': 'execution_failed',
                    'current_state': current_state,
                    'constraints': self.get_execution_constraints(current_state)
                }
            )

            # متبادل منصوبہ ایکسیکیوٹ کریں
            success = self.execute_low_level_actions(alternative_plan, robot_interface)

            adaptation['new_plan'] = alternative_plan
            adaptation['success'] = success
            adaptation['reason'] = 'plan_refined' if success else 'refinement_failed'

        except Exception as e:
            adaptation['reason'] = f'adaptation_error: {str(e)}'

        return adaptation
```

## ریزننگ فریم ورکس اور انضمام

### علامتی-نیورل انضمام

علامتی ریزننگ کو نیورل نیٹ ورک کی صلاحیت کے ساتھ جوڑنا:

```python
# علامتی-نیورل ریزننگ انضمام
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
        ریزننگ کے لیے نالج گراف بنائیں
        """
        G = nx.DiGraph()

        # آبجیکٹ کے تعلقات شامل کریں
        objects = ['cup', 'bottle', 'book', 'phone', 'table', 'chair', 'kitchen', 'living_room']

        for obj in objects:
            G.add_node(obj, type='object')

        # اسپیشل تعلقات شامل کریں
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

        # صلاحیت کے تعلقات شامل کریں
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
        پلاننگ کے لیے علامتی اور نیورل ریزننگ کو ضم کریں
        """
        # منطقی سٹرکچر کے لیے علامتی پلانر کا استعمال کریں
        symbolic_plan = self.symbolic_planner.generate_plan(goal, context)

        # تخلیقی مسئلہ حل کرنے کے لیے نیورل پلانر کا استعمال کریں
        neural_insights = self.neural_planner.decompose_task(goal, context)

        # دونوں ا approaches کو ملانا
        integrated_plan = self.combine_plans(symbolic_plan, neural_insights, context)

        # نالج گراف کے استعمال سے منصوبے کی توثیق کریں
        validated_plan = self.validate_plan_with_knowledge_graph(integrated_plan)

        return validated_plan

    def combine_plans(self, symbolic_plan: List[str], neural_insights: List[str],
                     context: Dict[str, Any]) -> List[str]:
        """
        علامتی اور نیورل پلاننگ کے نتائج کو ملانا
        """
        combined_plan = []

        # علامتی سٹرکچر کے ساتھ شروع کریں
        combined_plan.extend(symbolic_plan)

        # مناسب جگہوں پر نیورل بصیرتیں ضم کریں
        for insight in neural_insights:
            if self.is_insight_relevant(insight, combined_plan, context):
                # مناسب پوزیشن پر بصیرت داخل کریں
                position = self.find_appropriate_position(insight, combined_plan)
                combined_plan.insert(position, insight)

        return combined_plan

    def is_insight_relevant(self, insight: str, current_plan: List[str],
                           context: Dict[str, Any]) -> bool:
        """
        چیک کریں کہ نیورل بصیرت موجودہ منصوبے کے لیے متعلقہ ہے
        """
        # ہدف کے ساتھ سیمینٹک مماثلت کی جانچ کریں
        goal = context.get('goal', '')
        similarity = self.calculate_semantic_similarity(insight, goal)

        # چیک کریں کہ بصیرت موجودہ چیلنجوں کو حل کرتی ہے
        challenges = context.get('challenges', [])
        addresses_challenge = any(
            self.calculate_semantic_similarity(insight, challenge) > 0.3
            for challenge in challenges
        )

        return similarity > 0.2 or addresses_challenge

    def find_appropriate_position(self, insight: str, plan: List[str]) -> int:
        """
        منصوبے میں بصیرت داخل کرنے کے لیے مناسب پوزیشن تلاش کریں
        """
        # سادہ نقطہ نظر: متعلقہ تصور کے بعد داخل کریں
        insight_lower = insight.lower()

        for i, step in enumerate(plan):
            if self.calculate_semantic_similarity(insight, step) > 0.5:
                return i + 1  # متعلقہ اسٹیپ کے بعد داخل کریں

        # اگر کوئی متعلقہ اسٹیپ نہیں ملا تو آخر میں شامل کریں
        return len(plan)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        ٹیکسٹس کے درمیان سیمینٹک مماثلت کا حساب لگائیں
        """
        # جامع ایمبیڈنگز یا زیادہ جامع اقدار کے استعمال کے لیے سادہ لفظی اورلیپ
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def validate_plan_with_knowledge_graph(self, plan: List[str]) -> List[str]:
        """
        نالج گراف کے استعمال سے منصوبے کی توثیق کریں
        """
        validated_plan = []

        for step in plan:
            if self.is_step_valid_in_knowledge_graph(step):
                validated_plan.append(step)
            else:
                # کنٹرینٹس کو پورا کرنے والے متبادل کی تلاش کریں
                alternative = self.find_valid_alternative(step)
                if alternative:
                    validated_plan.append(alternative)

        return validated_plan

    def is_step_valid_in_knowledge_graph(self, step: str) -> bool:
        """
        چیک کریں کہ اسٹیپ نالج گراف کے مطابق درست ہے
        """
        # اسٹیپ سے اجزاء اور تعلقات کو پارس کریں
        entities = self.extract_entities_from_step(step)

        # چیک کریں کہ اجزاء نالج گراف میں موجود ہیں
        for entity in entities:
            if not self.knowledge_graph.has_node(entity):
                return False

        # چیک کریں کہ تعلقات درست ہیں
        relations = self.extract_relations_from_step(step)
        for rel in relations:
            if not self.is_valid_relation(rel):
                return False

        return True

    def extract_entities_from_step(self, step: str) -> List[str]:
        """
        پلاننگ اسٹیپ سے اجزاء نکالیں
        """
        # سادہ ایکٹریکشن - عمل میں، NER کا استعمال کریں
        words = step.lower().split()
        entities = [word for word in words if word in self.knowledge_graph.nodes()]
        return entities

    def extract_relations_from_step(self, step: str) -> List[str]:
        """
        پلاننگ اسٹیپ سے تعلقات نکالیں
        """
        # سادہ تعلقات ایکٹریکشن - عمل میں، زیادہ جامع تحلیل کا استعمال کریں
        relations = []
        words = step.lower().split()

        for i, word in enumerate(words):
            if word in ['on', 'in', 'at', 'to', 'from', 'with']:
                if i > 0 and i < len(words):
                    relations.append((words[i-1], word, words[i]))

        return relations

    def is_valid_relation(self, relation: Tuple[str, str, str]) -> bool:
        """
        چیک کریں کہ تعلق نالج گراف میں درست ہے
        """
        source, rel_type, target = relation
        if self.knowledge_graph.has_edge(source, target):
            edge_rel = self.knowledge_graph[source][target].get('relation')
            return edge_rel == rel_type or rel_type in edge_rel
        return False

    def find_valid_alternative(self, step: str) -> Optional[str]:
        """
        نامناسب اسٹیپ کے لیے درست متبادل تلاش کریں
        """
        # نیورل پلانر کو متبادل کی تجویز کے لیے استعمال کریں
        context = {'invalid_step': step, 'constraints': 'follow_knowledge_graph_rules'}
        alternatives = self.neural_planner.decompose_task(
            f"find alternative to: {step}", context
        )

        for alt in alternatives:
            if self.is_step_valid_in_knowledge_graph(alt):
                return alt

        return None
```

### کوگنیٹو ریزننگ ماڈلز

LLMs کا استعمال کرتے ہوئے ہائی-لیول تصوراتی سوچ:

```python
# کوگنیٹو ریزننگ کے لیے LLM کا استعمال
class CognitiveReasoning:
    def __init__(self):
        self.llm_model = LLMPlanner()
        self.memory_system = MemorySystem()
        self.reasoning_framework = ReasoningFramework()

    def perform_high_level_reasoning(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        ہائی-لیول تصوراتی سوچ کا اظہار کریں
        """
        # مسئلہ کو سمجھیں اور تجزیہ کریں
        problem_analysis = self.analyze_problem(problem, context)

        # ممکنہ حل کے طریقے تیار کریں
        solution_approaches = self.generate_solution_approaches(problem_analysis)

        # سب سے بہترین حل کا انتخاب کریں
        best_approach = self.evaluate_and_select_best_approach(
            solution_approaches, context
        )

        # مکمل حل کا منصوبہ بنائیں
        solution_plan = self.create_solution_plan(best_approach, context)

        return {
            'problem_analysis': problem_analysis,
            'solution_approaches': solution_approaches,
            'selected_approach': best_approach,
            'solution_plan': solution_plan,
            'confidence': self.calculate_reasoning_confidence(solution_plan)
        }

    def analyze_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        مسئلہ کا تصوراتی تجزیہ کریں
        """
        analysis_prompt = (
            f"مسئلہ: '{problem}' کا تجزیہ کریں۔ "
            f"سیاق و سباق: {json.dumps(context, indent=2)}. "
            f"مسئلے کے اجزاء، پیچیدگی، اور ممکنہ چیلنجوں کا تجزیہ کریں:"
        )

        analysis = self.llm_model.generate_response(analysis_prompt)
        return self.parse_problem_analysis(analysis)

    def generate_solution_approaches(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        مسئلے کے حل کے لیے متعدد طریقے تیار کریں
        """
        approaches_prompt = (
            f"مسئلہ کا تجزیہ: {json.dumps(problem_analysis, indent=2)}. "
            f"اس مسئلے کے حل کے لیے مختلف طریقے تیار کریں. "
            f"ہر نقطہ نظر کے فوائد، نقصانات، اور قابلیت پر غور کریں:"
        )

        approaches_text = self.llm_model.generate_response(approaches_prompt)
        approaches = self.parse_solution_approaches(approaches_text)

        return approaches

    def evaluate_and_select_best_approach(self, approaches: List[Dict[str, Any]],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        متبادل حل کے طریقے کا جائزہ لیں اور بہترین کا انتخاب کریں
        """
        evaluation_prompt = (
            f"حل کے طریقے: {json.dumps(approaches, indent=2)}. "
            f"سیاق و سباق: {json.dumps(context, indent=2)}. "
            f"ہر نقطہ نظر کا جائزہ لیں اور بہترین کا انتخاب کریں:"
        )

        evaluation = self.llm_model.generate_response(evaluation_prompt)
        best_approach = self.select_best_approach_from_evaluation(evaluation, approaches)

        return best_approach

    def create_solution_plan(self, approach: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """
        منتخب کردہ نقطہ نظر کے مطابق حل کا منصوبہ بنائیں
        """
        planning_prompt = (
            f"نقطہ نظر: {json.dumps(approach, indent=2)}. "
            f"سیاق و سباق: {json.dumps(context, indent=2)}. "
            f"اس نقطہ نظر کے مطابق ایک تفصیلی حل منصوبہ بنائیں:"
        )

        plan_text = self.llm_model.generate_response(planning_prompt)
        plan = self.parse_solution_plan(plan_text)

        return plan

    def calculate_reasoning_confidence(self, solution_plan: List[str]) -> float:
        """
        حل کے منصوبے کے لیے تصوراتی سوچ کا اعتماد حساب لگائیں
        """
        # سادہ اعتماد کیلکولیشن - عمل میں، زیادہ جامع ہوگی
        if solution_plan:
            return min(0.9, len(solution_plan) * 0.1)  # منصوبے کی لمبائی کے مطابق
        else:
            return 0.1  # کم اعتماد اگر کوئی منصوبہ نہیں

    def learn_from_reasoning(self, problem: str, solution: Dict[str, Any], outcome: str):
        """
        ریزننگ کے نتائج سے سیکھیں اور مستقبل کی سوچ کو بہتر بنائیں
        """
        learning_record = {
            'problem': problem,
            'reasoning_process': solution,
            'outcome': outcome,
            'timestamp': time.time()
        }

        self.memory_system.store_learning_experience(learning_record)

    def retrieve_similar_reasoning_cases(self, current_problem: str) -> List[Dict[str, Any]]:
        """
        مماثل مسائل کے لیے گزشتہ ریزننگ کیسز بازیافت کریں
        """
        # چیک کریں کہ مماثل مسئلہ یادداشت میں موجود ہے
        similar_cases = self.memory_system.find_similar_cases(
            current_problem, similarity_threshold=0.7
        )

        return similar_cases
```

## کارکردگی کا جائزہ اور بہتری

### LLM-مبنی پلاننگ کارکردگی کے میٹرکس

روبوٹکس کے سیاق و سباق میں LLM کارکردگی کا جائزہ لینا:

```python
# LLM-مبنی پلاننگ کارکردگی کے میٹرکس
class PlanningPerformanceEvaluator:
    def __init__(self):
        self.metrics = {
            'plan_success_rate': 0.0,
            'reasoning_accuracy': 0.0,
            'execution_time': 0.0,
            'adaptability_score': 0.0,
            'user_satisfaction': 0.0
        }
        self.evaluation_history = []

    def evaluate_planning_performance(self, planner, test_scenarios):
        """
        متعدد منظرناموں کے ساتھ پلاننگ کارکردگی کا جائزہ لیں
        """
        evaluation_results = []

        for scenario in test_scenarios:
            result = self.evaluate_single_scenario(planner, scenario)
            evaluation_results.append(result)

        # اجتماعی میٹرکس کا حساب لگائیں
        aggregate_metrics = self.calculate_aggregate_metrics(evaluation_results)

        # جائزہ کو محفوظ کریں
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
        ایک منظرنامے کے لیے پلانر کا جائزہ لیں
        """
        start_time = time.time()

        # منصوبہ جنریٹ کریں
        plan = planner.generate_plan(scenario['goal'], scenario['context'])

        plan_generation_time = time.time() - start_time

        # ایکسیکیوشن کا شبیہہ
        execution_result = self.simulate_execution(plan, scenario)

        # منظرنامہ-مخصوص میٹرکس کا حساب لگائیں
        scenario_metrics = {
            'success': execution_result['success'],
            'plan_length': len(plan) if isinstance(plan, list) else len(plan.get('high_level', [])),
            'generation_time': plan_generation_time,
            'execution_time': execution_result.get('execution_time', 0),
            'adaptations_needed': len(execution_result.get('adaptations', [])),
            'errors': len(execution_result.get('errors', []))
        }

        return scenario_metrics

    def simulate_execution(self, plan, scenario):
        """
        مؤثریت کا جائزہ لینے کے لیے منصوبے کی ایکسیکیوشن کا شبیہہ
        """
        import random

        # حقیقیت کے لیے کچھ بے ترتیبی کے ساتھ ایکسیکیوشن کا شبیہہ
        success_probability = 0.8  # بنیادی کامیابی کی شرح

        # منصوبے کی پیچیدگی کے مطابق ایڈجسٹ کریں
        plan_complexity = len(plan) if isinstance(plan, list) else 10  # تخمینہ
        adjusted_success = max(0.1, success_probability - (plan_complexity * 0.02))

        # ایکسیکیوشن کا شبیہہ
        execution_success = random.random() < adjusted_success
        execution_time = random.uniform(30, 300)  # 30-300 سیکنڈ

        # کچھ اڈاپٹیشنز کا شبیہہ
        adaptations_needed = 0 if execution_success else random.randint(1, 3)
        errors = 0 if execution_success else random.randint(1, 2)

        return {
            'success': execution_success,
            'execution_time': execution_time,
            'adaptations': ['adaptation_' + str(i) for i in range(adaptations_needed)],
            'errors': ['error_' + str(i) for i in range(errors)]
        }

    def calculate_aggregate_metrics(self, evaluation_results):
        """
        جائزہ کے نتائج سے اجتماعی میٹرکس کا حساب لگائیں
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
            'average_errors': sum(r['errors'] for r in evaluation_results) / total_scenarios if total_scenarios > 0 else 0
        }

        return aggregate_metrics

    def generate_performance_report(self):
        """
        جامع کارکردگی کا جائزہ رپورٹ جنریٹ کریں
        """
        if not self.evaluation_history:
            return "کوئی جائزے دستیاب نہیں."

        latest_evaluation = self.evaluation_history[-1]
        report = {
            'summary': self.generate_summary(latest_evaluation),
            'trends': self.analyze_trends(),
            'recommendations': self.generate_recommendations(latest_evaluation)
        }

        return report

    def generate_summary(self, evaluation):
        """
        تازہ ترین جائزے کا خلاصہ تیار کریں
        """
        metrics = evaluation['aggregate_metrics']

        summary = f"""
        پلاننگ کارکردگی کا خلاصہ:
        - کامیابی کی شرح: {metrics.get('plan_success_rate', 0):.2%}
        - اوسط منصوبہ لمبائی: {metrics.get('average_plan_length', 0):.1f} اسٹیپس
        - اوسط جنریشن ٹائم: {metrics.get('average_generation_time', 0):.2f}s
        - اوسط ایکسیکیوشن ٹائم: {metrics.get('average_execution_time', 0):.2f}s
        - اوسط اڈاپٹیشنز: {metrics.get('average_adaptations', 0):.1f} فی منصوبہ
        - اوسط ناکامیاں: {metrics.get('average_errors', 0):.1f} فی منصوبہ
        """

        return summary

    def analyze_trends(self):
        """
        وقت کے ساتھ کارکردگی کے رجحانات کا تجزیہ کریں
        """
        if len(self.evaluation_history) < 2:
            return "رجحانات کے تجزیہ کے لیے ناکافی ڈیٹا."

        # پہلے اور آخری جائزے کا موازنہ کریں
        first_metrics = self.evaluation_history[0]['aggregate_metrics']
        last_metrics = self.evaluation_history[-1]['aggregate_metrics']

        trends = {}
        for metric in first_metrics.keys():
            if metric in last_metrics:
                change = last_metrics[metric] - first_metrics[metric]
                direction = "improving" if change > 0 else "declining" if change < 0 else "stable"
                trends[metric] = {
                    'direction': direction,
                    'change': change,
                    'first_value': first_metrics[metric],
                    'last_value': last_metrics[metric]
                }

        return trends

    def generate_recommendations(self, evaluation):
        """
        جائزہ کے نتائج کی بنیاد پر تجاویز جنریٹ کریں
        """
        metrics = evaluation['aggregate_metrics']
        recommendations = []

        if metrics.get('plan_success_rate', 0) < 0.7:
            recommendations.append(
                "کامیابی کی شرح 70% سے کم ہے. منصوبہ جنریشن کو بہتر بنانے یا "
                "زیادہ مضبوط ناکامی کے انتظام کو شامل کرنے پر غور کریں."
            )

        if metrics.get('average_generation_time', float('inf')) > 5.0:
            recommendations.append(
                "منصوبہ جنریشن ٹائم 5 سیکنڈ سے زیادہ ہے. LLM انفریسنگ کو بہتر بنائیں "
                "یا عام منظرناموں کے لیے منصوبہ کیش کا استعمال کریں."
            )

        if metrics.get('average_adaptations', 0) > 2.0:
            recommendations.append(
                "زیادہ اڈاپٹیشن فریکوینسی ظاہر کرتی ہے کہ منصوبے شاید بہت سخت ہیں. "
                "ابتداء سے زیادہ اڈاپٹو منصوبے بنانے پر غور کریں."
            )

        return recommendations

    def benchmark_against_classical_planners(self, llm_planner, classical_planners, scenarios):
        """
        کلاسیکل پلاننگ ا approaches کے مقابلے میں LLM پلانر کا جائزہ لیں
        """
        results = {
            'llm_planner': self.evaluate_planning_performance(llm_planner, scenarios),
            'classical_planners': {}
        }

        for name, planner in classical_planners.items():
            results['classical_planners'][name] = self.evaluate_planning_performance(planner, scenarios)

        return results
```

## عملی مشق: کوگنیٹو پلاننگ سسٹم کا نفاذ

### مشق کے اہداف
- ہیومنوائڈ روبوٹ کے لیے LLM-مبنی پلاننگ سسٹم کا مکمل نفاذ کریں
- ہیئرآرکیکل پلاننگ کو اڈاپٹیشن صلاحیات کے ساتھ ضم کریں
- شبیہہ ماحول میں پلاننگ کارکردگی کی جانچ کریں
- جائزہ اور منصوبہ بندی کو بہتر بنائیں

### قدم وار ہدایات

1. **LLM پلاننگ انفراسٹرکچر** کو ماڈل لوڈنگ اور بنیادی صلاحیات کے ساتھ سیٹ اپ کریں
2. **ہیئرآرکیکل پلاننگ** متعدد ایبسٹریکشن سطحوں کے ساتھ نافذ کریں
3. **اڈاپٹیشن میکنزمز** منصوبہ ناکامیوں کو ہینڈل کرنے کے لیے شامل کریں
4. **روبوٹ سیمولیٹر** کے ساتھ انضمام کریں ایکسیکیوشن ٹیسٹنگ کے لیے
5. **کارکردگی کا جائزہ** متعدد منظرناموں کے ساتھ لیں
6. **پلاننگ پیرامیٹرز** کو جائزہ کے نتائج کے مطابق بہتر بنائیں

### متوقع نتائج
- کام کرتا ہوا LLM-مبنی کوگنیٹو پلاننگ سسٹم
- ہیئرآرکیکل پلاننگ کے تصورات کی سمجھ
- اڈاپٹیو پلاننگ اور سیکھنے کا تجربہ
- کارکردگی کے جائزہ کی صلاحیتیں

## علم کی چیک

1. LLM-مبنی پلاننگ کلاسیکل علامتی پلاننگ ا approaches سے کیسے مختلف ہے؟
2. ہیومنوائڈ روبوٹکس میں ہیئرآرکیکل پلاننگ کے تصور کی وضاحت کریں۔
3. روبوٹس کے لیے اڈاپٹو پلاننگ کو نافذ کرنے میں کیا کلیدی چیلنج ہیں؟
4. گزشتہ تجربات سے سیکھنا پلاننگ کارکردگی کو کیسے بہتر بنا سکتا ہے؟

## خلاصہ

اس باب نے ہیومنوائڈ روبوٹکس کے لیے کوگنیٹو پلاننگ سسٹمزم میں بڑے زبانی ماڈلز کے انضمام کو تلاش کیا۔ ہم نے ہیئرآرکیکل پلاننگ آرکیٹیکچرزم، علامتی-نیورل انضمام، اور اڈاپٹو پلاننگ میکنزمز کو کور کیا۔ LLM-مبنی پلاننگ روبوٹس کو پیچیدہ، متعدد اسٹیپس والے کاموں کو نیچرل لینگویج انٹرفیسز کے ساتھ ہینڈل کرنے کے قابل بناتا ہے جبکہ تبدیل ہوتی ہوئی حالات کے مطابق ایڈجسٹ ہوتا ہے اور تجربے سے سیکھتا ہے۔ بالا سطحی ریزننگ صلاحیتیں اور عملی ایکسیکیوشن کنٹرینٹس کا مجموعہ حقیقی دنیا کے ماحول میں کام کرنے کے قابل مضبوط پلاننگ سسٹمزم تخلیق کرتا ہے۔

## اگلے اقدامات

باب 20 میں، ہم کیپسٹون پراجیکٹ کا جائزہ لیں گے جہاں ہم اس کتاب کے تمام تصورات کو ایک مکمل ہیومنوائڈ روبوٹ سسٹم میں ضم کریں گے، جو پیچیدہ ٹاسک ایکسیکیوشن اور انسانی انٹرایکشن کے قابل ہو۔