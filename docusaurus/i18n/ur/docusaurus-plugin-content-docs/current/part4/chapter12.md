---
title: "باب 12: سیم ٹو ریل ٹرانسفر تکنیکس"
sidebar_label: "باب 12: سیم ٹو ریل ٹرانسفر"
---

# باب 12: سیم ٹو ریل ٹرانسفر تکنیکس

## سیکھنے کے اہداف
- ہیومنوائڈ روبوٹکس کے لیے سیم ٹو ریل ٹرانسفر کے اصولوں کو سمجھنا
- سیمولیشن اور حقیقی دنیا کی تنصیب کے درمیان حقیقت کے فرق کی شناخت اور اس کا سامنا کرنا
- ڈومین رینڈمائزیشن اور دیگر ٹرانسفر لرننگ میتھوڈولوجیز نافذ کرنا
- کامیاب سیمولیشن ٹو ریلیٹی تنصیب کے لیے بہترین طریقے نافذ کرنا

## تعارف

سیم ٹو ریل ٹرانسفر روبوٹکس کی ترقی کے سب سے چیلنجنگ پہلوؤں میں سے ایک ہے، خاص طور پر پیچیدہ ہیومنوائڈ سسٹمز کے لیے۔ "حقیقت کا فرق" - سیمولیشن اور حقیقی دنیا کے ماحول کے درمیان فرق - ایسے کنٹرول پالیسیز اور ادراک سسٹمز کو ناکام کر سکتا ہے جو سیمولیشن میں مکمل طور پر کام کرتے ہیں لیکن حقیقی دنیا میں برا طریقے سے ناکام ہو جاتے ہیں۔ یہ باب اس فرق کو پُر کرنے کی تکنیکس کو تلاش کرتا ہے، جو سیمولیشن ٹرینڈ سسٹمز کو جسمانی ہیومنوائڈ روبوٹس میں کامیابی کے ساتھ منتقل کرنے کے قابل بناتا ہے۔

## حقیقت کے فرق کو سمجھنا

### حقیقت کے فرق کے ذرائع

ہیومنوائڈ روبوٹکس میں حقیقت کا فرق متعدد ذرائع سے نکلتا ہے:

1. **ڈائینمکس ماڈلنگ کی غلطیاں**: جسمانی خصوصیات کی غلط سیمولیشن
2. **سینسر نوائس اور نقص**: سیمولیٹڈ اور حقیقی سینسرز کے درمیان فرق
3. **ایکچوایٹر کی حدود**: حقیقی ایکچوایٹرز میں تاخیر، نوائس، اور حدود ہوتی ہیں
4. **ماحولیاتی حالات**: روشنی، سطح کی خصوصیات، اور متغیرات
5. **ماڈل کی سادگی**: سیمولیشن میں کمپیوٹیشنل پابندیاں
6. **سسٹم شناخت کی غلطیاں**: غلط جسمانی پیرامیٹرز

### حقیقت کے فرق کو مقدار میں بیان کرنا

حقیقت کا فرق کو مختلف میٹرکس کا استعمال کرتے ہوئے مقدار میں بیان کیا جا سکتا ہے:

```python
# حقیقت کے فرق کی مقدار میں بیان کرنا کا مثال
import numpy as np
from scipy.spatial.distance import euclidean

class RealityGapAnalyzer:
    def __init__(self):
        self.simulation_data = []
        self.real_world_data = []
        self.gap_metrics = {}

    def calculate_dynamics_gap(self, sim_trajectory, real_trajectory):
        """متحرک رویے میں فرق کا حساب لگائیں"""
        if len(sim_trajectory) != len(real_trajectory):
            raise ValueError("ٹریجکٹریز کا ایک ہی لمبائی ہونا چاہیے")

        position_errors = []
        velocity_errors = []

        for i in range(len(sim_trajectory)):
            pos_error = euclidean(
                sim_trajectory[i]['position'],
                real_trajectory[i]['position']
            )
            position_errors.append(pos_error)

            vel_error = abs(
                sim_trajectory[i]['velocity'] -
                real_trajectory[i]['velocity']
            )
            velocity_errors.append(vel_error)

        avg_pos_error = np.mean(position_errors)
        avg_vel_error = np.mean(velocity_errors)

        return {
            'avg_position_error': avg_pos_error,
            'avg_velocity_error': avg_vel_error,
            'max_position_error': max(position_errors),
            'std_position_error': np.std(position_errors)
        }

    def calculate_sensor_gap(self, sim_sensor_data, real_sensor_data):
        """سینسر کے پڑھنے میں فروق کو مقدار میں بیان کریں"""
        # سینسر ڈیٹا کے درمیان احصائی فروق کا حساب لگائیں
        sim_mean = np.mean(sim_sensor_data)
        real_mean = np.mean(real_sensor_data)
        mean_diff = abs(sim_mean - real_mean)

        sim_std = np.std(sim_sensor_data)
        real_std = np.std(real_sensor_data)
        std_diff = abs(sim_std - real_std)

        # تعلق کا حساب لگائیں
        if len(sim_sensor_data) > 1:
            correlation = np.corrcoef(sim_sensor_data, real_sensor_data)[0, 1]
        else:
            correlation = 0.0

        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'correlation': correlation
        }

    def assess_reality_gap(self):
        """جامع حقیقت کا فرق کا جائزہ"""
        gap_report = {
            'dynamics_gap': self.calculate_dynamics_gap(
                self.simulation_data, self.real_world_data
            ),
            'sensor_gap': self.calculate_sensor_gap(
                [d['sensor'] for d in self.simulation_data],
                [d['sensor'] for d in self.real_world_data]
            )
        }

        return gap_report
```

### ہیومنوائڈ روبوٹکس پر اثر

حقیقت کا فرق ہیومنوائڈ روبوٹس کو مخصوص طریقے سے متاثر کرتا ہے:

- **توازن کنٹرول**: چھوٹی ماڈلنگ کی غلطیاں نمایاں توازن کے مسائل کا سبب بن سکتی ہیں
- **فوٹ اسٹیپ منصوبہ بندی**: سطح کی خصوصیات قدم کی استحکام کو متاثر کرتی ہیں
- **مینیپولیشن**: چیزوں کی خصوصیات گریسنگ کی کامیابی کو متاثر کرتی ہیں
- **نیویگیشن**: ڈائینامک رکاوٹیں اور زمین کی تبدیلیاں
- **ادراک**: روشنی اور ٹیکسچر کے فروق پہچان کو متاثر کرتے ہیں

## ڈومین رینڈمائزیشن

### ڈومین رینڈمائزیشن کے اصول

ڈومین رینڈمائزیشن ایک تکنیک ہے جو ماڈلز کو سیمولیشن میں رینڈمائزڈ پیرامیٹرز کے ساتھ تربیت دیتی ہے تاکہ حقیقی دنیا کی کارکردگی کو بہتر بنایا جا سکے:

```python
# ڈومین رینڈمائزیشن امپلیمنٹیشن
import numpy as np
import random

class DomainRandomizer:
    def __init__(self):
        # رینڈمائزیشن کے لیے پیرامیٹر کی حدیں متعین کریں
        self.param_ranges = {
            'robot_mass': (0.8, 1.2),  # حقیقی ماس کا فیکٹر
            'friction_coefficient': (0.1, 1.0),
            'restitution': (0.0, 0.3),
            'gravity': (9.5, 10.5),  # میٹر/سیکنڈ^2
            'sensor_noise_std': (0.001, 0.01),
            'actuator_delay': (0.005, 0.02),  # سیکنڈز
            'lighting_intensity': (0.5, 2.0),
            'texture_scale': (0.1, 2.0),
            'camera_intrinsics': (0.8, 1.2),  # فوکل لمبائی کا فیکٹر
        }

    def randomize_environment(self, sim_env):
        """سیمولیشن ماحول پر ڈومین رینڈمائزیشن لاگو کریں"""
        randomized_params = {}

        for param_name, (min_val, max_val) in self.param_ranges.items():
            if 'robot' in param_name:
                # روبوٹ-مخصوص پیرامیٹرز
                randomized_params[param_name] = random.uniform(min_val, max_val)
            elif 'sensor' in param_name:
                # سینسر-مخصوص پیرامیٹرز
                randomized_params[param_name] = random.uniform(min_val, max_val)
            elif 'lighting' in param_name:
                # روشنی کے پیرامیٹرز
                randomized_params[param_name] = random.uniform(min_val, max_val)
            else:
                # عام پیرامیٹرز
                randomized_params[param_name] = random.uniform(min_val, max_val)

        # سیمولیشن میں رینڈمائزڈ پیرامیٹرز لاگو کریں
        self.apply_parameters(sim_env, randomized_params)
        return randomized_params

    def apply_parameters(self, sim_env, params):
        """سیمولیشن ماحول میں رینڈمائزڈ پیرامیٹرز لاگو کریں"""
        # روبوٹ ماس رینڈمائزیشن لاگو کریں
        if 'robot_mass' in params:
            sim_env.robot.set_mass(params['robot_mass'])

        # فرکشن رینڈمائزیشن لاگو کریں
        if 'friction_coefficient' in params:
            sim_env.set_friction(params['friction_coefficient'])

        # سینسر نوائس رینڈمائزیشن لاگو کریں
        if 'sensor_noise_std' in params:
            sim_env.add_sensor_noise(params['sensor_noise_std'])

        # روشنی رینڈمائزیشن لاگو کریں
        if 'lighting_intensity' in params:
            sim_env.set_lighting(params['lighting_intensity'])

    def train_with_domain_randomization(self, policy, num_episodes=1000):
        """ڈومین رینڈمائزیشن کے ساتھ پالیسی تربیت دیں"""
        for episode in range(num_episodes):
            # ہر ایپی سوڈ کے لیے ماحول کو رینڈمائز کریں
            randomized_params = self.randomize_environment(self.sim_env)

            # رینڈمائزڈ ماحول میں پالیسی تربیت دیں
            episode_reward = self.run_episode(policy, randomized_params)

            # تربیت کی پیشرفت لاگ کریں
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}")

        return policy

# مثال استعمال
domain_rand = DomainRandomizer()
```

### اعلی درجے کی ڈومین رینڈمائزیشن تکنیکس

#### ٹیکسچر رینڈمائزیشن

```python
# ٹیکسچر اور ظہور رینڈمائزیشن
class TextureRandomizer:
    def __init__(self):
        self.texture_library = [
            'wood', 'metal', 'concrete', 'carpet', 'tile',
            'grass', 'sand', 'water', 'fabric'
        ]
        self.color_palettes = [
            [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)],  # گرے سکیل
            [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2)],  # لال-سبز
            [(0.2, 0.2, 0.8), (0.8, 0.8, 0.2)],  # نیلا-پیلا
        ]

    def randomize_textures(self, sim_env):
        """سیمولیشن میں ٹیکسچرز اور ظہور کو رینڈمائز کریں"""
        for surface in sim_env.get_surfaces():
            # ٹیکسچر بے ترتیب طور پر منتخب کریں
            texture = random.choice(self.texture_library)
            surface.set_texture(texture)

            # رنگوں کو بے ترتیب طور پر ایڈجسٹ کریں
            color_palette = random.choice(self.color_palettes)
            primary_color = random.choice(color_palette)
            surface.set_color(primary_color)

    def randomize_lighting(self, sim_env):
        """روشنی کی حالتیں رینڈمائز کریں"""
        # لائٹ کی پوزیشنز کو رینڈمائز کریں
        lights = sim_env.get_lights()
        for light in lights:
            # پوزیشن رینڈمائز کریں
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z = random.uniform(2, 10)
            light.set_position([x, y, z])

            # شدت اور رنگ رینڈمائز کریں
            intensity = random.uniform(0.5, 2.0)
            color = [random.uniform(0.8, 1.0) for _ in range(3)]
            light.set_intensity(intensity)
            light.set_color(color)
```

#### فزکس پیرامیٹر رینڈمائزیشن

```python
# فزکس پیرامیٹر رینڈمائزیشن
class PhysicsRandomizer:
    def __init__(self):
        self.physics_params = {
            'gravity': {'mean': 9.81, 'std': 0.1, 'range': (9.5, 10.1)},
            'air_resistance': {'mean': 0.01, 'std': 0.005, 'range': (0.005, 0.02)},
            'ground_friction': {'mean': 0.5, 'std': 0.2, 'range': (0.1, 0.9)},
            'joint_damping': {'mean': 0.1, 'std': 0.05, 'range': (0.01, 0.3)},
        }

    def randomize_physics(self, sim_env):
        """فزکس پیرامیٹرز کو رینڈمائز کریں"""
        for param_name, param_info in self.physics_params.items():
            # حد کے اندر نارمل ڈسٹری بیوشن سے نمونہ لیں
            while True:
                value = np.random.normal(param_info['mean'], param_info['std'])
                if param_info['range'][0] <= value <= param_info['range'][1]:
                    break

            # سیمولیشن میں لاگو کریں
            sim_env.set_physics_parameter(param_name, value)
```

## سسٹم شناخت اور ماڈل کیلیبریشن

### حقیقی روبوٹ پیرامیٹرز کی شناخت

سسٹم شناخت حقیقت کے فرق کو کم کرنے کے لیے اہم ہے:

```python
# ہیومنوائڈ روبوٹ کے لیے سسٹم شناخت
import scipy.optimize as opt
from scipy.integrate import odeint

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.identification_data = []
        self.optimized_params = {}

    def collect_identification_data(self, robot, trajectories):
        """سسٹم شناخت کے لیے ڈیٹا جمع کریں"""
        collected_data = []

        for trajectory in trajectories:
            # حقیقی روبوٹ پر ٹریجکٹری انجام دیں
            robot.execute_trajectory(trajectory)

            # حالت، ان پٹ، اور آؤٹ پٹ ڈیٹا ریکارڈ کریں
            states = robot.get_states()
            inputs = robot.get_inputs()
            outputs = robot.get_outputs()

            collected_data.append({
                'states': states,
                'inputs': inputs,
                'outputs': outputs,
                'trajectory': trajectory
            })

        return collected_data

    def dynamics_model(self, state, t, params):
        """نامعلوم پیرامیٹرز کے ساتھ ڈائینمکس ماڈل"""
        # مثال: سادہ پینڈولم ماڈل (ایک جوائنٹ کے لیے)
        theta, theta_dot = state
        g, l, b = params  # گریویٹی، لمبائی، ڈیمپنگ

        # ڈائینمکس مساواتیں
        theta_ddot = -(g/l) * np.sin(theta) - b * theta_dot

        return [theta_dot, theta_ddot]

    def simulate_system(self, params, initial_state, time_points):
        """دیے گئے پیرامیٹرز کے ساتھ سسٹم کی سیمولیشن کریں"""
        solution = odeint(
            self.dynamics_model,
            initial_state,
            time_points,
            args=(params,)
        )
        return solution

    def objective_function(self, params, time_points, real_states):
        """پیرامیٹر کی اصلاح کے لیے اہداف کا فنکشن"""
        # موجودہ پیرامیٹرز کے ساتھ سیمولیٹ کریں
        simulated_states = self.simulate_system(params, real_states[0], time_points)

        # حقیقی اور سیمولیٹڈ کے درمیان غلطی کا حساب لگائیں
        error = np.sum((real_states - simulated_states)**2)
        return error

    def identify_parameters(self, real_data):
        """اصلاح کے ذریعے سسٹم پیرامیٹرز کی شناخت کریں"""
        # متعلقہ ڈیٹا نکالیں
        time_points = real_data['time']
        real_states = real_data['states']
        initial_params = [9.81, 1.0, 0.1]  # [g, l, b]

        # پیرامیٹرز کی اصلاح کریں
        result = opt.minimize(
            self.objective_function,
            initial_params,
            args=(time_points, real_states),
            method='BFGS'
        )

        self.optimized_params = result.x
        return result.x

    def update_simulation_model(self):
        """شناخت شدہ پیرامیٹرز کے ساتھ سیمولیشن کو اپ ڈیٹ کریں"""
        for i, param_name in enumerate(['gravity', 'length', 'damping']):
            self.robot_model.set_parameter(param_name, self.optimized_params[i])
```

### ماڈل-مبنی ٹرانسفر لرننگ

```python
# ماڈل-مبنی ٹرانسفر لرننگ
class ModelBasedTransfer:
    def __init__(self, sim_model, real_model):
        self.sim_model = sim_model
        self.real_model = real_model
        self.transfer_matrix = None

    def learn_transfer_mapping(self, sim_data, real_data):
        """سیمولیشن سے حقیقی دنیا کے لیے میپنگ سیکھیں"""
        # ٹرانسفر فنکشن سیکھنے کے لیے مشین لرننگ کا استعمال کریں
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        # تربیت کا ڈیٹا تیار کریں
        X_train = sim_data  # سیمولیشن ان پٹس/آؤٹ پٹس
        y_train = real_data  # حقیقی دنیا کا متعلقہ ڈیٹا

        # کرنل کی وضاحت کریں
        kernel = ConstantKernel(1.0) * RBF(1.0)

        # گاؤسین پروسیس تربیت دیں
        self.transfer_gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10
        )
        self.transfer_gp.fit(X_train, y_train)

    def apply_transfer(self, sim_prediction):
        """سیمولیشن کی پیڈکشن پر سیکھی گئی ٹرانسفر لاگو کریں"""
        # تربیت شدہ ماڈل کا استعمال کرتے ہوئے سیمولیشن کو حقیقی میں میپ کریں
        real_prediction, uncertainty = self.transfer_gp.predict(
            sim_prediction.reshape(1, -1),
            return_std=True
        )
        return real_prediction[0], uncertainty
```

## مضبوط کنٹرول اور ایڈاپٹیشن

### مضبوط کنٹرول ڈیزائن

مضبوط کنٹرول تکنیکس ماڈل کی عدم یقینیوں کو سنبھالنے میں مدد کرتی ہیں:

```python
# ہیومنوائڈ روبوٹس کے لیے مضبوط کنٹرول
class RobustController:
    def __init__(self, nominal_model, uncertainty_bounds):
        self.nominal_model = nominal_model
        self.uncertainty_bounds = uncertainty_bounds
        self.controller_params = {}
        self.adaptive_components = []

    def design_robust_controller(self):
        """ماڈل کی عدم یقینیوں کے لیے مضبوط کنٹرولر ڈیزائن کریں"""
        # H-انFINITY کنٹرول ڈیزائن
        # یہ ایک سادہ مثال ہے
        import control as ctrl

        # نامزد سسٹم کی وضاحت کریں
        A, B, C, D = self.nominal_model.get_state_space()
        sys_nominal = ctrl.ss(A, B, C, D)

        # مضبوط کنٹرول تکنیکس کا استعمال کرتے ہوئے کنٹرولر ڈیزائن کریں
        # ہیومنوائڈ توازن کے لیے، LQR کو عدم یقینی کے ساتھ ڈیزائن کریں
        Q = np.eye(A.shape[0])  # حالت کی قیمت میٹرکس
        R = np.eye(B.shape[1])  # ان پٹ کی قیمت میٹرکس

        # Riccati مساوات حل کریں LQR کے لیے
        K, S, E = ctrl.lqr(A, B, Q, R)
        self.controller_params['K'] = K

        return K

    def adaptive_control(self, state_error, time_step):
        """تبدیل ہوتی حالت کو سنبھالنے کے لیے ایڈاپٹیو کنٹرول"""
        # پیرامیٹر ایڈاپٹیشن لا
        # یہ ہیومنوائڈ توازن کے لیے ایک سادہ مثال ہے
        adaptation_rate = 0.01

        # غلطی کے مطابق کنٹرولر پیرامیٹرز اپ ڈیٹ کریں
        if np.linalg.norm(state_error) > 0.1:  # غلطی کی حد
            # کنٹرول گینز ایڈجسٹ کریں
            self.controller_params['K'] *= (1 + adaptation_rate)

        return self.controller_params['K']

    def robust_balance_control(self, robot_state, target_state):
        """ہیومنوائڈ روبوٹ کے لیے مضبوط توازن کنٹرول"""
        # حالت کی غلطی کا حساب لگائیں
        state_error = target_state - robot_state

        # مضبوط کنٹرول لا لاگو کریں
        control_input = -np.dot(self.controller_params['K'], state_error)

        # ضرورت پڑنے پر ایڈاپٹیو کمپوننٹ لاگو کریں
        adaptive_input = self.adaptive_control(state_error, 0.01)
        control_input += adaptive_input

        # کنٹرول سیچوریشن حدیں لاگو کریں
        control_input = np.clip(control_input, -1.0, 1.0)

        return control_input
```

### آن لائن ایڈاپٹیشن تکنیکس

```python
# سیم ٹو ریل ٹرانسفر کے لیے آن لائن ایڈاپٹیشن
class OnlineAdaptation:
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.adaptation_strategy = None
        self.adaptation_history = []

    def monitor_performance(self, sim_performance, real_performance):
        """کارکردگی کا فرق مانیٹر کریں اور ایڈاپٹیشن ٹرگر کریں"""
        performance_gap = abs(sim_performance - real_performance)

        if performance_gap > self.performance_threshold:
            # ایڈاپٹیشن ٹرگر کریں
            adaptation_needed = True
            adaptation_type = self.determine_adaptation_type(
                sim_performance, real_performance
            )
        else:
            adaptation_needed = False
            adaptation_type = None

        return adaptation_needed, adaptation_type

    def determine_adaptation_type(self, sim_perf, real_perf):
        """مناسب ایڈاپٹیشن حکمت عملی کا تعین کریں"""
        if real_perf < sim_perf * 0.7:  # نمایاں کارکردگی میں کمی
            return "major_adaptation"
        elif real_perf < sim_perf * 0.9:  # معتدل کارکردگی میں کمی
            return "minor_adaptation"
        else:
            return "monitoring_only"

    def online_parameter_adaptation(self, current_params, performance_feedback):
        """حقیقی دنیا کی کارکردگی کے مطابق پیرامیٹرز ایڈاپٹ کریں"""
        # گریڈیئنٹ-فری اصلاح یا دیگر طریقے استعمال کریں
        learning_rate = 0.01

        # کارکردگی گریڈیئنٹ کے مطابق پیرامیٹر اپ ڈیٹس کا حساب لگائیں
        param_updates = self.estimate_parameter_gradient(
            current_params, performance_feedback
        )

        # پیرامیٹرز اپ ڈیٹ کریں
        updated_params = current_params + learning_rate * param_updates

        return updated_params

    def estimate_parameter_gradient(self, params, performance_data):
        """ایڈاپٹیشن کے لیے پیرامیٹر گریڈیئنٹ کا تخمینہ لگائیں"""
        # محدود فرق طریقہ
        gradient = np.zeros_like(params)
        epsilon = 0.001

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            perf_plus = self.evaluate_performance(params_plus)

            params_minus = params.copy()
            params_minus[i] -= epsilon
            perf_minus = self.evaluate_performance(params_minus)

            gradient[i] = (perf_plus - perf_minus) / (2 * epsilon)

        return gradient

    def evaluate_performance(self, params):
        """دیے گئے پیرامیٹرز کے ساتھ کارکردگی کا جائزہ لیں"""
        # اس میں پیرامیٹرز کے ساتھ سسٹم چلانا اور کارکردگی کے میٹرکس ناپنا شامل ہوگا
        pass
```

## ادراک ڈومین رینڈمائزیشن

### ویژوئل ڈومین رینڈمائزیشن

کیمرہ والے ہیومنوائڈ روبوٹس کے لیے، ویژوئل ڈومین رینڈمائزیشن اہم ہے:

```python
# ہیومنوائڈ ادراک کے لیے ویژوئل ڈومین رینڈمائزیشن
import cv2
import numpy as np

class VisualDomainRandomizer:
    def __init__(self):
        self.color_augmentations = [
            'brightness', 'contrast', 'saturation',
            'hue', 'gamma', 'noise'
        ]
        self.geometric_augmentations = [
            'blur', 'motion_blur', 'gaussian_noise',
            'jpeg_compression', 'pixelate'
        ]

    def randomize_image(self, image):
        """تصویر پر بے ترتیب ویژوئل اضافیات لاگو کریں"""
        augmented_image = image.copy()

        # بے ترتیب رنگ اضافیات لاگو کریں
        if random.random() > 0.3:  # 70% کا موقع
            augmented_image = self.random_color_augmentation(augmented_image)

        # بے ترتیب جیومیٹرک اضافیات لاگو کریں
        if random.random() > 0.4:  # 60% کا موقع
            augmented_image = self.random_geometric_augmentation(augmented_image)

        return augmented_image

    def random_color_augmentation(self, image):
        """بے ترتیب رنگ-مبنی اضافیات لاگو کریں"""
        augmented = image.astype(np.float32)

        # بے ترتیب چمک
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            augmented = augmented * brightness_factor

        # بے ترتیب کنٹراسٹ
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            augmented = 127 + (augmented - 127) * contrast_factor

        # بے ترتیب سیچوریشن
        if random.random() > 0.5:
            # سیچوریشن ایڈجسٹمنٹ کے لیے HSV میں تبدیل کریں
            hsv = cv2.cvtColor(augmented.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            saturation_factor = random.uniform(0.5, 1.5)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ویلیوز کو درست حد میں کلپ کریں
        augmented = np.clip(augmented, 0, 255)
        return augmented.astype(np.uint8)

    def random_geometric_augmentation(self, image):
        """بے ترتیب جیومیٹرک اضافیات لاگو کریں"""
        augmented = image.copy()

        # بے ترتیب بلر
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5, 7])
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)

        # بے ترتیب نوائس
        if random.random() > 0.6:
            noise = np.random.normal(0, random.uniform(5, 15), augmented.shape)
            augmented = augmented + noise
            augmented = np.clip(augmented, 0, 255)

        return augmented.astype(np.uint8)

    def simulate_camera_effects(self, image):
        """ مختلف کیمرہ اثرات اور نقصانات کی شبیہہ بنائیں"""
        # لینس کی ڈسٹورشن کی شبیہہ بنائیں
        augmented = self.simulate_lens_distortion(image)

        # موشن بلر کی شبیہہ بنائیں (متحرک ہیومنوائڈ کے لیے)
        if random.random() > 0.5:
            augmented = self.simulate_motion_blur(augmented)

        # فوکس بلر کی شبیہہ بنائیں
        if random.random() > 0.5:
            augmented = self.simulate_focus_blur(augmented)

        return augmented

    def simulate_lens_distortion(self, image):
        """لینس ڈسٹورشن اثرات کی شبیہہ بنائیں"""
        h, w = image.shape[:2]

        # بے ترتیب ڈسٹورشن کوائف تیار کریں
        k1 = random.uniform(-0.1, 0.1)  # ریڈیل ڈسٹورشن
        k2 = random.uniform(-0.01, 0.01)
        p1 = random.uniform(-0.01, 0.01)  # ٹینجنٹل ڈسٹورشن
        p2 = random.uniform(-0.01, 0.01)

        # کیمرہ میٹرکس تیار کریں
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([[w, 0, cx], [0, w, cy], [0, 0, 1]])

        # ڈسٹورشن لاگو کریں
        distorted = cv2.undistort(image, camera_matrix, np.array([k1, k2, p1, p2, 0]))

        return distorted
```

## ٹرانسفر لرننگ تکنیکس

### فائن ٹیوننگ کے طریقے

```python
# سیم ٹو ریل کے لیے ٹرانسفر لرننگ
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class SimToRealTransferLearner:
    def __init__(self, pretrained_model, real_data_size=100):
        self.pretrained_model = pretrained_model
        self.real_data_size = real_data_size
        self.transfer_strategy = "fine_tuning"

    def prepare_transfer_data(self, sim_data, real_data):
        """ٹرانسفر لرننگ کے لیے ڈیٹا تیار کریں"""
        # سیم اور ریل کے درمیان ڈیٹا فارمیٹس کو مطابق کریں
        aligned_sim_data = self.align_data_format(sim_data, "simulation")
        aligned_real_data = self.align_data_format(real_data, "real")

        # ٹرانسفر کے لیے مکسڈ ڈیٹا سیٹ تیار کریں
        mixed_dataset = self.create_mixed_dataset(
            aligned_sim_data, aligned_real_data
        )

        return mixed_dataset

    def align_data_format(self, data, source_type):
        """سیمولیشن اور حقیقی دنیا کے درمیان ڈیٹا فارمیٹ مطابق کریں"""
        # اس میں مندرجہ ذیل کے درمیان فروق کو سنبھالا جائے گا:
        # - ڈیٹا ٹائپس (مثلاً float32 بمقابلہ float64)
        # - کوآرڈینیٹ سسٹمز
        # - ناپنے کی اکائیاں
        # - سینسر کنفیگریشنز
        return data

    def create_mixed_dataset(self, sim_data, real_data):
        """سیمولیشن اور حقیقی ڈیٹا کو مکس کرنے والی ڈیٹا سیٹ تیار کریں"""
        # سیم سے ریل ڈیٹا کے مختلف تناسب استعمال کریں
        sim_ratio = 0.8  # 80% سیم ڈیٹا سے شروع کریں
        real_ratio = 0.2  # 20% ریل ڈیٹا

        mixed_data = []
        mixed_labels = []

        # ڈومین لیبل کے ساتھ سیمولیشن ڈیٹا شامل کریں
        for data_point in sim_data:
            mixed_data.append(data_point)
            mixed_labels.append("simulation")

        # ڈومین لیبل کے ساتھ ریل ڈیٹا شامل کریں
        for data_point in real_data:
            mixed_data.append(data_point)
            mixed_labels.append("real")

        return list(zip(mixed_data, mixed_labels))

    def fine_tune_model(self, train_loader, epochs=10):
        """حقیقی دنیا کے ڈیٹا کے ساتھ ماڈل کو فائن ٹیون کریں"""
        # ابتدائی لیئرز فریز کریں (ٹرانسفر لرنٹ فیچرز)
        for param in list(self.pretrained_model.parameters())[:-4]:  # آخری 4 لیئرز کے علاوہ سب فریز کریں
            param.requires_grad = False

        # فائن ٹیوننگ کے لیے کم سیکھنے کی شرح استعمال کریں
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.pretrained_model.parameters()),
            lr=1e-5  # سیکھنے کی کم شرح تاکہ سیکھے ہوئے فیچرز برقرار رہیں
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.pretrained_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

    def domain_adversarial_training(self, sim_loader, real_loader):
        """ڈومین کے فرق کو کم کرنے کے لیے ڈومین مخالف تربیت چلائیں"""
        # یہ ڈومین-مخالف نیورل نیٹ ورکس (DANN) کو نافذ کرتا ہے
        # ماڈل ڈومین-ان ویرینٹ فیچرز سیکھتا ہے
        pass
```

## توثیق اور ٹیسٹنگ کی حکمت عملیاں

### کراس ڈومین توثیق

```python
# کراس ڈومین توثیق کی تکنیکس
class CrossDomainValidator:
    def __init__(self):
        self.validation_metrics = []
        self.uncertainty_estimators = []

    def validate_transfer(self, sim_model, real_robot, test_scenarios):
        """منظر ناموں کے درمیان سیم ٹو ریل ٹرانسفر کی توثیق کریں"""
        results = {}

        for scenario in test_scenarios:
            # سیمولیشن میں ٹیسٹ کریں
            sim_performance = self.evaluate_in_simulation(
                sim_model, scenario
            )

            # حقیقی روبوٹ پر ٹیسٹ کریں
            real_performance = self.evaluate_on_real_robot(
                real_robot, scenario
            )

            # ٹرانسفر فرق کا حساب لگائیں
            gap = abs(sim_performance - real_performance) / sim_performance * 100

            results[scenario.name] = {
                'sim_performance': sim_performance,
                'real_performance': real_performance,
                'transfer_gap_percent': gap,
                'success_rate': self.calculate_success_rate(
                    real_performance, scenario.threshold
                )
            }

        return results

    def calculate_success_rate(self, performance, threshold):
        """کارکردگی کی حد کے مطابق کامیابی کی شرح کا حساب لگائیں"""
        if isinstance(performance, dict):
            # متعدد میٹرکس
            success_count = 0
            total_metrics = 0

            for metric_name, value in performance.items():
                if value >= threshold.get(metric_name, 0):
                    success_count += 1
                total_metrics += 1

            return success_count / total_metrics if total_metrics > 0 else 0
        else:
            # واحد میٹرک
            return 1.0 if performance >= threshold else 0.0

    def uncertainty_aware_validation(self, model, test_inputs):
        """اچنبھ کی مقدار کے ساتھ توثیق کریں"""
        predictions = []
        uncertainties = []

        for input_data in test_inputs:
            # اچنبھ کے ساتھ پیڈکشن حاصل کریں
            pred, uncertainty = self.predict_with_uncertainty(model, input_data)
            predictions.append(pred)
            uncertainties.append(uncertainty)

        # اچنبھ کے نمونے تجزیہ کریں
        avg_uncertainty = np.mean(uncertainties)
        uncertainty_std = np.std(uncertainties)

        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'avg_uncertainty': avg_uncertainty,
            'uncertainty_std': uncertainty_std
        }

    def predict_with_uncertainty(self, model, input_data):
        """اچنبھ کا تخمینہ کے ساتھ پیڈکشن حاصل کریں"""
        # مونٹی کارلو ڈراپ آؤٹ یا اینسمبل طریقے
        model.train()  # اچنبھ کا تخمینہ کے لیے ڈراپ آؤٹ فعال کریں

        predictions = []
        for _ in range(10):  # متعدد فارورڈ پاسز
            pred = model(input_data)
            predictions.append(pred.detach().cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)

        return mean_pred, uncertainty
```

## کامیاب ٹرانسفر کے لیے بہترین طریقے

### تدریجی ڈومین ٹرانسفر

```python
# تدریجی ڈومین ٹرانسفر کا طریقہ
class GradualDomainTransfer:
    def __init__(self):
        self.transfer_stages = []
        self.current_stage = 0

    def create_transfer_schedule(self):
        """تدریجی ڈومین ٹرانسفر کا شیڈول تیار کریں"""
        # مرحلہ 1: بنیادی سیمولیشن (مکمل حالات)
        stage1 = {
            'name': 'basic_sim',
            'domain_randomization': 0.0,
            'noise': 0.0,
            'complexity': 'low',
            'duration': 1000  # ایپی سوڈز
        }

        # مرحلہ 2: درمیانہ سیمولیشن (کچھ رینڈمائزیشن)
        stage2 = {
            'name': 'medium_sim',
            'domain_randomization': 0.3,
            'noise': 0.05,
            'complexity': 'medium',
            'duration': 2000
        }

        # مرحلہ 3: اعلی درجے کی سیمولیشن (مکمل رینڈمائزیشن)
        stage3 = {
            'name': 'advanced_sim',
            'domain_randomization': 0.8,
            'noise': 0.1,
            'complexity': 'high',
            'duration': 3000
        }

        # مرحلہ 4: مکسڈ ریلیٹی (سیمولیشن + حقیقی ڈیٹا)
        stage4 = {
            'name': 'mixed_reality',
            'domain_randomization': 0.5,
            'real_data_ratio': 0.3,
            'complexity': 'high',
            'duration': 1000
        }

        # مرحلہ 5: حقیقی دنیا (کم سے کم سیمولیشن)
        stage5 = {
            'name': 'real_world',
            'domain_randomization': 0.0,
            'real_data_ratio': 1.0,
            'complexity': 'real',
            'duration': 5000
        }

        self.transfer_stages = [stage1, stage2, stage3, stage4, stage5]

    def advance_to_next_stage(self):
        """اگلے ٹرانسفر مرحلے پر بڑھیں"""
        if self.current_stage < len(self.transfer_stages) - 1:
            self.current_stage += 1
            current_stage = self.transfer_stages[self.current_stage]
            print(f"Advancing to stage: {current_stage['name']}")
            return True
        else:
            print("Transfer complete - reached final stage")
            return False

    def evaluate_stage_progress(self, performance_metrics):
        """اگلے مرحلے پر بڑھنے کے لیے تیاری کا جائزہ لیں"""
        current_stage = self.transfer_stages[self.current_stage]

        # چیک کریں کہ کیا موجودہ مرحلے میں کارکردگی مستحکم ہے
        if self.is_performance_stable(performance_metrics):
            # چیک کریں کہ کیا کم از کم مدت پوری ہو گئی ہے
            if self.current_stage_duration_met(current_stage['duration']):
                return True

        return False

    def is_performance_stable(self, metrics):
        """چیک کریں کہ کیا کارکردگی مستحکم ہے"""
        # اس کا مدار مخصوص میٹرکس پر ہے
        # عموماً: کم تغیر اور مسلسل بہتری
        if len(metrics) < 100:
            return False

        recent_performance = metrics[-50:]  # آخری 50 ایپی سوڈز
        avg_performance = np.mean(recent_performance)
        std_performance = np.std(recent_performance)

        # مستحکم سمجھا جائے اگر تغیر چھوٹا ہو مین کا مقابلہ میں
        stability_threshold = 0.1  # 10% مین کا
        return std_performance / avg_performance < stability_threshold

    def current_stage_duration_met(self, required_duration):
        """چیک کریں کہ کیا موجودہ مرحلے کی مدت کی ضرورت پوری ہو گئی ہے"""
        # یہ اصل تربیتی ایپی سوڈز ٹریک کرے گا
        return True  # سادہ
```

## حفاظتی امور اور خطرہ کم کرنا

### محفوظ ٹرانسفر پروٹوکول

```python
# سیم ٹو ریل ٹرانسفر کے لیے محفظ پروٹوکول
class SafeTransferProtocol:
    def __init__(self, robot_safety_limits):
        self.safety_limits = robot_safety_limits
        self.emergency_stop = False
        self.safety_monitors = []

    def safety_check(self, action, robot_state):
        """چیک کریں کہ کیا ایکشن موجودہ حالت کے مطابق محفوظ ہے"""
        # جوائنٹ حدود چیک کریں
        if not self.check_joint_limits(action):
            return False, "Joint limit violation"

        # رفتار کی حدود چیک کریں
        if not self.check_velocity_limits(action):
            return False, "Velocity limit violation"

        # فورس/ٹورک کی حدود چیک کریں
        if not self.check_force_limits(action):
            return False, "Force limit violation"

        # توازن استحکام چیک کریں
        if not self.check_balance_stability(robot_state):
            return False, "Balance instability"

        return True, "Action is safe"

    def check_joint_limits(self, action):
        """چیک کریں کہ کیا جوائنٹ کمانڈز حدود کے اندر ہیں"""
        for joint_idx, command in enumerate(action):
            if (command < self.safety_limits['joint_min'][joint_idx] or
                command > self.safety_limits['joint_max'][joint_idx]):
                return False
        return True

    def check_velocity_limits(self, action):
        """چیک کریں کہ کیا رفتار کمانڈز محفوظ ہیں"""
        # یہ پچھلی حالت کے مقابلہ میں رفتار چیک کرے گا
        return True  # سادہ

    def check_force_limits(self, action):
        """چیک کریں کہ کیا فورس/ٹورک کمانڈز محفوظ ہیں"""
        return True  # سادہ

    def check_balance_stability(self, robot_state):
        """چیک کریں کہ کیا روبوٹ مستحکم ترتیب میں ہے"""
        # زیرو مومینٹ پوائنٹ (ZMP) یا سینٹر آف ماس (CoM) چیک کریں
        # یہ ایک سادہ چیک ہے
        com_position = robot_state['com_position']
        support_polygon = robot_state['support_polygon']

        # چیک کریں کہ کیا CoM سپورٹ پولی گان کے اندر ہے
        return self.is_point_in_polygon(com_position, support_polygon)

    def is_point_in_polygon(self, point, polygon):
        """چیک کریں کہ کیا پوائنٹ پولی گان (سپورٹ ایریا) کے اندر ہے"""
        # رے کاسٹنگ الگورتھم
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def emergency_procedure(self):
        """ایمرجنسی سٹاپ کا عمل انجام دیں"""
        self.emergency_stop = True
        # تمام ایکچوایٹرز پر صفر کمانڈز بھیجیں
        # ایمرجنسی واقعہ لاگ کریں
        # حفاظتی پروٹوکولز ٹرگر کریں
        pass
```

## ہاتھوں سے مشق: سیم ٹو ریل ٹرانسفر نافذ کرنا

### مشق کے اہداف

- ہیومنوائڈ روبوٹ سیمولیشن کے لیے ڈومین رینڈمائزیشن نافذ کریں
- سیمولیشن پیرامیٹرز کیلیبریٹ کرنے کے لیے سسٹم شناخت لاگو کریں
- ٹرانسفر لرننگ کا طریقہ ڈیزائن اور ٹیسٹ کریں
- حفاظتی امور کے ساتھ ٹرانسفر کارکردگی کی توثیق کریں

### مرحلہ وار ہدایات

1. **سیمولیشن ماحول سیٹ اپ کریں** ڈومین رینڈمائزیشن کی صلاحیتوں کے ساتھ
2. **سسٹم شناخت کا ڈیٹا جمع کریں** سیمولیشن اور حقیقی روبوٹ دونوں سے
3. **پیرامیٹر کیلیبریشن نافذ کریں** حقیقت کا فرق کم کرنے کے لیے
4. **ٹرانسفر لرننگ کا طریقہ ڈیزائن کریں** مناسب توثیق کے ساتھ
5. **سیمولیشن میں ٹیسٹ کریں** بڑھتی ہوئی ڈومین رینڈمائزیشن کے ساتھ
6. **حقیقی روبوٹ پر توثیق کریں** حفاظتی مانیٹرنگ کے ساتھ
7. **ٹرانسفر کارکردگی کا تجزیہ کریں** اور بہتری کے علاقوں کی شناخت کریں

### متوقع نتائج

- کام کرتا ڈومین رینڈمائزیشن سسٹم
- کیلیبریٹڈ سیمولیشن ماڈل
- کامیاب سیم ٹو ریل ٹرانسفر
- کارکردگی کا تجزیہ اور بہتری کی حکمت عملیاں

## نالج چیک

1. ہیومنوائڈ روبوٹکس میں حقیقت کے فرق کے کیا اہم ذرائع ہیں؟
2. وضاحت کریں کہ ڈومین رینڈمائزیشن سیم ٹو ریل کے فرق کو پُر کرنے میں کیسے مدد کرتا ہے۔
3. سسٹم شناخت کیا ہے اور ٹرانسفر کے لیے یہ کیوں اہم ہے؟
4. سیم ٹو ریل ٹرانسفر کے لیے حفاظتی امور کی وضاحت کریں۔

## خلاصہ

اس باب نے ہیومنوائڈ روبوٹس پر سیمولیشن ٹرینڈ سسٹمز کی تنصیب کے لیے ضروری سیم ٹو ریل ٹرانسفر تکنیکس کو تلاش کیا۔ ڈومین رینڈمائزیشن، سسٹم شناخت، مضبوط کنٹرول، اور احتیاط سے توثیق کے ذریعے، ہم حقیقت کا فرق کو کافی حد تک کم کر سکتے ہیں اور پیچیدہ ہیومنوائڈ رویوں کی حقیقی دنیا میں کامیاب تنصیب حاصل کر سکتے ہیں۔

## اگلے اقدامات

پارٹ V میں، ہم مخصوص ہیومنوائڈ روبوٹ کی ترقی کے موضوعات میں گہرائی سے جائیں گے جن میں کنیمیٹکس، لوکوموشن، مینیپولیشن، اور انسان-روبوٹ تعامل شامل ہیں، یہاں قائم کردہ ٹرانسفر لرننگ بنیاد پر تعمیر کرتے ہوئے۔