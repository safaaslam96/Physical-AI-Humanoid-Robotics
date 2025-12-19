---
title: "باب 13: ہیومنوائڈ روبوٹ کنیمیٹکس اور ڈائینمکس"
sidebar_label: "باب 13: کنیمیٹکس اور ڈائینمکس"
---

# باب 13: ہیومنوائڈ روبوٹ کنیمیٹکس اور ڈائینمکس

## سیکھنے کے اہداف
- ہیومنوائڈ روبوٹ کنیمیٹکس کی ریاضی کی بنیاد سمجھنا
- ملٹی-لنک ہیومنوائڈ سسٹمز کے لیے فارورڈ اور انورس کنیمیٹکس میں مہارت حاصل کرنا
- بائی پیڈل حرکت اور توازن کے لیے ڈائینمک ماڈلنگ کا تجزیہ کرنا
- کنیمیٹک اور ڈائینمک اصولوں کی بنیاد پر کنٹرول سسٹمز نافذ کرنا

## تعارف

ہیومنوائڈ روبوٹ کنیمیٹکس اور ڈائینمکس انسان نما روبوٹک سسٹمز کی پیچیدہ حرکات کو سمجھنے اور کنٹرول کرنے کے لیے ریاضی کی بنیاد فراہم کرتے ہیں۔ سادہ روبوٹس کے برعکس، ہیومنوائڈ روبوٹس کو بائی پیڈل لوکوموشن، متعدد ڈگریوں کی آزادی والے مینیپولیشن، اور توازن برقرار رکھنے کے چیلنجز کا سامنا کرنا پڑتا ہے۔ یہ باب ہیومنوائڈ روبوٹ موشن کو م govern کرنے والے ریاضی کے اصولوں کو تلاش کرتا ہے، بنیادی کنیمیٹک ریلیشن شپس سے لے کر مستحکم بائی پیڈل حرکت کے لیے ضروری پیچیدہ ڈائینمک ماڈلز تک۔

## ہیومنوائڈ روبوٹکس کے لیے ریاضی کی بنیادیں

### کوآرڈینیٹ سسٹمز اور ٹرانسفارمیشنز

ہیومنوائڈ روبوٹس کو ان کی پیچیدہ سٹرکچر کی وضاحت کے لیے متعدد کوآرڈینیٹ سسٹمز کی ضرورت ہوتی ہے:

```python
# ہیومنوائڈ روبوٹس کے لیے کوآرڈینیٹ سسٹم ٹرانسفارمیشنز
import numpy as np
from scipy.spatial.transform import Rotation as R

class CoordinateSystem:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation = np.eye(3)  # 3x3 ریوٹیشن میٹرکس
        self.transform_matrix = np.eye(4)  # 4x4 ہومو جینس ٹرانسفارمیشن

    def set_pose(self, position, orientation):
        """پوزیشن اور اورینٹیشن سیٹ کریں (ریوٹیشن میٹرکس)"""
        self.position = np.array(position)
        self.orientation = np.array(orientation)

        # ہومو جینس ٹرانسفارمیشن میٹرکس اپ ڈیٹ کریں
        self.transform_matrix[:3, :3] = self.orientation
        self.transform_matrix[:3, 3] = self.position

    def get_transform_to_parent(self):
        """پیرنٹ کوآرڈینیٹ سسٹم کے لیے ٹرانسفارمیشن میٹرکس حاصل کریں"""
        return self.transform_matrix

    def transform_point(self, point):
        """اس کوآرڈینیٹ سسٹم سے پیرنٹ میں پوائنٹ ٹرانسفارم کریں"""
        point_homogeneous = np.append(point, 1)
        transformed = self.transform_matrix @ point_homogeneous
        return transformed[:3]

class HumanoidKinematicChain:
    def __init__(self):
        self.links = []
        self.joints = []
        self.coordinate_frames = {}

    def add_link(self, name, length, joint_type='revolute'):
        """کنیمیٹک چین میں ایک لنک شامل کریں"""
        link = {
            'name': name,
            'length': length,
            'joint_type': joint_type,
            'dof': 1 if joint_type == 'revolute' else 3
        }
        self.links.append(link)

    def create_dh_parameters(self):
        """روبوٹ کے لیے ڈیناویٹ-ہارٹنبرگ پیرامیٹرز تخلیق کریں"""
        # DH پیرامیٹرز: [theta, d, a, alpha]
        # theta: جوائنٹ اینگل
        # d: لنک آف سیٹ
        # a: لنک لمبائی
        # alpha: لنک ٹوسٹ
        dh_params = []

        for i, link in enumerate(self.links):
            # ایک سادہ سیریل چین کے لیے ڈیفالٹ DH پیرامیٹرز
            dh_param = {
                'theta': 0,  # جوائنٹ اینگل (متغیر)
                'd': 0,      # لنک آف سیٹ (revolute کے لیے مستقل)
                'a': link['length'],  # لنک لمبائی
                'alpha': 0   # لنک ٹوسٹ
            }
            dh_params.append(dh_param)

        return dh_params
```

### ویکٹر اور میٹرکس ریاضی

ہیومنوائڈ روبوٹکس لکیری الجبرا پر بہت زیادہ انحصار کرتے ہیں:

```python
# ہیومنوائڈ روبوٹکس کے لیے ریاضی کے یوٹیلیٹیز
class MathUtils:
    @staticmethod
    def skew_symmetric(vector):
        """3D ویکٹر سے سکیو-سمیٹرک میٹرکس تخلیق کریں"""
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    @staticmethod
    def rotation_matrix_from_axis_angle(axis, angle):
        """ایکسیس-اینگل ریپریزنٹیشن سے ریوٹیشن میٹرکس تخلیق کریں"""
        axis = axis / np.linalg.norm(axis)  # ایکسیس نارملائز کریں
        kx, ky, kz = axis
        c = np.cos(angle)
        s = np.sin(angle)
        v = 1 - c

        return np.array([
            [kx*kx*v + c, kx*ky*v - kz*s, kx*kz*v + ky*s],
            [kx*ky*v + kz*s, ky*ky*v + c, ky*kz*v - kx*s],
            [kx*kz*v - ky*s, ky*kz*v + kx*s, kz*kz*v + c]
        ])

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """کووٹرینشن کو ریوٹیشن میٹرکس میں تبدیل کریں"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    @staticmethod
    def homogeneous_transform(rotation, translation):
        """4x4 ہومو جینس ٹرانسفارمیشن میٹرکس تخلیق کریں"""
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
```

## فارورڈ کنیمیٹکس

### فارورڈ کنیمیٹکس کو سمجھنا

فارورڈ کنیمیٹکس دی گئی جوائنٹ اینگلز سے اینڈ ایفیکٹر کی پوزیشن اور اورینٹیشن کا حساب لگاتا ہے۔ ہیومنوائڈ روبوٹس کے لیے، اس میں متعدد کنیمیٹک چینز شامل ہیں (بازو، ٹانگ، ٹورسو، سر)۔

```python
# فارورڈ کنیمیٹکس امپلیمنٹیشن
class ForwardKinematics:
    def __init__(self, dh_parameters):
        self.dh_params = dh_parameters  # DH پیرامیٹر ڈیکشنریز کی فہرست

    def dh_transform(self, theta, d, a, alpha):
        """ڈیناویٹ-ہارٹنبرگ ٹرانسفارمیشن میٹرکس کا حساب لگائیں"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def calculate_forward_kinematics(self, joint_angles):
        """پوری چین کے لیے فارورڈ کنیمیٹکس کا حساب لگائیں"""
        if len(joint_angles) != len(self.dh_params):
            raise ValueError("جوائنٹ اینگلز کی تعداد DH پیرامیٹرز سے مماثل ہونی چاہیے")

        # شناخت ٹرانسفارمیشن کے ساتھ شروع کریں
        cumulative_transform = np.eye(4)
        transforms = [cumulative_transform.copy()]  # تمام ٹرانسفارمیشنز اسٹور کریں

        for i, angle in enumerate(joint_angles):
            # اس جوائنٹ کے لیے DH پیرامیٹرز حاصل کریں
            dh = self.dh_params[i]
            theta = dh['theta'] + angle  # جوائنٹ اینگل کو آف سیٹ میں شامل کریں
            d = dh['d']
            a = dh['a']
            alpha = dh['alpha']

            # انفرادی جوائنٹ ٹرانسفارمیشن کا حساب لگائیں
            joint_transform = self.dh_transform(theta, d, a, alpha)

            # ٹرانسفارمیشن کو جمع کریں
            cumulative_transform = cumulative_transform @ joint_transform
            transforms.append(cumulative_transform.copy())

        # اینڈ ایفیکٹر پوزیشن اور اورینٹیشن نکالیں
        end_effector_pos = cumulative_transform[:3, 3]
        end_effector_rot = cumulative_transform[:3, :3]

        return {
            'position': end_effector_pos,
            'orientation': end_effector_rot,
            'all_transforms': transforms
        }

    def calculate_link_positions(self, joint_angles):
        """چین میں تمام لنکس کی پوزیشنز کا حساب لگائیں"""
        result = self.calculate_forward_kinematics(joint_angles)
        transforms = result['all_transforms']

        link_positions = []
        for transform in transforms:
            pos = transform[:3, 3]
            link_positions.append(pos)

        return link_positions
```

### ہیومنوائڈ-مخصوص فارورڈ کنیمیٹکس

ہیومنوائڈ روبوٹس کے پاس متعدد کنیمیٹک چینز ہیں جنہیں ایک ساتھ سمجھا جانا چاہیے:

```python
# ہیومنوائڈ-مخصوص فارورڈ کنیمیٹکس
class HumanoidFK:
    def __init__(self):
        # مختلف جسم کے حصوں کے لیے کنیمیٹک چینز کی وضاحت کریں
        self.chains = {
            'left_arm': self.create_arm_dh_params('left'),
            'right_arm': self.create_arm_dh_params('right'),
            'left_leg': self.create_leg_dh_params('left'),
            'right_leg': self.create_leg_dh_params('right'),
            'torso': self.create_torso_dh_params()
        }

        # ہر چین کے لیے FK کیلکولیٹرز
        self.fk_calculators = {}
        for chain_name, dh_params in self.chains.items():
            self.fk_calculators[chain_name] = ForwardKinematics(dh_params)

    def create_arm_dh_params(self, side):
        """ہیومنوائڈ بازو کے لیے DH پیرامیٹرز تخلیق کریں"""
        # سادہ 7-DOF بازو ماڈل
        sign = -1 if side == 'left' else 1

        dh_params = [
            {'theta': 0, 'd': 0.1, 'a': 0, 'alpha': -np.pi/2},  # شانہ جوائنٹ
            {'theta': 0, 'd': 0, 'a': 0.2, 'alpha': 0},         # شانہ فلیکشن
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': -np.pi/2},    # شانہ ریوٹیشن
            {'theta': 0, 'd': 0.3, 'a': 0, 'alpha': -np.pi/2},  # کہنی جوائنٹ
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': np.pi/2},     # کہنی ریوٹیشن
            {'theta': 0, 'd': 0.25, 'a': 0, 'alpha': -np.pi/2}, # کلائی جوائنٹ
            {'theta': 0, 'd': 0.05, 'a': 0, 'alpha': 0}         # کلائی ریوٹیشن
        ]
        return dh_params

    def create_leg_dh_params(self, side):
        """ہیومنوائڈ ٹانگ کے لیے DH پیرامیٹرز تخلیق کریں"""
        # سادہ 6-DOF ٹانگ ماڈل
        sign = -1 if side == 'left' else 1

        dh_params = [
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': -np.pi/2},    # ہپ ایبڈکشن
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': np.pi/2},     # ہپ فلیکشن
            {'theta': 0, 'd': -0.4, 'a': 0, 'alpha': 0},        # ہپ ریوٹیشن
            {'theta': 0, 'd': -0.4, 'a': 0, 'alpha': 0},        # گھٹنا جوائنٹ
            {'theta': 0, 'd': 0, 'a': 0, 'alpha': 0},           # انکل پچ
            {'theta': 0, 'd': -0.05, 'a': 0, 'alpha': 0}        # انکل رول
        ]
        return dh_params

    def calculate_humanoid_pose(self, joint_angles):
        """جوائنٹ اینگلز سے مکمل ہیومنوائڈ پوز کا حساب لگائیں"""
        poses = {}

        for chain_name, fk_calc in self.fk_calculators.items():
            # اس چین کے لیے متعلقہ جوائنٹ اینگلز نکالیں
            chain_angles = self.extract_chain_angles(chain_name, joint_angles)

            # فارورڈ کنیمیٹکس کا حساب لگائیں
            pose = fk_calc.calculate_forward_kinematics(chain_angles)
            poses[chain_name] = pose

        return poses

    def extract_chain_angles(self, chain_name, all_angles):
        """مخصوص چین کے لیے جوائنٹ اینگلز نکالیں"""
        # یہ مکمل جوائنٹ اینگل ویکٹر سے چین-مخصوص اینگلز میں میپ کرے گا
        # امپلیمنٹیشن روبوٹ میں جوائنٹ آرڈرنگ پر منحصر ہے
        chain_map = {
            'left_arm': slice(0, 7),      # جوائنٹس 0-6
            'right_arm': slice(7, 14),    # جوائنٹس 7-13
            'left_leg': slice(14, 20),    # جوائنٹس 14-19
            'right_leg': slice(20, 26),   # جوائنٹس 20-25
            'torso': slice(26, 29)        # جوائنٹس 26-28
        }

        return all_angles[chain_map[chain_name]]
```

## انورس کنیمیٹکس

### انورس کنیمیٹکس کو سمجھنا

انورس کنیمیٹکس (IK) مطلوبہ اینڈ ایفیکٹر پوزیشن اور اورینٹیشن حاصل کرنے کے لیے ضروری جوائنٹ اینگلز کا حساب لگاتا ہے۔ یہ ہیومنوائڈ روبوٹس کے لیے اہم ہے تاکہ وہ ریچنگ، چلنا، اور مینیپولیشن جیسے کام انجام دے سکیں۔

```python
# انورس کنیمیٹکس امپلیمنٹیشن
class InverseKinematics:
    def __init__(self, dh_parameters):
        self.dh_params = dh_parameters
        self.forward_kin = ForwardKinematics(dh_parameters)

    def jacobian(self, joint_angles):
        """جیومیٹرک جیکوبین میٹرکس کا حساب لگائیں"""
        # FK کا استعمال کرتے ہوئے موجودہ اینڈ ایفیکٹر پوزیشن کا حساب لگائیں
        fk_result = self.forward_kin.calculate_forward_kinematics(joint_angles)
        current_pos = fk_result['position']

        # جیکوبین کالمز کا حساب لگائیں
        jacobian = np.zeros((6, len(joint_angles)))  # 6 DOF (پوزیشن + اورینٹیشن)

        # تمام لنک پوزیشنز حاصل کریں
        link_positions = self.forward_kin.calculate_link_positions(joint_angles)

        for i in range(len(joint_angles)):
            # جوائنٹ i کے لیے ریوٹیشن ایکسز کا حساب لگائیں
            # ریوولوٹ جوائنٹس کے لیے: ایکسز جوائنٹ فریم کے z پر ہے
            transform_to_joint = self.forward_kin.calculate_forward_kinematics(
                joint_angles[:i+1]
            )['all_transforms'][-1] if i > 0 else np.eye(4)

            # جوائنٹ ایکسز ورلڈ کوآرڈینیٹس میں
            z_axis = transform_to_joint[:3, 2]  # جوائنٹ فریم کا z-axis
            joint_pos = link_positions[i]

            # اینڈ ایفیکٹر تک پوزیشن ویکٹر
            r = current_pos - joint_pos

            # پوزیشن کے لیے جیکوبین کالم
            jacobian[:3, i] = np.cross(z_axis, r)

            # اورینٹیشن کے لیے جیکوبین کالم
            jacobian[3:, i] = z_axis

        return jacobian

    def inverse_kinematics_analytical(self, target_pose, initial_angles):
        """تجزیاتی IK حل (سادہ چینز کے لیے)"""
        # یہ سادہ مثال 2-DOF پلینر بازو کے لیے ہے
        # حقیقی ہیومنوائڈ IK کو عددی طریقے کی ضرورت ہے

        # ٹارگٹ پوزیشن نکالیں
        target_pos = target_pose[:3, 3]

        # 2-DOF بازو کے لیے: جوائنٹ اینگلز کا تجزیاتی حل
        # یہ صرف ایک مثال ہے - ہیومنوائڈ روبوٹس کو مزید پیچیدہ حل کی ضرورت ہے
        x, y = target_pos[0], target_pos[1]

        # بیس سے ٹارگٹ تک فاصلہ کا حساب لگائیں
        distance = np.sqrt(x**2 + y**2)

        # چیک کریں کہ کیا ٹارگٹ قابل رسائی ہے
        link_lengths = [self.dh_params[0]['a'], self.dh_params[1]['a']]
        max_reach = sum(link_lengths)

        if distance > max_reach:
            # ٹارگٹ قابل رسائی سے باہر ہے - قریب ترین ممکن حاصل کریں
            scale = max_reach / distance
            target_pos = np.array([x * scale, y * scale, target_pos[2]])
            distance = max_reach

        # 2-DOF بازو کا تجزیاتی حل
        # (یہ سادہ ہے اور پیچیدہ ہیومنوائڈ چینز کے لیے کام نہیں کرے گا)
        pass

    def inverse_kinematics_numerical(self, target_pose, initial_angles,
                                   max_iterations=100, tolerance=1e-6):
        """جیکوبین ٹرانسپوز/پسوڈو-انورس کا استعمال کرتے ہوئے عددی IK حل"""
        current_angles = np.array(initial_angles, dtype=float)

        for iteration in range(max_iterations):
            # موجودہ اینڈ ایفیکٹر پوز کا حساب لگائیں
            current_pose = self.forward_kin.calculate_forward_kinematics(
                current_angles
            )

            # غلطی کا حساب لگائیں
            pos_error = target_pose[:3, 3] - current_pose['position']
            rot_error = self.rotation_error(
                target_pose[:3, :3],
                current_pose['orientation']
            )

            # پوزیشن اور ریوٹیشن غلطیوں کو جوڑیں
            error = np.concatenate([pos_error, rot_error])

            # کنورجنس چیک کریں
            if np.linalg.norm(error) < tolerance:
                return current_angles, True  # کامیابی

            # جیکوبین کا حساب لگائیں
            jacobian = self.jacobian(current_angles)

            # استحکام کے لیے ڈیمپڈ لیسٹ سکوئیئرز (لیونبرگ-مارکویڈٹ) کا استعمال کریں
            damping = 0.01
            j_pinv = np.linalg.pinv(jacobian + damping * np.eye(jacobian.shape[0]))

            # جوائنٹ اینگل اپ ڈیٹ کا حساب لگائیں
            delta_theta = j_pinv @ error

            # جوائنٹ اینگلز اپ ڈیٹ کریں
            current_angles += delta_theta

        return current_angles, False  # کنورجنس میں ناکامی

    def rotation_error(self, target_rot, current_rot):
        """دو ریوٹیشن میٹرکسز کے درمیان ریوٹیشن غلطی کا حساب لگائیں"""
        # ریوٹیشن ویکٹر ریپریزنٹیشن کا استعمال کریں
        target_r = R.from_matrix(target_rot)
        current_r = R.from_matrix(current_rot)

        # ریلیٹو ریوٹیشن کا حساب لگائیں
        relative_r = target_r * current_r.inv()

        # ریوٹیشن ویکٹر حاصل کریں
        rot_vec = relative_r.as_rotvec()

        return rot_vec
```

### ہیومنوائڈ-مخصوص انورس کنیمیٹکس

ہیومنوائڈ روبوٹس کو ان کی پیچیدہ سٹرکچر کی وجہ سے خصوصی IK ایپروچز کی ضرورت ہوتی ہے:

```python
# ہیومنوائڈ-مخصوص انورس کنیمیٹکس
class HumanoidIK:
    def __init__(self):
        self.chains = HumanoidFK()
        self.ik_solvers = {}

        # ہر چین کے لیے IK سالورز کا آغاز کریں
        for chain_name, dh_params in self.chains.chains.items():
            self.ik_solvers[chain_name] = InverseKinematics(dh_params)

    def solve_arm_ik(self, arm_side, target_pose, current_angles):
        """مخصوص بازو کے لیے انورس کنیمیٹکس حل کریں"""
        chain_name = f"{arm_side}_arm"
        ik_solver = self.ik_solvers[chain_name]

        # اس چین کے لیے موجودہ اینگلز نکالیں
        chain_angles = self.chains.extract_chain_angles(chain_name, current_angles)

        # IK حل کریں
        solution, success = ik_solver.inverse_kinematics_numerical(
            target_pose, chain_angles
        )

        return solution, success

    def solve_leg_ik(self, leg_side, target_pose, current_angles):
        """مخصوص ٹانگ کے لیے انورس کنیمیٹکس حل کریں"""
        chain_name = f"{leg_side}_leg"
        ik_solver = self.ik_solvers[chain_name]

        # اس چین کے لیے موجودہ اینگلز نکالیں
        chain_angles = self.chains.extract_chain_angles(chain_name, current_angles)

        # IK حل کریں
        solution, success = ik_solver.inverse_kinematics_numerical(
            target_pose, chain_angles
        )

        return solution, success

    def whole_body_ik(self, targets, current_angles, weights=None):
        """متعدد ٹارگٹس کے ساتھ وہول-بodies انورس کنیمیٹکس حل کریں"""
        if weights is None:
            weights = {'left_arm': 1.0, 'right_arm': 1.0, 'left_leg': 1.0, 'right_leg': 1.0}

        # یہ ایک مزید پیچیدہ آپٹیمائزیشن ایپروچ نافذ کرے گا
        # تمام ٹارگٹس کو ایک ساتھ مدنظر رکھتے ہوئے
        # عام ایپروچز: ترجیح دی گئی IK، آپٹیمائزیشن-مبنی IK

        # سادہ ایپروچ: ہر چین کو الگ سے حل کریں مطابقت کے ساتھ
        result_angles = current_angles.copy()

        for chain_name, target_pose in targets.items():
            if chain_name in self.ik_solvers:
                ik_solver = self.ik_solvers[chain_name]

                # چین-مخصوص اینگلز نکالیں
                chain_slice = self.get_chain_slice(chain_name)
                chain_angles = current_angles[chain_slice]

                # اس چین کے لیے IK حل کریں
                new_chain_angles, success = ik_solver.inverse_kinematics_numerical(
                    target_pose, chain_angles
                )

                if success:
                    result_angles[chain_slice] = new_chain_angles

        return result_angles

    def get_chain_slice(self, chain_name):
        """چین-مخصوص اینگلز نکالنے کے لیے سلائس حاصل کریں"""
        chain_map = {
            'left_arm': slice(0, 7),
            'right_arm': slice(7, 14),
            'left_leg': slice(14, 20),
            'right_leg': slice(20, 26),
            'torso': slice(26, 29)
        }
        return chain_map[chain_name]

    def balance_aware_ik(self, targets, current_angles, com_target=None):
        """توازن برقرار رکھتے ہوئے IK حل کریں"""
        # یہ سینٹر آف ماس کے ا considerations کو ضم کرے گا
        # IK حل کے ساتھ تاکہ مستحکم توازن برقرار رہے

        # پہلے بنیادی IK حل کریں
        ik_solution = self.whole_body_ik(targets, current_angles)

        # پھر ضرورت پڑنے پر توازن کے لیے ایڈجسٹ کریں
        if com_target is not None:
            # موجودہ CoM کا حساب لگائیں
            current_com = self.calculate_center_of_mass(ik_solution)

            # CoM کو ٹارگٹ کی طرف لے جانے کے لیے حل ایڈجسٹ کریں
            adjusted_solution = self.adjust_for_balance(
                ik_solution, current_com, com_target
            )

            return adjusted_solution

        return ik_solution

    def calculate_center_of_mass(self, joint_angles):
        """جوائنٹ کنفیگریشن کے مطابق سینٹر آف ماس کا حساب لگائیں"""
        # یہ روبوٹ کی ماس خصوصیات اور موجودہ کنفیگریشن کا استعمال کرے گا
        # CoM پوزیشن کا حساب لگانے کے لیے
        pass

    def adjust_for_balance(self, current_config, current_com, target_com):
        """توازن کو بہتر بنانے کے لیے کنفیگریشن ایڈجسٹ کریں"""
        # یہ آپٹیمائزیشن کا استعمال کرے گا جوائنٹ اینگلز ایڈجسٹ کرنے کے لیے
        # جبکہ ٹاسک کی ضروریات برقرار رکھتے ہوئے اور توازن کو بہتر بناتے ہوئے
        pass
```

## ڈائینمک ماڈلنگ

### روبوٹ ڈائینمکس کو سمجھنا

روبوٹ ڈائینمکس روبوٹ پر کام کرنے والے فورسز اور نتیجے کے موشن کے درمیان تعلق کی وضاحت کرتا ہے۔ ہیومنوائڈ روبوٹس کے لیے، ڈائینمکس مستحکم بائی پیڈل لوکوموشن اور مینیپولیشن کے لیے اہم ہے۔

```python
# ہیومنوائڈ روبوٹس کے لیے ریجڈ باڈی ڈائینمکس
class RigidBodyDynamics:
    def __init__(self, mass, inertia_tensor):
        self.mass = mass
        self.inertia_tensor = np.array(inertia_tensor)  # 3x3 میٹرکس
        self.inertia_tensor_inv = np.linalg.inv(self.inertia_tensor)

    def rigid_body_equation(self, pose, twist, wrench):
        """وrench (فورس/ٹورک) سے ایکسلریشن کا حساب لگائیں"""
        # pose: [position, orientation]
        # twist: [linear_velocity, angular_velocity]
        # wrench: [force, torque]

        linear_force = wrench[:3]
        torque = wrench[3:]

        # لینیئر ایکسلریشن (F = ma)
        linear_accel = linear_force / self.mass

        # اینگولر ایکسلریشن (tau = I*alpha + omega x (I*omega))
        angular_velocity = twist[3:]
        inertia_omega = self.inertia_tensor @ angular_velocity
        angular_accel = (torque - np.cross(angular_velocity, inertia_omega)) @ self.inertia_tensor_inv

        acceleration = np.concatenate([linear_accel, angular_accel])

        return acceleration

class HumanoidDynamics:
    def __init__(self, robot_description):
        self.links = robot_description['links']
        self.joints = robot_description['joints']
        self.mass_properties = self.calculate_mass_properties()

    def calculate_mass_properties(self):
        """ہر لنک کے لیے ماس خصوصیات کا حساب لگائیں"""
        mass_props = {}

        for link_name, link_data in self.links.items():
            mass_props[link_name] = {
                'mass': link_data.get('mass', 1.0),
                'com': np.array(link_data.get('center_of_mass', [0, 0, 0])),
                'inertia': np.array(link_data.get('inertia', np.eye(3)))
            }

        return mass_props

    def euler_lagrange_dynamics(self, joint_positions, joint_velocities, joint_torques):
        """Euler-Lagrange فارمولیشن کا استعمال کرتے ہوئے جوائنٹ ایکسلریشنز کا حساب لگائیں"""
        # M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau
        # جہاں:
        # M(q) = ماس میٹرکس
        # C(q, q_dot) = کوریولس اور سینٹریفوگل فورسز
        # g(q) = گریویٹی فورسز
        # tau = جوائنٹ ٹورکس

        M = self.mass_matrix(joint_positions)
        C = self.coriolis_matrix(joint_positions, joint_velocities)
        g = self.gravity_vector(joint_positions)

        # حل: M * q_ddot = tau - C * q_dot - g
        q_ddot = np.linalg.solve(M, joint_torques - C @ joint_velocities - g)

        return q_ddot

    def mass_matrix(self, joint_positions):
        """ماس میٹرکس M(q) کا حساب لگائیں"""
        # یہ کمپوسیٹ ریجڈ باڈی الگورتھم یا دیگر طریقے استعمال کرے گا
        # ماس میٹرکس کا حساب لگانے کے لیے
        n = len(joint_positions)
        M = np.zeros((n, n))

        # سادہ حساب - عمل میں یہ میں
        # پیچیدہ ریکریسیو الگورتھم شامل ہوتے ہیں
        for i in range(n):
            for j in range(n):
                M[i, j] = self.calculate_inertial_coupling(i, j, joint_positions)

        return M

    def coriolis_matrix(self, joint_positions, joint_velocities):
        """کوریولس اور سینٹریفوگل میٹرکس C(q, q_dot) کا حساب لگائیں"""
        # یہ میٹرکس ولسٹی-متعلقہ فورسز کا احاطہ کرتا ہے
        n = len(joint_positions)
        C = np.zeros((n, n))

        # کرسٹوفل علامات کا حساب لگائیں اور C میٹرکس تشکیل دیں
        # یہ ماس میٹرکس کے جزوی مشتق کے متعلق پیچیدہ حساب ہے
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    christoffel = self.christoffel_symbol(i, j, k, joint_positions)
                    C[i, j] += christoffel * joint_velocities[k]

        return C

    def gravity_vector(self, joint_positions):
        """گریویٹی ویکٹر g(q) کا حساب لگائیں"""
        # جوائنٹ اسپیس میں گریویٹی اثرات کا حساب لگائیں
        n = len(joint_positions)
        g = np.zeros(n)

        # یہ گریویٹیشنل فورسز کو ٹرانسفارم کرے گا
        # کنیمیٹک چین کے ذریعے
        return g

    def christoffel_symbol(self, i, j, k, q):
        """ڈائینمک ایکویشنز کے لیے کرسٹوفل علامت کا حساب لگائیں"""
        # کرسٹوفل علامات ماس میٹرکس کے جزوی مشتق سے حساب لگائی جاتی ہیں
        # Gamma^i_jk = 0.5 * (dM_ik/dq_j + dM_jk/dq_i - dM_ij/dq_k)

        # یہ ایک سادہ پلیس ہولڈر ہے
        return 0.0

    def calculate_inertial_coupling(self, i, j, q):
        """جوائنٹس i اور j کے درمیان اینرشل کوپلنگ کا حساب لگائیں"""
        # یہ روبوٹ کنیمیٹکس اور ماس تقسیم کی بنیاد پر
        # پیچیدہ حسابات میں شامل ہوگا
        return 0.0 if i != j else 1.0  # سادہ
```

### بائی پیڈل ڈائینمکس اور توازن

ہیومنوائڈ روبوٹس کے پاس بائی پیڈل لوکوموشن کے لیے منفرد ڈائینمک امور ہیں:

```python
# بائی پیڈل ڈائینمکس اور توازن کنٹرول
class BipedalDynamics:
    def __init__(self):
        self.gravity = 9.81
        self.total_mass = 70.0  # kg، تقریبی ہیومنوائڈ ماس
        self.com_height = 0.8  # m، تقریبی CoM اونچائی
        self.step_length = 0.6  # m، عام قدم کی لمبائی

    def linear_inverted_pendulum_model(self, com_position, com_velocity, zmp_position):
        """توازن کے لیے لکیری انورٹڈ پینڈولم ماڈل"""
        # LIPM: x_ddot = omega^2 * (x - x_zmp)
        # جہاں omega^2 = g / h (h = CoM اونچائی)

        omega_sq = self.gravity / self.com_height

        # توازن کے لیے ضروری CoM ایکسلریشن کا حساب لگائیں
        com_acceleration = omega_sq * (com_position - zmp_position)

        return com_acceleration

    def zero_moment_point(self, com_position, com_velocity, com_acceleration):
        """توازن کے لیے Zero Moment Point (ZMP) کا حساب لگائیں"""
        # ZMP = CoM position - g/CoM_ddot * (CoM_height - foot_height)
        # 2D کیس کے لیے سادہ

        zmp_x = com_position[0] - (self.gravity / com_acceleration[0]) * (
            self.com_height - 0  # فرض کریں کہ پاؤں زمین کی سطح پر ہے
        )

        zmp_y = com_position[1] - (self.gravity / com_acceleration[1]) * (
            self.com_height - 0
        )

        return np.array([zmp_x, zmp_y, 0])

    def capture_point(self, com_position, com_velocity):
        """توازن کی بازیابی کے لیے کیپچر پوائنٹ کا حساب لگائیں"""
        # کیپچر پوائنٹ = CoM position + CoM velocity / omega
        # جہاں omega = sqrt(g / CoM_height)

        omega = np.sqrt(self.gravity / self.com_height)

        capture_point = com_position + com_velocity / omega

        return capture_point

    def balance_controller(self, current_com, desired_com, current_zmp, support_polygon):
        """ZMP اور کیپچر پوائنٹ تصورات کا استعمال کرتے ہوئے توازن کنٹرولر"""
        # غلطی کا حساب لگائیں
        com_error = desired_com - current_com
        zmp_error = self.calculate_zmp_error(current_zmp, support_polygon)

        # PID کنٹرول توازن کے لیے
        kp = 10.0  # م_PROPOR
        ki = 1.0   # INTEGRAL گین
        kd = 5.0   # DERIVATIVE گین

        # کنٹرول آؤٹ پٹ کا حساب لگائیں
        control_output = kp * com_error + kd * zmp_error

        return control_output

    def calculate_zmp_error(self, current_zmp, support_polygon):
        """سپورٹ پولی گان کے مقابلے میں ZMP غلطی کا حساب لگائیں"""
        # چیک کریں کہ کیا ZMP سپورٹ پولی گان کے اندر ہے
        if self.is_zmp_in_support_polygon(current_zmp, support_polygon):
            return np.zeros(2)  # سپورٹ میں ہونے پر کوئی غلطی نہیں

        # سپورٹ پولی گان میں قریب ترین پوائنٹ کا حساب لگائیں
        closest_point = self.closest_point_in_polygon(current_zmp, support_polygon)
        zmp_error = closest_point - current_zmp[:2]

        return zmp_error

    def is_zmp_in_support_polygon(self, zmp, polygon):
        """چیک کریں کہ کیا ZMP سپورٹ پولی گان کے اندر ہے"""
        # رے کاسٹنگ الگورتھم یا دیگر پوائنٹ-ان-پولی گان ٹیسٹ کا استعمال کریں
        return self.point_in_polygon(zmp[:2], polygon)

    def point_in_polygon(self, point, polygon):
        """رے کاسٹنگ کا استعمال کرتے ہوئے چیک کریں کہ کیا پوائنٹ پولی گان کے اندر ہے"""
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

    def closest_point_in_polygon(self, point, polygon):
        """دی گئی ZMP پوزیشن کے لیے پولی گان کی حد کا قریب ترین پوائنٹ تلاش کریں"""
        # یہ دیے گئے ZMP پوزیشن کے لیے پولی گان کی حد پر قریب ترین پوائنٹ تلاش کرے گا
        pass
```

## کنٹرول سسٹم امپلیمنٹیشن

### ہیومنوائڈ روبوٹس کے لیے PD کنٹرول

```python
# ہیومنوائڈ روبوٹس کے لیے کنٹرول سسٹمز
class HumanoidController:
    def __init__(self):
        self.joint_controllers = {}
        self.balance_controller = BipedalDynamics()
        self.trajectory_generators = {}

    def setup_joint_controller(self, joint_name, kp=100, ki=0, kd=10):
        """مخصوص جوائنٹ کے لیے PD کنٹرولر سیٹ اپ کریں"""
        self.joint_controllers[joint_name] = {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'prev_error': 0,
            'integral_error': 0
        }

    def compute_joint_control(self, joint_name, desired_position, current_position,
                            desired_velocity=0, current_velocity=0, dt=0.001):
        """سنگل جوائنٹ کے لیے کنٹرول آؤٹ پٹ کا حساب لگائیں"""
        controller = self.joint_controllers[joint_name]

        # غلطیاں کا حساب لگائیں
        position_error = desired_position - current_position
        velocity_error = desired_velocity - current_velocity

        # انٹیگرل اپ ڈیٹ کریں (اینٹی-وائیڈ اَپ کے ساتھ)
        controller['integral_error'] += position_error * dt
        max_integral = 10.0  # اینٹی-وائیڈ اَپ حد
        controller['integral_error'] = np.clip(
            controller['integral_error'], -max_integral, max_integral
        )

        # ڈیریویٹیوز کا حساب لگائیں
        derivative_error = (position_error - controller['prev_error']) / dt
        controller['prev_error'] = position_error

        # کنٹرول آؤٹ پٹ کا حساب لگائیں
        p_term = controller['kp'] * position_error
        i_term = controller['ki'] * controller['integral_error']
        d_term = controller['kd'] * (velocity_error + derivative_error)

        control_output = p_term + i_term + d_term

        return control_output

    def operational_space_control(self, task_jacobian, task_desired, task_current):
        """ٹاسک-اسپیس کنٹرول کے لیے آپریشنل اسپیس کنٹرول"""
        # ٹاسک غلطی کا حساب لگائیں
        task_error = task_desired - task_current

        # جیکوبین پسوڈو-انورس کا استعمال کرتے ہوئے جوائنٹ ویلوسٹیز کا حساب لگائیں
        # q_dot = J# * x_dot + (I - J# * J) * q_dot_null
        # جہاں J# جیکوبین کا پسوڈو-انورس ہے

        j_pinv = np.linalg.pinv(task_jacobian)
        joint_velocities = j_pinv @ task_error

        return joint_velocities

    def inverse_dynamics_control(self, joint_positions, joint_velocities,
                               joint_accelerations, gravity_compensation=True):
        """انورس ڈائینمکس کا استعمال کرتے ہوئے ضروری جوائنٹ ٹورکس کا حساب لگائیں"""
        # ریکریسیو نیوٹن-ایولر الگورتھم یا دیگر انورس ڈائینمکس طریقے استعمال کریں
        # ضروری جوائنٹ ٹورکس کا حساب لگانے کے لیے

        # یہ مکمل انورس ڈائینمکس ایکویشن نافذ کرے گا:
        # tau = M(q) * q_ddot + C(q, q_dot) * q_dot + g(q)

        # سادہ ورژن
        dynamics_model = HumanoidDynamics({})  # حقیقی روبوٹ کی وضاحت کا استعمال کرے گا
        coriolis_centrifugal = dynamics_model.coriolis_matrix(
            joint_positions, joint_velocities
        ) @ joint_velocities
        gravity_effects = dynamics_model.gravity_vector(joint_positions)
        inertial_effects = dynamics_model.mass_matrix(joint_positions) @ joint_accelerations

        required_torques = inertial_effects + coriolis_centrifugal + gravity_effects

        return required_torques
```

## عملی امپلیمنٹیشن مثالیں

### چلنے کا پیٹرن جنریشن

```python
# کنیمیٹکس اور ڈائینمکس کا استعمال کرتے ہوئے چلنے کا پیٹرن جنریٹ کرنا
class WalkingPatternGenerator:
    def __init__(self):
        self.step_height = 0.05  # m
        self.step_length = 0.6   # m
        self.step_duration = 1.0 # s
        self.zmp_reference = np.array([0.0, 0.0])  # مطلوبہ ZMP

    def generate_walk_trajectory(self, num_steps, step_width=0.2):
        """مکمل چلنے کا ٹریجکٹری تخلیق کریں"""
        trajectory = []

        for step in range(num_steps):
            # ڈبل سپورٹ فیز تخلیق کریں
            dsp_trajectory = self.generate_double_support_phase()

            # سنگل سپورٹ فیز تخلیق کریں (بائیں پاؤں کی سپورٹ)
            if step % 2 == 0:  # بائیں پاؤں کی سپورٹ جفت اسٹیپس کے لیے
                ssp_trajectory = self.generate_single_support_phase(
                    'left', step_width
                )
            else:  # جفت اسٹیپس کے لیے دائیں پاؤں کی سپورٹ
                ssp_trajectory = self.generate_single_support_phase(
                    'right', step_width
                )

            trajectory.extend(dsp_trajectory)
            trajectory.extend(ssp_trajectory)

        return trajectory

    def generate_double_support_phase(self, duration=0.1):
        """ڈبل سپورٹ فیز ٹریجکٹری تخلیق کریں"""
        # دونوں پاؤں زمین پر - وزن منتقل کریں
        steps = int(duration / 0.01)  # فرض کریں 100Hz کنٹرول
        phase_trajectory = []

        for i in range(steps):
            t = i / steps  # نارملائزڈ ٹائم (0 سے 1)

            # دوسرے پاؤں میں ہموار وزن ٹرانسفر
            # یہ CoM ٹریجکٹری، ZMP ٹریجکٹری، وغیرہ تخلیق کرے گا
            phase_trajectory.append({
                'time': t,
                'com_position': self.calculate_com_trajectory(t),
                'zmp_position': self.calculate_zmp_trajectory(t),
                'joint_angles': self.calculate_joint_trajectory(t)
            })

        return phase_trajectory

    def generate_single_support_phase(self, support_foot, step_width):
        """سنگل سپورٹ فیز ٹریجکٹری تخلیق کریں"""
        steps = int(self.step_duration / 0.01)
        phase_trajectory = []

        for i in range(steps):
            t = i / steps  # نارملائزڈ ٹائم

            # CoM ٹریجکٹری کو انورٹڈ پینڈولم ماڈل کے مطابق کیلکولیٹ کریں
            com_x = self.calculate_swing_foot_position(t, support_foot)
            com_y = self.balance_lateral_motion(t, support_foot, step_width)
            com_z = self.step_height_profile(t)  # CoM اونچائی برقرار رکھیں

            phase_trajectory.append({
                'time': t,
                'com_position': np.array([com_x, com_y, com_z]),
                'support_foot': support_foot,
                'swing_foot': 'right' if support_foot == 'left' else 'left'
            })

        return phase_trajectory

    def calculate_swing_foot_position(self, t, support_foot):
        """چلتے ہوئے سوئنگ فُٹ کی پوزیشن کا حساب لگائیں"""
        # ٹائم t پر سوئنگ فُٹ کہاں ہونا چاہیے اس کا حساب لگائیں
        if support_foot == 'left':
            # دائیں فُٹ سامنے بڑھ رہا ہے
            swing_pos = self.step_length * t  # لکیری ترقی
        else:
            # بائیں فُٹ سامنے بڑھ رہا ہے
            swing_pos = self.step_length * t

        return swing_pos

    def balance_lateral_motion(self, t, support_foot, step_width):
        """توازن کے لیے لیٹرل CoM موشن کا حساب لگائیں"""
        # CoM کو سپورٹ فُٹ کے اوپر شفٹ کریں
        if support_foot == 'left':
            target_y = step_width / 2  # CoM بائیں فُٹ کے اوپر
        else:
            target_y = -step_width / 2  # CoM دائیں فُٹ کے اوپر

        # sine فنکشن کا استعمال کرتے ہوئے ہموار ٹرانزیشن
        smooth_factor = np.sin(t * np.pi)  # 0 سے 1 سے 0
        return target_y * smooth_factor

    def step_height_profile(self, t):
        """سوئنگ فُٹ کے لیے قدم کی اونچائی کا پروفائل"""
        # پیرابولک قدم ٹریجکٹری تخلیق کریں
        height_factor = 4 * t * (1 - t)  # پیرابولک: 0->1->0
        return self.com_height + self.step_height * height_factor
```

## سیمولیشن اور توثیق

### ڈائینمکس سیمولیشن

```python
# ہیومنوائڈ روبوٹس کے لیے ڈائینمکس سیمولیشن
class HumanoidDynamicsSimulator:
    def __init__(self, robot_description):
        self.dynamics_model = HumanoidDynamics(robot_description)
        self.integration_dt = 0.001  # 1ms انٹیگریشن سٹیپ

    def simulate_step(self, current_state, joint_torques):
        """روبوٹ ڈائینمکس کا ایک ٹائم سٹیپ سیمولیٹ کریں"""
        # اسٹیٹ ویریبلز نکالیں
        joint_positions = current_state['joint_positions']
        joint_velocities = current_state['joint_velocities']

        # انورس ڈائینمکس کا استعمال کرتے ہوئے جوائنٹ ایکسلریشنز کا حساب لگائیں
        joint_accelerations = self.dynamics_model.euler_lagrange_dynamics(
            joint_positions, joint_velocities, joint_torques
        )

        # نئی ویلوسٹیز اور پوزیشنز حاصل کرنے کے لیے انٹیگریٹ کریں
        new_velocities = joint_velocities + joint_accelerations * self.integration_dt
        new_positions = joint_positions + new_velocities * self.integration_dt

        # اینڈ ایفیکٹر پوزیشنز کے لیے فارورڈ کنیمیٹکس کا حساب لگائیں
        fk_calculator = HumanoidFK()
        end_effector_poses = fk_calculator.calculate_humanoid_pose(new_positions)

        # سینٹر آف ماس کا حساب لگائیں
        com_position = self.calculate_com_position(new_positions)

        # ZMP کا حساب لگائیں
        zmp_position = self.calculate_zmp(com_position, new_positions, new_velocities)

        new_state = {
            'joint_positions': new_positions,
            'joint_velocities': new_velocities,
            'end_effector_poses': end_effector_poses,
            'com_position': com_position,
            'zmp_position': zmp_position
        }

        return new_state

    def calculate_com_position(self, joint_positions):
        """سینٹر آف ماس کی پوزیشن کا حساب لگائیں"""
        # یہ روبوٹ کی ماس خصوصیات اور کنیمیٹکس کا استعمال کرے گا
        # مجموعی سینٹر آف ماس کا حساب لگانے کے لیے
        pass

    def calculate_zmp(self, com_position, joint_positions, joint_velocities):
        """Zero Moment Point کا حساب لگائیں"""
        # ZMP کو حساب لگانے کے لیے ڈائینمک ایکویشنز استعمال کریں
        # ZMP = CoM - (g / CoM_z_ddot) * (CoM - foot_position)
        pass

    def validate_stability(self, state_trajectory):
        """موشن ٹریجکٹری کی استحکام کی توثیق کریں"""
        stability_metrics = {
            'zmp_in_support': [],
            'com_bounded': [],
            'energy_consumption': []
        }

        for state in state_trajectory:
            # چیک کریں کہ کیا ZMP سپورٹ پولی گان میں رہتا ہے
            zmp_in_support = self.is_zmp_stable(state['zmp_position'])
            stability_metrics['zmp_in_support'].append(zmp_in_support)

            # چیک کریں کہ کیا CoM باونڈڈ رہتا ہے
            com_bounded = self.is_com_stable(state['com_position'])
            stability_metrics['com_bounded'].append(com_bounded)

            # توانائی کی کھپت کا حساب لگائیں
            energy = self.calculate_energy(state)
            stability_metrics['energy_consumption'].append(energy)

        return stability_metrics

    def is_zmp_stable(self, zmp_position):
        """چیک کریں کہ کیا ZMP مستحکم علاقے میں ہے"""
        # یہ سپورٹ پولی گان کے مقابلے میں چیک کرے گا
        return True  # سادہ

    def is_com_stable(self, com_position):
        """چیک کریں کہ کیا CoM مستحکم حدود میں ہے"""
        # چیک کریں کہ کیا CoM مناسب حدود میں ہے
        return True  # سادہ

    def calculate_energy(self, state):
        """توانائی کی کھپت کا حساب لگائیں"""
        # کائنیٹک اور پوٹینشل توانائی کا حساب لگائیں
        # یہ جوائنٹ ویلوسٹیز اور پوزیشنز میں شامل ہوگا
        pass
```

## ہاتھوں سے مشق: ہیومنوائڈ کنیمیٹکس اور ڈائینمکس کا امپلیمنٹ کرنا

### مشق کے اہداف

- سادہ ہیومنوائڈ ماڈل کے لیے فارورڈ اور انورس کنیمیٹکس نافذ کریں
- ڈائینمک ماڈل تخلیق کریں اور بنیادی حرکات سیمولیٹ کریں
- ZMP اور توازن معیار کا استعمال کرتے ہوئے استحکام کی توثیق کریں
- کنیمیٹکس اور ڈائینمکس کے درمیان رشتہ کا تجزیہ کریں

### مرحلہ وار ہدایات

1. **ایک سادہ ہیومنوائڈ ماڈل تخلیق کریں** بنیادی کنیمیٹک چینز کے ساتھ
2. **بازو اور ٹانگ کی حرکات** کے لیے فارورڈ کنیمیٹکس نافذ کریں
3. **ریچنگ ٹاسکس** کے لیے انورس کنیمیٹکس سالور تیار کریں
4. **Euler-Lagrange فارمولیشن** کا استعمال کرتے ہوئے ڈائینمک ماڈل تخلیق کریں
5. **ریچنگ اور اسٹیپنگ** جیسی بنیادی حرکات سیمولیٹ کریں
6. **ZMP اور CoM معیار** کا استعمال کرتے ہوئے استحکام کا تجزیہ کریں
7. **متوقع رویے** کے مقابلے میں نتائج کی توثیق کریں

### متوقع نتائج

- کام کرتا کنیمیٹکس امپلیمنٹیشن
- ڈائینمک سیمولیشن کی صلاحیت
- استحکام کے معیار کی سمجھ
- ہیومنوائڈ کنٹرول کا عملی تجربہ

## نالج چیک

1. فارورڈ اور انورس کنیمیٹکس کے درمیان کیا کلیدی فرق ہیں؟
2. ہیومنوائڈ توازن میں Zero Moment Point (ZMP) کے تصور کی وضاحت کریں۔
3. لکیری انورٹڈ پینڈولم ماڈل توازن کنٹرول کو کیسے آسان بناتا ہے؟
4. ہیومنوائڈ روبوٹ ڈائینمکس میں کیا بنیادی چیلنجز ہیں؟

## خلاصہ

اس باب نے ہیومنوائڈ روبوٹ کنٹرول کے لیے ضروری بنیادی کنیمیٹکس اور ڈائینمکس اصولوں کو احاطہ کیا۔ فارورڈ اور انورس کنیمیٹکس کے ساتھ ساتھ ڈائینمک ماڈلنگ کو سمجھنا ہیومنوائڈ روبوٹس کو مستحکم اور قادر بنانے کے لیے اہم ہے۔ کنیمیٹک حل کو ڈائینمک امور کے ساتھ ضم کرنا بائی پیڈل لوکوموشن اور انسان نما مینیپولیشن کے لیے ضروری پیچیدہ حرکات کو فعال کرتا ہے۔

## اگلے اقدامات

باب 14 میں، ہم بائی پیڈل لوکوموشن اور توازن کنٹرول کو تفصیل سے تلاش کریں گے، یہاں قائم کردہ کنیمیٹکس اور ڈائینمکس کی بنیاد پر تعمیر کرتے ہوئے۔