---
title: "باب 9: NVIDIA Isaac SDK اور Isaac Sim"
sidebar_label: "باب 9: NVIDIA Isaac SDK"
---

# باب 9: NVIDIA Isaac SDK اور Isaac Sim

## سیکھنے کے اہداف
- NVIDIA Isaac پلیٹ فارم ایکو سسٹم اور اس کے اجزاء کو سمجھنا
- فوٹو ریلائزٹک سمولیشن کے لیے Isaac Sim کا قیام اور کنفیگریشن کرنا
- روبوٹکس کے لیے مصنوعی ڈیٹا جنریشن پائپ لائنز نافذ کرنا
- Isaac ٹولز کا استعمال کرتے ہوئے AI-پاورڈ ادراک اور مینیپولیشن سسٹمز کو ضم کرنا

## تعارف

NVIDIA Isaac پلیٹ فارم AI-پاورڈ روبوٹکس سسٹمز کی ترقی، سمولیشن، اور تنصیب کے لیے ایک جامع ایکو سسٹم کی نمائندگی کرتا ہے۔ اس کے مرکز میں، Isaac Sim فوٹو ریلائزٹک سمولیشن کی صلاحیات فراہم کرتا ہے جو سمولیشن اور حقیقی دنیا کی تنصیب کے درمیان حقیقت کا فرق پُر کرتا ہے۔ یہ باب Isaac پلیٹ فارم کو متعارف کراتا ہے، Isaac Sim پر توجہ مرکز کرتا ہے اعلی درجے کی روبوٹکس سمولیشن اور مصنوعی ڈیٹا جنریشن کے لیے، ہیومنوائڈ روبوٹکس اطلاقات میں AI سسٹمز کی تربیت کے لیے ضروری ہے۔

## NVIDIA Isaac ایکو سسٹم کو سمجھنا

### Isaac پلیٹ فارم اجزاء کا جائزہ

NVIDIA Isaac پلیٹ فارم متعدد مربوط اجزاء پر مشتمل ہے:

1. **Isaac Sim**: NVIDIA Omniverse پر تعمیر کردہ فوٹو ریلائزٹک سمولیشن ماحول
2. **Isaac ROS**: ہارڈ ویئر-تیز ادراک اور نیویگیشن پیکجز
3. **Isaac ایپس**: عام روبوٹکس کاموں کے لیے پیش ساختہ ایپلیکیشنز
4. **Isaac جم**: GPU-تیز رینفورسمنٹ لرننگ ماحول
5. **DeepGraph**: AI-پاورڈ میپنگ اور منصوبہ بندی کے ٹولز

### Isaac Sim آرکیٹیکچر

Isaac Sim NVIDIA کے Omniverse پلیٹ فارم کا فائدہ اٹھاتا ہے، یہ فراہم کرتا ہے:
- **USD (یونیورسل سین ڈیسکرپشن)**: منظر کی نمائندگی کا فارمیٹ
- **PhysX**: NVIDIA کا فزکس سمولیشن انجن
- **RTX رینڈرنگ**: فوٹو ریلائزٹک ویژولز کے لیے حقیقی وقت کا رے ٹریسنگ
- **Omniverse کنیکٹرز**: دیگر ٹولز اور پلیٹ فارمز کے ساتھ انضمام

### دیگر سمولیشن پلیٹ فارمز کے ساتھ موازنہ

| خصوصیت | Isaac Sim | Gazebo | یونیٹی |
|---------|-----------|--------|-------|
| بصری معیار | فوٹو ریلائزٹک (RTX) | اچھا | عمدہ |
| فزکس | PhysX (NVIDIA) | متعدد انجن | تعمیر شدہ فزکس |
| AI انضمام | نیٹیو (TensorRT, cuDNN) | پلگ انز کے ذریعے | پلگ انز کے ذریعے |
| ہارڈ ویئر تیزی | مکمل GPU تیزی | محدود | محدود |
| مصنوعی ڈیٹا | جامع ادراک ٹولز | بنیادی | ادراک پیکج |

## Isaac Sim کا قیام اور کنفیگریشن

### سسٹم کی ضروریات

Isaac Sim کو مانگنے والی ہارڈ ویئر کی ضروریات ہیں:
- **GPU**: NVIDIA RTX سیریز (RTX 3080 یا بہتر تجویز کردہ)
- **VRAM**: 8GB+ کم از کم، 16GB+ تجویز کردہ
- **CPU**: ملٹی کور پروسیسر (8+ کور)
- **RAM**: 16GB+ کم از کم، 32GB+ تجویز کردہ
- **OS**: Ubuntu 20.04/22.04 یا Windows 10/11
- **CUDA**: CUDA 11.8+ مطابق ڈرائیورز کے ساتھ

### انسٹالیشن کا عمل

```bash
# طریقہ 1: Isaac Sim Docker کا استعمال کریں (تجویز کردہ)
docker pull nvcr.io/nvidia/isaac-sim:latest
docker run --gpus all -it --rm \
  --network=host \
  --env "NVIDIA_VISIBLE_DEVICES=all" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all" \
  nvcr.io/nvidia/isaac-sim:latest

# طریقہ 2: Omniverse Launcher کے ذریعے Isaac Sim
# ڈاؤن لوڈ کریں اور Omniverse Launcher انسٹال کریں
# Isaac Sim ایکسٹینشن کو لانچر کے ذریعے انسٹال کریں
```

### ابتدائی کنفیگریشن

انسٹالیشن کے بعد، تصدیق کریں کہ Isaac Sim کام کر رہا ہے:

```bash
# Isaac Sim لانچ کریں
isaac-sim

# یا خودکار ٹیسٹنگ کے لیے ہیڈ لیس چلائیں
isaac-sim --/headless
```

## Isaac Sim بنیادیات

### USD منظر کی تفصیل

Isaac Sim USD (یونیورسل سین ڈیسکرپشن) کا استعمال منظر کی نمائندگی کے لیے کرتا ہے:

```python
# OmniGraph کا استعمال کرتے ہوئے پائی تھون مثال
import omni
from pxr import Usd, UsdGeom, Gf

# ایک نیا اسٹیج تخلیق کریں
stage = Usd.Stage.CreateNew("robot_scene.usd")

# ایک پریم (بنیادی چیز) شامل کریں
xform = UsdGeom.Xform.Define(stage, "/Robot")
xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

# ایک میش شامل کریں
mesh = UsdGeom.Mesh.Define(stage, "/Robot/Body")
mesh.CreatePointsAttr([(0,0,0), (1,0,0), (0,1,0)])
mesh.CreateFaceVertexCountsAttr([3])
mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

# اسٹیج محفوظ کریں
stage.GetRootLayer().Save()
```

### فزکس کنفیگریشن

Isaac Sim NVIDIA PhysX کا استعمال فزکس سمولیشن کے لیے کرتا ہے:

```python
# فزکس خصوصیات کنفیگر کریں
from omni.isaac.core.utils.physics import set_gpu_max_steps
from omni.isaac.core.utils.stage import add_reference_to_stage

# فزکس پیرامیٹرز سیٹ کریں
set_gpu_max_steps(1)  # GPU کا استعمال فزکس سمولیشن کے لیے کریں
```

### میٹریل اور لائٹنگ سیٹ اپ

```python
# فزیکلی-مبنی میٹریلز تخلیق کریں
from omni.isaac.core.utils.materials import create_diffuse_material
from omni.isaac.core.utils.stage import add_reference_to_stage

# حقیقی خصوصیات کے ساتھ میٹریلز تخلیق کریں
robot_material = create_diffuse_material(
    prim_path="/World/Looks/RobotMaterial",
    color=(0.7, 0.7, 0.7)  # روبوٹ باڈی کے لیے میٹلک گرے
)

# فوٹو ریلائزٹک رینڈرنگ کے لیے لائٹنگ کنفیگر کریں
from omni.isaac.core.utils.prims import create_prim
create_prim(
    prim_path="/World/Light",
    prim_type="DistantLight",
    position=(0, 0, 10),
    attributes={"color": (1, 1, 1), "intensity": 3000}
)
```

## فوٹو ریلائزٹک سمولیشن ماحول تخلیق کرنا

### ماحول کی ڈیزائن کے اصول

Isaac Sim ماحول کو درج ذیل کو شامل کرنا چاہیے:
- **حقیقی میٹریلز**: درست خصوصیات کے ساتھ PBR میٹریلز
- **ڈائینامک لائٹنگ**: حقیقی سایوں کے ساتھ متعدد لائٹ ذرائع
- **اعلی معیار کی ٹیکسچریز**: تفصیل کے لیے 4K+ ٹیکسچریز
- **فزکس-درست خصوصیات**: حقیقی فرکشن، ریسٹیٹوشن، وغیرہ

### 3D ماڈلز درآمد کرنا

Isaac Sim مختلف ماڈل فارمیٹس کی حمایت کرتا ہے:
- **USD**: مکمل فیچر سپورٹ کے ساتھ مقامی فارمیٹ
- **FBX**: اینیمیشن سپورٹ کے ساتھ عام 3D فارمیٹ
- **OBJ**: سادہ جیومیٹری فارمیٹ
- **GLTF**: PBR میٹریلز کے ساتھ جدید فارمیٹ

### منظر کمپوزیشن

```python
# Isaac Sim میں منظر سیٹ اپ کی مثال
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.kit.commands

# دنیا کو شروع کریں
world = World(stage_units_in_meters=1.0)

# روبوٹ کو منظر میں شامل کریں
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Isaac Sim اثاثے نہیں مل سکے۔ کیا آپ نے Isaac Sim درست طریقے سے انسٹال کیا؟")
else:
    # ایک نمونہ روبوٹ شامل کریں
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
        prim_path="/World/Robot"
    )

# ماحول شامل کریں
add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd",
    prim_path="/World/Room"
)

# تبدیلیاں لاگو کرنے کے لیے دنیا کو ری سیٹ کریں
world.reset()
```

## مصنوعی ڈیٹا جنریشن پائپ لائن

### ادراک ڈیٹا جنریشن

Isaac Sim مصنوعی ادراک ڈیٹا جنریٹ کرنے میں بہتر ہے:

```python
# مصنوعی ڈیٹا جنریشن کے لیے سینسرز کنفیگر کریں
from omni.isaac.sensor import Camera, LidarRtx
import numpy as np

# RGB کیمرہ شامل کریں
camera = Camera(
    prim_path="/World/Robot/Camera",
    position=np.array([0.5, 0.0, 1.0]),
    orientation=np.array([0, 0, 0, 1])
)

# LIDAR سینسر شامل کریں
lidar = LidarRtx(
    prim_path="/World/Robot/Lidar",
    translation=np.array([0.0, 0.0, 1.2]),
    config="Example_Rotary"
)

# مصنوعی ڈیٹا کیپچر کریں
rgb_data = camera.get_rgb()
depth_data = camera.get_depth()
lidar_data = lidar.get_linear_depth_data()
```

### گراؤنڈ ٹرو اینوٹیشن

```python
# گراؤنڈ ٹرو ڈیٹا تخلیق کریں
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.core.utils.prims import get_prim_at_path

# سیگمینٹیشن ماسک حاصل کریں
semantic_sensor = world.scene.get_sensor("semantic")
semantic_data = semantic_sensor.get_semantic_data()

# 3D باؤنڈنگ باکس حاصل کریں
bbox_sensor = world.scene.get_sensor("bounding_box_3d")
bbox_data = bbox_sensor.get_bounding_box_3d_data()

# 6D پوز معلومات حاصل کریں
pose_sensor = world.scene.get_sensor("pose")
pose_data = pose_sensor.get_pose_data()
```

### ڈومین رینڈمائزیشن

```python
# ڈومین رینڈمائزیشن نافذ کریں
from omni.isaac.core.utils.prims import randomize_instanceable_assets
import random

def randomize_environment():
    # لائٹنگ رینڈمائز کریں
    light_prim = get_prim_at_path("/World/Light")
    light_prim.GetAttribute("inputs:intensity").Set(
        random.uniform(1000, 5000)
    )

    # میٹریلز رینڈمائز کریں
    materials = ["/World/Looks/FloorMaterial", "/World/Looks/WallMaterial"]
    for mat_path in materials:
        mat_prim = get_prim_at_path(mat_path)
        # رنگ، کٹورتا، وغیرہ رینڈمائز کریں
        mat_prim.GetAttribute("inputs:diffuse_tint").Set(
            (random.random(), random.random(), random.random())
        )

    # اشیاء کی پوزیشنز رینڈمائز کریں
    objects = ["/World/Objects/Object1", "/World/Objects/Object2"]
    for obj_path in objects:
        obj_prim = get_prim_at_path(obj_path)
        obj_prim.GetAttribute("xformOp:translate").Set(
            (random.uniform(-2, 2), random.uniform(-2, 2), 0)
        )
```

## AI-پاورڈ ادراک اور مینیپولیشن

### Isaac ROS انضمام

Isaac ROS ہارڈ ویئر-تیز ادراک پیکجز فراہم کرتا ہے:

```bash
# Isaac ROS پیکجز انسٹال کریں
sudo apt install ros-humble-isaac-ros-* ros-humble-novatel-octopus-*
```

### ادراک پائپ لائن کی مثال

```python
# Isaac ROS کا استعمال کرتے ہوئے ادراک پائپ لائن کی مثال
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray

class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Isaac Sim سے کیمرہ ڈیٹا کے لیے سبسکرائب کریں
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.camera_callback,
            10
        )

        # ڈیٹیکشن نتائج کے لیے سبسکرائب کریں
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/isaac_ros_detection',
            self.detection_callback,
            10
        )

        # اشیاء کی پوز کے لیے پبلشر
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/detected_object_pose',
            10
        )

    def camera_callback(self, msg):
        # Isaac ROS ادراک کا استعمال کرتے ہوئے کیمرہ ڈیٹا پروسیس کریں
        # یہ عام طور پر Isaac ROS DNN نوڈس سے جڑے گا
        pass

    def detection_callback(self, msg):
        # ڈیٹیکشن نتائج کو پروسیس کریں اور پوز شائع کریں
        for detection in msg.detections:
            pose_msg = PoseStamped()
            # 2D ڈیٹیکشن سے 3D پوز کا حساب لگائیں
            # ROS ٹاپک پر شائع کریں
            self.pose_pub.publish(pose_msg)
```

### مینیپولیشن منصوبہ بندی

```python
# Isaac ٹولز کا استعمال کرتے ہوئے مینیپولیشن منصوبہ بندی کی مثال
from omni.isaac.motion_generation import RmpFlow
from omni.isaac.core.articulations import ArticulationView
import numpy as np

class IsaacManipulationController:
    def __init__(self, robot_name):
        # موشن جنریشن کے لیے RMPFlow شروع کریں
        self.rmp_flow = RmpFlow(
            robot_description_path="/path/to/robot/urdf",
            end_effector_frame_name="end_effector"
        )

        # روبوٹ ارٹیکولیشن ویو حاصل کریں
        self.robot = ArticulationView(prim_path=f"/World/{robot_name}")

    def move_to_pose(self, target_position, target_orientation):
        # ہدف کی پوز کے لیے جوائنٹ پوزیشنز کا حساب لگائیں
        joint_positions = self.rmp_flow.compute_joints(
            target_position=target_position,
            target_orientation=target_orientation
        )

        # جوائنٹ پوزیشنز روبوٹ پر لاگو کریں
        self.robot.set_joint_positions(joint_positions)
```

## Isaac Sim ایکسٹینشنز اور کسٹم ٹولز

### کسٹم ایکسٹینشنز تخلیق کرنا

```python
# Isaac Sim ایکسٹینشن کی مثال
import omni.ext
import omni.ui as ui
from pxr import Gf

class IsaacSimRobotExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[isaac_sim_robot_extension] روبوٹ ایکسٹینشن اسٹارٹ اپ")

        # ونڈو تخلیق کریں
        self._window = ui.Window("روبوٹ کنٹرول", width=300, height=300)

        with self._window.frame:
            with ui.VStack():
                ui.Label("روبوٹ کنٹرول پینل")
                ui.Button("روبوٹ حرکت دیں", clicked_fn=self._move_robot)
                ui.Button("سمولیشن ری سیٹ کریں", clicked_fn=self._reset_simulation)

    def _move_robot(self):
        # کسٹم روبوٹ حرکت لاگک
        print("سمولیشن میں روبوٹ حرکت دے رہا ہے")

    def _reset_simulation(self):
        # سمولیشن کو ابتدائی حالت میں ری سیٹ کریں
        print("سمولیشن ری سیٹ ہو رہا ہے")

    def on_shutdown(self):
        print("[isaac_sim_robot_extension] روبوٹ ایکسٹینشن شٹ ڈاؤن")
```

### کسٹم سینسرز اور ایکچویٹرز

```python
# کسٹم سینسر امپلیمنٹیشن کی مثال
from omni.isaac.core.sensors import Sensor
import numpy as np

class CustomForceTorqueSensor(Sensor):
    def __init__(self, prim_path, name, position, orientation):
        super().__init__(prim_path=prim_path, name=name)
        self._position = position
        self._orientation = orientation
        self._force_data = np.zeros(3)
        self._torque_data = np.zeros(3)

    def get_sensor_data(self):
        # فورس/ٹورک پیمائش کی شبیہہ بنائیں
        # یہ فزکس سمولیشن ڈیٹا سے جڑے گا
        return {
            'force': self._force_data,
            'torque': self._torque_data,
            'timestamp': self._world.get_physics_dt() * self._world.current_frame
        }

    def update(self):
        # سمولیشن کی بنیاد پر سینسر ڈیٹا اپ ڈیٹ کریں
        # نوائس، فلٹرنگ، وغیرہ لاگو کریں
        pass
```

## Isaac Sim میں کارکردگی کی بہتری

### GPU تیزی

Isaac Sim متعدد GPU خصوصیات کا فائدہ اٹھاتا ہے:
- **RTX رینڈرنگ**: فوٹو ریلائزٹک ویژولز کے لیے حقیقی وقت کا رے ٹریسنگ
- **PhysX GPU**: تیز فزکس سمولیشن
- **CUDA کرنلز**: کسٹم کمپیوٹ آپریشنز
- **TensorRT**: بہتر AI انفرینس

### ملٹی GPU کنفیگریشن

```python
# Isaac Sim میں ملٹی GPU استعمال کنفیگر کریں
import omni
from omni.isaac.core.utils.settings import set_simulation_settings

# رینڈرنگ اور فزکس کو GPU استعمال کرنے کے لیے سیٹ کریں
set_simulation_settings(
    stage_units_in_meters=1.0,
    render_physics_thread=True,
    enable_gpu_physics=True,
    gpu_max_steps=1
)
```

### میموری مینجمنٹ

- **سٹریمنگ**: اثاثے متحرک طور پر لوڈ/ان لوڈ کریں
- **LOD سسٹم**: فاصلے کی بنیاد پر مختلف تفصیل کی سطحیں استعمال کریں
- **ٹیکسچر کمپریشن**: ٹیکسچر میموری کے استعمال کو بہتر بنائیں
- **انسٹینس رینڈرنگ**: دہرائے گئے اشیاء کے لیے انسٹینسنگ استعمال کریں

## Isaac Sim ترقی کے لیے بہترین طریقے

### سمولیشن فیڈلٹی

- **توثیق**: سمولیشن نتائج کو حقیقی دنیا کے ڈیٹا کے ساتھ موازنہ کریں
- **کیلیبریشن**: فزیکل روبوٹس کے مطابق پیرامیٹرز کو فائن ٹیون کریں
- **توثیق**: متعدد منظار اور حالات کے ساتھ ٹیسٹ کریں

### مصنوعی ڈیٹا کی معیار

- **تنوع**: مختلف نقطہ نظر اور حالات سے ڈیٹا جنریٹ کریں
- **اینوٹیشن**: درست گراؤنڈ ٹرو لیبلز کو یقینی بنائیں
- **توثیق**: تربیت سے پہلے مصنوعی ڈیٹا کی معیار کی تصدیق کریں

### انضمام کے حکمت عملیاں

- **ماڈیولر ڈیزائن**: سمولیشن اور حقیقی دنیا کے اجزاء کو الگ رکھیں
- **API مطابقت**: سمولیشن اور حقیقی دنیا کے درمیان مطابق انٹرفیسز استعمال کریں
- **کارکردگی کی نگرانی**: سمولیشن کارکردگی کے معیار کو ٹریک کریں

## ہاتھوں سے مشق: ہیومنوائڈ روبوٹکس کے لیے Isaac Sim کا قیام

### مشق کے اہداف

- Isaac Sim انسٹال اور کنفیگر کریں
- ایک سادہ ہیومنوائڈ روبوٹ سمولیشن تخلیق کریں
- بنیادی ادراک سینسرز نافذ کریں
- مصنوعی تربیتی ڈیٹا جنریٹ کریں

### مرحلہ وار ہدایات

1. **Isaac Sim انسٹال کریں** Docker یا Omniverse Launcher کا استعمال کرتے ہوئے
2. **ایک ہیومنوائڈ روبوٹ ماڈل** سمولیشن میں درآمد کریں
3. **ادراک سینسرز کنفیگر کریں** (کیمرہ، LIDAR)
4. **ڈومین رینڈمائزیشن** ڈیٹا ایگزیمیشن کے لیے نافذ کریں
5. **مصنوعی ڈیٹا سیٹس** AI تربیت کے لیے جنریٹ کریں
6. **سمولیشن کی توثیق** متوقع رویوں کے خلاف کریں

### متوقع نتائج

- کام کرتا Isaac Sim ماحول
- سمولیشن میں سینسرز کے ساتھ ہیومنوائڈ روبوٹ
- مصنوعی ڈیٹا جنریشن پائپ لائن
- فوٹو ریلائزٹک سمولیشن کی سمجھ

## نالج چیک

1. NVIDIA Isaac پلیٹ فارم کے کلیدی اجزاء کیا ہیں؟
2. Isaac Sim اور Gazebo جیسے روایتی سمولیٹرز کے درمیان فرق کی وضاحت کریں۔
3. ڈومین رینڈمائزیشن مصنوعی ڈیٹا کی معیار کو کیسے بہتر بناتا ہے؟
4. Isaac Sim کو مؤثر طریقے سے چلانے کے لیے ہارڈ ویئر کی ضروریات کیا ہیں؟

## خلاصہ

اس باب نے NVIDIA Isaac پلیٹ فارم کو متعارف کرایا، جو Isaac Sim پر توجہ مرکز کرتا ہے فوٹو ریلائزٹک سمولیشن اور مصنوعی ڈیٹا جنریشن کے لیے۔ Isaac Sim کی اعلی درجے کی رینڈرنگ کی صلاحیتوں کو جوڑ کر، GPU تیزی اور AI انضمام کے ساتھ یہ اعلی معیار کے تربیتی ڈیٹا جنریٹ کرنے اور AI-پاورڈ روبوٹکس سسٹمز کو ٹیسٹ کرنے کے لیے ایک مثالی پلیٹ فارم ہے۔ سمولیشن اور حقیقی دنیا کی تنصیب کے درمیان حقیقت کا فرق پُر کرنے کی پلیٹ فارم کی صلاحیت ہیومنوائڈ روبوٹکس اطلاقات کے لیے اہم ہے۔

## اگلے اقدامات

باب 10 میں، ہم Isaac ROS اور ہارڈ ویئر-تیز ادراک کو تلاش کریں گے، Isaac ٹولز اور ROS 2 کے درمیان انضمام میں گہرائی میں جاتے ہوئے اعلی درجے کی روبوٹکس اطلاقات کے لیے۔