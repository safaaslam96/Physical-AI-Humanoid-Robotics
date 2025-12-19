---
title: "باب 11: Nav2 اور راستہ منصوبہ بندی"
sidebar_label: "باب 11: Nav2 راستہ منصوبہ بندی"
---

# باب 11: Nav2 اور راستہ منصوبہ بندی

## سیکھنے کے اہداف
- Nav2 نیویگیشن اسٹیک اور اس کے آرکیٹیکچر کو سمجھنا
- بائی پیڈل ہیومنوائڈ حرکت کے لیے راستہ منصوبہ بندی الگورتھم نافذ کرنا
- ہیومنوائڈ روبوٹس کے لیے نیویگیشن سسٹم کی ترتیب جو منفرد کنیمیٹکس کا احترام کرتی ہو
- روبوٹ کنٹرول اور نیویگیشن کے لیے مضبوط سیکھنے کی تکنیکس کا اطلاق

## تعارف

نیویگیشن اسٹیک 2 (Nav2) روبوٹ نیویگیشن میں جدید ترین کا نمائندہ ہے، جو راستہ منصوبہ بندی، مقام کی دریافت اور موشن کنٹرول کے لیے ایک جامع فریم ورک فراہم کرتا ہے۔ ہیومنوائڈ روبوٹس کے لیے، Nav2 کو بائی پیڈل لوکوموشن، توازن کی پابندیوں، اور انسان نما نیویگیشن پیٹرن کا احترام کرنے کے لیے مخصوص ترتیب کی ضرورت ہوتی ہے۔ یہ باب Nav2 کے آرکیٹیکچر، ہیومنوائڈ روبوٹس کے لیے کنفیگریشن، اور جدید راستہ منصوبہ بندی کی تکنیکس کو تلاش کرتا ہے۔

## Nav2 آرکیٹیکچر کو سمجھنا

### Nav2 سسٹم کا جائزہ

Nav2 ایک مکمل نیویگیشن سسٹم ہے جس میں شامل ہیں:
- **نیویگیشن سرور**: مرکزی ربط اور اسٹیٹ مینجمنٹ
- **پلانرز**: گلوبل اور مقامی راستہ منصوبہ بندی الگورتھم
- **کنٹرولرز**: ٹریجکٹری کو فالو کرنے کے لیے موشن کنٹرول
- **ریکوری بیヘویئرز**: نیویگیشن ناکامیوں کے سامنے کارروائی کے لیے حکمت عملیاں
- **ٹرانسفارم مینجمنٹ**: کوآرڈینیٹ فریم ہینڈلنگ اور TF ٹریز
- **لائف سائیکل مینجمنٹ**: جزو کی حالت کا انتظام

### کلیدی اجزاء اور انٹرفیسز

```python
# Nav2 آرکیٹیکچر اجزاء
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
import rclpy.action

class Nav2ClientNode(Node):
    def __init__(self):
        super().__init__('nav2_client')

        # نیویگیشن کے لیے ایکشن کلائنٹ
        self.nav_client = rclpy.action.ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # نیویگیشن کمانڈز کے لیے پبلیشرز
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # نیویگیشن فیڈ بیک کے لیے سبسکرائبرز
        self.feedback_sub = self.create_subscription(
            String, '/navigation_feedback', self.feedback_callback, 10
        )

    def navigate_to_pose(self, pose):
        # Nav2 سرور کو نیویگیشن گول بھیجیں
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.nav_client.wait_for_server()
        return self.nav_client.send_goal_async(goal_msg)

    def feedback_callback(self, msg):
        # نیویگیشن فیڈ بیک کو ہینڈل کریں
        self.get_logger().info(f'نیویگیشن فیڈ بیک: {msg.data}')
```

### Nav2 بمقابلہ Nav1 میں بہتری

| خصوصیت | Nav1 | Nav2 |
|---------|------|------|
| آرکیٹیکچر | مونولیتھک | ماڈولر، پلگ ان-بیسڈ |
| کنفیگریشن | سٹیٹک | ڈائنامک، لائف سائیکل |
| لچک | محدود | انتہائی قابل ترتیب |
| کارکردگی | CPU-بیسڈ | GPU-تیزی کے اختیارات |
| حفاظت | بنیادی | جدید حفاظتی خصوصیات |
| ریکوری | سادہ | جامع ریکوری بیヘویئرز |

## ہیومنوائڈ روبوٹس کے لیے گلوبل راستہ منصوبہ بندی

### منصوبہ بندی الگورتھم کا جائزہ

Nav2 متعدد گلوبل پلانرز کی حمایت کرتا ہے:
- **NavFn**: پوٹینشل فیلڈ-بیسڈ پلینر
- **گلوبل پلینر**: A* اور Dijkstra کے اطلاقات
- **TEB پلینر**: ٹائمڈ الیسٹک بینڈ برائے متحرک ماحول
- **SMAC پلینر**: سپارس مارکوو چین برائے SE2 اور 3D منصوبہ بندی

### ہیومنوائڈ-مخصوص منصوبہ بندی کے امور

ہیومنوائڈ روبوٹس کو مخصوص راستہ منصوبہ بندی کی ضرورت ہوتی ہے کیونکہ:
- **فُٹ اسٹیپ منصوبہ بندی**: بائی پیڈل لوکوموشن کے لیے ڈسکریٹ قدم کی جگہ
- **توازن کی پابندیاں**: توازن کو سپورٹ پولی گون کے اندر برقرار رکھنا
- **قدم کی اونچائی کی حدود**: محفوظ لوکوموشن کے لیے زیادہ سے زیادہ قدم کی اونچائی
- **موڑ کا رداس**: وہیلڈ روبوٹس کے مقابلے میں محدود موڑ کی اہلیت

### کسٹم گلوبل پلینر کا اطلاق

```python
# ہیومنوائڈ-آگاہ گلوبل پلینر
from nav2_core.global_planner import GlobalPlanner
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
import numpy as np

class HumanoidGlobalPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.logger = None
        self.costmap_ros = None
        self.planner_frequency = 1.0
        self.step_size = 0.3  # میٹر میں ہیومنوائڈ قدم کا سائز
        self.turn_threshold = 0.2  # موڑنے کے لیے کم از کم فاصلہ

    def configure(self, tf_buffer, costmap_ros, autostart):
        """کوسٹ میپ اور ٹرانسفارم کے ساتھ پلینر کی ترتیب دیں"""
        self.logger = self.get_logger()
        self.costmap_ros = costmap_ros
        self.tf_buffer = tf_buffer

    def cleanup(self):
        """پلینر وسائل صاف کریں"""
        pass

    def set_costmap_topic(self, topic):
        """منصوبہ بندی کے لیے کوسٹ میپ ٹاپک سیٹ کریں"""
        pass

    def create_plan(self, start, goal):
        """ہیومنوائڈ پابندیوں کو مدنظر رکھتے ہوئے شروع سے گول تک راستہ تخلیق کریں"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        # چیک کریں کہ آیا شروع اور گول درست ہیں
        if not self.is_valid_pose(start) or not self.is_valid_pose(goal):
            self.logger.warn("غلط شروع یا گول پوز")
            return path

        # ہیومنوائڈ-مخصوص پابندیوں کے ساتھ راستہ منصوبہ بندی کریں
        planned_path = self.plan_humanoid_path(start, goal)

        if planned_path:
            path.poses = planned_path
            self.logger.info(f"{len(path.poses)} ویز پوائنٹس کے ساتھ راستہ منصوبہ بندی کیا گیا")
        else:
            self.logger.warn("راستہ منصوبہ بندی میں ناکامی")

        return path

    def plan_humanoid_path(self, start, goal):
        """ہیومنوائڈ لوکوموشن پابندیوں کو مدنظر رکھتے ہوئے راستہ منصوبہ بندی کریں"""
        # ہیومنوائڈ-مخصوص قیمت فنکشن کے ساتھ A* یا Dijkstra نافذ کریں
        # قدم کا سائز، توازن، اور موڑ کی پابندیوں کو مدنظر رکھیں

        planned_path = []

        # سادہ مثال - عمل میں، یہ فُٹ اسٹیپ منصوبہ بندی استعمال کرے گا
        current_pos = start.pose.position
        goal_pos = goal.pose.position

        # درمیانی ویز پوائنٹس کا حساب لگائیں
        distance = np.sqrt(
            (goal_pos.x - current_pos.x)**2 +
            (goal_pos.y - current_pos.y)**2
        )

        num_steps = int(distance / self.step_size)

        for i in range(num_steps + 1):
            ratio = i / num_steps if num_steps > 0 else 0
            waypoint = PoseStamped()
            waypoint.header.frame_id = "map"
            waypoint.pose.position.x = current_pos.x + ratio * (goal_pos.x - current_pos.x)
            waypoint.pose.position.y = current_pos.y + ratio * (goal_pos.y - current_pos.y)
            waypoint.pose.position.z = 0.0  # زمین کی سطح

            # گول کی طرف اورینٹیشن سیٹ کریں
            angle = np.arctan2(goal_pos.y - current_pos.y, goal_pos.x - current_pos.x)
            waypoint.pose.orientation.z = np.sin(angle / 2)
            waypoint.pose.orientation.w = np.cos(angle / 2)

            planned_path.append(waypoint)

        return planned_path

    def is_valid_pose(self, pose):
        """چیک کریں کہ آیا ہیومنوائڈ نیویگیشن کے لیے پوز درست ہے"""
        # چیک کریں کہ آیا پوز مفت جگہ میں ہے اور قابل رسائی ہے
        costmap = self.costmap_ros.get_costmap()
        map_x, map_y = self.pose_to_map_coords(pose)

        if not (0 <= map_x < costmap.size_x and 0 <= map_y < costmap.size_y):
            return False

        cost = costmap.get_cost(map_x, map_y)
        return cost < 253  # مہلک رکاوٹ نہیں
```

### فُٹ اسٹیپ منصوبہ بندی کا انضمام

اصل ہیومنوائڈ نیویگیشن کے لیے، فُٹ اسٹیپ منصوبہ بندی ضروری ہے:

```python
# فُٹ اسٹیپ منصوبہ بندی انٹرفیس
from geometry_msgs.msg import Point
from std_msgs.msg import Header

class FootstepPlanner:
    def __init__(self):
        self.support_polygon = []  # سپورٹ فُٹس کا کن ویکس ہل
        self.step_limit = 0.3  # زیادہ سے زیادہ قدم کی دوری
        self.foot_width = 0.1
        self.foot_length = 0.25

    def plan_footsteps(self, path, robot_state):
        """ہائی-لیول راستہ کو فُٹ اسٹیپ سیکوئنس میں تبدیل کریں"""
        footsteps = []

        for i in range(len(path) - 1):
            start_pose = path[i]
            end_pose = path[i + 1]

            # پوزز کے درمیان ضروری فُٹ اسٹیپس کا حساب لگائیں
            steps = self.calculate_intermediate_steps(start_pose, end_pose)
            footsteps.extend(steps)

        return footsteps

    def calculate_intermediate_steps(self, start, end):
        """دو پوزز کے درمیان ضروری فُٹ اسٹیپس کا حساب لگائیں"""
        # فُٹ اسٹیپ منصوبہ بندی الگورتھم نافذ کریں
        # توازن، قدم کا سائز، اور زمین کی پابندیوں کو مدنظر رکھیں
        pass

    def validate_footstep(self, foot_pose, terrain_map):
        """چیک کریں کہ آیا فُٹ اسٹیپ محفوظ اور مستحکم ہے"""
        # رکاوٹوں، ڈھلوان، اور سطح کی استحکام کے لیے چیک کریں
        pass
```

## مقامی راستہ منصوبہ بندی اور کنٹرول

### مقامی پلینر اجزاء

Nav2 کی مقامی منصوبہ بندی میں شامل ہیں:
- **ٹریجکٹری رول آؤٹ**: امکانی ٹریجکٹریز تخلیق کریں
- **کولیژن چیکنگ**: یہ چیک کریں کہ ٹریجکٹریز رکاوٹ سے پاک ہیں
- **کنٹرول ایگزیکوشن**: روبوٹ کو رفتار کی کمانڈز بھیجیں
- **ریکوری بیヘویئرز**: مقامی نیویگیشن ناکامیوں کے سامنے کارروائی

### ہیومنوائڈ مقامی منصوبہ بندی کے امور

ہیومنوائڈ روبوٹس کے لیے منفرد مقامی منصوبہ بندی کی ضروریات ہیں:
- **توازن برقرار رکھنا**: حرکت کے دوران مستقل توازن
- **قدم کا ٹائمنگ**: مطابقت شدہ فُٹ اسٹیپ ٹائمنگ
- **ZMP کنٹرول**: زیرو مومنٹ پوائنٹ استحکام
- **پش ریکوری**: غیر متوقع قوتوں کو سنبھالنا

### مقامی پلینر کا اطلاق

```python
# ہیومنوائڈ-آگاہ مقامی پلینر
from nav2_core.local_planner import LocalPlanner
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import math

class HumanoidLocalPlanner(LocalPlanner):
    def __init__(self):
        super().__init__()
        self.current_cmd_vel = Twist()
        self.balance_controller = BalanceController()
        self.footstep_generator = FootstepGenerator()
        self.max_linear_speed = 0.3  # استحکام کے لیے محتاط
        self.max_angular_speed = 0.5
        self.lookahead_distance = 0.5

    def setPlan(self, plan):
        """مقامی انجام دہی کے لیے گلوبل منصوبہ سیٹ کریں"""
        self.global_plan = plan
        self.plan_index = 0

    def computeVelocityCommands(self, pose, velocity):
        """ہیومنوائڈ پابندیوں کو مدنظر رکھتے ہوئے رفتار کمانڈز کا حساب لگائیں"""
        cmd_vel = Twist()

        # گلوبل منصوبہ سے اگلا ہدف حاصل کریں
        target = self.get_next_waypoint(pose)
        if target is None:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel, self.get_goal_status()

        # ہدف کی طرف مطلوبہ رفتار کا حساب لگائیں
        distance = self.calculate_distance(pose.pose.position, target.pose.position)
        angle_to_target = self.calculate_angle_to_target(pose, target)

        # ہیومنوائڈ-مخصوص پابندیاں لاگو کریں
        cmd_vel.linear.x = min(
            self.max_linear_speed,
            distance * 0.5  # فاصلے کے متناسب
        )

        cmd_vel.angular.z = min(
            self.max_angular_speed,
            angle_to_target * 2.0  # زاویہ غلطی کے متناسب
        )

        # توازن کی تصحیحات لاگو کریں
        balance_correction = self.balance_controller.get_correction()
        cmd_vel.linear.x *= (1 - abs(balance_correction))

        # مستحکم چلنے کا پیٹرن یقینی بنائیں
        cmd_vel = self.apply_stability_constraints(cmd_vel)

        return cmd_vel, self.get_goal_status()

    def get_next_waypoint(self, current_pose):
        """گلوبل منصوبہ سے اگلا متعلقہ ویز پوائنٹ حاصل کریں"""
        # ویز پوائنٹ فالو کرنے کے لاجک نافذ کریں
        # موجودہ پوز اور لُک ایہیڈ فاصلے کو مدنظر رکھیں
        pass

    def apply_stability_constraints(self, cmd_vel):
        """ہیومنوائڈ-مخصوص استحکام پابندیاں لاگو کریں"""
        # گرنے سے بچنے کے لیے ایکسلریشن کو محدود کریں
        # ریتھمک چلنے کا پیٹرن لاگو کریں
        # توازن کے فیڈ بیک کو مدنظر رکھیں
        return cmd_vel

    def isGoalReached(self):
        """چیک کریں کہ آیا گول تک پہنچ گئے"""
        # ہیومنوائڈ روبوٹس کے لیے گول تک پہنچنے کا معیار نافذ کریں
        pass
```

## ہیومنوائڈ روبوٹس کے لیے نیویگیشن کنفیگریشن

### کوسٹ میپ کنفیگریشن

ہیومنوائڈ روبوٹس کو مخصوص کوسٹ میپ ترتیبات کی ضرورت ہوتی ہے:

```yaml
# costmap_common_params.yaml
robot_radius: 0.4  # ہیومنوائڈ روبوٹ رداس
footprint: []      # واضح فُٹ پرنٹ کے بجائے robot_radius استعمال کریں

obstacle_range: 2.5
raytrace_range: 3.0

# حفاظت کے لیے رکاوٹ سُفخن
inflation_radius: 0.55
cost_scaling_factor: 5.0

# ہیومنوائڈ-مخصوص مشاہدہ کے ذرائع
observation_sources: scan camera
scan:
  sensor_frame: base_scan
  data_type: LaserScan
  topic: /scan
  marking: true
  clearing: true
camera:
  sensor_frame: camera_link
  data_type: PointCloud2
  topic: /camera/depth/points
  marking: true
  clearing: true
  min_obstacle_height: 0.2
  max_obstacle_height: 2.0
```

### کنٹرولر کنفیگریشن

```yaml
# controller_server_params.yaml
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # ہیومنوائڈ-مخصوص کنٹرولر
    FollowPath:
      plugin: "nav2_mppi_controller::MppiController"
      time_steps: 20
      control_horizon: 10
      model_dt: 0.1
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_bounds_x: [-1.0, 1.0]
      state_bounds_y: [-1.0, 1.0]
      state_bounds_theta: [-1.57, 1.57]
      control_bounds_vx: [-0.5, 0.5]
      control_bounds_vy: [-0.1, 0.1]
      control_bounds_wz: [-0.6, 0.6]
      reference_cost_multiplier: 1.0
      goal_cost_multiplier: 24.0
      obstacle_cost_multiplier: 50.0
      control_cost_multiplier: 10.0
      nonholonomic_cost_multiplier: 100.0
```

### بیヘویئر ٹری کنفیگریشن

```xml
<!-- humanoid_navigate_to_pose_w_replanning_and_recovery.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <RecoveryNode number_of_retries="6" name="NavigateRecovery">
            <PipelineSequence name="NavigateWithReplanning">
                <RateController hz="1.0">
                    <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                </RateController>
                <RecoveryNode number_of_retries="1" name="FollowPathRecovery">
                    <FollowPath path="{path}" controller_id="FollowPath"/>
                    <ReactiveFallback name="FollowPathWithRecovery">
                        <GoalReached goal="{goal}"/>
                        <ClearEntirely name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap">
                            <RecoveryNode number_of_retries="1" name="FollowPathWithRecovery-Context">
                                <PipelineSequence name="FollowPathWithRecovery-Sequence">
                                    <ControlRate hz="20.0"/>
                                    <IsPathValid path="{path}"/>
                                    <FollowPath path="{path}" controller_id="FollowPath"/>
                                </PipelineSequence>
                                <ReactiveFallback name="RecoveryFallback">
                                    <GoalReached goal="{goal}"/>
                                    <ClearEntirely name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                                </ReactiveFallback>
                            </RecoveryNode>
                        </ClearEntirely>
                    </ReactiveFallback>
                </RecoveryNode>
            </PipelineSequence>
            <ReactiveFallback name="RecoveryFallback">
                <GoalReached goal="{goal}"/>
                <ClearEntirely name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap">
                    <RecoveryNode number_of_retries="1" name="ClearGlobalCostmap-Context-Recovery">
                        <BackUp backup_dist="0.15" backup_speed="0.05"/>
                        <Spin spin_dist="1.57"/>
                        <Wait wait_duration="5"/>
                    </RecoveryNode>
                </ClearEntirely>
            </ReactiveFallback>
        </RecoveryNode>
    </BehaviorTree>
</root>
```

## نیویگیشن کے لیے مضبوط سیکھنا

### RL-بیسڈ راستہ منصوبہ بندی

مضبوط سیکھنا نیویگیشن کی صلاحیتوں کو بہتر بنا سکتا ہے:

```python
# RL-بیسڈ نیویگیشن کنٹرولر
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class NavigationDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NavigationDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLNavigationAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # ایکسپلوریشن کی شرح
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr

        # نیورل نیٹ ورکس
        self.q_network = NavigationDQN(state_size, action_size)
        self.target_network = NavigationDQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def act(self, state):
        """ایپسیلون-گریڈی پالیسی کا استعمال کرتے ہوئے ایکشن منتخب کریں"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        """ریپلے میموری میں تجربہ ذخیرہ کریں"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """تجربوں کے ایک بیچ پر ماڈل کو تربیت دیں"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RLNavigationController:
    def __init__(self):
        # حالت: [robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_distances...]
        self.state_size = 20  # مثال کے طور پر حالت کا سائز
        self.action_size = 9  # ہیومنوائڈ حرکت کے لیے ڈسکریٹ ایکشنز
        self.agent = RLNavigationAgent(self.state_size, self.action_size)

        # ایکشن اسپیس: ہیومنوائڈ حرکت کے لیے [vx, vy, wz] کمبی نیشنز
        self.action_space = [
            [0.2, 0.0, 0.0],    # آگے
            [-0.2, 0.0, 0.0],   # پیچھے
            [0.0, 0.1, 0.0],    # بائیں
            [0.0, -0.1, 0.0],   # دائیں
            [0.0, 0.0, 0.3],    # بائیں موڑ
            [0.0, 0.0, -0.3],   # دائیں موڑ
            [0.1, 0.0, 0.2],    # آگے-بائیں
            [0.1, 0.0, -0.2],   # آگے-دائیں
            [0.0, 0.0, 0.0]     # رکیں
        ]

    def get_state(self, robot_pose, goal_pose, sensor_data):
        """موجودہ معلومات سے حالت ویکٹر تخلیق کریں"""
        state = []

        # روبوٹ کی پوزیشن اور اورینٹیشن
        state.extend([robot_pose.x, robot_pose.y, robot_pose.theta])

        # گول کی ریلیٹو پوزیشن
        state.extend([goal_pose.x - robot_pose.x, goal_pose.y - robot_pose.y])

        # سینسرز سے رکاوٹ کی دوریاں
        state.extend(sensor_data[:15])  # پہلے 15 سینسر ریڈنگس استعمال کریں

        # ضرورت کے مطابق پیڈ کریں
        while len(state) < self.state_size:
            state.append(0.0)

        return np.array(state[:self.state_size])

    def compute_navigation_command(self, robot_pose, goal_pose, sensor_data):
        """RL ایجنٹ کا استعمال کرتے ہوئے نیویگیشن کمانڈ کا حساب لگائیں"""
        state = self.get_state(robot_pose, goal_pose, sensor_data)
        action_idx = self.agent.act(state)

        # ایکشن انڈیکس کو رفتار کمانڈ میں تبدیل کریں
        action = self.action_space[action_idx]
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.linear.y = action[1]
        cmd_vel.angular.z = action[2]

        return cmd_vel
```

### ہیومنوائڈ-مخصوص RL امور

ہیومنوائڈ نیویگیشن کے لیے مضبوط سیکھنا کو مندرجہ ذیل کو مدنظر رکھنا چاہیے:
- **توازن برقرار رکھنا**: توازن برقرار رکھنے کے لیے انعامات
- **توانائی کی کارآمدی**: کارآمد حرکت کے لیے انعامات
- **حفاظت**: غیر مستحکم حالت کے لیے جرمانے
- **قدم کے پیٹرنز**: قدرتی چلنے کے گیٹس سیکھنا

## عملی اطلاق کی مثالیں

### سادہ نیویگیشن مثال

```python
# ہیومنوائڈ روبوٹ کے لیے مکمل نیویگیشن مثال
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
import rclpy.action

class HumanoidNavigator(Node):
    def __init__(self):
        super().__init__('humanoid_navigator')

        # نیویگیشن ایکشن کلائنٹ
        self.nav_client = rclpy.action.ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # رفتار کمانڈ پبلیشر
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # روبوٹ اسٹیٹ پبلیشر
        self.robot_state_pub = self.create_publisher(
            String, '/robot_state', 10
        )

        # اسٹیٹ مانیٹرنگ کے لیے ٹائمر
        self.timer = self.create_timer(0.1, self.monitor_navigation)

    def navigate_to_goal(self, x, y, theta=0.0):
        """مخصوص ہدف پوزیشن پر نیویگیٹ کریں"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2)

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

        return future

    def goal_response_callback(self, future):
        """نیویگیشن ہدف کے جواب کو ہینڈل کریں"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('ہدف مسترد کر دیا گیا')
            return

        self.get_logger().info('ہدف قبول کر لیا گیا')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """نیویگیشن کا نتیجہ ہینڈل کریں"""
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f'نیویگیشن مکمل ہوا بطور حیثیت: {status}')

    def monitor_navigation(self):
        """نیویگیشن کی پیشرفت اور روبوٹ اسٹیٹ کی نگرانی کریں"""
        # توازن، قدم کا ٹائمنگ، اور پیشرفت چیک کریں
        # مانیٹرنگ کے لیے روبوٹ اسٹیٹ شائع کریں
        pass

def main():
    rclpy.init()
    navigator = HumanoidNavigator()

    # مثال: مخصوص مقام پر نیویگیٹ کریں
    future = navigator.navigate_to_goal(5.0, 3.0, 0.0)

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## کارکردگی کی بہتری اور ٹیوننگ

### پیرامیٹر ٹیوننگ حکمت عملیاں

ہیومنوائڈ نیویگیشن کے لیے، ٹیون کرنے کے لیے کلیدی پیرامیٹرز شامل ہیں:
- **کوسٹ میپ ریزولوشن**: درستگی اور کارکردگی کے درمیان توازن
- **انفلیشن رداس**: رکاوٹوں کے ارد گرد حفاظتی مارجن
- **کنٹرولر فریکوئنسی**: ہمواری اور جوابدہی کے درمیان توازن
- **ٹالرینس ویلیوز**: پوزیشن اور اورینٹیشن ٹالرینسز
- **رفتار کی حدود**: استحکام کے لیے محتاط حدود

### حقیقی وقت کارکردگی کی بہتری

```python
# کارکردگی کی بہتری کی تکنیکس
class OptimizedNavigationManager:
    def __init__(self):
        self.planning_cache = {}  # حالیہ منصوبے کیش کریں
        self.multi_threading = True
        self.gpu_acceleration = True

    def optimize_planning(self, start, goal):
        """کارکردگی کے لیے راستہ منصوبہ بندی کو بہتر بنائیں"""
        # پہلے کیش چیک کریں
        cache_key = (start, goal)
        if cache_key in self.planning_cache:
            return self.planning_cache[cache_key]

        # راستہ منصوبہ بندی کریں
        path = self.plan_path(start, goal)

        # نتیجہ کیش کریں
        self.planning_cache[cache_key] = path

        return path

    def plan_path(self, start, goal):
        """بہتری کی تکنیکس کے ساتھ راستہ منصوبہ بندی کریں"""
        # ہائیرارکیکل منصوبہ بندی استعمال کریں
        # ملٹی-ریزولوشن سرچ نافذ کریں
        # پرُننگ تکنیکس لاگو کریں
        pass
```

## ٹربل شوٹنگ اور بہترین طریقے

### عام نیویگیشن مسائل

1. **آسیلیشن**: روبوٹ آگے پیچھے چلتا ہے
   - حل: کنٹرولر پیرامیٹرز کو ایڈجسٹ کریں، لُک ایہیڈ فاصلہ بڑھائیں

2. **پھنسنا**: روبوٹ رکاوٹوں کے گرد نیویگیٹ کرنے میں ناکام ہو جاتا ہے
   - حل: کوسٹ میپ انفلیشن بہتر کریں، ریکوری بیヘویئرز شامل کریں

3. **غیر موثر راستے**: روبوٹ غیر ضروری طویل راستے لیتا ہے
   - حل: گلوبل پلینر پیرامیٹرز کو ٹیون کریں، کوسٹ میپ ترتیبات ایڈجسٹ کریں

4. **توازن کے مسائل**: نیویگیشن کے دوران ہیومنوائڈ گرتا ہے
   - حل: توازن کنٹرولر نافذ کریں، رفتار کی حدود کم کریں

### حفاظتی امور

- **ایمرجنسی سٹاپ**: فوری سٹاپ کی صلاحیت نافذ کریں
- **محفوظ رفتاریں**: استحکام کے لیے محتاط رفتار کی حدود
- **رکاوٹ کا پتہ لگانا**: قابل اعتماد رکاوٹ کا پتہ لگانا اور اس سے بچنا
- **گرنے کی ریکوری**: گرنے کے معاملات کے لیے طریقے

## ہاتھوں سے مشق: ہیومنوائڈ روبوٹ کے لیے Nav2 کی ترتیب

### مشق کے اہداف
- ہیومنوائڈ-مخصوص نیویگیشن کے لیے Nav2 کی ترتیب
- توازن کو مدنظر رکھتے ہوئے کسٹم راستہ منصوبہ بندی نافذ کرنا
- موافق رویے کے لیے RL-بیسڈ نیویگیشن کا انضمام
- نیویگیشن کارکردگی کی جانچ اور توثیق

### مرحلہ وار ہدایات

1. **Nav2 انسٹال کریں اور ترتیب دیں** ہیومنوائڈ-مخصوص پیرامیٹرز کے ساتھ
2. **ہیومنوائڈ پابندیوں کے لیے کسٹم کوسٹ میپ کنفیگریشن** تخلیق کریں
3. **توازن کو مدنظر رکھتے ہوئے بنیادی راستہ فالو کرنا** نافذ کریں
4. **موافق نیویگیشن کے لیے RL جزو** شامل کریں
5. **مختلف منظرناموں کے ساتھ سیمولیشن میں جانچ** کریں
6. **کارکردگی کا تجزیہ** اور پیرامیٹرز ٹیون کریں

### متوقع نتائج
- ہیومنوائڈ روبوٹ کے لیے کام کرتی Nav2 کنفیگریشن
- ہیومنوائڈ نیویگیشن چیلنجز کی سمجھ
- پیرامیٹر ٹیوننگ کا تجربہ
- کارکردگی کا تجزیہ اور بہتری

## نالج چیک

1. Nav2 اور Nav1 آرکیٹیکچر کے درمیان کلیدی فرق کیا ہیں؟
2. بائی پیڈل ہیومنوائڈ روبوٹس کے لیے راستہ منصوبہ بندی کے چیلنجز کی وضاحت کریں۔
3. مضبوط سیکھنا نیویگیشن کی صلاحیتوں کو کیسے بہتر بنا سکتا ہے؟
4. ہیومنوائڈ روبوٹ نیویگیشن کے لیے کون سے حفاظتی امور منفرد ہیں؟

## خلاصہ

اس باب میں ہم نے ہیومنوائڈ روبوٹس کے لیے Nav2 اور راستہ منصوبہ بندی کا جائزہ لیا، جس میں بائی پیڈل لوکوموشن، توازن برقرار رکھنا، اور انسان نما نیویگیشن پیٹرنز کے منفرد چیلنجز کو سامنے کیا گیا۔ روایتی راستہ منصوبہ بندی کو مضبوط سیکھنے کے ساتھ ضم کرنا ہیومنوائڈ روبوٹس کو حقیقی دنیا کے انتظام کے لیے ضروری موافق نیویگیشن کی صلاحیتوں سے لیس کرتا ہے۔

## اگلے اقدامات

باب 12 میں، ہم سیم-ٹو-ریل ٹرانسفير تکنیکس کا جائزہ لیں گے، جو نیویگیشن کی بنیاد کو ابھرتے ہوئے حقیقی دنیا کے ماحول میں کامیاب انتظام کے قابل بنانے کے لیے استعمال کریں گے۔