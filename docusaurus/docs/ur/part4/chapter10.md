---
title: "چیپٹر 10: SLAM اور نیوی گیشن"
sidebar_label: "چیپٹر 10: SLAM اور نیوی گیشن"
---

# چیپٹر 10: SLAM اور نیوی گیشن

## سیکھنے کے اہداف
- SLAM (Simultaneous Localization and Mapping) کے تصورات کو سمجھنا
- ROS 2 میں Nav2 فریم ورک کا نفاذ
- بائی پیڈل انسان نما روبوٹ کے لیے نیوی گیشن کا ڈیزائن
- VSLAM (Visual SLAM) کا استعمال کرنا
- نیوی گیشن کے لیے ٹریجکٹری پلاننگ کا تجزیہ

## SLAM کی معرفت

### SLAM کیا ہے؟

SLAM (Simultaneous Localization and Mapping) ایک الگورتھم ہے جو روبوٹ کو اس کے ماحول کو ناپا جانے والے ماحول میں اس کی مقام کو ساتھ ساتھ نقشہ بناتے ہوئے متعین کرنے کی اجازت دیتا ہے۔

### SLAM کے اجزاء

1. **Localization**: روبوٹ کا پتہ لگانا کہ وہ ماحول میں کہاں ہے
2. **Mapping**: ماحول کا نقشہ بنانا
3. **Data Association**: ڈیٹا کو متعلقہ اشیاء سے جوڑنا
4. **Filtering**: شور کو فلٹر کرنا اور ڈیٹا کو ہموار کرنا

### SLAM کی قسمیں

1. **Laser SLAM**: لیزر سکینرز کا استعمال کرتا ہے
2. **Visual SLAM**: کیمرہ کا استعمال کرتا ہے
3. **Visual-Inertial SLAM**: کیمرہ اور IMU کا استعمال کرتا ہے
4. **Multi-sensor SLAM**: متعدد سینسرز کو ضم کرتا ہے

## VSLAM (Visual SLAM)

### VSLAM کیا ہے؟

VSLAM (Visual Simultaneous Localization and Mapping) تصویر کے ڈیٹا کا استعمال کرتا ہے تاکہ روبوٹ کی مقام کو تعین کیا جا سکے اور ماحول کا نقشہ بنایا جا سکے۔

### VSLAM کے مراحل

1. **Feature Detection**: اہم خصوصیات کو تصویر میں تلاش کرنا
2. **Feature Matching**: مختلف تصاویر میں خصوصیات کا میچ بنانا
3. **Pose Estimation**: روبوٹ کی پوزیشن کا حساب لگانا
4. **Mapping**: ماحول کا نقشہ بنانا
5. **Loop Closure**: پہلے سے دیکھے گئے علاقوں کو پہچاننا

### ORB-SLAM

ORBSLAM ایک مقبول VSLAM سسٹم ہے:

```python
import cv2
import numpy as np

class ORB_SLAM:
    def __init__(self):
        # ORB ڈیٹکٹر کو شروع کریں
        self.orb = cv2.ORB_create(nfeatures=1000)

        # FLANN میچر کو شروع کریں
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
        search_params = {}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # پچھلی تصویر کو ذخیرہ کریں
        self.prev_image = None
        self.prev_kps = None

        # کیمرہ کے پیرامیٹرز
        self.fx = 525.0  # فوکل لمبائی x
        self.fy = 525.0  # فوکل لمبائی y
        self.cx = 319.5  # اصل پوائنٹ x
        self.cy = 239.5  # اصل پوائنٹ y

    def process_frame(self, current_image):
        """ایک نئی تصویر کو پروسیس کریں"""
        # گرے اسکیل میں تبدیل کریں
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

        # خصوصیات کو ڈیٹکٹ کریں
        kps, des = self.orb.detectAndCompute(gray, None)

        if self.prev_image is not None and self.prev_kps is not None:
            # خصوصیات کو میچ کریں
            matches = self.match_features(self.prev_des, des)

            # ہومو گرافی کا حساب لگائیں
            transformation = self.estimate_motion(matches, self.prev_kps, kps)

            # روبوٹ کی پوزیشن کا حساب لگائیں
            pose = self.update_pose(transformation)

            # نتیجہ لوپ کو چیک کریں
            if self.loop_closure_needed():
                self.perform_loop_closure()

        # موجودہ ڈیٹا کو اگلے فریم کے لیے محفوظ کریں
        self.prev_image = gray
        self.prev_kps = kps
        self.prev_des = des

        return pose

    def match_features(self, prev_des, curr_des):
        """خصوصیات کو میچ کریں"""
        if len(prev_des) >= 2 and len(curr_des) >= 2:
            matches = self.flann.knnMatch(prev_des, curr_des, k=2)

            # گوڈ میچز کو فلٹر کریں
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            return good_matches
        return []

    def estimate_motion(self, matches, prev_kps, curr_kps):
        """روبوٹ کی حرکت کا تخمینہ لگائیں"""
        if len(matches) >= 10:
            src_pts = np.float32([prev_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # ہومو گرافی کا حساب لگائیں
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        return None

    def update_pose(self, transformation):
        """روبوٹ کی پوزیشن کو اپ ڈیٹ کریں"""
        # ہومو گرافی کو استعمال کرکے پوزیشن کا حساب لگائیں
        # یہ ایک سادہ تخمینہ ہے، اصل سسٹم پیچیدہ ہوگا
        if transformation is not None:
            # پوزیشن کا حساب لگائیں
            dx = transformation[0, 2]  # x ڈیسپلیسمنٹ
            dy = transformation[1, 2]  # y ڈیسپلیسمنٹ

            # زاویہ کا تخمینہ لگائیں
            angle = np.arctan2(transformation[1, 0], transformation[0, 0])

            return np.array([dx, dy, angle])

        return np.array([0, 0, 0])
```

## ROS 2 میں Nav2 فریم ورک

### Nav2 کیا ہے؟

Nav2 (Navigation 2) ROS 2 کے لیے نیوی گیشن فریم ورک ہے جو روبوٹ کو ایک ماحول میں نیوی گیٹ کرنے کے قابل بناتا ہے۔

### Nav2 کے اجزاء

1. **Behavior Tree Navigator**: ٹاسک کو انجام دینے کے لیے بیہیوئر ٹریز
2. **Global Planner**: بڑے پیمانے پر راستہ منصوبہ بندی
3. **Local Planner**: رکاوٹوں سے بچنا اور راستہ پر رہنا
4. **Controller**: روبوٹ کو حرکت دینا
5. **Recovery Behaviors**: ناکامی سے بازیافت کے لیے

### Nav2 کنفیگریشن

```yaml
# Nav2 کنفیگریشن فائل
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "nav2_mppi_controller::MppiController"
      # MPPI کنٹرولر کے لیے پیرامیٹرز
      time_steps: 16
      control_horizon: 8
      vx_samples: 21
      vy_samples: 1
      wz_samples: 21
      rollouts: 1000
      dt: 0.05
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_bounds_x: [-1.0, 1.0]
      state_bounds_y: [-1.0, 1.0]
      state_bounds_theta: [-1.0, 1.0]
      control_bounds_vx: [-0.5, 0.5]
      control_bounds_vy: [-0.5, 0.5]
      control_bounds_wz: [-0.5, 0.5]
      Q_matrix: [10.0, 10.0, 1.0]
      R_matrix: [0.1, 0.1, 0.1]
      P_matrix: [0.025, 0.025, 0.025]
      control_delay: 2
      collision_cost: 1000.0
      goal_angle_cost: 1.0
      goal_dist_cost: 5.0
      xy_threshold: 0.1
      trans_stopped_velocity: 0.25
      rot_stopped_velocity: 0.25
      heading_lookahead_scale: 0.5
      inflation_cost_scaling_factor: 3.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.3  # انسان نما روبوٹ کا رداس

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

## انسان نما روبوٹ کے لیے نیوی گیشن

### انسان نما نیوی گیشن کی خصوصیات

1. **Bipedal Navigation**: دو پاؤں والی چلنے کے لیے مناسب
2. **Balance Constraints**: توازن کو برقرار رکھنے کے لیے پابندیاں
3. **Step Planning**: اقدامات کی منصوبہ بندی
4. **Terrain Adaptation**: زمین کی قسم کے مطابق ایڈجسٹمنٹ

### انسان نما نیوی گیشن کنٹرولر

```python
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
import numpy as np

class HumanoidNavigationController(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_controller')

        # نیوی گیشن کے لیے سبسکرائبرز اور پبلشرز
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # انسان نما نیوی گیشن کے لیے پیرامیٹرز
        self.step_length = 0.3  # میٹر
        self.step_width = 0.2   # میٹر
        self.step_height = 0.05 # میٹر
        self.balance_margin = 0.1 # توازن کے لیے مارجن

        # موجودہ راستہ اور ہدف
        self.current_path = None
        self.current_waypoint_idx = 0
        self.current_pose = None

        # نیوی گیشن کے لیے ٹائمر
        self.nav_timer = self.create_timer(0.1, self.navigation_callback)

    def path_callback(self, msg):
        """راستہ کو محفوظ کریں"""
        self.current_path = msg.poses
        self.current_waypoint_idx = 0

    def navigation_callback(self):
        """نیوی گیشن کنٹرول لوپ"""
        if self.current_path is not None and len(self.current_path) > 0:
            # موجودہ ہدف حاصل کریں
            target_pose = self.current_path[self.current_waypoint_idx].pose
            current_pose = self.get_current_pose()

            # انسان نما مخصوص نیوی گیشن کنٹرول
            cmd_vel = self.humanoid_navigation_control(current_pose, target_pose)

            # کمانڈ پبلش کریں
            self.cmd_vel_pub.publish(cmd_vel)

            # چیک کریں کہ آیا ہدف تک پہنچ گئے
            if self.is_at_waypoint(current_pose, target_pose):
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.current_path):
                    # تمام ہدف مکمل ہو گئے
                    self.stop_robot()

    def humanoid_navigation_control(self, current_pose, target_pose):
        """انسان نما مخصوص نیوی گیشن کنٹرول"""
        # پوزیشن کے درمیان فاصلہ حساب لگائیں
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # ہدف کی طرف جانے کے لیے زاویہ
        target_angle = np.arctan2(dy, dx)
        current_angle = self.get_current_yaw(current_pose)

        # زاویہ کی غلطی
        angle_error = self.normalize_angle(target_angle - current_angle)

        # کمانڈ تیار کریں
        cmd_vel = Twist()

        if distance > self.step_length:
            # راستہ میں جاری رکھیں
            cmd_vel.linear.x = min(distance, 0.3)  # 0.3 m/s تک
            cmd_vel.angular.z = max(min(angle_error * 2.0, 0.5), -0.5)  # 0.5 rad/s تک
        else:
            # قریب ہدف، آہستہ حرکت
            cmd_vel.linear.x = min(distance, 0.1)  # 0.1 m/s تک
            cmd_vel.angular.z = max(min(angle_error, 0.2), -0.2)

        # انسان نما مخصوص توازن کی پابندیاں
        cmd_vel = self.apply_balance_constraints(cmd_vel)

        return cmd_vel

    def apply_balance_constraints(self, cmd_vel):
        """توازن کی پابندیاں لگائیں"""
        # انسان نما روبوٹ کے لیے توازن کی پابندیاں
        max_linear_speed = 0.3  # m/s
        max_angular_speed = 0.4  # rad/s

        cmd_vel.linear.x = max(min(cmd_vel.linear.x, max_linear_speed), -max_linear_speed)
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, max_angular_speed), -max_angular_speed)

        return cmd_vel

    def is_at_waypoint(self, current_pose, target_pose, tolerance=0.2):
        """چیک کریں کہ آیا ہدف تک پہنچ گئے"""
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)
        return distance < tolerance

    def normalize_angle(self, angle):
        """زاویہ کو نارملائز کریں"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    def get_current_pose(self):
        """موجودہ پوزیشن حاصل کریں"""
        # اصل میں، آپ TF یا odometry سے پوزیشن حاصل کریں گے
        # یہ ایک نمونہ ہے
        return self.current_pose

    def get_current_yaw(self, pose):
        """کوائف سے yaw حاصل کریں"""
        # کوائف سے yaw زاویہ حاصل کریں
        qw = pose.orientation.w
        qz = pose.orientation.z
        return np.arctan2(2*(qw*qz), 1-2*(qz*qz))
```

## ٹریجکٹری پلاننگ

### ٹریجکٹری پلاننگ کیا ہے؟

ٹریجکٹری پلاننگ ایک ایسے راستے کو تیار کرنا ہے جس پر روبوٹ کو حرکت کرنا چاہیے تاکہ یہ ایک شروع کے پوائنٹ سے ہدف تک جا سکے۔

### ٹریجکٹری پلاننگ کے طریقے

1. **A* Algorithm**: ہیورسٹک کے ساتھ گراف سرچ
2. **RRT (Rapidly-exploring Random Trees)**: رینڈم ٹریز کے ساتھ ایکسپلوریشن
3. **Dijkstra's Algorithm**: سب سے چھوٹا راستہ
4. **Potential Fields**: طاقت کے میدان کے ذریعے رہنمائی

### RRT (Rapidly-exploring Random Trees)

```python
import numpy as np
import random

class RRTPlanner:
    def __init__(self, start, goal, bounds, obstacle_list):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds  # [(x_min, x_max), (y_min, y_max)]
        self.obstacles = obstacle_list
        self.nodes = [self.start]
        self.parent = {tuple(self.start): None}
        self.max_iter = 1000
        self.step_size = 0.1

    def plan(self):
        """راستہ تلاش کریں"""
        for i in range(self.max_iter):
            # رینڈم پوائنٹ نمونہ لیں
            rand_point = self.sample_random_point()

            # قریب ترین نوڈ تلاش کریں
            nearest_node = self.nearest_node(rand_point)

            # نوڈ کی طرف ایک قدم بڑھائیں
            new_node = self.steer(nearest_node, rand_point)

            # چیک کریں کہ کیا نیا نوڈ جائزہ ہے
            if self.is_valid(new_node):
                self.nodes.append(new_node)
                self.parent[tuple(new_node)] = tuple(nearest_node)

                # چیک کریں کہ کیا ہدف تک پہنچ گئے
                if np.linalg.norm(new_node - self.goal) < self.step_size:
                    return self.extract_path()

        # راستہ نہیں ملا
        return None

    def sample_random_point(self):
        """رینڈم پوائنٹ نمونہ لیں"""
        if random.random() < 0.05:  # 5% کی صورت میں ہدف کی طرف
            return self.goal
        else:
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            return np.array([x, y])

    def nearest_node(self, point):
        """قریب ترین نوڈ تلاش کریں"""
        distances = [np.linalg.norm(point - node) for node in self.nodes]
        idx = np.argmin(distances)
        return self.nodes[idx]

    def steer(self, from_node, to_point):
        """نقطہ کی طرف ایک قدم بڑھائیں"""
        direction = to_point - from_node
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_point
        else:
            direction_unit = direction / distance
            return from_node + direction_unit * self.step_size

    def is_valid(self, node):
        """چیک کریں کہ نوڈ جائزہ ہے"""
        # چیک کریں کہ نوڈ باؤنڈز کے اندر ہے
        if not (self.bounds[0][0] <= node[0] <= self.bounds[0][1] and
                self.bounds[1][0] <= node[1] <= self.bounds[1][1]):
            return False

        # چیک کریں کہ نوڈ رکاوٹوں کے اندر نہیں ہے
        for obs in self.obstacles:
            if self.point_in_obstacle(node, obs):
                return False

        return True

    def point_in_obstacle(self, point, obstacle):
        """چیک کریں کہ پوائنٹ رکاوٹ کے اندر ہے"""
        # یہاں مربع رکاوٹ کے لیے چیک
        x, y = point
        x_min, y_min, x_max, y_max = obstacle
        return x_min <= x <= x_max and y_min <= y <= y_max

    def extract_path(self):
        """راستہ نکالیں"""
        path = [self.goal]
        current = tuple(self.goal)

        while current in self.parent and self.parent[current] is not None:
            current = self.parent[current]
            path.append(np.array(current))

        path.reverse()
        return path
```

## جائزہ

SLAM اور نیوی گیشن انسان نما روبوٹکس کا ایک اہم حصہ ہے۔ VSLAM، Nav2 فریم ورک، اور ٹریجکٹری پلاننگ کے تصورات کو سمجھنا روبوٹ کو ناپا جانے والے ماحول میں نیوی گیٹ کرنے کے قابل بناتا ہے۔ انسان نما روبوٹ کے لیے مخصوص نیوی گیشن کنٹرولرز توازن اور بائی پیڈل چلنے کے مسائل کو حل کرتے ہیں۔

