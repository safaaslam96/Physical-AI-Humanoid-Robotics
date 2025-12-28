---
title: "چیپٹر 9: کمپیوٹر وژن اور تاثر"
sidebar_label: "چیپٹر 9: کمپیوٹر وژن اور تاثر"
---

# چیپٹر 9: کمپیوٹر وژن اور تاثر

## سیکھنے کے اہداف
- کمپیوٹر وژن کے بنیادی تصورات کو سمجھنا
- تصویر کی پروسیسنگ اور شناخت کے الگورتھم تیار کرنا
- ROS 2 میں سینسر فیوژن کا نفاذ
- انسان نما روبوٹ کے لیے تاثر کے نظام کو ڈیزائن کرنا

## کمپیوٹر وژن کی معرفت

### کمپیوٹر وژن کیا ہے؟

کمپیوٹر وژن کمپیوٹر سسٹم کے ذریعے تصاویر اور ویڈیوز کی تشریح کا مطالعہ ہے۔ یہ روبوٹکس میں اشیاء کی شناخت، نیوی گیشن، اور تعامل کے لیے اہم ہے۔

### کمپیوٹر وژن کے اطلاقات

1. **Object Detection**: اشیاء کو تصویر میں تلاش کرنا اور شناخت کرنا
2. **Object Tracking**: ٹریک میں اشیاء کی حرکت کو متعین کرنا
3. **Pose Estimation**: اشیاء یا انسانوں کے ڈھانچے کی پوزیشن اور اورینٹیشن کا تعین
4. **Scene Understanding**: ماحول کو سمجھنا اور نقشہ بنانا
5. **Visual SLAM**: وژول سیمولٹینیئس لوکلائزیشن اور میپنگ

## تصویر کی پروسیسنگ کے بنیادی تصورات

### تصویر کی نمائندگی

کمپیوٹر میں تصاویر کو پکسلز کے میٹرکس کے طور پر نمائندہ کیا جاتا ہے:

```
Gray Scale Image: M x N x 1 (0-255 values)
RGB Image: M x N x 3 (Red, Green, Blue channels)
```

### تصویر کی پروسیسنگ کے اوزار

1. **OpenCV**: کمپیوٹر وژن کے لیے کھلا ماخذ لائبریری
2. **PIL (Pillow)**: تصویر کی دیکھ بھال کے لیے
3. **NumPy**: تصویر کے ڈیٹا کو مینوپولیٹ کرنے کے لیے

### بنیادی تصویر کی پروسیسنگ کے کام

```python
import cv2
import numpy as np

def image_processing_example(image_path):
    """تصویر کی پروسیسنگ کے بنیادی کام"""

    # تصویر لوڈ کریں
    img = cv2.imread(image_path)

    # گرے اسکیل میں تبدیل کریں
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # بلر کریں (نوائز کو کم کرنے کے لیے)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # کنارے ڈیٹکٹ کریں
    edges = cv2.Canny(blurred, 50, 150)

    # کنٹورز تلاش کریں
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # کنٹورز ڈرا کریں
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)

    return img_with_contours
```

## اشیاء کی شناخت

### اشیاء کی شناخت کی مетодز

1. **Classical Methods**: Haar Cascades, HOG, SIFT
2. **Deep Learning Methods**: CNN, YOLO, SSD, Faster R-CNN

### Haar Cascades

Haar Cascades کلاسیکل کمپیوٹر وژن کا طریقہ ہے چہرے یا اشیاء کو تلاش کرنے کے لیے:

```python
import cv2

def face_detection_haar_cascade():
    """چہرے کی شناخت کے لیے Haar Cascade"""

    # Haar cascade لوڈ کریں
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # تصویر لوڈ کریں
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # چہرے تلاش کریں
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # چہرے کے گرد ڈبے ڈالیں
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img
```

### CNN-based Object Detection

CNNs (Convolutional Neural Networks) اشیاء کی شناخت کے لیے زیادہ مؤثر ہیں:

```python
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectDetector:
    def __init__(self):
        # پہلے سے تربیت یافتہ ماڈل لوڈ کریں
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        # تصویر کو تبدیل کرنے کے لیے ٹرانسفارم
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect_objects(self, image):
        """تصویر میں اشیاء کو تلاش کریں"""

        # تصویر کو ٹرانسفارم کریں
        img_tensor = self.transform(image).unsqueeze(0)

        # ماڈل کے ذریعے ڈیٹکشن
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # نتائج کو فلٹر کریں
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        # صرف اعلی یقین والے نتائج کو واپس کریں
        high_confidence = scores > 0.5

        return {
            'boxes': boxes[high_confidence],
            'labels': labels[high_confidence],
            'scores': scores[high_confidence]
        }
```

## ROS 2 میں کمپیوٹر وژن

### ROS 2 میں تصویر کی مساجنگ

ROS 2 میں تصاویر sensor_msgs/Image میسج کے ذریعے بھیجی جاتی ہیں:

```python
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # تصویر کو پروسیس کرنے کے لیے CvBridge
        self.cv_bridge = CvBridge()

        # تصویر کو سبسکرائب کریں
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # پروسیسڈ تصویر کو پبلش کریں
        self.result_pub = self.create_publisher(Image, '/vision/result', 10)

    def image_callback(self, msg):
        """تصویر کے مسج کو ہینڈل کریں"""

        # تصویر کو OpenCV فارمیٹ میں تبدیل کریں
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # تصویر کو پروسیس کریں
        processed_image = self.process_image(cv_image)

        # نتیجہ پبلش کریں
        result_msg = self.cv_bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
        self.result_pub.publish(result_msg)

    def process_image(self, image):
        """تصویر کو پروسیس کریں"""
        # یہاں کمپیوٹر وژن الگورتھم لگائیں
        # اشیاء کی شناخت، کنارے ڈیٹکشن، وغیرہ

        # مثال کے طور پر، ہم تصویر کو بلر کریں گے
        processed = cv2.GaussianBlur(image, (15, 15), 0)
        return processed
```

## سینسر فیوژن

### سینسر فیوژن کیا ہے؟

سینسر فیوژن متعدد سینسرز کے ڈیٹا کو ضم کرنا تاکہ زیادہ درست اور قابل اعتماد معلومات حاصل کی جا سکیں۔

### ROS 2 میں سینسر فیوژن

```python
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # مختلف سینسرز کے لیے سبسکرائبرز
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # ضم شدہ نتائج کے لیے پبلشر
        self.fused_pub = self.create_publisher(PoseWithCovarianceStamped, '/fused_pose', 10)

        # سینسر ڈیٹا کو ذخیرہ کرنے کے لیے
        self.image_data = None
        self.laser_data = None
        self.imu_data = None

        # فیوژن الگورتھم کے لیے ٹائمر
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)

    def image_callback(self, msg):
        """تصویر کا ڈیٹا محفوظ کریں"""
        self.image_data = msg

    def laser_callback(self, msg):
        """لیزر سکین کا ڈیٹا محفوظ کریں"""
        self.laser_data = msg

    def imu_callback(self, msg):
        """IMU کا ڈیٹا محفوظ کریں"""
        self.imu_data = msg

    def fusion_callback(self):
        """سینسر ڈیٹا کو ضم کریں"""
        if self.image_data and self.laser_data and self.imu_data:
            # سینسر ڈیٹا کو ضم کریں
            fused_result = self.perform_sensor_fusion()

            # نتیجہ پبلش کریں
            self.fused_pub.publish(fused_result)

    def perform_sensor_fusion(self):
        """سینسر ڈیٹا کو ضم کریں"""
        # یہاں سینسر فیوژن الگورتھم نافذ کریں
        # مثال کے طور پر، Kalman Filter یا Particle Filter

        # ہم ایک سادہ مثال کا استعمال کرتے ہیں
        fused_pose = PoseWithCovarianceStamped()
        # ضم شدہ پوز کا حساب لگائیں
        return fused_pose
```

## SLAM (Simultaneous Localization and Mapping)

### SLAM کیا ہے؟

SLAM (Simultaneous Localization and Mapping) ایک الگورتھم ہے جو روبوٹ کو اس کے ماحول کو ناپا جانے والے ماحول میں اس کی مقام کو ساتھ ساتھ نقشہ بناتے ہوئے متعین کرنے کی اجازت دیتا ہے۔

### ROS 2 میں SLAM

```python
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np

class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # سینسرز کے لیے سبسکرائبرز
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # SLAM کا نتیجہ (میپ) پبلش کریں
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # SLAM الگورتھم کے لیے
        self.map_resolution = 0.05  # میٹر/سیل
        self.map_size = (100, 100)  # سیلز میں
        self.current_map = np.zeros(self.map_size)
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta

    def laser_callback(self, msg):
        """لیزر سکین کا استعمال کرکے SLAM کریں"""
        # لیزر ڈیٹا کو استعمال کرکے میپ اپ ڈیٹ کریں
        self.update_map_from_laser(msg)

    def image_callback(self, msg):
        """تصویر کا استعمال کرکے SLAM کو بہتر کریں"""
        # تصویر کے ڈیٹا کو استعمال کرکے میپ کو بہتر کریں
        # مثال کے طور پر، وژول SLAM
        pass

    def update_map_from_laser(self, laser_scan):
        """لیزر سکین سے میپ اپ ڈیٹ کریں"""
        # لیزر سکین کے ڈیٹا کو استعمال کرکے میپ کو اپ ڈیٹ کریں
        # یہاں ایک سادہ مثال ہے
        for i, range_val in enumerate(laser_scan.ranges):
            if range_val > laser_scan.range_min and range_val < laser_scan.range_max:
                # رینج کے مطابق زاویہ حساب لگائیں
                angle = laser_scan.angle_min + i * laser_scan.angle_increment

                # ڈیٹکٹ کی گئی چیز کی پوزیشن کا حساب لگائیں
                x = self.robot_pose[0] + range_val * np.cos(self.robot_pose[2] + angle)
                y = self.robot_pose[1] + range_val * np.sin(self.robot_pose[2] + angle)

                # میپ میں اس مقام کو اپ ڈیٹ کریں
                grid_x = int((x - self.map_size[0]/2 * self.map_resolution) / self.map_resolution)
                grid_y = int((y - self.map_size[1]/2 * self.map_resolution) / self.map_resolution)

                if 0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]:
                    self.current_map[grid_x, grid_y] = 100  # 100 = wall/obstacle
```

## انسان نما روبوٹ کے لیے وژن سسٹم

### انسان نما وژن سسٹم کی ضروریات

1. **Real-time Processing**: ریل ٹائم تصویر کی پروسیسنگ
2. **Multiple Cameras**: ڈوبل آئیں، ڈیپتھ کیمرہ
3. **Object Recognition**: اشیاء کی قدرتی شناخت
4. **Human Interaction**: انسانوں کی شناخت اور تعامل
5. **Navigation**: راستہ تلاش کرنے اور رکاوٹوں کو ڈیٹکٹ کرنے کے لیے

### انسان نما وژن سسٹم کا ڈیزائن

```python
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class HumanoidVisionSystem(Node):
    def __init__(self):
        super().__init__('humanoid_vision_system')

        # CvBridge کو شروع کریں
        self.cv_bridge = CvBridge()

        # کیمرہ سبسکرائبرز
        self.left_eye_sub = self.create_subscription(Image, '/left_camera/image_raw', self.left_eye_callback, 10)
        self.right_eye_sub = self.create_subscription(Image, '/right_camera/image_raw', self.right_eye_callback, 10)
        self.depth_camera_sub = self.create_subscription(Image, '/depth_camera/image_raw', self.depth_callback, 10)

        # نتائج کے لیے پبلشرز
        self.object_detection_pub = self.create_publisher(String, '/detected_objects', 10)
        self.human_detection_pub = self.create_publisher(String, '/detected_humans', 10)
        self.navigation_pub = self.create_publisher(String, '/navigation_targets', 10)

        # تصاویر کو ذخیرہ کرنے کے لیے
        self.left_image = None
        self.right_image = None
        self.depth_image = None

    def left_eye_callback(self, msg):
        """بائیں آنکھ کی تصویر کو ہینڈل کریں"""
        self.left_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_vision_data()

    def right_eye_callback(self, msg):
        """دائیں آنکھ کی تصویر کو ہینڈل کریں"""
        self.right_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_vision_data()

    def depth_callback(self, msg):
        """ڈیپتھ کیمرہ کا ڈیٹا ہینڈل کریں"""
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_vision_data()

    def process_vision_data(self):
        """وژن ڈیٹا کو پروسیس کریں"""
        if self.left_image is not None and self.right_image is not None:
            # اشیاء کی شناخت
            objects = self.detect_objects(self.left_image)
            if objects:
                obj_msg = String()
                obj_msg.data = str(objects)
                self.object_detection_pub.publish(obj_msg)

            # انسانوں کی شناخت
            humans = self.detect_humans(self.left_image)
            if humans:
                human_msg = String()
                human_msg.data = str(humans)
                self.human_detection_pub.publish(human_msg)

            # نیوی گیشن ٹارگٹس
            targets = self.find_navigation_targets(self.left_image, self.depth_image)
            if targets:
                nav_msg = String()
                nav_msg.data = str(targets)
                self.navigation_pub.publish(nav_msg)

    def detect_objects(self, image):
        """تصویر میں اشیاء کو تلاش کریں"""
        # اشیاء کی شناخت کا الگورتھم
        # یہاں ہم ایک سادہ کنارا ڈیٹکشن کا استعمال کرتے ہیں
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # چھوٹے کنٹورز کو فلٹر کریں
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({'x': x, 'y': y, 'width': w, 'height': h})

        return objects

    def detect_humans(self, image):
        """تصویر میں انسانوں کو تلاش کریں"""
        # ہیومن ڈیٹکشن کا الگورتھم
        # ہم OpenCV کا HOG ڈیسکرپٹر استعمال کرتے ہیں
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        boxes, weights = hog.detectMultiScale(image, winStride=(8,8))
        humans = [{'x': x, 'y': y, 'width': w, 'height': h} for x, y, w, h in boxes]

        return humans

    def find_navigation_targets(self, image, depth_image):
        """نیوی گیشن کے ٹارگٹس تلاش کریں"""
        # رکاوٹوں اور راستوں کو تلاش کریں
        if depth_image is not None:
            # ڈیپتھ امیج کو استعمال کرکے رکاوٹوں کو ڈیٹکٹ کریں
            avg_depth = np.mean(depth_image)
            safe_paths = avg_depth > 0.5  # 0.5 میٹر سے زیادہ گہرائی والے علاقے محفوظ ہیں

            return {'safe_paths': safe_paths, 'avg_depth': avg_depth}

        return None
```

## جائزہ

کمپیوٹر وژن اور تاثر انسان نما روبوٹکس کا ایک اہم حصہ ہے۔ تصویر کی پروسیسنگ، اشیاء کی شناخت، SLAM، اور سینسر فیوژن کے تصورات کو سمجھنا روبوٹ کو اس کے ماحول کو سمجھنے اور اس کے ساتھ مؤثر طریقے سے تعامل کرنے کے قابل بناتا ہے۔ ROS 2 کے ساتھ کمپیوٹر وژن کا نفاذ روبوٹک سسٹم کے انضمام کے لیے ضروری ہے۔

