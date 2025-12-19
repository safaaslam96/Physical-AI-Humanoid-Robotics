---
sidebar_position: 10
title: "Chapter 10: Isaac ROS and Hardware-Accelerated Perception"
---

# Chapter 10: Isaac ROS and Hardware-Accelerated Perception

## Learning Objectives
- Understand Isaac ROS for hardware-accelerated perception
- Implement Visual SLAM (VSLAM) systems for humanoid robots
- Develop advanced perception techniques using GPU acceleration
- Integrate computer vision with ROS 2 for humanoid robotics

## Introduction to Isaac ROS

Isaac ROS is NVIDIA's collection of GPU-accelerated packages that seamlessly integrate with ROS 2, providing high-performance perception and navigation capabilities. These packages leverage NVIDIA's CUDA cores and Tensor cores to accelerate computationally intensive tasks like visual SLAM, stereo vision, and deep learning inference.

### Key Isaac ROS Packages

- **Isaac ROS Visual SLAM**: GPU-accelerated simultaneous localization and mapping
- **Isaac ROS Stereo**: Hardware-accelerated stereo vision processing
- **Isaac ROS Image Pipeline**: GPU-accelerated image processing pipeline
- **Isaac ROS DNN**: Deep learning inference with TensorRT optimization
- **Isaac ROS Navigation**: GPU-accelerated navigation stack
- **Isaac ROS Manipulation**: Perception-driven manipulation

### Hardware Requirements

To fully utilize Isaac ROS:

- **GPU**: NVIDIA RTX 3080/4080, RTX A4000/A5000/A6000, or better
- **CUDA**: Version 11.8 or later
- **TensorRT**: For deep learning acceleration
- **OpenCV**: GPU-accelerated computer vision
- **NVIDIA Drivers**: Latest Game Ready or Studio drivers

## Isaac ROS Visual SLAM (VSLAM)

### Overview of Visual SLAM

Visual SLAM (Simultaneous Localization and Mapping) is crucial for humanoid robots to navigate unknown environments. Isaac ROS VSLAM provides GPU-accelerated processing for real-time performance.

### Setting Up Isaac ROS VSLAM

```python
# Isaac ROS VSLAM Node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import torch
import os

class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vslam_node')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'vslam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'vslam/pose', 10)

        # Subscribers
        self.left_image_sub = self.create_subscription(
            Image, 'stereo/left/image_rect_color', self.left_image_callback, 10)
        self.right_image_sub = self.create_subscription(
            Image, 'stereo/right/image_rect_color', self.right_image_callback, 10)
        self.left_info_sub = self.create_subscription(
            CameraInfo, 'stereo/left/camera_info', self.left_info_callback, 10)
        self.right_info_sub = self.create_subscription(
            CameraInfo, 'stereo/right/camera_info', self.right_info_callback, 10)

        # Initialize components
        self.bridge = CvBridge()
        self.vslam_system = self.initialize_vslam_system()

        # Camera parameters
        self.left_camera_matrix = None
        self.right_camera_matrix = None
        self.baseline = None

        # Frame counter and timing
        self.frame_count = 0
        self.last_process_time = self.get_clock().now()

        self.get_logger().info('Isaac ROS VSLAM Node initialized')

    def initialize_vslam_system(self):
        """Initialize GPU-accelerated VSLAM system"""
        try:
            # Import Isaac ROS VSLAM components
            from isaac_ros_visual_slam import VisualSLAMNode
            vslam = VisualSLAMNode()
            return vslam
        except ImportError:
            self.get_logger().warn('Isaac ROS VSLAM not available, using fallback')
            return self.fallback_vslam()

    def fallback_vslam(self):
        """Fallback CPU-based VSLAM implementation"""
        class FallbackVSLAM:
            def __init__(self):
                self.position = np.array([0.0, 0.0, 0.0])
                self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # w, x, y, z
                self.keyframes = []

            def process_stereo_pair(self, left_img, right_img, camera_matrix, baseline):
                """Process stereo image pair for depth and pose estimation"""
                # Simplified stereo processing
                # In real implementation, this would use ORB-SLAM, LSD-SLAM, etc.
                depth_map = self.compute_depth_map(left_img, right_img)
                pose_change = self.estimate_motion(left_img, right_img)

                # Update position and orientation
                self.position += pose_change[:3]
                # Simplified orientation update
                self.orientation = self.update_orientation(self.orientation, pose_change[3:])

                return self.position, self.orientation, depth_map

            def compute_depth_map(self, left, right):
                """Compute depth map from stereo pair"""
                # Using OpenCV stereo matcher (GPU accelerated if available)
                stereo = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=128,
                    blockSize=5,
                    P1=8 * 3 * 5**2,
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=15,
                    speckleWindowSize=0,
                    speckleRange=2,
                    preFilterCap=63,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )

                disparity = stereo.compute(left, right).astype(np.float32) / 16.0
                # Convert disparity to depth
                depth_map = baseline * camera_matrix[0, 0] / (disparity + 1e-6)
                return depth_map

            def estimate_motion(self, prev_img, curr_img):
                """Estimate motion between frames"""
                # Feature-based motion estimation
                # Extract ORB features
                orb = cv2.ORB_create(nfeatures=1000)
                kp1, des1 = orb.detectAndCompute(prev_img, None)
                kp2, des2 = orb.detectAndCompute(curr_img, None)

                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)

                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) >= 10:
                    # Extract matched points
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    # Compute fundamental matrix
                    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 4, 0.999)
                    F = F/F[2,2]  # Normalize

                    # Extract essential matrix (assuming known camera intrinsics)
                    E = camera_matrix.T @ F @ camera_matrix

                    # Decompose essential matrix to get rotation and translation
                    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)

                    # Convert to 6DOF pose change
                    rvec, _ = cv2.Rodrigues(R)
                    pose_change = np.concatenate([t.flatten(), rvec.flatten()])
                    return pose_change
                else:
                    return np.zeros(6)  # No significant motion

            def update_orientation(self, current_quat, angular_change):
                """Update orientation quaternion with angular change"""
                # Convert angular change to quaternion
                angle = np.linalg.norm(angular_change)
                if angle > 1e-6:
                    axis = angular_change / angle
                    dq = np.array([
                        np.cos(angle/2),
                        np.sin(angle/2) * axis[0],
                        np.sin(angle/2) * axis[1],
                        np.sin(angle/2) * axis[2]
                    ])

                    # Multiply quaternions
                    w1, x1, y1, z1 = current_quat
                    w2, x2, y2, z2 = dq

                    new_quat = np.array([
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2
                    ])
                    return new_quat / np.linalg.norm(new_quat)
                else:
                    return current_quat

        return FallbackVSLAM()

    def left_info_callback(self, msg):
        """Handle left camera info"""
        self.left_camera_matrix = np.array(msg.k).reshape(3, 3)

    def right_info_callback(self, msg):
        """Handle right camera info"""
        self.right_camera_matrix = np.array(msg.k).reshape(3, 3)
        # Extract baseline from projection matrix (P[3]/fx)
        if hasattr(msg, 'p'):
            p = np.array(msg.p).reshape(3, 4)
            self.baseline = abs(p[0, 3] / p[0, 0])  # Assuming P[0,3] = -fx*T

    def left_image_callback(self, msg):
        """Process left camera image"""
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing left image: {e}')

    def right_image_callback(self, msg):
        """Process right camera image"""
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.process_stereo_pair()
        except Exception as e:
            self.get_logger().error(f'Error processing right image: {e}')

    def process_stereo_pair(self):
        """Process stereo pair for VSLAM"""
        if not hasattr(self, 'left_image') or not hasattr(self, 'right_image'):
            return

        if self.left_camera_matrix is None or self.baseline is None:
            return

        # Process with VSLAM system
        position, orientation, depth_map = self.vslam_system.process_stereo_pair(
            self.left_image, self.right_image, self.left_camera_matrix, self.baseline
        )

        # Publish odometry
        self.publish_odometry(position, orientation)

        # Publish pose
        self.publish_pose(position, orientation)

        self.frame_count += 1
        current_time = self.get_clock().now()
        dt = (current_time - self.last_process_time).nanoseconds / 1e9
        self.last_process_time = current_time

        if self.frame_count % 30 == 0:  # Log every 30 frames
            self.get_logger().info(f'Processed frame {self.frame_count}, '
                                 f'Position: {position}, FPS: {1.0/dt if dt > 0 else 0:.1f}')

    def publish_odometry(self, position, orientation):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Position
        odom_msg.pose.pose.position.x = float(position[0])
        odom_msg.pose.pose.position.y = float(position[1])
        odom_msg.pose.pose.position.z = float(position[2])

        # Orientation
        odom_msg.pose.pose.orientation.w = float(orientation[0])
        odom_msg.pose.pose.orientation.x = float(orientation[1])
        odom_msg.pose.pose.orientation.y = float(orientation[2])
        odom_msg.pose.pose.orientation.z = float(orientation[3])

        # Publish
        self.odom_pub.publish(odom_msg)

    def publish_pose(self, position, orientation):
        """Publish pose message"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        pose_msg.pose.orientation.w = float(orientation[0])
        pose_msg.pose.orientation.x = float(orientation[1])
        pose_msg.pose.orientation.y = float(orientation[2])
        pose_msg.pose.orientation.z = float(orientation[3])

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacVSLAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Stereo Processing

```python
# Isaac ROS Stereo Processing Node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import struct

class IsaacStereoNode(Node):
    def __init__(self):
        super().__init__('isaac_stereo_node')

        # Publishers
        self.disparity_pub = self.create_publisher(DisparityImage, 'stereo/disparity', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'stereo/pointcloud', 10)

        # Subscribers using message filters for synchronization
        self.left_sub = Subscriber(self, Image, 'stereo/left/image_rect')
        self.right_sub = Subscriber(self, Image, 'stereo/right/image_rect')
        self.left_info_sub = Subscriber(self, CameraInfo, 'stereo/left/camera_info')
        self.right_info_sub = Subscriber(self, CameraInfo, 'stereo/right/camera_info')

        # Synchronize stereo pairs
        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.left_info_sub, self.right_info_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.stereo_callback)

        # Initialize components
        self.bridge = CvBridge()
        self.stereo_matcher = self.initialize_stereo_matcher()
        self.q_matrix = None

        self.get_logger().info('Isaac ROS Stereo Node initialized')

    def initialize_stereo_matcher(self):
        """Initialize GPU-accelerated stereo matcher"""
        try:
            # Use CUDA-accelerated stereo matcher if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Initialize CUDA stereo matcher
                left_matcher = cv2.cuda.StereoBM_create(numDisparities=128, blockSize=21)
                right_matcher = cv2.cuda.StereoBM_create(numDisparities=128, blockSize=21)
                return (left_matcher, right_matcher, True)
            else:
                # Fallback to CPU stereo matcher
                stereo = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=128,
                    blockSize=5,
                    P1=8 * 3 * 5**2,
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=15,
                    speckleWindowSize=0,
                    speckleRange=2,
                    preFilterCap=63,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )
                return (stereo, None, False)
        except Exception as e:
            self.get_logger().warn(f'Could not initialize GPU stereo: {e}')
            # Fallback stereo matcher
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=128,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=0,
                speckleRange=2,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            return (stereo, None, False)

    def stereo_callback(self, left_msg, right_msg, left_info_msg, right_info_msg):
        """Process synchronized stereo pair"""
        try:
            # Convert ROS images to OpenCV
            left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')

            # Compute disparity
            disparity = self.compute_disparity(left_img, right_img)

            # Compute Q matrix from camera info
            self.q_matrix = self.compute_q_matrix(left_info_msg, right_info_msg)

            # Publish disparity image
            self.publish_disparity(disparity, left_msg.header)

            # Generate and publish point cloud
            pointcloud = self.generate_pointcloud(disparity, left_img)
            self.publish_pointcloud(pointcloud, left_msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in stereo callback: {e}')

    def compute_disparity(self, left_img, right_img):
        """Compute disparity map from stereo pair"""
        left_matcher, right_matcher, is_cuda = self.stereo_matcher

        if is_cuda:
            # GPU processing
            left_cuda = cv2.cuda_GpuMat()
            right_cuda = cv2.cuda_GpuMat()
            left_cuda.upload(left_img)
            right_cuda.upload(right_img)

            # Compute disparity using CUDA
            disp_left = left_matcher.compute(left_cuda, right_cuda)
            disp_right = right_matcher.compute(right_cuda, left_cuda)

            # Convert back to CPU
            disparity = disp_left.download().astype(np.float32) / 16.0
        else:
            # CPU processing
            disparity = left_matcher.compute(left_img, right_img).astype(np.float32) / 16.0

        return disparity

    def compute_q_matrix(self, left_info, right_info):
        """Compute Q matrix for 3D reconstruction"""
        # Extract camera parameters
        fx = left_info.k[0]  # Focal length x
        fy = left_info.k[4]  # Focal length y
        cx = left_info.k[2]  # Principal point x
        cy = left_info.k[5]  # Principal point y

        # Baseline (distance between cameras)
        # Extract from projection matrix P[3] = -fx * Tx where Tx is baseline
        baseline = abs(right_info.p[3] / fx) if right_info.p else 0.1

        # Create Q matrix for reprojectImageTo3D
        Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1/baseline, 0]
        ], dtype=np.float32)

        return Q

    def generate_pointcloud(self, disparity, left_img):
        """Generate 3D point cloud from disparity map"""
        if self.q_matrix is None:
            return None

        # Reproject disparity to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.q_matrix)

        # Create colored point cloud
        points = []
        colors = []

        height, width = disparity.shape
        for v in range(height):
            for u in range(width):
                if disparity[v, u] > 0:  # Valid disparity
                    x, y, z = points_3d[v, u]
                    if z > 0 and z < 10:  # Filter out invalid depths
                        points.append([x, y, z])
                        # Get color from left image
                        b, g, r = left_img[v, u], left_img[v, u], left_img[v, u]
                        colors.append([r, g, b])

        return np.array(points), np.array(colors)

    def publish_disparity(self, disparity, header):
        """Publish disparity image"""
        # Convert disparity to 8-bit for visualization
        disp_8bit = cv2.convertScaleAbs(disparity, alpha=255.0/128.0)

        # Create disparity message
        disp_msg = DisparityImage()
        disp_msg.header = header
        disp_msg.image = self.bridge.cv2_to_imgmsg(disp_8bit, encoding='mono8')
        disp_msg.f = float(self.q_matrix[2, 3]) if self.q_matrix is not None else 1.0
        disp_msg.T = float(abs(self.q_matrix[3, 2])) if self.q_matrix is not None else 0.1

        self.disparity_pub.publish(disp_msg)

    def publish_pointcloud(self, pointcloud_data, header):
        """Publish 3D point cloud"""
        if pointcloud_data is None:
            return

        points, colors = pointcloud_data

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        # Pack points and colors into binary format
        points_list = []
        for i in range(len(points)):
            # Pack RGB as single float
            rgb = struct.unpack('I', struct.pack('BBBB', int(colors[i][2]),
                                                int(colors[i][1]),
                                                int(colors[i][0]), 0))[0]
            points_list.append(struct.pack('fffI', points[i][0], points[i][1], points[i][2], rgb))

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16  # 3*4 bytes for xyz + 4 bytes for rgb
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.data = b''.join(points_list)
        cloud_msg.is_dense = True

        self.pointcloud_pub.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacStereoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Perception Techniques

### Isaac ROS DNN Inference

```python
# Isaac ROS Deep Learning Node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
import numpy as np
import time

class IsaacDNNNode(Node):
    def __init__(self):
        super().__init__('isaac_dnn_node')

        # Publishers
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        # Initialize components
        self.bridge = CvBridge()
        self.model = self.load_model()
        self.transform = self.get_transform()

        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()

        self.get_logger().info('Isaac ROS DNN Node initialized')

    def load_model(self):
        """Load pre-trained model with TensorRT optimization"""
        try:
            # Load model from Isaac ROS DNN package or custom model
            import torchvision
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                self.get_logger().info('Model loaded on GPU')
            else:
                self.get_logger().warn('CUDA not available, using CPU')

            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading model: {e}')
            return None

    def get_transform(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),  # Resize for model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def image_callback(self, msg):
        """Process incoming image with DNN"""
        if self.model is None:
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Preprocess image
            input_tensor = self.transform(rgb_image).unsqueeze(0)

            # Move to GPU if available
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # For accurate timing

                predictions = self.model(input_tensor)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # For accurate timing

            inference_time = time.time() - start_time

            # Process predictions
            detections = self.process_predictions(predictions, cv_image.shape, msg.header)

            # Publish detections
            self.detection_pub.publish(detections)

            # Performance logging
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                avg_fps = self.frame_count / (time.time() - self.start_time)
                self.get_logger().info(f'Processed {self.frame_count} frames, '
                                     f'Avg FPS: {avg_fps:.2f}, '
                                     f'Inference time: {inference_time*1000:.2f}ms')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_predictions(self, predictions, image_shape, header):
        """Process model predictions into ROS format"""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        # Get predictions for first image in batch (usually batch size = 1)
        pred = predictions[0]

        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        height, width = image_shape[:2]

        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                detection = Detection2D()
                detection.header = header

                # Convert box coordinates to center + size format
                x1, y1, x2, y2 = boxes[i]
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                size_x = x2 - x1
                size_y = y2 - y1

                detection.bbox.center.x = float(center_x)
                detection.bbox.center.y = float(center_y)
                detection.bbox.size_x = float(size_x)
                detection.bbox.size_y = float(size_y)

                # Add hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = int(labels[i])
                hypothesis.score = float(scores[i])
                detection.results.append(hypothesis)

                detections_msg.detections.append(detection)

        return detections_msg

def main(args=None):
    rclpy.init(args=args)
    node = IsaacDNNNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Navigation with Perception

```python
# Isaac ROS Navigation Node with Perception
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformListener, Buffer
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class IsaacNavigationNode(Node):
    def __init__(self):
        super().__init__('isaac_navigation_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.path_pub = self.create_publisher(Path, 'local_plan', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'navigation_markers', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(Image, 'camera/image_raw', self.camera_callback, 10)

        # TF listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize components
        self.bridge = CvBridge()
        self.local_planner = LocalPlanner()
        self.obstacle_detector = ObstacleDetector()
        self.path_planner = PathPlanner()

        # Navigation state
        self.robot_pose = None
        self.goal_pose = None
        self.obstacles = []
        self.path = []

        # Navigation parameters
        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.safety_distance = 0.5

        # Timer for navigation loop
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

        self.get_logger().info('Isaac ROS Navigation Node initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        try:
            # Convert scan to obstacle points
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            ranges = np.array(msg.ranges)

            # Filter valid ranges
            valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
            valid_angles = angles[valid_mask]
            valid_ranges = ranges[valid_mask]

            # Convert to Cartesian coordinates relative to robot
            x_points = valid_ranges * np.cos(valid_angles)
            y_points = valid_ranges * np.sin(valid_angles)

            # Transform to global coordinates
            if self.robot_pose is not None:
                robot_x, robot_y, robot_yaw = self.get_robot_state()
                cos_yaw = np.cos(robot_yaw)
                sin_yaw = np.sin(robot_yaw)

                global_x = robot_x + x_points * cos_yaw - y_points * sin_yaw
                global_y = robot_y + x_points * sin_yaw + y_points * cos_yaw

                self.obstacles = np.column_stack((global_x, global_y))

        except Exception as e:
            self.get_logger().error(f'Error processing scan: {e}')

    def camera_callback(self, msg):
        """Process camera data for visual obstacle detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect obstacles in image
            visual_obstacles = self.obstacle_detector.detect_obstacles(cv_image)

            # Convert image coordinates to world coordinates
            world_obstacles = self.camera_to_world(visual_obstacles)

            # Merge with LIDAR obstacles
            if len(self.obstacles) > 0 and len(world_obstacles) > 0:
                self.obstacles = np.vstack((self.obstacles, world_obstacles))
            elif len(world_obstacles) > 0:
                self.obstacles = world_obstacles

        except Exception as e:
            self.get_logger().error(f'Error processing camera: {e}')

    def get_robot_state(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            # Convert quaternion to yaw
            quat = transform.transform.rotation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            yaw = r.as_euler('xyz')[2]  # Only yaw for 2D navigation

            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(f'Could not get robot transform: {e}')
            return 0.0, 0.0, 0.0

    def navigation_loop(self):
        """Main navigation loop"""
        if self.goal_pose is None:
            return

        # Get current robot state
        robot_x, robot_y, robot_yaw = self.get_robot_state()

        # Plan path to goal
        self.path = self.path_planner.plan_path(
            (robot_x, robot_y),
            (self.goal_pose.pose.position.x, self.goal_pose.pose.position.y),
            self.obstacles
        )

        # Follow path using local planner
        velocity_cmd = self.local_planner.compute_velocity(
            (robot_x, robot_y, robot_yaw),
            self.path,
            self.obstacles
        )

        # Publish velocity command
        cmd_msg = Twist()
        cmd_msg.linear.x = velocity_cmd[0]
        cmd_msg.angular.z = velocity_cmd[1]
        self.cmd_vel_pub.publish(cmd_msg)

        # Publish path for visualization
        self.publish_path()

    def camera_to_world(self, image_obstacles):
        """Convert image coordinates to world coordinates"""
        # This would use camera intrinsics and robot pose
        # Simplified implementation
        world_coords = []

        for obs in image_obstacles:
            # Convert image pixel to world coordinates using camera model
            # This requires camera calibration and robot pose
            world_x = obs[0]  # Placeholder
            world_y = obs[1]  # Placeholder
            world_coords.append([world_x, world_y])

        return np.array(world_coords)

    def publish_path(self):
        """Publish navigation path for visualization"""
        if len(self.path) == 0:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for point in self.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

class LocalPlanner:
    """Local path following planner"""
    def __init__(self):
        self.lookahead_distance = 1.0
        self.max_linear_speed = 0.5
        self.max_angular_speed = 0.5

    def compute_velocity(self, robot_state, path, obstacles):
        """Compute velocity command to follow path"""
        if len(path) < 2:
            return [0.0, 0.0]  # Stop if no path

        robot_x, robot_y, robot_yaw = robot_state

        # Find closest point on path
        closest_idx = 0
        min_dist = float('inf')
        for i, (x, y) in enumerate(path):
            dist = np.sqrt((x - robot_x)**2 + (y - robot_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Find lookahead point
        lookahead_point = None
        for i in range(closest_idx, len(path)):
            x, y = path[i]
            dist = np.sqrt((x - robot_x)**2 + (y - robot_y)**2)
            if dist >= self.lookahead_distance:
                lookahead_point = (x, y)
                break

        if lookahead_point is None:
            # Use last point if no point is far enough
            lookahead_point = path[-1]

        # Calculate desired direction
        dx = lookahead_point[0] - robot_x
        dy = lookahead_point[1] - robot_y
        desired_angle = np.arctan2(dy, dx)

        # Calculate angle error
        angle_error = desired_angle - robot_yaw
        # Normalize angle to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # Simple proportional controller
        angular_vel = max(-self.max_angular_speed,
                         min(self.max_angular_speed, 2.0 * angle_error))

        # Calculate linear velocity based on angular error
        linear_vel = self.max_linear_speed * max(0, 1 - abs(angle_error) / np.pi)

        # Check for obstacles
        if self.check_obstacles(robot_state, obstacles):
            linear_vel = 0.0  # Stop if obstacle ahead

        return [linear_vel, angular_vel]

    def check_obstacles(self, robot_state, obstacles):
        """Check if there are obstacles in the robot's path"""
        if len(obstacles) == 0:
            return False

        robot_x, robot_y, robot_yaw = robot_state

        # Check obstacles in front of robot (within cone)
        for obs_x, obs_y in obstacles:
            # Calculate relative position
            rel_x = obs_x - robot_x
            rel_y = obs_y - robot_y

            # Calculate distance and angle relative to robot heading
            distance = np.sqrt(rel_x**2 + rel_y**2)
            angle = np.arctan2(rel_y, rel_x) - robot_yaw

            # Normalize angle
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi

            # Check if obstacle is in front of robot (within 60 degrees cone)
            if distance < 1.0 and abs(angle) < np.pi/3:  # 60 degrees
                return True

        return False

class ObstacleDetector:
    """Visual obstacle detection"""
    def __init__(self):
        # Initialize detection models
        pass

    def detect_obstacles(self, image):
        """Detect obstacles in image"""
        # Simplified obstacle detection
        # In practice, this would use deep learning models
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find obstacles
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append((x + w/2, y + h/2))  # Center of bounding box

        return obstacles

class PathPlanner:
    """Global path planner"""
    def __init__(self):
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 100  # 10m x 10m grid

    def plan_path(self, start, goal, obstacles):
        """Plan path using A* algorithm with obstacle avoidance"""
        # Create occupancy grid
        grid = self.create_occupancy_grid(obstacles)

        # Implement A* path planning
        path = self.a_star(grid, start, goal)

        return path

    def create_occupancy_grid(self, obstacles):
        """Create occupancy grid from obstacle points"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        if len(obstacles) == 0:
            return grid

        # Convert obstacle coordinates to grid indices
        for obs_x, obs_y in obstacles:
            grid_x = int(obs_x / self.grid_resolution) + self.grid_size // 2
            grid_y = int(obs_y / self.grid_resolution) + self.grid_size // 2

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_y, grid_x] = 100  # Occupied

        return grid

    def a_star(self, grid, start, goal):
        """A* path planning algorithm"""
        import heapq

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        start_grid = (int(start[0] / self.grid_resolution) + self.grid_size // 2,
                     int(start[1] / self.grid_resolution) + self.grid_size // 2)
        goal_grid = (int(goal[0] / self.grid_resolution) + self.grid_size // 2,
                    int(goal[1] / self.grid_resolution) + self.grid_size // 2)

        # Check bounds
        if (not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size) or
            not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size)):
            return []

        # Check if start or goal is occupied
        if grid[start_grid[1], start_grid[0]] == 100 or grid[goal_grid[1], goal_grid[0]] == 100:
            return []

        # A* algorithm
        frontier = [(0, start_grid)]
        came_from = {start_grid: None}
        cost_so_far = {start_grid: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)

            if current == goal_grid:
                break

            for next_cell in self.get_neighbors(current, grid):
                new_cost = cost_so_far[current] + 1

                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(goal_grid, next_cell)
                    heapq.heappush(frontier, (priority, next_cell))
                    came_from[next_cell] = current

        # Reconstruct path
        path = []
        current = goal_grid
        if current in came_from:
            while current != start_grid:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            path.reverse()

        # Convert grid coordinates back to world coordinates
        world_path = []
        for grid_x, grid_y in path:
            world_x = (grid_x - self.grid_size // 2) * self.grid_resolution
            world_y = (grid_y - self.grid_size // 2) * self.grid_resolution
            world_path.append((world_x, world_y))

        return world_path

    def get_neighbors(self, pos, grid):
        """Get valid neighbors for A*"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),  # 4-connectivity
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # 8-connectivity
            x, y = pos[0] + dx, pos[1] + dy
            if (0 <= x < self.grid_size and 0 <= y < self.grid_size and
                grid[y, x] != 100):  # Not occupied
                neighbors.append((x, y))
        return neighbors

def main(args=None):
    rclpy.init(args=args)
    node = IsaacNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Computer Vision in Robotics

### Isaac ROS Image Pipeline

```python
# Isaac ROS Image Pipeline Node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacImagePipelineNode(Node):
    def __init__(self):
        super().__init__('isaac_image_pipeline_node')

        # Publishers
        self.processed_pub = self.create_publisher(Image, 'image_processed', 10)
        self.features_pub = self.create_publisher(Image, 'image_features', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.info_callback, 10)

        # Initialize components
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # Feature detection parameters
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous frame for feature tracking
        self.prev_frame = None
        self.prev_features = None

        self.get_logger().info('Isaac ROS Image Pipeline Node initialized')

    def info_callback(self, msg):
        """Handle camera info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Apply camera calibration if available
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                cv_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)

            # Process image
            processed_img = self.process_image(cv_image)
            features_img = self.extract_features(cv_image)

            # Publish results
            processed_msg = self.bridge.cv2_to_imgmsg(processed_img, encoding='bgr8')
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)

            features_msg = self.bridge.cv2_to_imgmsg(features_img, encoding='bgr8')
            features_msg.header = msg.header
            self.features_pub.publish(features_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_image(self, image):
        """Apply image processing pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Convert back to BGR for output
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced_bgr

    def extract_features(self, image):
        """Extract and visualize features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect features
        keypoints = self.feature_detector.detect(gray, None)
        keypoints, descriptors = self.feature_detector.compute(gray, keypoints)

        # Draw keypoints on image
        feature_img = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # If we have previous features, try to match them
        if self.prev_features is not None and len(keypoints) > 0:
            # Compute descriptors for current frame
            curr_kp, curr_desc = self.feature_detector.compute(gray, keypoints)

            if curr_desc is not None and self.prev_features[1] is not None:
                # Match features
                matches = self.matcher.match(self.prev_features[1], curr_desc)

                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)

                # Draw matches
                matched_img = cv2.drawMatches(
                    self.prev_frame, self.prev_features[0],
                    image, curr_kp,
                    matches[:50],  # Show top 50 matches
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                # Update feature visualization
                feature_img = matched_img

        # Store current frame features for next iteration
        self.prev_frame = image.copy()
        self.prev_features = (keypoints, descriptors)

        return feature_img

def main(args=None):
    rclpy.init(args=args)
    node = IsaacImagePipelineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### GPU Memory Management

```python
class IsaacPerceptionOptimizer:
    def __init__(self):
        self.gpu_memory_fraction = 0.8
        self.batch_size = 1
        self.precision = 'fp16'  # or 'fp32'
        self.async_processing = True

    def setup_gpu_memory(self):
        """Configure GPU memory allocation"""
        if torch.cuda.is_available():
            # Set memory fraction to prevent out-of-memory errors
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)

            # Enable TensorFloat32 for faster training on RTX cards
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)

    def optimize_model_inference(self, model):
        """Optimize model for inference"""
        # Convert to evaluation mode
        model.eval()

        # Use torch.jit for faster inference
        try:
            model = torch.jit.script(model)
        except Exception as e:
            self.get_logger().warn(f'Could not script model: {e}')

        # Use TensorRT if available
        if self.precision == 'fp16':
            try:
                import torch_tensorrt
                model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(
                        min_shape=[1, 3, 224, 224],
                        opt_shape=[self.batch_size, 3, 224, 224],
                        max_shape=[self.batch_size, 3, 224, 224],
                        dtype=torch.float
                    )],
                    enabled_precisions={torch.float16}
                )
            except ImportError:
                self.get_logger().warn('TensorRT not available')

        return model

    def optimize_pipeline(self):
        """Optimize the entire perception pipeline"""
        # Use pinned memory for faster CPU-GPU transfers
        def pin_memory_collate(batch):
            return [item.pin_memory() if hasattr(item, 'pin_memory') else item for item in batch]

        # Enable asynchronous data loading
        if self.async_processing:
            # Use multiple threads for data preprocessing
            pass  # Implementation would depend on specific pipeline

        # Optimize CUDA streams
        self.cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

# Example usage
def setup_optimized_perception():
    optimizer = IsaacPerceptionOptimizer()
    optimizer.setup_gpu_memory()

    # Initialize perception node with optimizations
    perception_node = IsaacDNNNode()
    perception_node.model = optimizer.optimize_model_inference(perception_node.model)

    return perception_node
```

## Troubleshooting Common Issues

### Performance Monitoring

```python
class IsaacPerformanceMonitor:
    def __init__(self, node):
        self.node = node
        self.frame_times = []
        self.max_samples = 100
        self.gpu_monitoring = torch.cuda.is_available()

    def start_frame(self):
        """Start timing a frame"""
        self.frame_start = time.time()
        if self.gpu_monitoring:
            torch.cuda.synchronize()  # Ensure accurate timing

    def end_frame(self):
        """End timing a frame and return performance metrics"""
        if self.gpu_monitoring:
            torch.cuda.synchronize()  # Ensure accurate timing

        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)

        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # GPU memory usage
        gpu_memory = 0
        if self.gpu_monitoring:
            gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB

        return {
            'fps': fps,
            'avg_frame_time': avg_frame_time,
            'gpu_memory_gb': gpu_memory,
            'cpu_usage': self.get_cpu_usage()
        }

    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        import psutil
        return psutil.cpu_percent()

    def log_performance(self, metrics):
        """Log performance metrics"""
        self.node.get_logger().info(
            f'Performance - FPS: {metrics["fps"]:.2f}, '
            f'Avg Frame Time: {metrics["avg_frame_time"]*1000:.2f}ms, '
            f'GPU Memory: {metrics["gpu_memory_gb"]:.2f}GB, '
            f'CPU: {metrics["cpu_usage"]:.1f}%'
        )
```

## Knowledge Check

1. What are the key advantages of Isaac ROS over traditional ROS perception packages?
2. How does GPU acceleration improve VSLAM performance in humanoid robotics?
3. What are the essential components for implementing stereo vision with Isaac ROS?
4. How do you optimize deep learning inference for real-time robotics applications?

## Summary

This chapter explored Isaac ROS and hardware-accelerated perception for humanoid robotics. We covered Visual SLAM systems, stereo vision processing, deep learning inference, and navigation with perception integration. The chapter also provided practical implementations of Isaac ROS components and optimization techniques for GPU-accelerated robotics applications.

## Next Steps

In the next chapter, we'll examine Nav2 and path planning specifically for bipedal humanoid movement, exploring navigation systems designed for legged robots and reinforcement learning approaches for robot control.