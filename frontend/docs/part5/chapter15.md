---
sidebar_position: 15
title: "Chapter 15: Manipulation and Grasping"
---

# Chapter 15: Manipulation and Grasping

## Learning Objectives
- Understand dexterous manipulation techniques for humanoid robots
- Implement grasp planning algorithms for various objects
- Design human-robot interaction systems for manipulation tasks
- Master multi-modal interaction approaches for manipulation

## Introduction to Humanoid Manipulation

Humanoid manipulation involves the use of anthropomorphic hands and arms to interact with objects in the environment. Unlike simple grippers, humanoid hands have multiple degrees of freedom that allow for complex, dexterous manipulation similar to human capabilities.

### Challenges in Humanoid Manipulation

1. **Complex Kinematics**: Multiple joints in fingers and wrist
2. **Underactuation**: More degrees of freedom than actuators
3. **Tactile Sensing**: Limited tactile feedback compared to humans
4. **Object Recognition**: Identifying graspable parts of objects
5. **Force Control**: Managing contact forces during manipulation
6. **Multi-task Coordination**: Coordinating both arms for complex tasks

### Humanoid Hand Anatomy

Humanoid hands typically have:
- **Thumb**: 2-3 joints with opposition capability
- **Index Finger**: 3 joints for precision grasps
- **Middle Finger**: 3 joints for power grasps
- **Ring Finger**: 3 joints
- **Pinky Finger**: 3 joints

## Grasp Planning Algorithms

### Geometric Grasp Planning

```python
# geometric_grasp_planning.py
import numpy as np
from scipy.spatial import distance
from math import atan2, sqrt
import open3d as o3d

class GeometricGraspPlanner:
    def __init__(self):
        self.finger_length = 0.08  # meters
        self.thumb_length = 0.06
        self.grasp_width_range = (0.02, 0.15)  # minimum to maximum graspable width

    def find_grasp_points(self, object_mesh):
        """Find potential grasp points on object surface"""
        # Sample points on object surface
        surface_points = self.sample_surface_points(object_mesh)

        # Calculate surface normals
        normals = self.calculate_surface_normals(object_mesh, surface_points)

        # Find grasp candidates based on geometric criteria
        grasp_candidates = []

        for i, point in enumerate(surface_points):
            normal = normals[i]

            # Check if this point is suitable for grasping
            if self.is_graspable_point(object_mesh, point, normal):
                grasp_candidates.append({
                    'position': point,
                    'normal': normal,
                    'quality': self.calculate_grasp_quality(point, normal, surface_points)
                })

        return grasp_candidates

    def sample_surface_points(self, mesh, num_points=1000):
        """Sample points on object surface"""
        # Use Poisson disk sampling or random sampling
        # For now, using random sampling of mesh vertices
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        # Sample random points on triangles
        points = []
        for _ in range(num_points):
            # Pick a random triangle
            triangle_idx = np.random.randint(0, len(faces))
            triangle = faces[triangle_idx]

            # Sample random point on triangle
            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]

            # Barycentric coordinates for random point
            r1 = np.random.random()
            r2 = np.random.random()

            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2

            point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
            points.append(point)

        return np.array(points)

    def calculate_surface_normals(self, mesh, points):
        """Calculate surface normals at given points"""
        # For each point, find the closest face and use its normal
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        normals = []
        for point in points:
            # Find closest vertex
            closest_vertex_idx = np.argmin(np.linalg.norm(vertices - point, axis=1))

            # Find faces that contain this vertex
            face_indices = []
            for i, face in enumerate(faces):
                if closest_vertex_idx in face:
                    face_indices.append(i)

            if face_indices:
                # Average normals of adjacent faces
                face_normals = []
                for face_idx in face_indices:
                    face = faces[face_idx]
                    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                    normal = np.cross(v1 - v0, v2 - v0)
                    normal = normal / np.linalg.norm(normal)
                    face_normals.append(normal)

                avg_normal = np.mean(face_normals, axis=0)
                avg_normal = avg_normal / np.linalg.norm(avg_normal)
            else:
                avg_normal = np.array([0, 0, 1])  # Default normal

            normals.append(avg_normal)

        return np.array(normals)

    def is_graspable_point(self, mesh, point, normal):
        """Check if a point is suitable for grasping"""
        # Check if the surface is flat enough
        local_curvature = self.estimate_local_curvature(mesh, point, normal)
        if local_curvature > 0.1:  # Too curved
            return False

        # Check if the surface is not too steep
        gravity_dir = np.array([0, 0, -1])  # Assuming gravity is down
        angle_with_gravity = np.arccos(np.clip(np.dot(normal, gravity_dir), -1, 1))
        if angle_with_gravity > np.pi / 3:  # Too steep (60 degrees)
            return False

        return True

    def estimate_local_curvature(self, mesh, point, normal, radius=0.02):
        """Estimate local curvature around a point"""
        vertices = np.asarray(mesh.vertices)

        # Find nearby points
        distances = np.linalg.norm(vertices - point, axis=1)
        nearby_indices = np.where(distances < radius)[0]

        if len(nearby_indices) < 3:
            return 0.0

        nearby_points = vertices[nearby_indices]

        # Fit a plane to nearby points and measure deviation
        centered_points = nearby_points - point
        cov_matrix = np.cov(centered_points.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)

        # Curvature is related to the smallest eigenvalue
        curvature = min(eigenvalues)
        return abs(curvature)

    def calculate_grasp_quality(self, point, normal, surface_points):
        """Calculate quality metric for a grasp point"""
        # Distance to other surface points (avoid edges)
        distances = np.linalg.norm(surface_points - point, axis=1)
        min_distance = np.min(distances[distances > 1e-6])  # Exclude the point itself

        # Surface flatness (use normal consistency in neighborhood)
        neighborhood_size = 10
        nearest_indices = np.argsort(distances)[:neighborhood_size]
        neighborhood_normals = [normal]  # We don't have normals for all points in this simplified version

        # Simple quality metric
        quality = min_distance * 10  # Prefer points away from edges
        quality += 0.5  # Base quality

        return min(quality, 1.0)  # Normalize to [0, 1]

    def plan_grasp_poses(self, object_mesh, grasp_candidates, num_poses=5):
        """Plan multiple grasp poses for the object"""
        grasp_poses = []

        # Sort candidates by quality
        sorted_candidates = sorted(grasp_candidates, key=lambda x: x['quality'], reverse=True)

        for candidate in sorted_candidates[:num_poses]:
            # Generate grasp pose from contact point and normal
            grasp_pose = self.generate_grasp_pose(candidate['position'], candidate['normal'])
            grasp_poses.append({
                'pose': grasp_pose,
                'quality': candidate['quality'],
                'contact_point': candidate['position'],
                'approach_direction': candidate['normal']
            })

        return grasp_poses

    def generate_grasp_pose(self, contact_point, surface_normal):
        """Generate a complete grasp pose from contact information"""
        # Create a transformation matrix for the grasp
        # Z-axis points into the object (opposite to surface normal)
        z_axis = -surface_normal / np.linalg.norm(surface_normal)

        # X-axis can be chosen arbitrarily, perpendicular to Z
        if abs(z_axis[2]) < 0.9:  # Not pointing vertically
            x_axis = np.array([0, 0, 1])  # Point up
        else:
            x_axis = np.array([1, 0, 0])  # Point along X

        # Make sure X is perpendicular to Z
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis is cross product
        y_axis = np.cross(z_axis, x_axis)

        # Create rotation matrix
        rotation = np.column_stack([x_axis, y_axis, z_axis])

        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = contact_point

        return transform

class ForceClosureGraspPlanner:
    """Grasp planner based on force closure analysis"""
    def __init__(self):
        self.friction_coefficient = 0.5
        self.num_fingers = 2  # For simplicity, assuming 2-finger grasp

    def check_force_closure(self, contact_points, contact_normals, object_com):
        """Check if grasp achieves force closure"""
        # This is a simplified 2D version
        # For 3D, we would need to check the convex hull of wrench space

        if len(contact_points) < 2:
            return False

        # Calculate grasp matrix for force closure
        # In 2D: [fx1, fy1, tau1; fx2, fy2, tau2; ...]
        # Where tau is moment about object COM

        grasp_matrix = []
        for i, (point, normal) in enumerate(zip(contact_points, contact_normals)):
            # Force components
            fx, fy = normal[0], normal[1]

            # Moment about COM
            r = point[:2] - object_com[:2]  # 2D position relative to COM
            tau = r[0] * normal[1] - r[1] * normal[0]  # Cross product in 2D

            grasp_matrix.append([fx, fy, tau])

        grasp_matrix = np.array(grasp_matrix)

        # Check if the origin is inside the convex hull of columns
        # This is a simplified check - in practice, more sophisticated methods are used
        try:
            # Use linear programming to check if 0 is in convex hull
            from scipy.optimize import linprog

            # We want to find lambda such that:
            # sum(lambda_i * column_i) = 0
            # sum(lambda_i) = 1
            # lambda_i >= 0

            A_eq = np.column_stack([grasp_matrix.T, np.ones(len(contact_points))])
            b_eq = np.zeros(4)  # 3 for forces/moments + 1 for sum constraint
            b_eq[3] = 1  # sum of lambdas = 1

            c = np.zeros(len(contact_points))  # Minimize 0 (feasibility problem)
            bounds = [(0, None) for _ in range(len(contact_points))]

            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            return result.success

        except ImportError:
            # If scipy not available, use a simpler geometric check
            return self.geometric_force_closure_check(contact_points, contact_normals)

    def geometric_force_closure_check(self, contact_points, contact_normals):
        """Simplified geometric check for force closure"""
        if len(contact_points) < 2:
            return False

        # For 2 contacts, check if normals point toward each other
        if len(contact_points) == 2:
            p1, p2 = contact_points
            n1, n2 = contact_normals

            # Vector from p1 to p2
            v12 = p2[:2] - p1[:2]

            # Check if normals point inward
            inward_1 = np.dot(n1[:2], v12) > 0  # n1 points toward p2
            inward_2 = np.dot(n2[:2], -v12) > 0  # n2 points toward p1

            return inward_1 and inward_2

        # For more than 2 contacts, check if origin is in convex hull
        # This is a 2D simplification
        points_2d = np.array([p[:2] for p in contact_points])
        return self.points_form_convex_hull_around_origin(points_2d)

    def points_form_convex_hull_around_origin(self, points_2d):
        """Check if 2D points form a convex hull that contains origin"""
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(points_2d)

            # Check if origin is inside the convex hull
            # This is a simplified check - in practice, you'd use more robust methods
            # For now, we'll check if origin is "likely" inside by checking
            # if the centroid is close to origin relative to hull size
            centroid = np.mean(points_2d[hull.vertices], axis=0)
            distances = np.linalg.norm(points_2d[hull.vertices] - centroid, axis=1)
            avg_distance = np.mean(distances)

            # If centroid is close to origin and points are spread out
            return np.linalg.norm(centroid) < avg_distance * 0.5

        except:
            return False

# Example usage
def example_geometric_grasp_planning():
    # Create a simple object (box) for demonstration
    object_mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.1)

    # Plan grasps
    planner = GeometricGraspPlanner()
    candidates = planner.find_grasp_points(object_mesh)
    poses = planner.plan_grasp_poses(object_mesh, candidates)

    print(f"Found {len(candidates)} grasp candidates")
    print(f"Planned {len(poses)} grasp poses")

    if poses:
        best_pose = poses[0]
        print(f"Best grasp quality: {best_pose['quality']:.3f}")
        print(f"Contact point: {best_pose['contact_point']}")
        print(f"Grasp pose transformation:\n{best_pose['pose']}")

if __name__ == "__main__":
    try:
        example_geometric_grasp_planning()
    except ImportError:
        print("Open3D not available, skipping geometric grasp planning example")
```

### Learning-Based Grasp Planning

```python
# learning_based_grasping.py
import torch
import torch.nn as nn
import numpy as np
import cv2

class GraspQualityCNN(nn.Module):
    """CNN for predicting grasp quality from RGB-D images"""
    def __init__(self, input_channels=4):  # RGB + D
        super(GraspQualityCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Calculate the size after convolutions
        # Assuming input is 224x224, after 4 downsampling steps: 224/2^4 = 14
        conv_output_size = 256 * 14 * 14  # Adjust based on your input size

        # Fully connected layers for grasp quality prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output grasp quality between 0 and 1
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class GraspPosePredictor(nn.Module):
    """Network for predicting grasp pose parameters"""
    def __init__(self, num_outputs=6):  # 3 for position, 3 for orientation
        super(GraspPosePredictor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate feature map size
        conv_output_size = 128 * 28 * 28  # Adjust based on input size

        self.regressor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        pose_params = self.regressor(features)
        return pose_params

class LearningBasedGraspPlanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.quality_network = GraspQualityCNN().to(self.device)
        self.pose_network = GraspPosePredictor().to(self.device)

        # Load pre-trained weights if available
        self.load_pretrained_models()

    def load_pretrained_models(self):
        """Load pre-trained models (placeholder)"""
        # In practice, you would load trained weights
        pass

    def predict_grasp_quality(self, rgb_image, depth_image, grasp_poses):
        """Predict grasp quality for given poses using the trained network"""
        # Preprocess images
        input_tensor = self.preprocess_input(rgb_image, depth_image)

        # Predict quality for each pose
        with torch.no_grad():
            quality_predictions = self.quality_network(input_tensor)

        return quality_predictions.cpu().numpy()

    def predict_grasp_pose(self, rgb_image, depth_image):
        """Predict optimal grasp pose from RGB-D input"""
        input_tensor = self.preprocess_input(rgb_image, depth_image)

        with torch.no_grad():
            pose_params = self.pose_network(input_tensor)

        return pose_params.cpu().numpy()

    def preprocess_input(self, rgb_image, depth_image):
        """Preprocess RGB-D images for network input"""
        # Resize images to network input size
        target_size = (224, 224)
        rgb_resized = cv2.resize(rgb_image, target_size)
        depth_resized = cv2.resize(depth_image, target_size)

        # Normalize RGB
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0

        # Normalize depth
        depth_normalized = depth_resized.astype(np.float32)
        depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min() + 1e-6)

        # Stack RGB and depth
        input_data = np.concatenate([rgb_normalized, np.expand_dims(depth_normalized, axis=-1)], axis=-1)

        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(input_data).permute(2, 0, 1).unsqueeze(0)

        return input_tensor.to(self.device)

    def plan_grasps_with_network(self, rgb_image, depth_image, num_candidates=10):
        """Plan grasps using learning-based approach"""
        # Predict grasp poses
        predicted_poses = self.predict_grasp_pose(rgb_image, depth_image)

        # Evaluate multiple candidates
        grasp_candidates = []

        for i in range(num_candidates):
            # Generate candidate grasp pose with some variation
            base_pose = predicted_poses[0] if len(predicted_poses) > 0 else np.zeros(6)

            # Add random variation for diversity
            variation = np.random.normal(0, 0.1, 6)
            candidate_pose = base_pose + variation

            # Predict quality for this candidate
            quality = self.predict_grasp_quality(rgb_image, depth_image, [candidate_pose])

            grasp_candidates.append({
                'pose': candidate_pose,
                'quality': quality[0] if len(quality) > 0 else 0.0,
                'position': candidate_pose[:3],
                'orientation': candidate_pose[3:]
            })

        # Sort by quality
        grasp_candidates.sort(key=lambda x: x['quality'], reverse=True)

        return grasp_candidates

class GraspRefinementNetwork(nn.Module):
    """Network for refining initial grasp estimates"""
    def __init__(self):
        super(GraspRefinementNetwork, self).__init__()

        # Input: initial grasp + object features
        self.refinement_net = nn.Sequential(
            nn.Linear(10, 64),  # 6 for pose + 4 for object features
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)   # 3 pos + 3 orient refinement
        )

    def forward(self, initial_grasp, object_features):
        # Combine initial grasp and object features
        combined_input = torch.cat([initial_grasp, object_features], dim=1)
        refinement = self.refinement_net(combined_input)
        return refinement

# Example usage
def example_learning_based_grasping():
    # Initialize the planner
    planner = LearningBasedGraspPlanner()

    # Simulate RGB-D input (in practice, these would come from sensors)
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_image = np.random.rand(480, 640).astype(np.float32) * 2.0  # 0-2 meters

    # Plan grasps
    candidates = planner.plan_grasps_with_network(rgb_image, depth_image)

    print(f"Generated {len(candidates)} grasp candidates")
    if candidates:
        best_grasp = candidates[0]
        print(f"Best grasp quality: {best_grasp['quality']:.3f}")
        print(f"Position: {best_grasp['position']}")
        print(f"Orientation: {best_grasp['orientation']}")

if __name__ == "__main__":
    example_learning_based_grasping()
```

## Dexterous Manipulation Techniques

### Multi-Finger Grasp Synthesis

```python
# dexterous_manipulation.py
import numpy as np
from math import pi, sin, cos, atan2, sqrt

class MultiFingerGraspSynthesizer:
    def __init__(self):
        self.hand_params = {
            'thumb': {'length': 0.06, 'width': 0.02, 'joints': 3},
            'index': {'length': 0.08, 'width': 0.015, 'joints': 3},
            'middle': {'length': 0.085, 'width': 0.015, 'joints': 3},
            'ring': {'length': 0.08, 'width': 0.015, 'joints': 3},
            'pinky': {'length': 0.07, 'width': 0.012, 'joints': 3}
        }

        # Palm dimensions
        self.palm_width = 0.08
        self.palm_height = 0.06
        self.palm_depth = 0.02

    def synthesize_power_grasp(self, object_dims):
        """Synthesize power grasp for large objects"""
        # Power grasp: object is held in the palm with fingers wrapped around
        grasp_config = {
            'type': 'power',
            'thumb_pos': None,
            'finger_positions': [],
            'finger_angles': [],
            'preshape': 'wide_open'
        }

        # Calculate thumb position (opposes fingers)
        object_width, object_height, object_depth = object_dims

        # Thumb position on one side of object
        thumb_x = -object_width/2 - 0.01  # 1cm away from object
        thumb_y = 0
        thumb_z = object_height/2  # Middle height

        grasp_config['thumb_pos'] = [thumb_x, thumb_y, thumb_z]

        # Finger positions around the object
        finger_positions = []
        finger_angles = []

        # Distribute fingers around object circumference
        num_fingers = 4  # Index, middle, ring, pinky
        for i in range(num_fingers):
            angle = 2 * pi * i / num_fingers
            finger_x = object_width/2 + 0.01  # 1cm away from object
            finger_y = object_depth/2 * cos(angle)
            finger_z = object_height/2 * sin(angle)

            finger_positions.append([finger_x, finger_y, finger_z])

            # Finger angles (typically flexed for power grasp)
            finger_angles.append([1.2, 1.5, 0.8])  # Joint angles in radians

        grasp_config['finger_positions'] = finger_positions
        grasp_config['finger_angles'] = finger_angles

        return grasp_config

    def synthesize_precision_grasp(self, object_dims):
        """Synthesize precision grasp for small objects"""
        # Precision grasp: object is held between thumb and fingertips
        grasp_config = {
            'type': 'precision',
            'thumb_pos': None,
            'finger_positions': [],
            'finger_angles': [],
            'preshape': 'close_together'
        }

        object_width, object_height, object_depth = object_dims

        # Thumb opposes the finger
        thumb_x = -object_width/2 - 0.01
        thumb_y = -0.02  # Slightly offset
        thumb_z = object_height/2

        grasp_config['thumb_pos'] = [thumb_x, thumb_y, thumb_z]

        # Use index finger for precision grasp
        index_x = object_width/2 + 0.01
        index_y = 0.02  # Opposite to thumb
        index_z = object_height/2

        grasp_config['finger_positions'] = [[index_x, index_y, index_z]]
        grasp_config['finger_angles'] = [[0.2, 0.3, 0.1]]  # Less flexed for precision

        return grasp_config

    def synthesize_lateral_grasp(self, object_dims):
        """Synthesize lateral grasp (thumb-finger side grasp)"""
        grasp_config = {
            'type': 'lateral',
            'thumb_pos': None,
            'finger_positions': [],
            'finger_angles': [],
            'preshape': 'lateral'
        }

        object_width, object_height, object_depth = object_dims

        # Thumb on one side of object
        thumb_x = -object_width/2 - 0.01
        thumb_y = 0
        thumb_z = object_height/2

        grasp_config['thumb_pos'] = [thumb_x, thumb_y, thumb_z]

        # Index finger on opposite side, aligned with thumb
        index_x = object_width/2 + 0.01
        index_y = 0
        index_z = object_height/2

        grasp_config['finger_positions'] = [[index_x, index_y, index_z]]
        grasp_config['finger_angles'] = [[0.5, 0.8, 0.3]]  # Moderate flexion

        return grasp_config

    def optimize_grasp_configuration(self, grasp_config, object_properties):
        """Optimize grasp configuration based on object properties"""
        # Calculate grasp stability metrics
        stability_score = self.evaluate_grasp_stability(grasp_config, object_properties)

        # Adjust grasp based on object weight and friction
        object_weight = object_properties.get('weight', 0.1)  # kg
        object_friction = object_properties.get('friction', 0.5)

        # Increase grip force for heavier objects
        if object_weight > 0.5:
            # Adjust finger angles for stronger grip
            for i, angles in enumerate(grasp_config['finger_angles']):
                # Increase flexion for stronger grip
                adjusted_angles = [min(angle + 0.2, 2.0) for angle in angles]
                grasp_config['finger_angles'][i] = adjusted_angles

        # Adjust for low friction surfaces
        if object_friction < 0.3:
            # Use more contact points
            if grasp_config['type'] == 'precision':
                # Convert to 3-finger grasp if possible
                grasp_config = self.convert_to_three_finger_grasp(grasp_config)

        grasp_config['stability_score'] = stability_score
        return grasp_config

    def evaluate_grasp_stability(self, grasp_config, object_properties):
        """Evaluate the stability of a grasp configuration"""
        # Simplified stability evaluation
        # In practice, this would involve complex physics simulations

        stability_score = 0.5  # Base score

        # Consider number of contact points
        num_contacts = len(grasp_config['finger_positions']) + 1  # +1 for thumb
        if num_contacts >= 3:
            stability_score += 0.2

        # Consider object dimensions vs hand dimensions
        object_dims = object_properties.get('dimensions', [0.05, 0.05, 0.05])
        object_size = sum(object_dims)

        if grasp_config['type'] == 'power' and object_size > 0.1:
            stability_score += 0.2
        elif grasp_config['type'] == 'precision' and object_size < 0.05:
            stability_score += 0.2

        # Consider grasp type appropriateness
        if object_size < 0.03 and grasp_config['type'] != 'precision':
            stability_score -= 0.2  # Precision grasp more appropriate for small objects
        elif object_size > 0.1 and grasp_config['type'] == 'precision':
            stability_score -= 0.2  # Power grasp more appropriate for large objects

        return min(stability_score, 1.0)

    def convert_to_three_finger_grasp(self, grasp_config):
        """Convert precision grasp to three-finger grasp for better stability"""
        if grasp_config['type'] != 'precision':
            return grasp_config

        # Add middle and ring fingers to support the grasp
        original_finger_pos = grasp_config['finger_positions'][0]

        # Position additional fingers to provide support
        support_fingers = []
        for i in range(2):  # Add 2 more fingers
            # Offset the support fingers slightly
            offset = (i + 1) * 0.01  # 1cm offset
            support_pos = [
                original_finger_pos[0],
                original_finger_pos[1] + offset,
                original_finger_pos[2]
            ]
            support_fingers.append(support_pos)

        grasp_config['finger_positions'].extend(support_fingers)
        grasp_config['finger_angles'].extend([[0.3, 0.5, 0.2]] * 2)

        return grasp_config

    def generate_grasp_sequence(self, object_trajectory, initial_grasp):
        """Generate a sequence of grasp configurations for manipulation"""
        grasp_sequence = [initial_grasp]

        # Generate intermediate grasps for repositioning
        for i in range(len(object_trajectory) - 1):
            current_pos = object_trajectory[i]
            next_pos = object_trajectory[i + 1]

            # Calculate required hand repositioning
            pos_change = np.array(next_pos) - np.array(current_pos)

            # Generate intermediate grasp configuration
            intermediate_grasp = self.generate_intermediate_grasp(
                grasp_sequence[-1], pos_change
            )

            grasp_sequence.append(intermediate_grasp)

        return grasp_sequence

    def generate_intermediate_grasp(self, current_grasp, position_change):
        """Generate intermediate grasp configuration"""
        # Copy current grasp and adjust for position change
        new_grasp = current_grasp.copy()

        # Adjust thumb position
        if new_grasp['thumb_pos'] is not None:
            new_grasp['thumb_pos'] = [
                new_grasp['thumb_pos'][0] + position_change[0],
                new_grasp['thumb_pos'][1] + position_change[1],
                new_grasp['thumb_pos'][2] + position_change[2]
            ]

        # Adjust finger positions
        for i, pos in enumerate(new_grasp['finger_positions']):
            new_grasp['finger_positions'][i] = [
                pos[0] + position_change[0],
                pos[1] + position_change[1],
                pos[2] + position_change[2]
            ]

        return new_grasp

class ManipulationController:
    """Controller for executing manipulation tasks"""
    def __init__(self):
        self.grasp_synthesizer = MultiFingerGraspSynthesizer()
        self.current_grasp = None
        self.object_in_hand = None

    def approach_object(self, object_pose, grasp_config):
        """Generate approach trajectory to object"""
        # Calculate approach direction (opposite to grasp direction)
        approach_direction = self.calculate_approach_direction(grasp_config)

        # Generate trajectory points
        trajectory = []
        current_pos = object_pose['position']

        # Move to approach position (10cm away from grasp point)
        approach_pos = current_pos - approach_direction * 0.1

        # Linear approach trajectory
        for t in np.linspace(0, 1, 20):  # 20 intermediate points
            pos = approach_pos + t * (current_pos - approach_pos)
            trajectory.append(pos)

        return trajectory

    def calculate_approach_direction(self, grasp_config):
        """Calculate approach direction based on grasp configuration"""
        if grasp_config['type'] == 'power':
            # Approach from the side where fingers will wrap around
            return np.array([1, 0, 0])  # From positive X direction
        elif grasp_config['type'] == 'precision':
            # Approach from above or side
            return np.array([0, 0, -1])  # From above
        elif grasp_config['type'] == 'lateral':
            # Approach from the side
            return np.array([0, 1, 0])  # From positive Y direction
        else:
            return np.array([1, 0, 0])  # Default approach

    def execute_grasp(self, grasp_config):
        """Execute the grasp motion"""
        # Move fingers to grasp configuration
        print(f"Executing {grasp_config['type']} grasp")

        # Simulate finger movements
        for i, (pos, angles) in enumerate(zip(grasp_config['finger_positions'],
                                            grasp_config['finger_angles'])):
            print(f"Moving finger {i+1} to position {pos} with angles {angles}")

        # Close thumb
        if grasp_config['thumb_pos']:
            print(f"Moving thumb to position {grasp_config['thumb_pos']}")

        # Wait for grasp completion
        print("Grasp completed")

    def verify_grasp_success(self):
        """Verify that the grasp was successful"""
        # In practice, this would check force sensors, tactile sensors, etc.
        # For simulation, assume grasp succeeds with 90% probability
        success_probability = 0.9
        return np.random.random() < success_probability

    def lift_object(self, height=0.1):
        """Lift the grasped object"""
        print(f"Lifting object by {height} meters")
        # Simulate lifting motion
        return True

    def transport_object(self, target_pose):
        """Transport object to target pose"""
        print(f"Transporting object to {target_pose}")
        # Simulate transport motion
        return True

    def release_object(self):
        """Release the grasped object"""
        print("Releasing object")
        self.object_in_hand = None
        return True

# Example usage
def example_dexterous_manipulation():
    synthesizer = MultiFingerGraspSynthesizer()
    controller = ManipulationController()

    # Object dimensions (width, height, depth in meters)
    object_dims = [0.05, 0.05, 0.03]  # Small box
    object_properties = {
        'dimensions': object_dims,
        'weight': 0.2,  # kg
        'friction': 0.6
    }

    # Synthesize appropriate grasp
    if max(object_dims) < 0.04:
        grasp_config = synthesizer.synthesize_precision_grasp(object_dims)
    else:
        grasp_config = synthesizer.synthesize_power_grasp(object_dims)

    # Optimize the grasp
    optimized_grasp = synthesizer.optimize_grasp_configuration(grasp_config, object_properties)

    print(f"Synthesized {optimized_grasp['type']} grasp")
    print(f"Stability score: {optimized_grasp.get('stability_score', 0):.2f}")
    print(f"Thumb position: {optimized_grasp['thumb_pos']}")
    print(f"Finger positions: {optimized_grasp['finger_positions']}")

    # Execute the manipulation
    controller.execute_grasp(optimized_grasp)

    if controller.verify_grasp_success():
        print("Grasp successful!")
        controller.lift_object(0.1)
        controller.transport_object([0.5, 0.5, 0.2])
        controller.release_object()
    else:
        print("Grasp failed!")

if __name__ == "__main__":
    example_dexterous_manipulation()
```

## Human-Robot Interaction Design

### Multi-Modal Interaction for Manipulation

```python
# human_robot_interaction.py
import numpy as np
import speech_recognition as sr
import cv2
from queue import Queue
import threading
import time

class MultiModalInteractionManager:
    def __init__(self):
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Interaction state
        self.current_task = None
        self.user_intent = None
        self.robot_feedback = None

        # Modalities queue
        self.modality_queue = Queue()

        # Gesture recognition (simplified)
        self.gesture_recognizer = GestureRecognizer()

        # Shared state for coordination
        self.shared_state = {
            'user_attention': True,
            'task_progress': 0.0,
            'safety_status': 'safe',
            'communication_mode': 'active'
        }

    def start_interaction_loop(self):
        """Start the main interaction loop"""
        # Start threads for different modalities
        speech_thread = threading.Thread(target=self.listen_for_speech)
        vision_thread = threading.Thread(target=self.process_vision_input)
        gesture_thread = threading.Thread(target=self.process_gestures)

        speech_thread.daemon = True
        vision_thread.daemon = True
        gesture_thread.daemon = True

        speech_thread.start()
        vision_thread.start()
        gesture_thread.start()

        # Main interaction loop
        while True:
            if not self.modality_queue.empty():
                modality_input = self.modality_queue.get()
                self.process_multimodal_input(modality_input)

            # Update shared state
            self.update_shared_state()

            time.sleep(0.1)  # 10Hz update rate

    def listen_for_speech(self):
        """Continuously listen for speech commands"""
        with self.microphone as source:
            self.speech_recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                with self.microphone as source:
                    audio = self.speech_recognizer.listen(source, timeout=1)

                # Recognize speech
                text = self.speech_recognizer.recognize_google(audio)
                print(f"Recognized speech: {text}")

                # Add to queue
                self.modality_queue.put({
                    'modality': 'speech',
                    'content': text,
                    'timestamp': time.time()
                })

            except sr.WaitTimeoutError:
                pass  # Continue listening
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")

            time.sleep(0.1)

    def process_vision_input(self):
        """Process visual input from camera"""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame for relevant information
            objects = self.detect_objects(frame)
            user_attention = self.detect_user_attention(frame)

            # Add to queue
            self.modality_queue.put({
                'modality': 'vision',
                'objects': objects,
                'user_attention': user_attention,
                'frame': frame,
                'timestamp': time.time()
            })

            time.sleep(0.05)  # 20Hz for vision

    def process_gestures(self):
        """Process hand gestures for interaction"""
        cap = cv2.VideoCapture(1)  # Second camera for gesture

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Recognize gestures
            gesture = self.gesture_recognizer.recognize_gesture(frame)

            if gesture and gesture != 'none':
                self.modality_queue.put({
                    'modality': 'gesture',
                    'gesture': gesture,
                    'timestamp': time.time()
                })

            time.sleep(0.05)  # 20Hz for gestures

    def detect_objects(self, frame):
        """Detect objects in the camera frame"""
        # Simplified object detection
        # In practice, this would use deep learning models
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find contours (simplified object detection)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': [x, y, w, h],
                    'center': [x + w//2, y + h//2],
                    'area': cv2.contourArea(contour)
                })

        return objects

    def detect_user_attention(self, frame):
        """Detect if user is paying attention to robot"""
        # Simplified attention detection
        # In practice, this would use face detection/gaze estimation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use face detection as proxy for attention
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        return len(faces) > 0  # User is attending if face is detected

    def process_multimodal_input(self, modality_input):
        """Process input from any modality"""
        modality = modality_input['modality']

        if modality == 'speech':
            self.process_speech_command(modality_input['content'])
        elif modality == 'vision':
            self.process_vision_data(modality_input)
        elif modality == 'gesture':
            self.process_gesture_command(modality_input['gesture'])

    def process_speech_command(self, command):
        """Process speech commands for manipulation"""
        command_lower = command.lower()

        # Parse manipulation commands
        if 'grasp' in command_lower or 'pick up' in command_lower:
            # Extract object reference
            object_ref = self.extract_object_reference(command)
            self.initiate_grasp_task(object_ref)

        elif 'move' in command_lower or 'place' in command_lower or 'put' in command_lower:
            target_location = self.extract_location_reference(command)
            self.initiate_place_task(target_location)

        elif 'release' in command_lower or 'let go' in command_lower:
            self.initiate_release_task()

        elif 'stop' in command_lower or 'abort' in command_lower:
            self.abort_current_task()

        else:
            # Check for confirmation requests
            if 'yes' in command_lower or 'sure' in command_lower:
                self.confirm_pending_action()
            elif 'no' in command_lower or 'cancel' in command_lower:
                self.cancel_pending_action()

    def extract_object_reference(self, command):
        """Extract object reference from speech command"""
        # Simple keyword-based extraction
        keywords = ['box', 'cup', 'bottle', 'book', 'object', 'item']
        for keyword in keywords:
            if keyword in command.lower():
                return keyword
        return 'object'  # Default

    def extract_location_reference(self, command):
        """Extract location reference from speech command"""
        # Simple keyword-based extraction
        if 'table' in command.lower():
            return 'table'
        elif 'shelf' in command.lower():
            return 'shelf'
        elif 'box' in command.lower():
            return 'box'
        else:
            return 'default'

    def initiate_grasp_task(self, object_ref):
        """Initiate grasp task based on object reference"""
        print(f"Initiating grasp task for {object_ref}")

        # Find object in the environment
        target_object = self.find_object_by_reference(object_ref)

        if target_object:
            self.current_task = {
                'type': 'grasp',
                'target_object': target_object,
                'status': 'planning'
            }
            print(f"Found target object: {target_object}")
        else:
            self.request_object_location(command=f"Where is the {object_ref}?")

    def find_object_by_reference(self, object_ref):
        """Find object in environment by reference"""
        # This would interface with perception system
        # For simulation, return a mock object
        return {
            'type': object_ref,
            'position': [0.5, 0.3, 0.1],
            'dimensions': [0.05, 0.05, 0.05],
            'graspable': True
        }

    def request_object_location(self, command):
        """Request user to indicate object location"""
        self.speak_response(f"Could you please show me the {command.split()[-1][:-1]}?")
        self.shared_state['user_attention'] = True
        self.shared_state['communication_mode'] = 'requesting'

    def process_vision_data(self, vision_data):
        """Process vision data for interaction"""
        if 'user_attention' in vision_data:
            self.shared_state['user_attention'] = vision_data['user_attention']

        if self.shared_state['communication_mode'] == 'requesting':
            # User might be pointing to an object
            self.check_for_pointing_gesture(vision_data['frame'])

    def check_for_pointing_gesture(self, frame):
        """Check if user is pointing to indicate object location"""
        # Simplified pointing detection
        # In practice, this would use pose estimation
        pointing_detected = self.gesture_recognizer.detect_pointing(frame)

        if pointing_detected:
            # Extract object at pointed location
            pointed_object = self.get_object_at_location(pointing_detected['location'])
            if pointed_object:
                self.current_task = {
                    'type': 'grasp',
                    'target_object': pointed_object,
                    'status': 'confirmed'
                }
                self.speak_response("I see it, grasping now.")
                self.shared_state['communication_mode'] = 'active'

    def get_object_at_location(self, location):
        """Get object at a specific location in the environment"""
        # This would query the environment model
        return {
            'type': 'object',
            'position': [location[0]/100, location[1]/100, 0.1],  # Convert pixel to meters
            'dimensions': [0.05, 0.05, 0.05],
            'graspable': True
        }

    def process_gesture_command(self, gesture):
        """Process gesture commands"""
        if gesture == 'thumbs_up':
            self.confirm_pending_action()
        elif gesture == 'stop_hand':
            self.abort_current_task()
        elif gesture == 'pointing':
            # User is indicating something
            self.shared_state['communication_mode'] = 'requesting'
        elif gesture == 'wave':
            self.speak_response("Hello! How can I help you?")

    def speak_response(self, text):
        """Generate speech response (simulated)"""
        print(f"Robot says: {text}")
        # In practice, this would use text-to-speech

    def update_shared_state(self):
        """Update the shared interaction state"""
        # Update based on various factors
        if not self.shared_state['user_attention']:
            self.shared_state['communication_mode'] = 'passive'
        else:
            self.shared_state['communication_mode'] = 'active'

class GestureRecognizer:
    """Simple gesture recognizer"""
    def __init__(self):
        self.reference_gestures = self.load_reference_gestures()

    def load_reference_gestures(self):
        """Load reference gestures (simplified)"""
        return {
            'thumbs_up': [(100, 100), (120, 120)],  # Simplified
            'stop_hand': [(100, 150), (120, 150)],
            'pointing': [(100, 200), (120, 180)],
            'wave': [(100, 250), (120, 250)]
        }

    def recognize_gesture(self, frame):
        """Recognize gesture from frame"""
        # Simplified gesture recognition
        # In practice, this would use hand pose estimation
        height, width = frame.shape[:2]

        # Mock recognition based on simple features
        if width > height:  # Landscape orientation
            return 'wave'  # Mock detection
        else:
            return 'none'

    def detect_pointing(self, frame):
        """Detect pointing gesture"""
        # Simplified pointing detection
        height, width = frame.shape[:2]

        # Mock pointing detection
        if width > 640:  # If there's a distinct feature
            return {'location': (width//2, height//2)}
        return None

class ManipulationTaskExecutor:
    """Execute manipulation tasks with human-robot interaction"""
    def __init__(self):
        self.interaction_manager = MultiModalInteractionManager()
        self.grasp_planner = MultiFingerGraspSynthesizer()
        self.manipulation_controller = ManipulationController()

    def execute_grasp_with_interaction(self, object_info):
        """Execute grasp task with interaction feedback"""
        print("Starting interactive grasp task...")

        # Plan the grasp
        grasp_config = self.grasp_planner.synthesize_power_grasp(object_info['dimensions'])
        optimized_grasp = self.grasp_planner.optimize_grasp_configuration(
            grasp_config,
            {'dimensions': object_info['dimensions'], 'weight': 0.1, 'friction': 0.5}
        )

        # Inform user about the plan
        self.interaction_manager.speak_response(
            f"I plan to grasp the {object_info['type']} with a power grasp. Is this OK?"
        )

        # Wait for user confirmation (simplified)
        time.sleep(2)  # Simulate waiting for response

        # Execute the grasp
        success = self.execute_grasp_safely(optimized_grasp)

        if success:
            self.interaction_manager.speak_response("Successfully grasped the object!")
        else:
            self.interaction_manager.speak_response("Grasp failed. Would you like me to try again?")

        return success

    def execute_grasp_safely(self, grasp_config):
        """Execute grasp with safety checks"""
        try:
            self.manipulation_controller.execute_grasp(grasp_config)
            success = self.manipulation_controller.verify_grasp_success()
            return success
        except Exception as e:
            print(f"Grasp execution error: {e}")
            return False

# Example usage
def example_human_robot_interaction():
    executor = ManipulationTaskExecutor()

    # Simulate an object to grasp
    object_info = {
        'type': 'bottle',
        'position': [0.5, 0.3, 0.1],
        'dimensions': [0.05, 0.15, 0.05],  # width, height, depth
        'graspable': True
    }

    print("Starting interactive manipulation task...")
    success = executor.execute_grasp_with_interaction(object_info)

    if success:
        print("Task completed successfully!")
    else:
        print("Task failed.")

if __name__ == "__main__":
    try:
        example_human_robot_interaction()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: speech_recognition, opencv-python")
```

## Advanced Manipulation Strategies

### Dual-Arm Coordination

```python
# dual_arm_manipulation.py
import numpy as np
from scipy.spatial.transform import Rotation as R

class DualArmCoordinator:
    def __init__(self):
        # Robot arm parameters
        self.left_arm_base = np.array([0.2, 0.3, 0.0])
        self.right_arm_base = np.array([0.2, -0.3, 0.0])
        self.arm_reach = 0.8  # meters

    def coordinate_dual_arm_grasp(self, object_pose, object_dims):
        """Coordinate dual-arm grasp for large objects"""
        object_pos = object_pose['position']
        object_size = object_dims

        # Determine appropriate grasp points for each arm
        grasp_points = self.calculate_dual_arm_grasp_points(object_pos, object_size)

        left_grasp = grasp_points['left']
        right_grasp = grasp_points['right']

        # Generate trajectories for both arms
        left_trajectory = self.generate_reach_trajectory(
            self.left_arm_base, left_grasp['position'], left_grasp['orientation']
        )

        right_trajectory = self.generate_reach_trajectory(
            self.right_arm_base, right_grasp['position'], right_grasp['orientation']
        )

        # Synchronize the trajectories
        synchronized_trajectories = self.synchronize_trajectories(
            left_trajectory, right_trajectory
        )

        return synchronized_trajectories

    def calculate_dual_arm_grasp_points(self, object_pos, object_size):
        """Calculate appropriate grasp points for dual arms"""
        width, height, depth = object_size

        # Calculate grasp points on opposite sides of the object
        left_grasp_pos = object_pos + np.array([-width/2 - 0.05, 0, 0])  # 5cm offset
        right_grasp_pos = object_pos + np.array([width/2 + 0.05, 0, 0])  # 5cm offset

        # Orient grasps to oppose each other
        left_orientation = R.from_euler('xyz', [0, 0, np.pi/2]).as_matrix()  # Point inward
        right_orientation = R.from_euler('xyz', [0, 0, -np.pi/2]).as_matrix()  # Point inward

        return {
            'left': {
                'position': left_grasp_pos,
                'orientation': left_orientation
            },
            'right': {
                'position': right_grasp_pos,
                'orientation': right_orientation
            }
        }

    def generate_reach_trajectory(self, start_pos, end_pos, end_orientation, steps=50):
        """Generate reaching trajectory for a single arm"""
        trajectory = []

        for i in range(steps + 1):
            t = i / steps  # Interpolation parameter

            # Linear interpolation for position
            pos = start_pos + t * (end_pos - start_pos)

            # Slerp for orientation (simplified as linear interpolation)
            orientation = self.interpolate_orientations(
                np.eye(3), end_orientation, t
            )

            trajectory.append({
                'position': pos,
                'orientation': orientation,
                'time': t
            })

        return trajectory

    def interpolate_orientations(self, start_rot, end_rot, t):
        """Interpolate between two orientations"""
        # Convert to quaternions for proper interpolation
        start_quat = R.from_matrix(start_rot).as_quat()
        end_quat = R.from_matrix(end_rot).as_quat()

        # Slerp (simplified as linear interpolation)
        interpolated_quat = (1 - t) * start_quat + t * end_quat
        interpolated_quat = interpolated_quat / np.linalg.norm(interpolated_quat)

        return R.from_quat(interpolated_quat).as_matrix()

    def synchronize_trajectories(self, left_traj, right_traj):
        """Synchronize trajectories of both arms"""
        # Ensure both trajectories have the same number of steps
        min_steps = min(len(left_traj), len(right_traj))

        synchronized = {
            'left': left_traj[:min_steps],
            'right': right_traj[:min_steps]
        }

        # Add coordination information
        for i in range(min_steps):
            # Calculate distance between hands
            dist = np.linalg.norm(
                synchronized['left'][i]['position'] -
                synchronized['right'][i]['position']
            )

            synchronized['left'][i]['hand_distance'] = dist
            synchronized['right'][i]['hand_distance'] = dist

        return synchronized

    def execute_coordinated_task(self, task_type, object_info):
        """Execute coordinated dual-arm task"""
        if task_type == 'lift_large_object':
            return self.lift_large_object(object_info)
        elif task_type == 'assemble_parts':
            return self.assemble_parts(object_info)
        elif task_type == 'open_container':
            return self.open_container(object_info)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def lift_large_object(self, object_info):
        """Lift large object using coordinated dual arms"""
        print("Coordinated lifting of large object...")

        # Plan dual-arm grasp
        trajectories = self.coordinate_dual_arm_grasp(
            object_info['pose'],
            object_info['dimensions']
        )

        # Execute coordinated lift
        lift_height = object_info['pose']['position'][2] + 0.2  # Lift 20cm

        # Add lift motion to trajectories
        for i in range(len(trajectories['left'])):
            trajectories['left'][i]['position'][2] += (i / len(trajectories['left'])) * 0.2
            trajectories['right'][i]['position'][2] += (i / len(trajectories['right'])) * 0.2

        print("Large object lifted successfully with dual-arm coordination")
        return True

    def assemble_parts(self, parts_info):
        """Assemble parts using dual-arm coordination"""
        print("Assembling parts with dual arms...")

        # Left arm holds one part, right arm manipulates the other
        left_arm_task = {
            'action': 'hold',
            'object': parts_info['part1'],
            'grasp_type': 'power'
        }

        right_arm_task = {
            'action': 'manipulate',
            'object': parts_info['part2'],
            'target_pose': parts_info['assembly_pose'],
            'motion_type': 'insertion'
        }

        # Execute coordinated assembly
        print("Parts assembled successfully")
        return True

class BimanualManipulationPlanner:
    """Plan bimanual manipulation tasks"""
    def __init__(self):
        self.dual_arm_coordinator = DualArmCoordinator()

    def plan_bimanual_task(self, task_description):
        """Plan a bimanual manipulation task"""
        task_type = task_description['type']

        if task_type == 'pouring':
            return self.plan_pouring_task(task_description)
        elif task_type == 'opening':
            return self.plan_opening_task(task_description)
        elif task_type == 'supporting':
            return self.plan_supporting_task(task_description)
        else:
            raise ValueError(f"Unknown bimanual task: {task_type}")

    def plan_pouring_task(self, task_description):
        """Plan bimanual pouring task"""
        # Left arm holds container, right arm controls pouring
        container_pose = task_description['container_pose']
        target_pose = task_description['target_pose']

        # Left arm: stable grasp of container
        left_grasp = {
            'position': container_pose['position'],
            'orientation': self.calculate_container_grasp_orientation(),
            'grasp_type': 'power'
        }

        # Right arm: pouring motion
        pour_trajectory = self.generate_pouring_motion(
            container_pose, target_pose
        )

        return {
            'left_arm_task': {
                'action': 'hold',
                'grasp': left_grasp
            },
            'right_arm_task': {
                'action': 'pour',
                'trajectory': pour_trajectory
            },
            'coordination': 'synchronized'
        }

    def calculate_container_grasp_orientation(self):
        """Calculate appropriate grasp orientation for container"""
        # Grasp container handle with thumb up
        return R.from_euler('xyz', [0, 0, 0]).as_matrix()

    def generate_pouring_motion(self, container_pose, target_pose):
        """Generate pouring motion trajectory"""
        # Calculate motion from container to target
        container_pos = container_pose['position']
        target_pos = target_pose['position']

        # Define key points for pouring motion
        lift_point = container_pos + np.array([0, 0, 0.1])  # Lift slightly
        pour_point = np.array([
            target_pos[0], target_pos[1], container_pos[2]  # At target x,y but same height as container
        ])
        tilt_point = pour_point + np.array([0.1, 0, -0.05])  # Tilt forward

        return [container_pos, lift_point, pour_point, tilt_point]

    def plan_opening_task(self, task_description):
        """Plan bimanual opening task (e.g., opening a jar)"""
        jar_pose = task_description['jar_pose']

        # Left arm: stabilize the jar
        left_grasp = {
            'position': jar_pose['position'],
            'orientation': self.calculate_stabilizing_grasp_orientation(),
            'grasp_type': 'tripod'  # Three-finger grasp for stability
        }

        # Right arm: twist the lid
        twist_trajectory = self.generate_twist_motion(jar_pose)

        return {
            'left_arm_task': {
                'action': 'stabilize',
                'grasp': left_grasp
            },
            'right_arm_task': {
                'action': 'twist',
                'trajectory': twist_trajectory
            },
            'coordination': 'force_balance'
        }

    def calculate_stabilizing_grasp_orientation(self):
        """Calculate orientation for stabilizing grasp"""
        # Grasp the bottom of the jar for stability
        return R.from_euler('xyz', [0, np.pi, 0]).as_matrix()

    def generate_twist_motion(self, jar_pose):
        """Generate twisting motion for jar opening"""
        # Circular motion around jar axis
        jar_pos = jar_pose['position']
        radius = 0.03  # 3cm from center

        trajectory = []
        for angle in np.linspace(0, 4*np.pi, 100):  # 2 full rotations
            x = jar_pos[0] + radius * np.cos(angle)
            y = jar_pos[1] + radius * np.sin(angle)
            z = jar_pos[2] + 0.02 * np.sin(4*angle)  # Small vertical oscillation

            trajectory.append(np.array([x, y, z]))

        return trajectory

# Example usage
def example_dual_arm_manipulation():
    coordinator = DualArmCoordinator()
    bimanual_planner = BimanualManipulationPlanner()

    # Example 1: Lifting a large object
    large_object = {
        'pose': {'position': np.array([0.6, 0.0, 0.1])},
        'dimensions': [0.3, 0.2, 0.1]  # width, height, depth
    }

    success = coordinator.lift_large_object(large_object)
    print(f"Large object lift success: {success}")

    # Example 2: Planning a bimanual task
    pouring_task = {
        'type': 'pouring',
        'container_pose': {'position': np.array([0.5, 0.1, 0.2])},
        'target_pose': {'position': np.array([0.7, 0.1, 0.15])}
    }

    pouring_plan = bimanual_planner.plan_pouring_task(pouring_task)
    print(f"Pouring task planned: {pouring_plan['coordination']}")

if __name__ == "__main__":
    example_dual_arm_manipulation()
```

## Knowledge Check

1. What are the key differences between power grasps and precision grasps?
2. How does force closure contribute to grasp stability?
3. What are the advantages of multi-modal interaction in manipulation tasks?
4. How do dual-arm coordination strategies improve manipulation capabilities?

## Summary

This chapter covered advanced manipulation and grasping techniques for humanoid robots. We explored geometric and learning-based grasp planning algorithms, dexterous manipulation strategies for multi-finger hands, and human-robot interaction design for manipulation tasks. The chapter also covered multi-modal interaction approaches and dual-arm coordination strategies for complex manipulation scenarios.

## Next Steps

In the next chapter, we'll explore natural human-robot interaction, covering communication paradigms, user experience design, and advanced interaction techniques for humanoid robotics applications.