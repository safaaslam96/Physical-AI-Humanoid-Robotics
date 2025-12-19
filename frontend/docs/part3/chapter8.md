---
sidebar_position: 8
title: "Chapter 8: Unity Visualization and Sensor Simulation"
---

# Chapter 8: Unity Visualization and Sensor Simulation

## Learning Objectives
- Understand Unity's role in robotics visualization
- Set up Unity for robot visualization and simulation
- Implement physics simulation and sensor simulation in Unity
- Create high-fidelity visualizations for humanoid robots

## Introduction to Unity for Robotics

Unity is a powerful 3D development platform that has gained significant traction in robotics for creating high-fidelity visualizations, photorealistic simulations, and immersive environments. Unlike Gazebo, which is primarily physics-focused, Unity excels in visual quality and can be integrated with robotics frameworks for advanced visualization and simulation capabilities.

### Unity vs Gazebo for Robotics

| Aspect | Unity | Gazebo |
|--------|-------|--------|
| **Visual Quality** | Photorealistic graphics | Good but not photorealistic |
| **Physics Simulation** | Good (Unity Physics) | Excellent (ODE, Bullet, DART) |
| **Robotics Integration** | Through ROS# or Unity Robotics Package | Native ROS integration |
| **Development Environment** | Visual IDE with C# scripting | Text-based SDF/URDF |
| **Real-time Performance** | Excellent for visualization | Optimized for physics |
| **Learning Curve** | Moderate (C# programming) | Moderate (SDF/URDF) |

### Unity Robotics Package

Unity provides the Unity Robotics Package (URP) and Unity ML-Agents Toolkit specifically for robotics applications:

- **ROS#**: Bridge between Unity and ROS/ROS 2
- **ML-Agents**: Machine learning framework for training agents
- **Simulation Framework**: Tools for creating simulation environments
- **Sensor Simulation**: Realistic sensor simulation capabilities

## Setting Up Unity for Robotics

### Installation Requirements

1. **Unity Hub**: Download from unity.com
2. **Unity Editor**: Version 2021.3 LTS or later recommended
3. **Unity Robotics Package**: Available through Package Manager
4. **ROS/ROS 2 Bridge**: ROS# package for communication

### Creating a Robotics Project

1. Create a new 3D project in Unity
2. Install required packages through Package Manager:
   - Unity Robotics Package
   - ML-Agents (if needed for training)
   - ProBuilder (for quick environment creation)

3. Import the ROS# package for ROS/ROS 2 communication

### Basic Unity Robotics Scene Setup

```csharp
// RobotController.cs - Basic robot controller for Unity
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    [SerializeField] private float linearSpeed = 1.0f;
    [SerializeField] private float angularSpeed = 1.0f;

    private ROSConnection ros;
    private float linearVelocity = 0f;
    private float angularVelocity = 0f;

    void Start()
    {
        // Connect to ROS
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>("/cmd_vel");

        // Subscribe to velocity commands
        ros.Subscribe<TwistMsg>("/cmd_vel", ReceiveVelocityCommand);
    }

    void Update()
    {
        // Apply differential drive kinematics
        if (linearVelocity != 0 || angularVelocity != 0)
        {
            // Convert Twist to differential drive
            float leftWheel = linearVelocity - angularVelocity * 0.5f; // Assuming 1m wheelbase
            float rightWheel = linearVelocity + angularVelocity * 0.5f;

            // Apply to robot (assuming differential drive)
            transform.Translate(Vector3.forward * leftWheel * Time.deltaTime * linearSpeed);
            transform.Rotate(Vector3.up, (rightWheel - leftWheel) * angularSpeed * Time.deltaTime);
        }
    }

    void ReceiveVelocityCommand(TwistMsg cmd)
    {
        linearVelocity = (float)cmd.linear.x;
        angularVelocity = (float)cmd.angular.z;
    }

    void OnDestroy()
    {
        if (ros != null)
            ros.UnregisterPublisher<TwistMsg>("/cmd_vel");
    }
}
```

## Physics Simulation in Unity

### Unity Physics vs Traditional Robotics Physics

Unity uses its own physics engine (PhysX) which differs from traditional robotics simulators:

- **Unity Physics**: Optimized for games and visual quality
- **ROS Physics**: Optimized for accuracy and consistency
- **Hybrid Approach**: Use Unity for visualization, Gazebo for physics

### Setting Up Physics for Humanoid Robots

```csharp
// HumanoidPhysicsController.cs
using UnityEngine;

public class HumanoidPhysicsController : MonoBehaviour
{
    [Header("Balance Parameters")]
    [SerializeField] private float balanceThreshold = 0.1f;
    [SerializeField] private float balanceCorrectionStrength = 10f;

    [Header("Joint Configuration")]
    [SerializeField] private ConfigurableJoint[] joints;
    [SerializeField] private Transform centerOfMass;

    [Header("Sensor Simulation")]
    [SerializeField] private Transform[] feetSensors;
    [SerializeField] private Transform[] imuTransform;

    private Rigidbody rb;
    private Vector3 targetCOM;
    private bool[] feetContact = new bool[2]; // Left, Right

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (centerOfMass != null)
        {
            rb.centerOfMass = centerOfMass.localPosition;
        }

        SetupJoints();
    }

    void FixedUpdate()
    {
        UpdateSensors();
        ApplyBalanceControl();
        UpdateJointControl();
    }

    void SetupJoints()
    {
        foreach (var joint in joints)
        {
            // Configure joint limits and spring properties
            var jointDrive = joint.angularXDrive;
            jointDrive.positionSpring = 10000f; // Stiffness
            jointDrive.positionDamper = 1000f;  // Damping
            joint.angularXDrive = jointDrive;

            var yzDrive = joint.angularYZDrive;
            yzDrive.positionSpring = 10000f;
            yzDrive.positionDamper = 1000f;
            joint.angularYZDrive = yzDrive;
        }
    }

    void UpdateSensors()
    {
        // Ground contact sensors
        for (int i = 0; i < feetSensors.Length; i++)
        {
            RaycastHit hit;
            feetContact[i] = Physics.Raycast(feetSensors[i].position,
                Vector3.down, out hit, 0.1f);
        }

        // IMU simulation
        Vector3 angularVelocity = rb.angularVelocity;
        Vector3 linearAcceleration = rb.velocity / Time.fixedDeltaTime;
    }

    void ApplyBalanceControl()
    {
        // Simple balance control based on COM position
        Vector3 comOffset = rb.worldCenterOfMass - transform.position;
        comOffset.y = 0; // Ignore height

        if (comOffset.magnitude > balanceThreshold)
        {
            Vector3 correction = -comOffset.normalized * balanceCorrectionStrength * Time.fixedDeltaTime;
            rb.AddForceAtPosition(correction, rb.worldCenterOfMass, ForceMode.VelocityChange);
        }
    }

    void UpdateJointControl()
    {
        // Apply joint control based on desired positions
        // This would typically come from a higher-level controller
    }
}
```

### Collision Detection and Response

```csharp
// CollisionHandler.cs
using UnityEngine;

public class CollisionHandler : MonoBehaviour
{
    [Header("Collision Response")]
    [SerializeField] private float collisionThreshold = 10f;
    [SerializeField] private LayerMask collisionLayers;

    [Header("Damage Control")]
    [SerializeField] private float maxImpactForce = 100f;

    void OnCollisionEnter(Collision collision)
    {
        float impactForce = collision.impulse.magnitude / Time.fixedDeltaTime;

        if (impactForce > collisionThreshold)
        {
            Debug.Log($"High impact detected: {impactForce}N");

            if (impactForce > maxImpactForce)
            {
                HandleDamage(collision);
            }

            HandleImpact(collision);
        }
    }

    void HandleImpact(Collision collision)
    {
        // Play impact sound
        // Visual effects
        // Update robot state
    }

    void HandleDamage(Collision collision)
    {
        // Safety shutdown procedures
        Debug.LogWarning("Potential damage detected - safety shutdown initiated");
    }
}
```

## Sensor Simulation in Unity

### Camera Simulation

```csharp
// CameraSensor.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections;

public class CameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    [SerializeField] private Camera cameraComponent;
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 480;
    [SerializeField] private float updateRate = 30f; // Hz

    [Header("ROS Communication")]
    [SerializeField] private string imageTopic = "/camera/image_raw";

    private RenderTexture renderTexture;
    private Texture2D outputTexture;
    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1f / updateRate;

        SetupCamera();
        StartCoroutine(SendImages());
    }

    void SetupCamera()
    {
        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cameraComponent.targetTexture = renderTexture;

        outputTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    IEnumerator SendImages()
    {
        while (true)
        {
            if (Time.time - lastUpdateTime >= updateInterval)
            {
                CaptureAndSendImage();
                lastUpdateTime = Time.time;
            }
            yield return null;
        }
    }

    void CaptureAndSendImage()
    {
        // Set active render texture
        RenderTexture.active = renderTexture;

        // Read pixels from render texture
        outputTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        outputTexture.Apply();

        // Convert to ROS image message
        var imageMsg = CreateImageMessage(outputTexture);

        // Send to ROS
        ros.Publish(imageTopic, imageMsg);

        // Reset active render texture
        RenderTexture.active = null;
    }

    ImageMsg CreateImageMessage(Texture2D texture)
    {
        // Convert texture to byte array
        byte[] imageData = texture.EncodeToJPG();

        var imageMsg = new ImageMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = transform.name
            },
            height = (uint)texture.height,
            width = (uint)texture.width,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(texture.width * 3), // 3 bytes per pixel (RGB)
            data = imageData
        };

        return imageMsg;
    }
}
```

### LIDAR Simulation

```csharp
// LidarSensor.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    [SerializeField] private int numberOfRays = 720;
    [SerializeField] private float fieldOfView = 360f; // degrees
    [SerializeField] private float maxRange = 30f;
    [SerializeField] private float minRange = 0.1f;
    [SerializeField] private float updateRate = 10f; // Hz

    [Header("Noise Parameters")]
    [SerializeField] private float noiseStdDev = 0.01f;

    [Header("ROS Communication")]
    [SerializeField] private string scanTopic = "/scan";

    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1f / updateRate;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishLaserScan();
            lastUpdateTime = Time.time;
        }
    }

    void PublishLaserScan()
    {
        var scanMsg = new LaserScanMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = transform.name
            },
            angle_min = -fieldOfView * Mathf.Deg2Rad / 2,
            angle_max = fieldOfView * Mathf.Deg2Rad / 2,
            angle_increment = (fieldOfView * Mathf.Deg2Rad) / numberOfRays,
            time_increment = 0,
            scan_time = 1f / updateRate,
            range_min = minRange,
            range_max = maxRange
        };

        // Sample ranges
        List<float> ranges = new List<float>();
        float angleStep = fieldOfView / numberOfRays;

        for (int i = 0; i < numberOfRays; i++)
        {
            float angle = (i * angleStep - fieldOfView / 2) * Mathf.Deg2Rad;

            // Calculate ray direction
            Vector3 rayDirection = new Vector3(
                Mathf.Cos(angle),
                0,
                Mathf.Sin(angle)
            );

            rayDirection = transform.TransformDirection(rayDirection);

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, rayDirection, out hit, maxRange))
            {
                float range = hit.distance;
                // Add noise
                range += Random.Range(-noiseStdDev, noiseStdDev);
                ranges.Add(Mathf.Clamp(range, minRange, maxRange));
            }
            else
            {
                ranges.Add(float.PositiveInfinity);
            }
        }

        scanMsg.ranges = ranges.ToArray();

        // Intensities (optional)
        float[] intensities = new float[ranges.Count];
        for (int i = 0; i < intensities.Length; i++)
        {
            intensities[i] = 100f; // Default intensity
        }
        scanMsg.intensities = intensities;

        ros.Publish(scanTopic, scanMsg);
    }
}
```

### IMU Simulation

```csharp
// IMUSensor.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class IMUSensor : MonoBehaviour
{
    [Header("Noise Parameters")]
    [SerializeField] private float angularVelocityNoise = 0.01f;
    [SerializeField] private float linearAccelerationNoise = 0.1f;

    [Header("ROS Communication")]
    [SerializeField] private string imuTopic = "/imu/data";

    [Header("Reference Frame")]
    [SerializeField] private Transform referenceFrame;

    private ROSConnection ros;
    private Rigidbody rb;
    private float updateInterval = 0.01f; // 100Hz
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishIMUData();
            lastUpdateTime = Time.time;
        }
    }

    void PublishIMUData()
    {
        var imuMsg = new ImuMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = transform.name
            }
        };

        // Orientation (from Unity rotation)
        Quaternion rotation = transform.rotation;
        imuMsg.orientation.x = rotation.x;
        imuMsg.orientation.y = rotation.y;
        imuMsg.orientation.z = rotation.z;
        imuMsg.orientation.w = rotation.w;

        // Angular velocity
        Vector3 angularVel = rb.angularVelocity;
        angularVel.x += Random.Range(-angularVelocityNoise, angularVelocityNoise);
        angularVel.y += Random.Range(-angularVelocityNoise, angularVelocityNoise);
        angularVel.z += Random.Range(-angularVelocityNoise, angularVelocityNoise);

        imuMsg.angular_velocity.x = angularVel.x;
        imuMsg.angular_velocity.y = angularVel.y;
        imuMsg.angular_velocity.z = angularVel.z;

        // Linear acceleration (in world frame, then convert to body frame)
        Vector3 linearAcc = rb.velocity / Time.deltaTime;
        linearAcc = transform.InverseTransformDirection(linearAcc - Physics.gravity);

        linearAcc.x += Random.Range(-linearAccelerationNoise, linearAccelerationNoise);
        linearAcc.y += Random.Range(-linearAccelerationNoise, linearAccelerationNoise);
        linearAcc.z += Random.Range(-linearAccelerationNoise, linearAccelerationNoise);

        imuMsg.linear_acceleration.x = linearAcc.x;
        imuMsg.linear_acceleration.y = linearAcc.y;
        imuMsg.linear_acceleration.z = linearAcc.z;

        // Covariance matrices (set to 0 for now, but should be properly configured)
        imuMsg.orientation_covariance = new double[9] { -1, 0, 0, 0, 0, 0, 0, 0, 0 };
        imuMsg.angular_velocity_covariance = new double[9] { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        imuMsg.linear_acceleration_covariance = new double[9] { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        ros.Publish(imuTopic, imuMsg);
    }
}
```

## Advanced Visualization Techniques

### High-Fidelity Rendering

Unity excels at creating photorealistic environments and robot models:

```csharp
// HighFidelityRobot.cs
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Renderer))]
public class HighFidelityRobot : MonoBehaviour
{
    [Header("Material Properties")]
    [SerializeField] private Material robotMaterial;
    [SerializeField] private PhysicallyBasedMaterial robotPBR;

    [Header("Lighting Configuration")]
    [SerializeField] private Light[] robotLights;
    [SerializeField] private bool useRealisticShading = true;

    [Header("Reflection Probes")]
    [SerializeField] private ReflectionProbe[] reflectionProbes;

    void Start()
    {
        SetupMaterials();
        ConfigureLighting();
    }

    void SetupMaterials()
    {
        if (robotMaterial != null)
        {
            // Configure PBR properties
            robotMaterial.SetFloat("_Metallic", 0.7f);
            robotMaterial.SetFloat("_Smoothness", 0.8f);
            robotMaterial.SetColor("_Color", Color.gray);

            // Add wear and tear effects
            robotMaterial.SetFloat("_ScratchIntensity", 0.2f);
        }
    }

    void ConfigureLighting()
    {
        // Set up robot-specific lighting
        foreach (var light in robotLights)
        {
            light.shadows = LightShadows.Soft;
            light.intensity = 1.5f;
        }
    }

    public void UpdateRobotState(bool poweredOn, bool inMotion)
    {
        // Update material properties based on robot state
        if (robotMaterial != null)
        {
            float emission = poweredOn ? 0.2f : 0f;
            robotMaterial.SetColor("_EmissionColor",
                new Color(emission, emission, emission));

            // Add motion blur effect when moving
            if (inMotion)
            {
                // Apply motion blur shader or post-processing effect
            }
        }
    }
}
```

### Environment Creation

Creating realistic environments for humanoid robot testing:

```csharp
// EnvironmentManager.cs
using UnityEngine;
using System.Collections.Generic;

public class EnvironmentManager : MonoBehaviour
{
    [Header("Environment Prefabs")]
    [SerializeField] private GameObject[] indoorPrefabs;
    [SerializeField] private GameObject[] outdoorPrefabs;
    [SerializeField] private GameObject[] obstaclePrefabs;

    [Header("Terrain Configuration")]
    [SerializeField] private Terrain terrain;
    [SerializeField] private float terrainScale = 100f;

    [Header("Weather System")]
    [SerializeField] private bool enableWeather = true;
    [SerializeField] private float weatherChangeInterval = 300f; // 5 minutes

    private List<GameObject> spawnedObjects = new List<GameObject>();
    private float lastWeatherChange;

    void Start()
    {
        GenerateEnvironment();
        StartCoroutine(WeatherSystem());
    }

    void GenerateEnvironment()
    {
        // Create indoor environment
        if (indoorPrefabs.Length > 0)
        {
            CreateIndoorEnvironment();
        }

        // Add obstacles and interactive elements
        SpawnObstacles();

        // Configure terrain
        if (terrain != null)
        {
            ConfigureTerrain();
        }
    }

    void CreateIndoorEnvironment()
    {
        // Create rooms, corridors, furniture
        foreach (var prefab in indoorPrefabs)
        {
            Vector3 position = new Vector3(
                Random.Range(-terrainScale/2, terrainScale/2),
                0,
                Random.Range(-terrainScale/2, terrainScale/2)
            );

            GameObject instance = Instantiate(prefab, position, Quaternion.identity);
            spawnedObjects.Add(instance);
        }
    }

    void SpawnObstacles()
    {
        for (int i = 0; i < 20; i++) // Spawn 20 random obstacles
        {
            int prefabIndex = Random.Range(0, obstaclePrefabs.Length);
            Vector3 position = new Vector3(
                Random.Range(-terrainScale/3, terrainScale/3),
                0.5f, // Half the height of a human
                Random.Range(-terrainScale/3, terrainScale/3)
            );

            GameObject obstacle = Instantiate(obstaclePrefabs[prefabIndex], position, Quaternion.identity);
            spawnedObjects.Add(obstacle);
        }
    }

    void ConfigureTerrain()
    {
        // Configure terrain properties for realistic ground
        terrain.terrainData.size = new Vector3(terrainScale, 20f, terrainScale);

        // Add texture layers for different ground types
        SplatPrototype[] splatPrototypes = new SplatPrototype[3];

        // Grass
        splatPrototypes[0] = new SplatPrototype();
        splatPrototypes[0].texture = Resources.Load<Texture2D>("TerrainTextures/grass");
        splatPrototypes[0].tileSize = new Vector2(5f, 5f);

        // Concrete
        splatPrototypes[1] = new SplatPrototype();
        splatPrototypes[1].texture = Resources.Load<Texture2D>("TerrainTextures/concrete");
        splatPrototypes[1].tileSize = new Vector2(2f, 2f);

        // Dirt
        splatPrototypes[2] = new SplatPrototype();
        splatPrototypes[2].texture = Resources.Load<Texture2D>("TerrainTextures/dirt");
        splatPrototypes[2].tileSize = new Vector2(3f, 3f);

        terrain.terrainData.splatPrototypes = splatPrototypes;

        // Generate alphamap for texture blending
        float[,,] alphamap = new float[terrain.terrainData.alphamapWidth,
                                       terrain.terrainData.alphamapHeight,
                                       splatPrototypes.Length];

        // Simple pattern - grass in center, concrete around edges
        for (int y = 0; y < terrain.terrainData.alphamapHeight; y++)
        {
            for (int x = 0; x < terrain.terrainData.alphamapWidth; x++)
            {
                float normalizedX = (float)x / terrain.terrainData.alphamapWidth;
                float normalizedY = (float)y / terrain.terrainData.alphamapHeight;

                float distanceFromCenter = Mathf.Sqrt(
                    Mathf.Pow(normalizedX - 0.5f, 2) +
                    Mathf.Pow(normalizedY - 0.5f, 2)
                );

                if (distanceFromCenter < 0.3f)
                {
                    alphamap[x, y, 0] = 1f; // Grass
                    alphamap[x, y, 1] = 0f;
                    alphamap[x, y, 2] = 0f;
                }
                else if (distanceFromCenter < 0.4f)
                {
                    alphamap[x, y, 0] = 0.5f; // Mixed
                    alphamap[x, y, 1] = 0.5f;
                    alphamap[x, y, 2] = 0f;
                }
                else
                {
                    alphamap[x, y, 0] = 0f;
                    alphamap[x, y, 1] = 1f; // Concrete
                    alphamap[x, y, 2] = 0f;
                }
            }
        }

        terrain.terrainData.SetAlphamaps(0, 0, alphamap);
    }

    System.Collections.IEnumerator WeatherSystem()
    {
        while (true)
        {
            if (enableWeather)
            {
                ChangeWeather();
            }
            yield return new WaitForSeconds(weatherChangeInterval);
        }
    }

    void ChangeWeather()
    {
        // Simple weather system - could be expanded with particle systems
        float weatherType = Random.value;

        if (weatherType < 0.3f)
        {
            // Sunny
            RenderSettings.ambientLight = new Color(0.8f, 0.8f, 0.8f);
            RenderSettings.fog = false;
        }
        else if (weatherType < 0.6f)
        {
            // Cloudy
            RenderSettings.ambientLight = new Color(0.6f, 0.6f, 0.7f);
            RenderSettings.fog = true;
            RenderSettings.fogColor = new Color(0.7f, 0.7f, 0.8f);
            RenderSettings.fogDensity = 0.01f;
        }
        else
        {
            // Overcast/Rainy
            RenderSettings.ambientLight = new Color(0.4f, 0.4f, 0.5f);
            RenderSettings.fog = true;
            RenderSettings.fogColor = new Color(0.5f, 0.5f, 0.6f);
            RenderSettings.fogDensity = 0.02f;
        }
    }
}
```

## Integration with ROS 2

### ROS# Bridge Setup

```csharp
// ROSBridgeManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class ROSBridgeManager : MonoBehaviour
{
    [Header("ROS Connection")]
    [SerializeField] private string rosIPAddress = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;

    [Header("Robot Configuration")]
    [SerializeField] private string robotNamespace = "/humanoid_robot";

    private ROSConnection ros;
    private bool isConnected = false;

    void Start()
    {
        ConnectToROS();
        SetupROSCommunications();
    }

    void ConnectToROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);

        // Test connection
        InvokeRepeating("TestConnection", 1f, 5f);
    }

    void SetupROSCommunications()
    {
        // Publisher setup
        ros.RegisterPublisher<UInt8Msg>($"{robotNamespace}/status");
        ros.RegisterPublisher<StringMsg>($"{robotNamespace}/log");

        // Subscriber setup
        ros.Subscribe<StringMsg>($"{robotNamespace}/command", HandleCommand);
    }

    void TestConnection()
    {
        var testMsg = new UInt8Msg();
        testMsg.data = 1;
        ros.Publish($"{robotNamespace}/heartbeat", testMsg);
    }

    void HandleCommand(StringMsg cmd)
    {
        Debug.Log($"Received command: {cmd.data}");

        // Process command and update robot state
        ProcessRobotCommand(cmd.data);
    }

    void ProcessRobotCommand(string command)
    {
        switch (command.ToLower())
        {
            case "move_forward":
                // Trigger movement
                break;
            case "turn_left":
                // Trigger turn
                break;
            case "home_position":
                // Return to home position
                break;
            default:
                Debug.LogWarning($"Unknown command: {command}");
                break;
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.Disconnect();
        }
    }
}
```

### TF (Transform) Broadcasting

```csharp
// TFBroadcaster.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class TFBroadcaster : MonoBehaviour
{
    [Header("TF Configuration")]
    [SerializeField] private Transform[] robotLinks;
    [SerializeField] private string[] linkNames;
    [SerializeField] private string baseFrame = "odom";

    private ROSConnection ros;
    private float tfUpdateInterval = 0.05f; // 20Hz
    private float lastTFUpdate;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        if (robotLinks.Length != linkNames.Length)
        {
            Debug.LogError("Robot links and link names arrays must have the same length!");
        }
    }

    void Update()
    {
        if (Time.time - lastTFUpdate >= tfUpdateInterval)
        {
            BroadcastTransforms();
            lastTFUpdate = Time.time;
        }
    }

    void BroadcastTransforms()
    {
        var tfArray = new RosMessageTypes.Geometry.TFMessageMsg();
        tfArray.transforms = new TransformStampedMsg[robotLinks.Length];

        for (int i = 0; i < robotLinks.Length; i++)
        {
            tfArray.transforms[i] = CreateTransformStamped(
                baseFrame,
                linkNames[i],
                robotLinks[i]
            );
        }

        ros.Publish("tf", tfArray);
    }

    TransformStampedMsg CreateTransformStamped(string parentFrame, string childFrame, Transform transform)
    {
        var tf = new TransformStampedMsg
        {
            header = new HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = parentFrame
            },
            child_frame_id = childFrame,
            transform = new TransformMsg
            {
                translation = new Vector3Msg
                {
                    x = transform.position.x,
                    y = transform.position.y,
                    z = transform.position.z
                },
                rotation = new QuaternionMsg
                {
                    x = transform.rotation.x,
                    y = transform.rotation.y,
                    z = transform.rotation.z,
                    w = transform.rotation.w
                }
            }
        };

        return tf;
    }
}
```

## Performance Optimization

### Unity Performance Considerations

When creating high-fidelity visualizations for robotics:

1. **LOD (Level of Detail)**: Use different models based on distance
2. **Occlusion Culling**: Hide objects not visible to cameras
3. **Texture Compression**: Use appropriate texture formats
4. **Shader Optimization**: Use mobile/desktop optimized shaders

```csharp
// LODManager.cs
using UnityEngine;

public class LODManager : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public string name;
        public GameObject model;
        public float distance;
        public Material[] materials;
    }

    [SerializeField] private LODLevel[] lodLevels;
    [SerializeField] private Transform playerCamera;
    [SerializeField] private float lodUpdateInterval = 0.1f;

    private float lastLODUpdate;
    private int currentLODLevel = 0;

    void Update()
    {
        if (Time.time - lastLODUpdate >= lodUpdateInterval)
        {
            UpdateLOD();
            lastLODUpdate = Time.time;
        }
    }

    void UpdateLOD()
    {
        if (playerCamera == null) return;

        float distance = Vector3.Distance(transform.position, playerCamera.position);

        // Find appropriate LOD level
        int newLODLevel = 0;
        for (int i = 0; i < lodLevels.Length; i++)
        {
            if (distance <= lodLevels[i].distance)
            {
                newLODLevel = i;
                break;
            }
        }

        // Switch to new LOD level if needed
        if (newLODLevel != currentLODLevel)
        {
            SwitchLOD(newLODLevel);
            currentLODLevel = newLODLevel;
        }
    }

    void SwitchLOD(int levelIndex)
    {
        // Hide all LOD models
        foreach (var lod in lodLevels)
        {
            if (lod.model != null)
                lod.model.SetActive(false);
        }

        // Show selected LOD model
        if (levelIndex < lodLevels.Length && lodLevels[levelIndex].model != null)
        {
            lodLevels[levelIndex].model.SetActive(true);

            // Apply appropriate materials
            ApplyMaterials(levelIndex);
        }
    }

    void ApplyMaterials(int levelIndex)
    {
        var materials = lodLevels[levelIndex].materials;
        var renderers = lodLevels[levelIndex].model.GetComponentsInChildren<Renderer>();

        for (int i = 0; i < renderers.Length && i < materials.Length; i++)
        {
            renderers[i].material = materials[i];
        }
    }
}
```

## Best Practices for Unity Robotics

### Architecture Best Practices

1. **Modular Design**: Separate visualization from physics simulation
2. **Performance Monitoring**: Monitor frame rates and optimize accordingly
3. **Realistic Sensor Models**: Include noise and limitations
4. **ROS Integration**: Properly handle ROS message types and timing
5. **Scalability**: Design systems that can handle multiple robots

### Visualization Best Practices

1. **Realistic Lighting**: Use physically-based rendering
2. **Environmental Context**: Create meaningful environments
3. **Sensor Visualization**: Show sensor data overlay when needed
4. **Debug Visualization**: Include tools for debugging robot state
5. **User Interface**: Provide clear feedback about robot status

## Troubleshooting Common Issues

### Performance Issues
```csharp
// PerformanceMonitor.cs
using UnityEngine;
using UnityEngine.UI;

public class PerformanceMonitor : MonoBehaviour
{
    [SerializeField] private Text performanceText;
    [SerializeField] private float updateInterval = 0.5f;

    private float lastUpdate;
    private int frameCount = 0;
    private float accumulatedFrameTime = 0f;

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            float avgFrameTime = accumulatedFrameTime / frameCount;
            int avgFPS = Mathf.RoundToInt(1f / avgFrameTime);

            performanceText.text = $"FPS: {avgFPS}\nFrame Time: {avgFrameTime * 1000f:F1}ms";

            frameCount = 0;
            accumulatedFrameTime = 0f;
            lastUpdate = Time.time;
        }

        frameCount++;
        accumulatedFrameTime += Time.unscaledDeltaTime;
    }
}
```

### Sensor Synchronization
```csharp
// SensorSynchronizer.cs
using UnityEngine;
using System.Collections.Generic;

public class SensorSynchronizer : MonoBehaviour
{
    [SerializeField] private MonoBehaviour[] sensors;
    [SerializeField] private float syncInterval = 0.01f; // 100Hz

    private float lastSyncTime;
    private Queue<float> syncQueue = new Queue<float>();

    void Update()
    {
        if (Time.time - lastSyncTime >= syncInterval)
        {
            SynchronizeSensors();
            lastSyncTime = Time.time;
        }
    }

    void SynchronizeSensors()
    {
        foreach (var sensor in sensors)
        {
            if (sensor != null)
            {
                // Trigger sensor update
                // This ensures all sensors update at the same time
            }
        }
    }
}
```

## Knowledge Check

1. What are the key advantages of using Unity over Gazebo for robotics visualization?
2. How do you implement realistic sensor simulation in Unity?
3. What are the essential components for ROS integration in Unity?
4. How do you optimize Unity performance for high-fidelity robot visualization?

## Summary

This chapter explored Unity's role in robotics visualization and simulation, covering physics simulation, sensor modeling, and high-fidelity rendering techniques. We learned how to set up Unity for robotics applications, implement realistic sensor simulation, and integrate with ROS 2 for complete robot simulation systems. The chapter also provided best practices for performance optimization and system architecture.

## Next Steps

In the next module, we'll explore the AI-Robot Brain with NVIDIA Isaac SDK and Isaac Sim, learning about photorealistic simulation, synthetic data generation, and advanced perception techniques for humanoid robotics.