---
title: "باب 8: یونیٹی ویژولائزیشن اور سینسر سمولیشن"
sidebar_label: "باب 8: یونیٹی ویژولائزیشن"
---

# باب 8: یونیٹی ویژولائزیشن اور سینسر سمولیشن

## سیکھنے کے اہداف
- روبوٹ ویژولائزیشن اور سمولیشن کے پلیٹ فارم کے طور پر یونیٹی کو سمجھنا
- یونیٹی میں ورچوئل سینسرز نافذ کرنا بشمول LIDAR، کیمرے، اور IMUs
- ہیومنوائڈ روبوٹس کے لیے زیادہ معیار کی ویژولائزیشن ماحول تخلیق کرنا
- ریل ٹائم روبوٹ سمولیشن کے لیے یونیٹی کو ROS 2 سے جوڑنا

## تعارف

یونیٹی روبوٹکس سمولیشن اور ویژولائزیشن کے لیے ایک طاقتور پلیٹ فارم کے طور پر سامنے آیا ہے، فوٹو ریلائزٹک رینڈرنگ کی صلاحیات اور جامع فزکس سمولیشن فراہم کرتا ہے۔ جبکہ گیزبو بہترین فزکس-مبنی سمولیشن فراہم کرتا ہے، یونیٹی وہ ماحول تخلیق کرنے میں ممتاز ہے جو سمولیشن اور حقیقی دنیا کی تنصیب کے درمیان حقیقت کا فرق پُر کر سکتے ہیں۔ یہ باب ہیومنوائڈ روبوٹکس کے لیے یونیٹی کی صلاحیات کو ویژولائزیشن اور سینسر سمولیشن کے حوالے سے تلاش کرتا ہے۔

## روبوٹکس کے لیے یونیٹی کو سمجھنا

### یونیٹی بمقابلہ روایتی روبوٹکس سمولیٹرز

یونیٹی کئی کلیدی پہلوؤں میں روایتی روبوٹکس سمولیٹرز سے مختلف ہے:
- **بصری معیار**: ڈوبنے والے تجربات کے لیے فوٹو ریلائزٹک رینڈرنگ
- **ریل ٹائم کارکردگی**: ریل ٹائم تعامل کے لیے بہتر بنایا گیا
- **اثاثہ ایکو سسٹم**: 3D ماڈلز اور ماحول کی وسیع لائبریری
- **ترقی کے ٹولز**: ذاتی ویژول ایڈیٹر اور اسکرپٹنگ ماحول

### یونیٹی روبوٹکس ایکو سسٹم

یونیٹی روبوٹکس کے لیے چند خاص ٹولز فراہم کرتا ہے:
- **یونیٹی روبوٹکس ہب**: روبوٹکس پیکجز کے لیے مرکزی رسائی
- **ROS#**: یونیٹی کے لیے ROS برج
- **ML-ایجنٹس**: روبوٹکس کے لیے مشین لرننگ فریم ورک
- **ادراک پیکج**: مصنوعی ڈیٹا جنریشن ٹولز

## روبوٹکس کے لیے یونیٹی کا قیام

### سسٹم کی ضروریات

- **آپریٹنگ سسٹم**: ونڈوز 10/11، macOS 10.14+، یا اوبنٹو 18.04+
- **RAM**: 8GB کم از کم، 16GB+ تجویز کردہ
- **GPU**: 2GB+ VRAM کے ساتھ DirectX 11 مطابق گرافکس کارڈ
- **اسٹوریج**: یونیٹی انسٹالیشن اور پروجیکٹس کے لیے 20GB+
- **یونیٹی ورژن**: 2021.3 LTS یا نیا تجویز کردہ

### یونیٹی اور روبوٹکس پیکجز انسٹال کرنا

1. **یونیٹی ہب انسٹال کریں**: یونیٹی کی سرکاری ویب سائٹ سے ڈاؤن لوڈ کریں
2. **یونیٹی ایڈیٹر انسٹال کریں**: ضروری ماڈیولز کے ساتھ LTS ورژن منتخب کریں
3. **روبوٹکس پیکجز انسٹال کریں**: یونیٹی پیکج مینیجر کے ذریعے
4. **ROS برج کنفیگر کریں**: ROS 2 کمیونیکیشن کے لیے ROS# سیٹ اپ کریں

### یونیٹی ROS# انضمام

یونیٹی ROS# یونیٹی اور ROS 2 کے درمیان رابطہ فراہم کرتا ہے:

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<UInt64Msg>("unity_robot_joint_states");
    }

    void Update()
    {
        // ROS کو روبوٹ کی حالت بھیجیں
        var stateMsg = new UInt64Msg();
        stateMsg.data = (ulong)Time.time;
        ros.Publish("unity_robot_joint_states", stateMsg);
    }
}
```

## یونیٹی میں زیادہ معیار کے روبوٹ ماڈلز تخلیق کرنا

### روبوٹ ماڈلز درآمد کرنا

یونیٹی متعدد 3D ماڈل فارمیٹس کی حمایت کرتا ہے:
- **FBX**: روبوٹ ماڈلز کے لیے سب سے عام فارمیٹ
- **OBJ**: سادہ جیومیٹری فارمیٹ
- **DAE**: کولیڈا فارمیٹ اینیمیشن کی حمایت کے ساتھ
- **GLTF**: PBR مواد کے ساتھ جدید فارمیٹ

### روبوٹ رگنگ اور اینیمیشن

ہیومنوائڈ روبوٹس کے لیے، مناسب رگنگ ضروری ہے:
- **اسکلیٹ سیٹ اپ**: URDF جوائنٹ سٹرکچر کے مطابق ہڈی کی سلسلہ بندی تخلیق کریں
- **اینورس کنیمیٹکس**: قدرتی حرکت کے نمونوں کو فعال کریں
- **اینیمیشن کنٹرولرز**: مختلف حرکت کی حالتیں منظم کریں
- **جسمانی پابندیاں**: حقیقی روبوٹ جوائنٹ حدود کے مطابق کریں

### میٹریل اور شیڈر کنفیگریشن

یونیٹی کا رینڈرنگ پائپ لائن کو دیکھ بھال کی ضرورت ہوتی ہے:
- **PBR میٹریلز**: حقیقت پسندانہ کے لیے فزیکلی بیسڈ رینڈرنگ
- **ٹیکسچر میپنگ**: تفصیلی سطحوں کے لیے UV کوآرڈینیٹس
- **شیڈر سلیکشن**: مختلف سطحوں کے لیے مناسب شیڈرز منتخب کریں
- **لائٹنگ سیٹ اپ**: حقیقی نظر کے لیے لائٹنگ کنفیگر کریں

## یونیٹی میں ورچوئل سینسرز نافذ کرنا

### کیمرہ سمولیشن

یونیٹی کا کیمرہ سسٹم مختلف روبوٹ کیمرے کی شبیہہ بنا سکتا ہے:

```csharp
using UnityEngine;
using Unity.Robotics.SensorData;

public class RobotCamera : MonoBehaviour
{
    public Camera mainCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fov = 60f;

    private RenderTexture renderTexture;
    private Texture2D texture2D;

    void Start()
    {
        // کیمرہ پیرامیٹرز کنفیگر کریں
        mainCamera.fieldOfView = fov;
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        mainCamera.targetTexture = renderTexture;

        // امیج کیپچر کے لیے ٹیکسچر تخلیق کریں
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    void CaptureImage()
    {
        // رینڈر ٹیکسچر کو 2D ٹیکسچر میں تبدیل کریں
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // ROS میسج فارمیٹ میں تبدیل کریں
        byte[] imageBytes = texture2D.EncodeToPNG();
        // ROS ٹاپک پر شائع کریں
    }
}
```

### LIDAR سمولیشن

یونیٹی raycasting کا استعمال کرتے ہوئے LIDAR سینسرز کی شبیہہ بنا سکتی ہے:

```csharp
using System.Collections.Generic;
using UnityEngine;

public class UnityLidar : MonoBehaviour
{
    public int numRays = 360;
    public float maxDistance = 10f;
    public float scanAngle = 360f;

    private List<float> ranges;

    void Start()
    {
        ranges = new List<float>(new float[numRays]);
    }

    void Update()
    {
        float angleStep = scanAngle / numRays;

        for (int i = 0; i < numRays; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(
                Mathf.Cos(angle),
                0,
                Mathf.Sin(angle)
            );

            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction), out hit, maxDistance))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = maxDistance;
            }
        }

        // ROS کو رینج شائع کریں
        PublishLidarData();
    }

    void PublishLidarData()
    {
        // رینج کو ROS LaserScan میسج میں تبدیل کریں
        // اور ROS# کنکشن کے ذریعے شائع کریں
    }
}
```

### IMU سمولیشن

یونیٹی میں IMU سینسرز کی شبیہہ بنانا:

```csharp
using UnityEngine;

public class UnityIMU : MonoBehaviour
{
    public float noiseLevel = 0.01f;

    void Update()
    {
        // گھماؤ سے زاویہ ویلوسٹی حاصل کریں
        Quaternion deltaRotation = transform.rotation * Quaternion.Inverse(transform.rotation);
        Vector3 angularVelocity = new Vector3(
            Random.Range(-noiseLevel, noiseLevel) + GetAngularVelocityX(),
            Random.Range(-noiseLevel, noiseLevel) + GetAngularVelocityY(),
            Random.Range(-noiseLevel, noiseLevel) + GetAngularVelocityZ()
        );

        // لکیری ایکسلریشن حاصل کریں
        Vector3 linearAcceleration = Physics.gravity + GetLinearAcceleration();

        // ROS کو شائع کریں
        PublishIMUData(angularVelocity, linearAcceleration);
    }

    Vector3 GetLinearAcceleration()
    {
        // حرکت کی بنیاد پر لکیری ایکسلریشن کا حساب لگائیں
        return Vector3.zero; // مثال کے لیے سادہ
    }

    float GetAngularVelocityX() { return 0; } // سادہ
    float GetAngularVelocityY() { return 0; }
    float GetAngularVelocityZ() { return 0; }

    void PublishIMUData(Vector3 angularVelocity, Vector3 linearAcceleration)
    {
        // ROS sensor_msgs/Imu میسج میں تبدیل کریں
        // اور ROS# کنکشن کے ذریعے شائع کریں
    }
}
```

### فورس/ٹورک سینسر سمولیشن

```csharp
using UnityEngine;

public class UnityForceTorque : MonoBehaviour
{
    public float maxForce = 100f;
    public float maxTorque = 50f;

    void OnCollisionEnter(Collision collision)
    {
        // کالیژن فورسز کا حساب لگائیں
        foreach (ContactPoint contact in collision.contacts)
        {
            Vector3 force = contact.normal * collision.impulse.magnitude;
            // فورس/ٹورک ڈیٹا شائع کریں
            PublishForceTorqueData(force, Vector3.zero);
        }
    }

    void PublishForceTorqueData(Vector3 force, Vector3 torque)
    {
        // ROS geometry_msgs/Wrench میسج میں تبدیل کریں
        // اور ROS# کنکشن کے ذریعے شائع کریں
    }
}
```

## سمولیشن ماحول تخلیق کرنا

### ماحول کی ڈیزائن کے اصول

روبوٹکس کے لیے یونیٹی ماحول کو درج ذیل باتوں کو مدنظر رکھنا چاہیے:
- **حقیقت پسندی**: حقیقی دنیا کے منظار کی درست نمائندگی
- **تنوع**: جامع ٹیسٹنگ کے لیے متنوع ماحول
- **اینٹرایکٹیویٹی**: وہ عناصر جو روبوٹ کی کارروائیوں کے جواب میں رد عمل کریں
- **.scalability**: حقیقی وقت کی سمولیشن کے لیے کارآمد وسائل کا استعمال

### پیچیدہ ماحول تخلیق کرنا

#### انڈور ماحول
- **دفتر**: فرنیچر، دروازے، لفٹس
- **گھر**: کمرے، سیڑھیاں، گھریلو اشیاء
- **فیکٹریز**: اسمبلی لائنز، اسٹوریج علاقوں، سامان

#### آؤٹ ڈور ماحول
- **شہری**: سڑکیں، فُٹ پاتھ، عمارتیں
- **قدرتی**: پارک، جنگل، ناہموار زمین
- **مخصوص**: تعمیراتی مقامات، آفت کے علاقے

### یونیٹی میں فزکس سمولیشن

یونیٹی کا فزکس انجن یہ شبیہہ بنا سکتا ہے:
- **ریجڈ باڈی ڈائینمکس**: درست کالیژن اور تعامل
- **سافٹ باڈی فزکس**: ڈیفورم ایبل اشیاء
- **فلوئیڈ سمولیشن**: پانی اور دیگر مائع
- **کالیژن مکینکس**: تفصیلی تعامل کی قوتیں

## یونیٹی ادراک پیکج

### مصنوعی ڈیٹا جنریشن

یونیٹی کا ادراک پیکج یہ فعال کرتا ہے:
- **گراؤنڈ ٹرو اینوٹیشن**: خودکار طور پر اشیاء کو لیبل کرنا
- **سینسر سمولیشن**: کیمرہ، LIDAR، اور دیگر سینسرز
- **ڈومین رینڈمائزیشن**: مضبوط تربیت کے لیے متغیرات
- **ڈیٹا سیٹ جنریشن**: ML کے لیے منظم ڈیٹا سیٹس

### ادراک کیمرہ سیٹ اپ

```csharp
using Unity.Perception.GroundTruth;
using UnityEngine;

public class PerceptionCameraSetup : MonoBehaviour
{
    public GameObject perceptionCamera;

    void Start()
    {
        // ادراک کیمرہ کمپونینٹس شامل کریں
        var camera = perceptionCamera.GetComponent<Camera>();
        camera.depthTextureMode = DepthTextureMode.Depth;

        // سیگمینٹیشن لیبلز شامل کریں
        var segmentationLabeler = perceptionCamera.AddComponent<SegLabeler>();

        // ڈیٹا سیٹ کیپچر شامل کریں
        var datasetCapture = perceptionCamera.AddComponent<DatasetCapture>();
        datasetCapture.outputDirectory = "path/to/dataset";
    }
}
```

## ROS 2 کے ساتھ انضمام

### ROS# برج کنفیگریشن

یونیٹی کے لیے ROS برج سیٹ اپ کرنا:

1. **ROS# پیکج انسٹال کریں**: یونیٹی پیکج مینیجر کے ذریعے
2. **نیٹ ورک ترتیبات کنفیگر کریں**: IP ایڈریسز اور پورٹس
3. **کنکشن ٹیسٹ کریں**: یونیٹی اور ROS کے درمیان کمیونیکیشن کی تصدیق کریں
4. **میسج ٹائپس میپ کریں**: سسٹمز کے درمیان مطابقت یقینی بنائیں

### میسج شائع کرنا اور سبسکرائب کرنا

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityROSIntegration : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // پبلشرز اور سبسکرائبرز رجسٹر کریں
        ros.RegisterPublisher<ImageMsg>("unity_camera/image_raw");
        ros.RegisterPublisher<LaserScanMsg>("unity_lidar/scan");
        ros.RegisterSubscriber<JointStateMsg>("joint_states", OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        // جوائنٹ حالت کی بنیاد پر روبوٹ ماڈل اپ ڈیٹ کریں
        UpdateRobotJoints(msg.position);
    }

    void UpdateRobotJoints(float[] positions)
    {
        // جوائنٹ پوزیشنز کو یونیٹی روبوٹ ماڈل پر لاگو کریں
    }
}
```

## کارکردگی کی بہتری

### رینڈرنگ کی بہتری

- **LOD سسٹم**: پیچیدہ ماڈلز کے لیے ڈیٹیل کی سطح
- **اکلیوژن کلینگ**: کیمرہ کے نظر میں نہ ہونے والی اشیاء چھپائیں
- **ٹیکسچر کمپریشن**: ٹیکسچر میموری کے استعمال کو بہتر بنائیں
- **شیڈر بہتری**: حقیقی وقت کی رینڈرنگ کے لیے کارآمد شیڈرز استعمال کریں

### فزکس کی بہتری

- **کالیژن لیئرنگ**: کالیژن ڈیٹیکشن کو بہتر بنائیں
- **فکسڈ ٹائم سٹیپ**: مسلسل فزکس اپ ڈیٹ کی شرح
- **سادہ کالیڈرز**: کارکردگی کے لیے سادہ اشکال استعمال کریں
- **فیزکس میٹریل**: سطح کی خصوصیات کو مؤثر طریقے سے کنفیگر کریں

## یونیٹی روبوٹکس کے لیے بہترین طریقے

### ماڈل کی بہتری

- **پولی گون کاؤنٹ**: تفصیل اور کارکردگی کا توازن
- **ٹیکسچر ایٹلسنگ**: متعدد ٹیکسچرز کو جوڑیں
- **اینیمیشن کمپریشن**: اینیمیشن ڈیٹا کو بہتر بنائیں
- **Prefab استعمال**: اجزاء کو مؤثر طریقے سے دوبارہ استعمال کریں

### منظر کا نظم

- **ماڈیولر مناظر**: پیچیدہ ماحول کو ماڈیولز میں توڑیں
- **اثاثہ بندلز**: اثاثے متحرک طور پر لوڈ کریں
- **سٹریمنگ**: ضرورت کے مطابق ماحول کے حصے لوڈ/ان لوڈ کریں
- **کیچنگ**: اکثر استعمال ہونے والے ڈیٹا کو میموری میں محفوظ کریں

## ہاتھوں سے مشق: یونیٹی روبوٹ ویژولائزیشن تخلیق کرنا

### مشق کے اہداف

- ROS# انضمام کے ساتھ یونیٹی سیٹ اپ کریں
- یونیٹی میں ایک سادہ روبوٹ ماڈل تخلیق کریں
- بنیادی سینسر سمولیشن نافذ کریں
- دو طرفہ رابطے کے لیے ROS 2 سے جوڑیں

### مرحلہ وار ہدایات

1. **یونیٹی ہب اور ایڈیٹر انسٹال کریں** ضروری ماڈیولز کے ساتھ
2. **روبوٹکس سمولیشن کے لیے ایک نیا یونیٹی پروجیکٹ تخلیق کریں**
3. **پیکج مینیجر کے ذریعے ROS# پیکج درآمد کریں**
4. **بنیادی جوائنٹس کے ساتھ ایک سادہ روبوٹ ماڈل تخلیق کریں**
5. **کیمرہ سینسر** سمولیشن نافذ کریں
6. **ROS کنکشن** اور میسج ایکسچینج ٹیسٹ کریں

### متوقع نتائج

- ROS# انضمام کے ساتھ یونیٹی پروجیکٹ
- سینسر سمولیشن کے ساتھ روبوٹ ماڈل
- ROS 2 کے ساتھ کامیاب رابطہ
- یونیٹی-ROS انضمام کی سمجھ

## نالج چیک

1. یونیٹی اور روایتی روبوٹکس سمولیٹرز کے درمیان کیا کلیدی فرق ہیں؟
2. raycasting کا استعمال کرتے ہوئے یونیٹی میں LIDAR سمولیشن کیسے نافذ کریں؟
3. یونیٹی ادراک پیکج کس کام کے لیے استعمال ہوتا ہے؟
4. یونیٹی کا فزکس انجن گیزبو کی فزکس صلاحیتوں کے مقابلہ میں کیسا ہے؟

## خلاصہ

اس باب نے یونیٹی کو زیادہ معیار کے روبوٹ ویژولائزیشن اور سینسر سمولیشن کے لیے ایک طاقتور پلیٹ فارم کے طور پر متعارف کرایا۔ یونیٹی کی فوٹو ریلائزٹک رینڈرنگ کی صلاحیتوں کو جوڑ کر، اس کا جامع فزکس انجن اور ROS انضمام کے ساتھ ہیومنوائڈ روبوٹس کے لیے جامع ڈیجیٹل ٹوئن تخلیق کرنے کے لیے ایک عمدہ ٹول ہے۔ گیزبو کی فزکس سمولیشن اور یونیٹی کی ویژولائزیشن کا مجموعہ جامع سمولیشن کی صلاحیات فراہم کرتا ہے۔

## اگلے اقدامات

پارٹ IV میں، ہم NVIDIA Isaac پلیٹ فارم کو تلاش کریں گے، Isaac Sim کے لیے فوٹو ریلائزٹک سمولیشن اور Isaac ROS کے لیے ہارڈ ویئر-تیز ادراک میں گہرائی میں جاتے ہوئے، اس باب میں قائم کردہ یونیٹی بنیاد پر تعمیر کرتے ہوئے۔