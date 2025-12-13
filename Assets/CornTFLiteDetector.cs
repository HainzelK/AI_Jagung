using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using TensorFlowLite;
using System.IO;
using System.Collections;
using TMPro;

public class CornTFLiteDetector : MonoBehaviour
{
    [Header("UI Assignments")]
    public RawImage cameraPreview;
    public TMP_Text resultText;
    public AspectRatioFitter fitter;
    public Button captureButton; // ← Assign in Unity Inspector

    [Header("Model Settings")]
    [Range(0f, 1f)] public float detectionThreshold = 0.6f;
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

#if UNITY_EDITOR
    [Header("Editor Testing")]
    public Texture2D testImage; // ← Assign a test image in Editor to simulate capture
#endif

    private WebCamTexture webcam;
    private Interpreter interpreter;

    private readonly int inputWidth = 224;
    private readonly int inputHeight = 224;
    private bool isModelReady = false;

    private float[] inputBuffer;
    private float[] outputBuffer;
    private string[] labels;

    private string lastCapturedImagePath = "";

    IEnumerator Start()
    {
        if (captureButton != null)
            captureButton.interactable = false;

        resultText.text = "Requesting camera permission...";

        // 1. Request camera permission (skipped in Editor for testing)
#if !UNITY_EDITOR
        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            resultText.text = "❌ Camera permission denied!";
            Debug.LogError("Camera permission denied.");
            yield break;
        }
#endif

        // 2. Load TensorFlow Lite model
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        byte[] modelData = null;

        if (modelPath.Contains("://")) // Android
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success)
                {
                    resultText.text = $"❌ Failed to load model:\n{www.error}";
                    Debug.LogError($"Model error: {www.error}");
                    yield break;
                }
                modelData = www.downloadHandler.data;
            }
        }
        else // Editor or iOS
        {
            try
            {
                modelData = File.ReadAllBytes(modelPath);
            }
            catch (System.Exception e)
            {
                resultText.text = "❌ Model file not found!";
                Debug.LogError($"Model load error: {e.Message}");
                yield break;
            }
        }

        // 3. Initialize TFLite interpreter
        try
        {
            var options = new InterpreterOptions();
#if !UNITY_EDITOR
            options.AddGpuDelegate(); // GPU delegate not available in Editor
#endif
            interpreter = new Interpreter(modelData, options);
            interpreter.AllocateTensors();

            inputBuffer = new float[inputWidth * inputHeight * 3];
            outputBuffer = new float[3];
            isModelReady = true;
        }
        catch (System.Exception e)
        {
            resultText.text = "❌ TFLite initialization failed!";
            Debug.LogError($"TFLite error: {e.Message}");
            yield break;
        }

        // 4. Load labels
        string labelPath = Path.Combine(Application.streamingAssetsPath, labelFileName);
        string labelData = null;

        if (labelPath.Contains("://"))
        {
            using (UnityWebRequest req = UnityWebRequest.Get(labelPath))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                    labelData = req.downloadHandler.text;
            }
        }
        else
        {
            try
            {
                labelData = File.ReadAllText(labelPath);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"Failed to load labels: {e.Message}");
            }
        }

        if (!string.IsNullOrEmpty(labelData))
        {
            labels = labelData.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (labels.Length != outputBuffer.Length)
            {
                Debug.LogWarning($"Label count ({labels.Length}) ≠ model outputs ({outputBuffer.Length}). Using indices.");
                labels = null;
            }
            else
            {
                for (int i = 0; i < labels.Length; i++)
                {
                    labels[i] = labels[i].Trim();
                    Debug.Log($"[LABEL] Index {i}: '{labels[i]}'");
                }
            }
        }

        // 5. Start camera (only on device)
#if UNITY_EDITOR
        if (captureButton != null)
            captureButton.interactable = true; // Enable immediately in Editor
        resultText.text = testImage != null 
            ? "✅ Ready! Click 'Capture' to test AI" 
            : "⚠️ Assign 'Test Image' in Inspector";
#else
        resultText.text = "Starting camera...";
        StartCamera();
#endif
    }

#if !UNITY_EDITOR
    void StartCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            resultText.text = "❌ No camera found!";
            return;
        }

        string deviceName = devices[0].name;
        foreach (var device in devices)
        {
            if (!device.isFrontFacing)
            {
                deviceName = device.name;
                break;
            }
        }

        webcam = new WebCamTexture(deviceName, 1280, 720, 30);
        cameraPreview.texture = webcam;
        webcam.Play();
    }
#endif

#if !UNITY_EDITOR
    void Update()
    {
        if (webcam == null || !webcam.isPlaying) return;

        if (fitter != null)
        {
            float aspect = (float)webcam.width / webcam.height;
            if (webcam.videoRotationAngle == 90 || webcam.videoRotationAngle == 270)
                aspect = 1f / aspect;
            fitter.aspectRatio = aspect;
        }

        if (cameraPreview != null)
        {
            cameraPreview.rectTransform.localEulerAngles = new Vector3(0, 0, -webcam.videoRotationAngle);
            cameraPreview.uvRect = new Rect(
                0,
                webcam.videoVerticallyMirrored ? 1 : 0,
                1,
                webcam.videoVerticallyMirrored ? -1 : 1
            );
        }

        if (isModelReady && webcam.isPlaying && captureButton != null && !captureButton.interactable)
        {
            captureButton.interactable = true;
            resultText.text = "✅ Ready! Tap 'Capture' to analyze a photo";
        }
    }
#endif

    // 📸 Unified capture: uses test image in Editor, real camera on device
public void CaptureAndAnalyze()
{
    Debug.Log("📸 [INFO] Capture button pressed."); // ← Always logs

#if UNITY_EDITOR
    // --- EDITOR: Use test image ---
    if (!isModelReady)
    {
        resultText.text = "❌ Model not ready!";
        return;
    }

    if (testImage == null)
    {
        resultText.text = "⚠️ Assign 'Test Image' in Inspector!";
        Debug.LogWarning("Test image not assigned.");
        return;
    }

    Debug.Log($"✅ Using test image: {testImage.width}x{testImage.height}");
    resultText.text = "🧪 Analyzing test image...";

    // Wrap in block to isolate variable scope
    {
        Texture2D resized = ResizeTexture(testImage, inputWidth, inputHeight);
        RunInferenceOnTexture(resized);
        Destroy(resized);
    }
    return;
#endif

    // --- DEVICE: Use real camera ---
    if (!isModelReady || webcam == null || !webcam.isPlaying)
    {
        string msg = $"⚠️ Not ready!\nisModelReady: {isModelReady}\nwebcam: {(webcam != null)}\nisPlaying: {(webcam?.isPlaying ?? false)}";
        Debug.Log(msg);
        resultText.text = "⚠️ Not ready yet!";
        return;
    }

    resultText.text = "📸 Capturing photo...";

    RenderTexture rt = RenderTexture.GetTemporary(webcam.width, webcam.height, 0);
    Graphics.Blit(webcam, rt);

    Texture2D fullPhoto = new Texture2D(webcam.width, webcam.height, TextureFormat.RGB24, false);
    RenderTexture.active = rt;
    fullPhoto.ReadPixels(new Rect(0, 0, webcam.width, webcam.height), 0, 0);
    fullPhoto.Apply();
    RenderTexture.active = null;
    RenderTexture.ReleaseTemporary(rt);

    // Save photo
    try
    {
        byte[] png = fullPhoto.EncodeToPNG();
        lastCapturedImagePath = Path.Combine(Application.persistentDataPath, $"corn_{System.DateTime.Now:yyyyMMdd_HHmmss}.png");
        File.WriteAllBytes(lastCapturedImagePath, png);
        Debug.Log("📸 Photo captured and saved to: " + lastCapturedImagePath);
    }
    catch (System.Exception e)
    {
        Debug.LogWarning("⚠️ Failed to save image: " + e.Message);
    }

    // Wrap in block to avoid scope conflict
    {
        Texture2D resized = ResizeTexture(fullPhoto, inputWidth, inputHeight);
        RunInferenceOnTexture(resized);
        Destroy(fullPhoto);
        Destroy(resized);
    }
}
    // Helper: Resize texture to target dimensions
    Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        RenderTexture rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(source, rt);
        RenderTexture.active = rt;
        Texture2D result = new Texture2D(width, height, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        result.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    // Run TFLite inference on a Texture2D
    void RunInferenceOnTexture(Texture2D tex)
    {
        resultText.text = "🧠 Analyzing...";

        Color32[] pixels = tex.GetPixels32();
        int idx = 0;
        for (int i = 0; i < pixels.Length; i++)
        {
            inputBuffer[idx++] = pixels[i].r / 255.0f;
            inputBuffer[idx++] = pixels[i].g / 255.0f;
            inputBuffer[idx++] = pixels[i].b / 255.0f;
        }

        try
        {
            interpreter.SetInputTensorData(0, inputBuffer);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputBuffer);

            int predictedClass = 0;
            float maxConfidence = outputBuffer[0];
            for (int i = 1; i < outputBuffer.Length; i++)
            {
                if (outputBuffer[i] > maxConfidence)
                {
                    maxConfidence = outputBuffer[i];
                    predictedClass = i;
                }
            }

            string className = (labels != null && predictedClass < labels.Length)
                ? labels[predictedClass]
                : $"Class {predictedClass}";

            string status = maxConfidence > detectionThreshold ? "✅ DETECTED!" : "🔍 Low confidence";
            resultText.text = $"{className}\n{status}\nConfidence: {maxConfidence:P1}";

            Debug.Log($"[RESULT] {className} | Conf: {maxConfidence:F4}");
        }
        catch (System.Exception e)
        {
            resultText.text = "❌ Analysis failed!";
            Debug.LogError("Inference error: " + e.Message);
        }
    }

    void OnDestroy()
    {
#if !UNITY_EDITOR
        if (webcam != null && webcam.isPlaying)
            webcam.Stop();
#endif
        interpreter?.Dispose();
    }
}