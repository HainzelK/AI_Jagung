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

    [Header("Model Settings")]
    [Range(0f, 1f)] public float detectionThreshold = 0.6f;
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

    // Removed: public int cornClassIndex = 1; — no longer needed

    private WebCamTexture webcam;
    private Interpreter interpreter;

    private Texture2D inputTex;
    private float[] inputBuffer;
    private float[] outputBuffer;
    private string[] labels;

    private readonly int width = 224;
    private readonly int height = 224;
    private bool isModelReady = false;

    IEnumerator Start()
    {
        resultText.text = "Requesting camera permission...";

        // 1. Request camera permission
        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            resultText.text = "❌ Camera permission denied!";
            Debug.LogError("Camera permission denied.");
            yield break;
        }

        // 2. Load model
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        byte[] modelData = null;

        if (modelPath.Contains("://")) // Android
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success)
                {
                    resultText.text = $"❌ Model load failed:\n{www.error}";
                    Debug.LogError($"Model error: {www.error}");
                    yield break;
                }
                modelData = www.downloadHandler.data;
            }
        }
        else // Editor / iOS
        {
            try
            {
                modelData = File.ReadAllBytes(modelPath);
            }
            catch (System.Exception e)
            {
                resultText.text = "❌ Model file not found!";
                Debug.LogError($"Model file error: {e.Message}");
                yield break;
            }
        }

        // 3. Initialize TFLite
        try
        {
            var options = new InterpreterOptions();
            options.AddGpuDelegate();
            interpreter = new Interpreter(modelData, options);
            interpreter.AllocateTensors();

            inputBuffer = new float[width * height * 3];
            outputBuffer = new float[3];
            isModelReady = true;
        }
        catch (System.Exception e)
        {
            resultText.text = "❌ TFLite init failed!";
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
                labelData = null;
            }
        }

        if (!string.IsNullOrEmpty(labelData))
        {
            labels = labelData.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (labels.Length != outputBuffer.Length)
            {
                Debug.LogWarning($"Label count ({labels.Length}) ≠ model outputs ({outputBuffer.Length}). Ignoring labels.");
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
        else
        {
            Debug.LogWarning("Labels not loaded.");
            labels = null;
        }

        resultText.text = "✅ Ready!\nStarting camera...";
        StartCamera();
    }

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

        inputTex = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    void Update()
    {
        if (webcam == null || !webcam.isPlaying) return;

        if (fitter != null && webcam.width > 100 && webcam.height > 100)
        {
            float aspect = (float)webcam.width / webcam.height;
            if (webcam.videoRotationAngle == 90 || webcam.videoRotationAngle == 270)
            {
                aspect = 1.0f / aspect;
            }
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

        if (isModelReady && webcam.didUpdateThisFrame)
        {
            Detect();
        }
    }

    void Detect()
    {
        RenderTexture rt = RenderTexture.GetTemporary(width, height, 0);
        Graphics.Blit(webcam, rt);

        RenderTexture.active = rt;
        inputTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        inputTex.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = inputTex.GetPixels32();
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

            // 🔍 DEBUG: Log raw outputs
            Debug.Log($"[MODEL OUTPUT] → [0] {outputBuffer[0]:F4} | [1] {outputBuffer[1]:F4} | [2] {outputBuffer[2]:F4}");

            // ✅ Find the ACTUAL top predicted class
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

            // Get label
            string className = (labels != null && predictedClass < labels.Length) 
                ? labels[predictedClass] 
                : $"Class {predictedClass}";

            // Display result
            string status = maxConfidence > detectionThreshold ? "✅ DETECTED!" : "🔍 Low confidence";
            resultText.text = $"{className}\n{status}\nConfidence: {maxConfidence:P1}";

            // Optional: log final prediction
            Debug.Log($"[RESULT] Showing: '{className}' (index {predictedClass}, conf: {maxConfidence:F4})");
        }
        catch (System.Exception e)
        {
            resultText.text = "❌ Inference error!";
            Debug.LogError("TFLite inference failed: " + e.Message);
        }
    }

    void OnDestroy()
    {
        if (webcam != null && webcam.isPlaying)
            webcam.Stop();

        interpreter?.Dispose();
    }
}