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
    public Button captureButton;
    public Button continueButton;

    [Header("Model Settings")]
    [Range(0f, 1f)] public float detectionThreshold = 0.6f;
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

    private WebCamTexture webcam;
    private Interpreter interpreter;

    private Texture2D frozenFrame; // ✅ Fixed: was "Texture224"
    private float[] inputBuffer;
    private float[] outputBuffer;
    private string[] labels;

    private readonly int width = 224;
    private readonly int height = 224;
    private bool isModelReady = false;
    private bool isFrozen = false;

    IEnumerator Start()
    {
        if (captureButton != null) captureButton.interactable = false;
        if (continueButton != null) continueButton.interactable = false;

        resultText.text = "Requesting camera permission...";

        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            resultText.text = "❌ Camera permission denied!";
            yield break;
        }

        yield return new WaitForSeconds(1.0f);

        // Load model
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        byte[] modelData = null;

        if (modelPath.Contains("://"))
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success)
                {
                    resultText.text = $"❌ Model load failed:\n{www.error}";
                    yield break;
                }
                modelData = www.downloadHandler.data;
            }
        }
        else
        {
            try
            {
                modelData = File.ReadAllBytes(modelPath);
            }
            catch (System.Exception e)
            {
                resultText.text = "❌ Model file not found!";
                Debug.LogError(e.Message);
                yield break;
            }
        }

        // Initialize TFLite
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

        // Load labels
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
            catch { /* ignore */ }
        }

        if (!string.IsNullOrEmpty(labelData))
        {
            labels = labelData.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (labels.Length != outputBuffer.Length)
                labels = null;
            else
                for (int i = 0; i < labels.Length; i++)
                    labels[i] = labels[i].Trim();
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

        // Use HIGH RESOLUTION for smooth preview
        webcam = new WebCamTexture(deviceName, 1280, 720, 30);
        cameraPreview.texture = webcam;
        webcam.Play();

        if (captureButton != null) captureButton.interactable = true;
        if (continueButton != null) continueButton.interactable = false;

        resultText.text = "✅ Ready! Tap 'Capture' to freeze and analyze";
    }

    void Update()
    {
        if (webcam == null || !webcam.isPlaying || isFrozen) return;

        if (fitter != null && webcam.width > 100 && webcam.height > 100)
        {
            float aspect = (float)webcam.width / webcam.height;
            if (webcam.videoRotationAngle == 90 || webcam.videoRotationAngle == 270)
                aspect = 1.0f / aspect;
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
    }

    public void CaptureAndAnalyze()
    {
        if (!isModelReady || webcam == null || !webcam.isPlaying) return;

        // ✅ SAFE capture using RenderTexture (same as real-time)
        RenderTexture rt = RenderTexture.GetTemporary(width, height, 0);
        Graphics.Blit(webcam, rt);

        frozenFrame = new Texture2D(width, height, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        frozenFrame.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        frozenFrame.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        cameraPreview.texture = frozenFrame;
        isFrozen = true;
        webcam.Stop();

        RunInferenceOnFrozenFrame();

        if (continueButton != null) continueButton.interactable = true;
        if (captureButton != null) captureButton.interactable = false;
    }

    public void ContinueCamera()
    {
        if (webcam == null) return;

        cameraPreview.texture = webcam;
        webcam.Play();
        isFrozen = false;

        if (continueButton != null) continueButton.interactable = false;
        if (captureButton != null) captureButton.interactable = true;

        resultText.text = "✅ Ready! Tap 'Capture' to freeze and analyze";
    }

    void RunInferenceOnFrozenFrame()
    {
        resultText.text = "🧠 Analyzing...";

        Color32[] pixels = frozenFrame.GetPixels32();
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
            resultText.text = "❌ Inference error!";
            Debug.LogError("TFLite inference failed: " + e.Message);
            Debug.LogException(e);
        }
    }

    void OnDestroy()
    {
        if (webcam != null && webcam.isPlaying)
            webcam.Stop();
        interpreter?.Dispose();
        if (frozenFrame != null)
            Destroy(frozenFrame);
    }
}