using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using TensorFlowLite;
using System.IO;
using System.Collections;
using TMPro;

public class CornTFLiteDetector : MonoBehaviour
{
    [Header("Script References")]
    public ResultDisplay resultDisplay; 

    [Header("UI Assignments")]
    public RawImage cameraPreview;
    public TMP_Text resultText; // Debug Text
    public AspectRatioFitter fitter;
    public Button captureButton;

    // (Removed 'continueButton' variable because we use the Back button now)

    [Header("Model Settings")]
    [Range(0f, 1f)] public float detectionThreshold = 0.6f;
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

    private WebCamTexture webcam;
    private Interpreter interpreter;
    private Texture2D frozenFrame;
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
        
        if(resultText != null) resultText.text = "Requesting camera permission...";

        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            if(resultText != null) resultText.text = "❌ Permission denied!";
            yield break;
        }

        // Load model
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        byte[] modelData = null;

        if (modelPath.Contains("://"))
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                modelData = www.downloadHandler.data;
            }
        }
        else
        {
            try { modelData = File.ReadAllBytes(modelPath); } catch {}
        }

        // Init TFLite
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
            Debug.LogError($"TFLite error: {e.Message}");
            yield break;
        }

        // Load labels
        string labelPath = Path.Combine(Application.streamingAssetsPath, labelFileName);
        string labelData = null;
        if (labelPath.Contains("://")) {
            using (UnityWebRequest req = UnityWebRequest.Get(labelPath)) {
                yield return req.SendWebRequest();
                labelData = req.downloadHandler.text;
            }
        } else {
            try { labelData = File.ReadAllText(labelPath); } catch { }
        }

        if (!string.IsNullOrEmpty(labelData)) {
            labels = labelData.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        }

        StartCamera();
    }

    void StartCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0) return;

        string deviceName = devices[0].name;
        foreach (var device in devices) {
            if (!device.isFrontFacing) {
                deviceName = device.name;
                break;
            }
        }

        webcam = new WebCamTexture(deviceName, 1280, 720, 30);
        cameraPreview.texture = webcam;
        webcam.Play();

        if (captureButton != null) captureButton.interactable = true;
        if (resultText != null) resultText.text = "✅ Ready!";
    }

    void Update()
    {
        if (webcam == null || !webcam.isPlaying || isFrozen) return;

        if (fitter != null) {
            float aspect = (float)webcam.width / webcam.height;
            if (webcam.videoRotationAngle == 90 || webcam.videoRotationAngle == 270)
                aspect = 1.0f / aspect;
            fitter.aspectRatio = aspect;
        }

        cameraPreview.rectTransform.localEulerAngles = new Vector3(0, 0, -webcam.videoRotationAngle);
        cameraPreview.uvRect = new Rect(0, webcam.videoVerticallyMirrored ? 1 : 0, 1, webcam.videoVerticallyMirrored ? -1 : 1);
    }

    // ------------------------------------------------
    // 1. CAPTURE LOGIC
    // ------------------------------------------------
    public void CaptureAndAnalyze()
    {
        if (!isModelReady || webcam == null || !webcam.isPlaying) return;

        // Freeze frame
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

        // DISABLE BUTTON immediately
        if (captureButton != null) captureButton.interactable = false;

        RunInferenceOnFrozenFrame();
    }

    // ------------------------------------------------
    // 2. RESTART LOGIC (Called by Back Button)
    // ------------------------------------------------
    public void ContinueCamera()
    {
        if (webcam == null) return;

        cameraPreview.texture = webcam;
        webcam.Play();
        isFrozen = false;

        // ✅ RE-ENABLE BUTTON
        if (captureButton != null) 
        {
            captureButton.interactable = true;
            Debug.Log("Capture button re-enabled.");
        }

        if (resultText != null) resultText.text = "✅ Ready!";
    }

    void RunInferenceOnFrozenFrame()
    {
        if (resultText != null) resultText.text = "Analyzing...";

        Color32[] pixels = frozenFrame.GetPixels32();
        int idx = 0;
        for (int i = 0; i < pixels.Length; i++) {
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
            for (int i = 1; i < outputBuffer.Length; i++) {
                if (outputBuffer[i] > maxConfidence) {
                    maxConfidence = outputBuffer[i];
                    predictedClass = i;
                }
            }

            string className = (labels != null && predictedClass < labels.Length) 
                ? labels[predictedClass] : $"Class {predictedClass}";

            Debug.Log($"[RESULT] {className}");

            // Trigger Result UI
            if (resultDisplay != null) {
                resultDisplay.ShowResult(className);
            } else {
                Debug.LogError("ResultDisplay not assigned in Detector script!");
            }
        }
        catch (System.Exception e) {
            Debug.LogError(e.Message);
        }
    }

    void OnDestroy() {
        if (webcam != null) webcam.Stop();
        interpreter?.Dispose();
    }
}