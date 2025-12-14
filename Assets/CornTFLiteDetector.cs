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

    [Header("Model Settings")]
    [Range(0f, 1f)] public float detectionThreshold = 0.6f;
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

    [Header("TFLite Settings")]
    public bool useGPUDelegate = false;

    private WebCamTexture webcam;
    private Interpreter interpreter;

    // We separate the AI image from the Display image to fix stretching/blur
    private Texture2D aiFrame;      // Small (224x224) for AI
    private Texture2D displayFrame; // HD (Full Screen) for User

    private float[] inputBuffer;
    private float[] outputBuffer;
    private string[] labels;

    // Dimensions read automatically from Model
    private int inputWidth;
    private int inputHeight;
    private int inputChannels;

    private bool isModelReady = false;
    private bool isFrozen = false;

    IEnumerator Start()
    {
        if (captureButton != null) captureButton.interactable = false;
        if (resultText != null) resultText.text = "Requesting permissions...";

        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            if (resultText != null) resultText.text = "❌ Permission denied!";
            yield break;
        }

        // Load Model
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
            if (useGPUDelegate) { try { options.AddGpuDelegate(); } catch { } }
            options.threads = 2;
            interpreter = new Interpreter(modelData, options);
            interpreter.AllocateTensors();

            // Get Model Input Size
            var inputInfo = interpreter.GetInputTensorInfo(0);
            int[] inputShape = inputInfo.shape;

            if (inputShape.Length == 4)
            {
                inputHeight = inputShape[1];
                inputWidth = inputShape[2];
                inputChannels = inputShape[3];
            }
            else
            {
                inputHeight = 224; inputWidth = 224; inputChannels = 3;
            }

            inputBuffer = new float[inputWidth * inputHeight * inputChannels];
            
            var outputInfo = interpreter.GetOutputTensorInfo(0);
            int outputClasses = outputInfo.shape[1];
            outputBuffer = new float[outputClasses];

            isModelReady = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TFLite error: {e.Message}");
            yield break;
        }

        yield return LoadLabels();
    }

    IEnumerator LoadLabels()
    {
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
        for (int i = 0; i < devices.Length; i++)
        {
            if (!devices[i].isFrontFacing) { deviceName = devices[i].name; break; }
        }

        // Fix for Zoom: Request Screen Resolution
        int w = Screen.height; 
        int h = Screen.width;
        
        webcam = new WebCamTexture(deviceName, w, h, 30);
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
        float scaleY = webcam.videoVerticallyMirrored ? -1f : 1f;
        cameraPreview.rectTransform.localScale = new Vector3(1f, scaleY, 1f);
    }

    // ------------------------------------------------
    // 1. CAPTURE LOGIC (FIXED)
    // ------------------------------------------------
    public void CaptureAndAnalyze()
    {
        if (!isModelReady || webcam == null || !webcam.isPlaying) return;
        StartCoroutine(CaptureRoutine());
    }

    IEnumerator CaptureRoutine()
    {
        yield return new WaitForEndOfFrame();

        // A. CAPTURE HD IMAGE FOR DISPLAY (Fixes Blur/Stretch)
        // ----------------------------------------------------
        if (displayFrame != null) Destroy(displayFrame);
        displayFrame = new Texture2D(webcam.width, webcam.height, TextureFormat.RGB24, false);
        displayFrame.SetPixels32(webcam.GetPixels32());
        displayFrame.Apply();
        
        // Show the HD image to the user
        cameraPreview.texture = displayFrame;

        // B. CAPTURE SMALL IMAGE FOR AI (Fixes 'width' error)
        // ---------------------------------------------------
        // Create temporary RenderTexture sized for AI (e.g. 224x224)
        RenderTexture rt = RenderTexture.GetTemporary(inputWidth, inputHeight, 0);
        
        // Squash webcam image into small square
        Graphics.Blit(webcam, rt); 
        
        // Save previous active render texture (Fixes 'oldRt' error)
        RenderTexture oldRt = RenderTexture.active; 
        
        // Read pixels from small RT
        RenderTexture.active = rt;
        if (aiFrame != null) Destroy(aiFrame);
        aiFrame = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);
        aiFrame.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        aiFrame.Apply();

        // Restore and Clean up
        RenderTexture.active = oldRt;
        RenderTexture.ReleaseTemporary(rt);

        // Stop camera now that we have the images
        isFrozen = true;
        webcam.Stop();

        if (captureButton != null) captureButton.interactable = false;

        RunInference();
    }

    public void ContinueCamera()
    {
        if (webcam == null) return;

        cameraPreview.texture = webcam;
        webcam.Play();
        isFrozen = false;

        // Cleanup Memory
        if (displayFrame != null) Destroy(displayFrame);
        if (aiFrame != null) Destroy(aiFrame);

        if (captureButton != null) captureButton.interactable = true;
        if (resultText != null) resultText.text = "✅ Ready!";
    }

    void RunInference()
    {
        if (resultText != null) resultText.text = "Analyzing...";

        // Use the SMALL AI FRAME for calculation
        Color32[] pixels = aiFrame.GetPixels32();

        int idx = 0;
        for (int i = 0; i < pixels.Length; i++) {
            inputBuffer[idx++] = pixels[i].r / 255.0f;
            inputBuffer[idx++] = pixels[i].g / 255.0f;
            inputBuffer[idx++] = pixels[i].b / 255.0f;
        }

        interpreter.SetInputTensorData(0, inputBuffer);

        try
        {
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

            if (resultDisplay != null) {
                resultDisplay.ShowResult(className);
            } else {
                Debug.LogError("ResultDisplay not assigned!");
            }
        }
        catch (System.Exception e) {
            Debug.LogError(e.Message);
        }
    }

    void OnDestroy() {
        if (webcam != null) webcam.Stop();
        interpreter?.Dispose();
        if (displayFrame != null) Destroy(displayFrame);
        if (aiFrame != null) Destroy(aiFrame);
    }
}