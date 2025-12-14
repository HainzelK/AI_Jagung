using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using TensorFlowLite;
using System.IO;
using System.Collections;
using TMPro;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Collections;

public class CornTFLiteDetector : MonoBehaviour
{
    [Header("Script References")]
    public ResultDisplay resultDisplay;
    public ARCameraManager arCameraManager;

    [Header("UI Assignments")]
    public RawImage freezePreview;
    public TMP_Text resultText;
    public Button captureButton;

    [Header("Model Settings")]
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

    [Header("TFLite Settings")]
    public bool useGPUDelegate = false;

    private Interpreter interpreter;
    private Texture2D aiTexture;      // 224x224
    private Texture2D displayTexture; // Full Res

    private float[] inputBuffer;
    private float[] outputBuffer;
    private string[] labels;

    private int inputWidth;
    private int inputHeight;
    private int inputChannels;

    private bool isModelReady = false;
    private bool isCaptured = false;

    IEnumerator Start()
    {
        // PERBAIKAN: Matikan layar putih di awal & reset transform
        if (freezePreview != null)
        {
            freezePreview.gameObject.SetActive(false);
            freezePreview.rectTransform.pivot = new Vector2(0.5f, 0.5f);
            freezePreview.rectTransform.anchorMin = new Vector2(0.5f, 0.5f);
            freezePreview.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
        }

        if (captureButton != null) captureButton.interactable = false;
        if (resultText != null) resultText.text = "Loading Model...";

        // 1. Load Model
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
            try { modelData = File.ReadAllBytes(modelPath); } catch { }
        }

        // 2. Init TFLite
        try
        {
            var options = new InterpreterOptions();
            if (useGPUDelegate) { try { options.AddGpuDelegate(); } catch { } }
            options.threads = 2;
            interpreter = new Interpreter(modelData, options);
            interpreter.AllocateTensors();

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
            Debug.LogError($"TFLite Init Error: {e.Message}");
            yield break;
        }

        yield return LoadLabels();
    }

    IEnumerator LoadLabels()
    {
        string labelPath = Path.Combine(Application.streamingAssetsPath, labelFileName);
        string labelData = null;
        if (labelPath.Contains("://"))
        {
            using (UnityWebRequest req = UnityWebRequest.Get(labelPath))
            {
                yield return req.SendWebRequest();
                labelData = req.downloadHandler.text;
            }
        }
        else
        {
            try { labelData = File.ReadAllText(labelPath); } catch { }
        }

        if (!string.IsNullOrEmpty(labelData))
        {
            labels = labelData.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        }

        if (captureButton != null) captureButton.interactable = true;
        if (resultText != null) resultText.text = "✅ Ready (AR)";
    }

    public void CaptureAndAnalyze()
    {
        if (!isModelReady || isCaptured) return;

        if (arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            StartCoroutine(ProcessARImage(image));
        }
        else
        {
            if (resultText != null) resultText.text = "Failed to acquire AR image";
        }
    }

    // INI ADALAH FUNGSI UTAMA (Versi Fixed & Synchronous)
    IEnumerator ProcessARImage(XRCpuImage image)
    {
        using (image)
        {
            isCaptured = true;
            if (captureButton != null) captureButton.interactable = false;

            // =========================================================
            // 1. FREEZE DISPLAY (HIGH QUALITY / FULL RESOLUTION)
            // =========================================================

            // HAPUS DOWNSAMPLING. Gunakan resolusi penuh.
            var displayParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = new Vector2Int(image.width, image.height), // FULL RES
                outputFormat = TextureFormat.RGB24,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            // Setup Texture
            if (displayTexture == null || displayTexture.width != displayParams.outputDimensions.x)
            {
                if (displayTexture != null) Destroy(displayTexture);
                displayTexture = new Texture2D(displayParams.outputDimensions.x, displayParams.outputDimensions.y, TextureFormat.RGB24, false);

                // PENTING: Gunakan Bilinear agar halus, bukan Point (kotak-kotak)
                displayTexture.filterMode = FilterMode.Bilinear;
            }

            // Convert Synchronous
            var displayBuffer = new NativeArray<byte>(image.GetConvertedDataSize(displayParams), Allocator.Temp);
            image.Convert(displayParams, displayBuffer);
            displayTexture.LoadRawTextureData(displayBuffer);
            displayTexture.Apply();
            displayBuffer.Dispose();

            // Tampilkan ke UI
            if (freezePreview != null)
            {
                freezePreview.texture = displayTexture;
                freezePreview.gameObject.SetActive(true);

                // Reset ukuran pixel asli
                freezePreview.SetNativeSize();

                // Rotasi 90 derajat (Portrait Mode)
                freezePreview.rectTransform.localEulerAngles = new Vector3(0, 0, 90);

                // --- LOGIKA ZOOM TO FILL (FULL SCREEN HD) ---
                float visualWidth = displayTexture.height; // Lebar visual = Tinggi texture (karena rotasi)
                float visualHeight = displayTexture.width; // Tinggi visual = Lebar texture

                float screenWidth = Screen.width;
                float screenHeight = Screen.height;

                float scaleX = screenWidth / visualWidth;
                float scaleY = screenHeight / visualHeight;

                // Ambil scale terbesar agar menutup seluruh layar tanpa sisa hitam
                float finalScale = Mathf.Max(scaleX, scaleY);

                freezePreview.rectTransform.localScale = new Vector3(finalScale, finalScale, 1f);
                freezePreview.rectTransform.anchoredPosition = Vector2.zero;
            }

            // =========================================================
            // 2. AI PROCESSING (TETAP KECIL: 224x224)
            // =========================================================
            // Kita pisahkan proses AI agar AI tetap cepat meskipun tampilan HD
            var aiParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = new Vector2Int(inputWidth, inputHeight),
                outputFormat = TextureFormat.RGB24,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            if (aiTexture == null) aiTexture = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

            var aiBuffer = new NativeArray<byte>(image.GetConvertedDataSize(aiParams), Allocator.Temp);
            image.Convert(aiParams, aiBuffer);
            aiTexture.LoadRawTextureData(aiBuffer);
            aiTexture.Apply();
            aiBuffer.Dispose();

            yield return null;
            RunInference();
        }
    }

    public void ContinueCamera()
    {
        if (freezePreview != null) freezePreview.gameObject.SetActive(false);
        isCaptured = false;
        if (captureButton != null) captureButton.interactable = true;
        if (resultText != null) resultText.text = "✅ Ready (AR)";
    }

    void RunInference()
    {
        if (resultText != null) resultText.text = "Analyzing...";

        Color32[] pixels = aiTexture.GetPixels32();
        int idx = 0;
        for (int i = 0; i < pixels.Length; i++)
        {
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
            for (int i = 1; i < outputBuffer.Length; i++)
            {
                if (outputBuffer[i] > maxConfidence)
                {
                    maxConfidence = outputBuffer[i];
                    predictedClass = i;
                }
            }

            string className = (labels != null && predictedClass < labels.Length)
                ? labels[predictedClass] : $"Class {predictedClass}";

            Debug.Log($"[RESULT] {className} ({maxConfidence:P1})");

            if (resultDisplay != null)
            {
                resultDisplay.ShowResult(className);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError(e.Message);
        }
    }

    void OnDestroy()
    {
        interpreter?.Dispose();
        if (displayTexture != null) Destroy(displayTexture);
        if (aiTexture != null) Destroy(aiTexture);
    }
}