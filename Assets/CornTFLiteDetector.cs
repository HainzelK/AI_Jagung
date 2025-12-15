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

    [Header("Size Estimation (Coin Method)")]
    [Tooltip("Masukkan GameObject yang memiliki script CornSizeWithCoin")]
    public CornSizeWithCoin coinCalculator;

    [Header("UI Assignments")]
    public RawImage freezePreview; // Image UI full screen untuk preview foto
    public TMP_Text resultText;    // Text status kecil di bawah/atas
    public Button captureButton;   // Tombol shutter

    [Header("Model Settings")]
    public string modelFileName = "corn_detector.tflite";
    public string labelFileName = "labels.txt";

    [Header("TFLite Settings")]
    public bool useGPUDelegate = false;

    // Internal Variables
    private Interpreter interpreter;
    private Texture2D aiTexture;      // Texture Kecil (misal 224x224) untuk AI
    private Texture2D displayTexture; // Texture Besar (Full HD) untuk Tampilan & Ukur Koin

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
        // 1. Setup UI Awal
        if (freezePreview != null)
        {
            freezePreview.gameObject.SetActive(false);
            // Reset transform agar tidak aneh saat start
            freezePreview.rectTransform.pivot = new Vector2(0.5f, 0.5f);
            freezePreview.rectTransform.anchorMin = new Vector2(0.5f, 0.5f);
            freezePreview.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
        }

        if (captureButton != null) captureButton.interactable = false;
        if (resultText != null) resultText.text = "Loading Model...";

        // 2. Load Model File
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        byte[] modelData = null;

        // Handle Android StreamingAssets (via WebRequest) vs Editor (File.Read)
        if (modelPath.Contains("://"))
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError("Error loading model: " + www.error);
                    yield break;
                }
                modelData = www.downloadHandler.data;
            }
        }
        else
        {
            try { modelData = File.ReadAllBytes(modelPath); } catch (System.Exception e) { Debug.LogError(e); }
        }

        // 3. Initialize TFLite Interpreter
        try
        {
            var options = new InterpreterOptions();
            if (useGPUDelegate) { try { options.AddGpuDelegate(); } catch { Debug.LogWarning("GPU Delegate not supported"); } }
            options.threads = 2; // Pakai 2 core CPU

            interpreter = new Interpreter(modelData, options);
            interpreter.AllocateTensors();

            // Get Input Shape
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
                // Fallback default
                inputHeight = 224; inputWidth = 224; inputChannels = 3;
            }

            inputBuffer = new float[inputWidth * inputHeight * inputChannels];

            // Get Output Shape
            var outputInfo = interpreter.GetOutputTensorInfo(0);
            int outputClasses = outputInfo.shape[1];
            outputBuffer = new float[outputClasses];

            isModelReady = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TFLite Init Error: {e.Message}");
            if (resultText != null) resultText.text = "Model Error!";
            yield break;
        }

        // 4. Load Labels
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

        // Selesai Loading
        if (captureButton != null) captureButton.interactable = true;
        if (resultText != null) resultText.text = "✅ Ready (Scan with Coin)";
    }

    // Dipanggil oleh tombol UI Button
    public void CaptureAndAnalyze()
    {
        if (!isModelReady || isCaptured) return;

        // Ambil gambar mentah dari AR Camera Manager
        if (arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            StartCoroutine(ProcessARImage(image));
        }
        else
        {
            if (resultText != null) resultText.text = "Camera Error";
        }
    }

    IEnumerator ProcessARImage(XRCpuImage image)
    {
        // Wajib menggunakan 'using' agar memory image dibersihkan
        using (image)
        {
            isCaptured = true;
            if (captureButton != null) captureButton.interactable = false;

            // =========================================================
            // 1. BUAT DISPLAY TEXTURE (HIGH RES)
            // =========================================================
            // Texture ini digunakan untuk ditampilkan di layar (Freeze) DAN untuk menghitung koin (butuh detail).

            var displayParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = new Vector2Int(image.width, image.height), // Resolusi Asli
                outputFormat = TextureFormat.RGB24,
                transformation = XRCpuImage.Transformation.MirrorY // Mirror agar seperti cermin/kamera selfie, atau sesuaikan
            };

            // Setup Texture Display jika belum ada
            if (displayTexture == null || displayTexture.width != displayParams.outputDimensions.x)
            {
                if (displayTexture != null) Destroy(displayTexture);
                displayTexture = new Texture2D(displayParams.outputDimensions.x, displayParams.outputDimensions.y, TextureFormat.RGB24, false);
                displayTexture.filterMode = FilterMode.Bilinear; // Agar halus
            }

            // Convert data mentah ke texture
            var displayBuffer = new NativeArray<byte>(image.GetConvertedDataSize(displayParams), Allocator.Temp);
            image.Convert(displayParams, displayBuffer);
            displayTexture.LoadRawTextureData(displayBuffer);
            displayTexture.Apply();
            displayBuffer.Dispose();

            // Tampilkan ke UI RawImage
            if (freezePreview != null)
            {
                freezePreview.texture = displayTexture;
                freezePreview.gameObject.SetActive(true);
                freezePreview.SetNativeSize();

                // Rotate 90 derajat karena orientasi kamera HP biasanya portrait
                freezePreview.rectTransform.localEulerAngles = new Vector3(0, 0, 90);

                // --- LOGIKA ZOOM-TO-FILL (Agar full screen tanpa bar hitam) ---
                float visualWidth = displayTexture.height; // Dibalik karena rotasi 90
                float visualHeight = displayTexture.width;

                float screenWidth = Screen.width;
                float screenHeight = Screen.height;

                // Hitung skala agar menutup layar
                float scale = Mathf.Max(screenWidth / visualWidth, screenHeight / visualHeight);

                freezePreview.rectTransform.localScale = new Vector3(scale, scale, 1f);
                freezePreview.rectTransform.anchoredPosition = Vector2.zero;
            }

            // =========================================================
            // 2. BUAT AI TEXTURE (LOW RES)
            // =========================================================
            // Texture ini dikecilkan (misal 224x224) agar TFLite ringan memprosesnya.

            var aiParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = new Vector2Int(inputWidth, inputHeight), // Resize ke ukuran model
                outputFormat = TextureFormat.RGB24,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            if (aiTexture == null) aiTexture = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

            var aiBuffer = new NativeArray<byte>(image.GetConvertedDataSize(aiParams), Allocator.Temp);
            image.Convert(aiParams, aiBuffer);
            aiTexture.LoadRawTextureData(aiBuffer);
            aiTexture.Apply();
            aiBuffer.Dispose();

            // Tunggu 1 frame agar UI sempat ter-render sebelum freeze (opsional)
            yield return null;

            // Jalankan Analisis
            RunInference();
        }
    }

    public void ContinueCamera()
    {
        // Dipanggil tombol "Back" di result screen
        if (freezePreview != null) freezePreview.gameObject.SetActive(false);
        isCaptured = false;
        if (captureButton != null) captureButton.interactable = true;
        if (resultText != null) resultText.text = "✅ Ready (Scan with Coin)";
    }

    void RunInference()
    {
        if (resultText != null) resultText.text = "Analyzing...";

        // ------------------------------------
        // A. PROSES TFLITE (Klasifikasi)
        // ------------------------------------
        Color32[] pixels = aiTexture.GetPixels32();
        int idx = 0;

        // Normalize 0-255 byte ke 0.0-1.0 float
        for (int i = 0; i < pixels.Length; i++)
        {
            inputBuffer[idx++] = pixels[i].r / 255.0f;
            inputBuffer[idx++] = pixels[i].g / 255.0f;
            inputBuffer[idx++] = pixels[i].b / 255.0f;
        }

        interpreter.SetInputTensorData(0, inputBuffer);

        string className = "Unknown";
        float maxConfidence = 0f;

        try
        {
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputBuffer);

            // Cari nilai confidence tertinggi
            int predictedClass = 0;
            maxConfidence = outputBuffer[0];
            for (int i = 1; i < outputBuffer.Length; i++)
            {
                if (outputBuffer[i] > maxConfidence)
                {
                    maxConfidence = outputBuffer[i];
                    predictedClass = i;
                }
            }

            className = (labels != null && predictedClass < labels.Length)
                ? labels[predictedClass] : $"Class {predictedClass}";
        }
        catch (System.Exception e)
        {
            Debug.LogError("Inference Error: " + e.Message);
        }

        // ------------------------------------
        // B. PROSES UKURAN (Metode Koin)
        // ------------------------------------
        float estimatedLength = 0f;
        if (coinCalculator != null)
        {
            estimatedLength = coinCalculator.CalculateLength(displayTexture);

            // --- TAMBAHAN UNTUK DEBUGGING ---
            // Jika gagal ukur (0) atau tidak pas, kita tampilkan visualisasinya
            Texture2D debugTex = coinCalculator.GetLastDebugTexture();
            if (freezePreview != null && debugTex != null)
            {
                // Tampilkan hasil threshold hitam-putih di layar utama sejenak
                // agar kita tahu apakah backgroundnya bersih atau kotor
                freezePreview.texture = debugTex;
            }
            // -------------------------------
        }
        else
        {
            Debug.LogWarning("CoinCalculator belum di-assign di Inspector!");
        }

        Debug.Log($"[RESULT] Class: {className} ({maxConfidence:P1}) | Length: {estimatedLength} cm");

        // ------------------------------------
        // C. TAMPILKAN HASIL
        // ------------------------------------
        if (resultDisplay != null)
        {
            // Kirim Label + Panjang ke UI Hasil
            resultDisplay.ShowResult(className, estimatedLength);
        }
    }

    void OnDestroy()
    {
        // Bersihkan memori saat aplikasi tutup/pindah scene
        interpreter?.Dispose();
        if (displayTexture != null) Destroy(displayTexture);
        if (aiTexture != null) Destroy(aiTexture);
    }
}