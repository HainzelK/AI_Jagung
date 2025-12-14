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

    [Header("TFLite Settings")]
    [Tooltip("Matikan jika aplikasi keluar sendiri (Crash) saat tombol ditekan")]
    public bool useGPUDelegate = false;

    private WebCamTexture webcam;
    private Interpreter interpreter;

    private Texture2D frozenFrame;
    private float[] inputBuffer;
    private float[] outputBuffer;
    private string[] labels;

    // Dimensi yang dibaca otomatis dari Model
    private int inputWidth;
    private int inputHeight;
    private int inputChannels;

    private bool isModelReady = false;
    private bool isFrozen = false;

    IEnumerator Start()
    {
        // Matikan tombol saat loading
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

        if (modelPath.Contains("://")) // Android (dalam APK)
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                modelData = www.downloadHandler.data;
            }
        }
        else // Editor (PC)
        {
            try { modelData = File.ReadAllBytes(modelPath); } catch {}
        }

        // Init TFLite
        try
        {
            var options = new InterpreterOptions();

            // GPU Delegate (Opsional)
            if (useGPUDelegate)
            {
                try { options.AddGpuDelegate(); }
                catch { Debug.LogWarning("GPU gagal, pakai CPU saja."); }
            }

            options.threads = 2;
            interpreter = new Interpreter(modelData, options);
            interpreter.AllocateTensors();

            // --- 3. BACA UKURAN MODEL OTOMATIS (Anti Crash) ---
            var inputInfo = interpreter.GetInputTensorInfo(0);
            int[] inputShape = inputInfo.shape; // contoh: [1, 224, 224, 3]

            // Ambil Lebar, Tinggi, Channel dari info model
            if (inputShape.Length == 4)
            {
                inputHeight = inputShape[1];
                inputWidth = inputShape[2];
                inputChannels = inputShape[3];
            }
            else
            {
                // Jaga-jaga jika format model beda, pakai standar default
                inputHeight = 224;
                inputWidth = 224;
                inputChannels = 3;
            }

            Debug.Log($"Ukuran Model: {inputWidth}x{inputHeight}, Channel: {inputChannels}");

            // Siapkan Buffer Data
            inputBuffer = new float[inputWidth * inputHeight * inputChannels];

            // Siapkan Output
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

        // --- 4. Load Labels ---
        yield return LoadLabels();

        resultText.text = "✅ Siap!\nMenyalakan kamera...";
        StartCamera();
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

    // --- LOGIKA UTAMA FULL SCREEN DI SINI ---
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
        StartCoroutine(CaptureRoutine());
    }

    // Coroutine untuk mengambil gambar dengan aman
    IEnumerator CaptureRoutine()
    {
        yield return new WaitForEndOfFrame();

        // Freeze frame
        RenderTexture rt = RenderTexture.GetTemporary(width, height, 0);
        Graphics.Blit(webcam, rt);
        frozenFrame = new Texture2D(width, height, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        frozenFrame.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        frozenFrame.Apply();

        // Kembalikan settingan render
        RenderTexture.active = oldRt;
        RenderTexture.ReleaseTemporary(rt);

        // Hentikan kamera dan tampilkan hasil beku
        webcam.Stop();
        cameraPreview.texture = frozenFrame;

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

        // Kembali ke mode live kamera
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

    void RunInference()
    {
        if (resultText != null) resultText.text = "Analyzing...";

        // Ambil warna pixel
        Color32[] pixels = frozenFrame.GetPixels32();

        // Normalisasi data (0-255 menjadi 0.0-1.0) untuk model Float32
        int idx = 0;
        for (int i = 0; i < pixels.Length; i++) {
            inputBuffer[idx++] = pixels[i].r / 255.0f;
            inputBuffer[idx++] = pixels[i].g / 255.0f;
            inputBuffer[idx++] = pixels[i].b / 255.0f;
        }

        // Masukkan data ke TFLite
        interpreter.SetInputTensorData(0, inputBuffer);

        try
        {
            // Jalankan Deteksi
            interpreter.Invoke();

            // Ambil Hasil
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