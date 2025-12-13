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
    public AspectRatioFitter fitter; // Pastikan ini diisi di Inspector!
    public Button captureButton;
    public Button continueButton;

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
        if (continueButton != null) continueButton.interactable = false;

        resultText.text = "Meminta izin kamera...";

        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            resultText.text = "❌ Izin kamera ditolak!";
            yield break;
        }

        // Tunggu sebentar
        yield return new WaitForSeconds(0.5f);

        // --- 1. Load Model ---
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        byte[] modelData = null;

        if (modelPath.Contains("://")) // Android (dalam APK)
        {
            using (UnityWebRequest www = UnityWebRequest.Get(modelPath))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success)
                {
                    resultText.text = $"❌ Gagal load model:\n{www.error}";
                    yield break;
                }
                modelData = www.downloadHandler.data;
            }
        }
        else // Editor (PC)
        {
            if (File.Exists(modelPath)) modelData = File.ReadAllBytes(modelPath);
            else { resultText.text = "❌ Model tidak ditemukan di StreamingAssets!"; yield break; }
        }

        // --- 2. Inisialisasi TFLite ---
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
            resultText.text = "❌ Error Inisialisasi TFLite!";
            Debug.LogError($"TFLite Error: {e}");
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
        string labelData = "";

        if (labelPath.Contains("://"))
        {
            using (UnityWebRequest req = UnityWebRequest.Get(labelPath))
            {
                yield return req.SendWebRequest();
                labelData = req.downloadHandler.text;
            }
        }
        else if (File.Exists(labelPath))
        {
            labelData = File.ReadAllText(labelPath);
        }

        if (!string.IsNullOrEmpty(labelData))
        {
            labels = labelData.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        }
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

        // PERUBAHAN DI SINI:
        // Gunakan ukuran layar saat ini untuk meminta resolusi kamera.
        // Kita tukar Width dan Height karena sensor kamera HP biasanya Landscape fisik,
        // tapi layar kita Portrait. Ini membantu mendapatkan FOV (Field of View) maksimal.
        int w = Screen.height;
        int h = Screen.width;

        // Minta resolusi tinggi sesuai layar
        webcam = new WebCamTexture(deviceName, w, h, 30);

        cameraPreview.texture = webcam;
        webcam.Play();

        if (captureButton) captureButton.interactable = true;
        if (continueButton) continueButton.interactable = false;

        resultText.text = $"✅ Siap Deteksi";
    }

    // --- LOGIKA UTAMA FULL SCREEN DI SINI ---
    void Update()
    {
        if (webcam == null || !webcam.isPlaying || isFrozen) return;

        if (webcam.width > 100)
        {
            // --- PERBAIKAN LOGIKA RASIO ---
            if (fitter != null)
            {
                // Cek rotasi kamera. Jika 90 atau 270 derajat, berarti orientasi Portrait.
                // Kita harus menukar rasio Width dan Height.
                int rotation = webcam.videoRotationAngle;
                float ratio;

                if (rotation % 180 == 0)
                {
                    // Landscape (0 atau 180)
                    ratio = (float)webcam.width / (float)webcam.height;
                }
                else
                {
                    // Portrait (90 atau 270)
                    // Rasio dibalik agar Fitter menyesuaikan dengan layar tegak
                    ratio = (float)webcam.height / (float)webcam.width;
                }

                fitter.aspectRatio = ratio;
            }

            // Atur Rotasi (Agar Portrait tegak lurus)
            int orient = -webcam.videoRotationAngle;
            cameraPreview.rectTransform.localEulerAngles = new Vector3(0, 0, orient);

            // Atur Mirroring
            float scaleY = webcam.videoVerticallyMirrored ? -1f : 1f;
            cameraPreview.rectTransform.localScale = new Vector3(1f, scaleY, 1f);
        }
    }

    public void CaptureAndAnalyze()
    {
        if (!isModelReady || webcam == null || !webcam.isPlaying) return;
        StartCoroutine(CaptureRoutine());
    }

    // Coroutine untuk mengambil gambar dengan aman
    IEnumerator CaptureRoutine()
    {
        yield return new WaitForEndOfFrame();

        // Buat Texture sementara di GPU untuk resize otomatis
        // Kita resize gambar kamera HD ke ukuran kecil model (misal 224x224)
        RenderTexture rt = RenderTexture.GetTemporary(inputWidth, inputHeight, 0, RenderTextureFormat.ARGB32);
        RenderTexture oldRt = RenderTexture.active;

        // Copy gambar webcam ke RenderTexture kecil
        Graphics.Blit(webcam, rt);

        // Bersihkan memory lama
        if (frozenFrame != null) Destroy(frozenFrame);
        frozenFrame = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        // Baca pixel dari RenderTexture
        RenderTexture.active = rt;
        frozenFrame.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        frozenFrame.Apply();

        // Kembalikan settingan render
        RenderTexture.active = oldRt;
        RenderTexture.ReleaseTemporary(rt);

        // Hentikan kamera dan tampilkan hasil beku
        webcam.Stop();
        cameraPreview.texture = frozenFrame;

        // Karena texture kita sekarang kecil (224x224) dan mungkin orientasinya berubah setelah dibaca,
        // Kita reset rotasi preview agar hasil crop terlihat tegak (Opsional, tergantung hasil di HP)
        // Jika hasil freeze miring, hapus baris di bawah ini:
        cameraPreview.rectTransform.localEulerAngles = Vector3.zero;
        cameraPreview.rectTransform.localScale = Vector3.one;

        isFrozen = true;
        if (captureButton) captureButton.interactable = false;
        if (continueButton) continueButton.interactable = true;

        RunInference();
    }

    public void ContinueCamera()
    {
        if (webcam == null) return;

        // Kembali ke mode live kamera
        cameraPreview.texture = webcam;
        webcam.Play();
        isFrozen = false;

        if (captureButton) captureButton.interactable = true;
        if (continueButton) continueButton.interactable = false;
        resultText.text = "✅ Siap Deteksi";
    }

    void RunInference()
    {
        if (interpreter == null) return;

        resultText.text = "🧠 Menganalisa...";

        // Ambil warna pixel
        Color32[] pixels = frozenFrame.GetPixels32();

        // Normalisasi data (0-255 menjadi 0.0-1.0) untuk model Float32
        int idx = 0;
        for (int i = 0; i < pixels.Length; i++)
        {
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

            // Cari nilai tertinggi (Confidence tertinggi)
            float maxVal = 0f;
            int maxIdx = -1;
            for (int i = 0; i < outputBuffer.Length; i++)
            {
                if (outputBuffer[i] > maxVal)
                {
                    maxVal = outputBuffer[i];
                    maxIdx = i;
                }
            }

            // Tampilkan Text
            string label = (labels != null && maxIdx >= 0 && maxIdx < labels.Length) ? labels[maxIdx] : $"Class {maxIdx}";

            // Tampilkan dalam persen
            float confidencePercent = maxVal * 100f;

            if (maxVal >= detectionThreshold)
                resultText.text = $"Hasil: {label}\nAkurasi: {confidencePercent:F1}%";
            else
                resultText.text = $"Kurang Yakin: {label}\nAkurasi: {confidencePercent:F1}%";
        }
        catch (System.Exception e)
        {
            Debug.LogError(e);
            resultText.text = "Gagal Deteksi";
        }
    }

    void OnDestroy()
    {
        if (webcam != null) webcam.Stop();
        if (interpreter != null) interpreter.Dispose();
        if (frozenFrame != null) Destroy(frozenFrame);
    }
}