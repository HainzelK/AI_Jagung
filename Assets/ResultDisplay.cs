using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ResultDisplay : MonoBehaviour
{
    [Header("Script References")]
    [Tooltip("Hubungkan ke script CornQualityCalculator di scene")]
    public CornQualityCalculator calculator;

    [Tooltip("Hubungkan ke script CornTFLiteDetector di scene")]
    public CornTFLiteDetector detector;

    [Header("UI Panels")]
    public GameObject resultPanel; // Panel utama hasil

    [Header("UI Text Elements")]
    public TMP_Text titleText;      // Menampilkan Label (Healthy/Diseased)
    public TMP_Text percentageText; // Menampilkan Skor Akhir (0-100%)
    public Image progressCircle;    // Lingkaran loading bar skor

    [Header("Detail Text")]
    public TMP_Text jagungLength;   // Menampilkan panjang jagung (cm)
    public TMP_Text benihKetahanan; // Menampilkan status ketahanan benih
    public TMP_Text benihKondisi;   // Menampilkan kondisi fisik (sama dengan label)
    public TMP_Text tanahKondisi;   // Menampilkan kondisi tanah

    /// <summary>
    /// Dipanggil oleh CornTFLiteDetector setelah analisa selesai.
    /// </summary>
    /// <param name="aiLabel">Hasil klasifikasi AI (e.g. Corn_Healthy)</param>
    /// <param name="measuredLengthCm">Hasil pengukuran panjang via Koin</param>
    public void ShowResult(string aiLabel, float measuredLengthCm)
    {
        // 1. Validasi Panjang (Pastikan tidak negatif)
        float finalLength = measuredLengthCm > 0.1f ? measuredLengthCm : 0f;

        // 2. Kirim Data ke Calculator untuk Hitung Skor Total
        if (calculator != null)
        {
            calculator.cornLengthCm = finalLength; // Update panjang
            calculator.CalculateQuality(aiLabel);  // Hitung skor berdasarkan bobot
        }
        else
        {
            Debug.LogError("CornQualityCalculator belum di-assign di ResultDisplay!");
            return;
        }

        // 3. Update Text UI
        if (titleText != null) titleText.text = aiLabel; // Contoh: "Corn_Healthy"

        if (percentageText != null)
            percentageText.text = $"{calculator.finalWeightedScore:F0}%"; // Contoh: "85%"

        if (jagungLength != null)
            jagungLength.text = finalLength > 0 ? $"{finalLength:F1} cm" : "Gagal Ukur";

        if (benihKetahanan != null)
            benihKetahanan.text = calculator.seedResistance.ToString();

        if (tanahKondisi != null)
            tanahKondisi.text = calculator.soilCondition.ToString();

        if (benihKondisi != null)
            benihKondisi.text = aiLabel; // Kondisi fisik diambil dari label AI

        // 4. Update Visual Lingkaran (Progress Circle)
        if (progressCircle != null)
        {
            // Fill Amount (0.0 sampai 1.0)
            progressCircle.fillAmount = calculator.finalWeightedScore / 100f;

            // Logika Warna: Biru jika Bagus, Merah jika Buruk
            if (calculator.finalWeightedScore >= 50)
            {
                // Warna Biru (#0099F8)
                progressCircle.color = new Color32(0, 153, 248, 255);
            }
            else
            {
                // Warna Merah (#EE0000)
                progressCircle.color = new Color32(238, 0, 0, 255);
            }
        }

        // 5. Tampilkan Panel Hasil
        if (resultPanel != null)
        {
            resultPanel.SetActive(true);

            // Jika menggunakan CanvasGroup untuk animasi fade/interaksi
            CanvasGroup cg = resultPanel.GetComponent<CanvasGroup>();
            if (cg != null)
            {
                cg.alpha = 1f;
                cg.interactable = true;
                cg.blocksRaycasts = true;
            }
        }

        // 6. Matikan Tombol Capture (Agar user menekan tombol Back dulu untuk foto lagi)
        if (detector != null && detector.captureButton != null)
        {
            detector.captureButton.interactable = false;
        }
    }

    // ------------------------------------------------------------
    // HUBUNGKAN FUNGSI INI KE TOMBOL "BACK" / "CLOSE" DI INSPECTOR
    // ------------------------------------------------------------
    public void OnBackButtonPressed()
    {
        Debug.Log("Back Button Pressed.");

        // 1. Sembunyikan Panel Hasil
        if (resultPanel != null)
        {
            CanvasGroup cg = resultPanel.GetComponent<CanvasGroup>();
            if (cg != null)
            {
                cg.alpha = 0f;
                cg.interactable = false;
                cg.blocksRaycasts = false;
            }
            else
            {
                resultPanel.SetActive(false);
            }
        }

        // 2. Perintahkan Kamera untuk Mulai Lagi (Unfreeze & Reset State)
        if (detector != null)
        {
            detector.ContinueCamera();
        }
        else
        {
            Debug.LogError("Detector reference missing in ResultDisplay!");
        }
    }
}