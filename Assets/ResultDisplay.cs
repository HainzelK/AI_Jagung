using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ResultDisplay : MonoBehaviour
{
    [Header("References")]
    public CornQualityCalculator calculator;
    public CornTFLiteDetector detector; // <--- MUST BE ASSIGNED

    [Header("UI Panels")]
    public GameObject resultPanel;

    [Header("UI Text Elements")]
    public TMP_Text titleText;
    public TMP_Text percentageText;
    public Image progressCircle;
    public TMP_Text jagungLength;
    public TMP_Text benihKetahanan;
    public TMP_Text benihKondisi;
    public TMP_Text tanahKondisi;

    public void ShowResult(string aiLabel)
    {
        // 1. Calculations
        if (calculator != null) calculator.CalculateQuality(aiLabel);

        // 2. Text Updates
        if (titleText != null) titleText.text = "Result: " + aiLabel;
        if (percentageText != null) percentageText.text = $"{calculator.finalWeightedScore:F0}%";
        if (jagungLength != null) jagungLength.text = $"{calculator.cornLengthCm} cm";
        if (benihKetahanan != null) benihKetahanan.text = calculator.seedResistance.ToString();
        if (tanahKondisi != null) tanahKondisi.text = calculator.soilCondition.ToString();
        if (benihKondisi != null) benihKondisi.text = aiLabel;

        // 3. Update Progress Circle (Fill & Color)
        if (progressCircle != null) 
        {
            // Set Fill Amount
            progressCircle.fillAmount = calculator.finalWeightedScore / 100f;

            // --- COLOR LOGIC ADDED HERE ---
            if (calculator.finalWeightedScore >= 50)
            {
                // Hex: #0099F8 (Blue) -> R:0, G:153, B:248
                progressCircle.color = new Color32(0, 153, 248, 255);
            }
            else
            {
                // Hex: #EE0000 (Red) -> R:238, G:0, B:0
                progressCircle.color = new Color32(238, 0, 0, 255);
            }
        }

        // 4. Show Panel
        if (resultPanel != null)
        {
            resultPanel.SetActive(true);
            CanvasGroup cg = resultPanel.GetComponent<CanvasGroup>();
            if (cg != null) { cg.alpha = 1f; cg.interactable = true; cg.blocksRaycasts = true; }
        }

        // 5. Force Capture Button OFF
        if (detector != null && detector.captureButton != null)
        {
            detector.captureButton.interactable = false;
        }
    }

    // ------------------------------------------------------------
    // LINK THIS TO YOUR BACK BUTTON IN INSPECTOR
    // ------------------------------------------------------------
    public void OnBackButtonPressed()
    {
        Debug.Log("Back Button Pressed.");

        // 1. Hide Panel
        if (resultPanel != null)
        {
            CanvasGroup cg = resultPanel.GetComponent<CanvasGroup>();
            if (cg != null) { cg.alpha = 0f; cg.interactable = false; cg.blocksRaycasts = false; }
            else { resultPanel.SetActive(false); }
        }

        // 2. Resume Camera & Enable Button
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