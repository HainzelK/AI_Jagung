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
        if (progressCircle != null) progressCircle.fillAmount = calculator.finalWeightedScore / 100f;

        // 3. Show Panel
        if (resultPanel != null)
        {
            resultPanel.SetActive(true);
            CanvasGroup cg = resultPanel.GetComponent<CanvasGroup>();
            if (cg != null) { cg.alpha = 1f; cg.interactable = true; cg.blocksRaycasts = true; }
        }

        // 4. Force Capture Button OFF
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
            detector.ContinueCamera(); // <--- This enables the button
        }
        else
        {
            Debug.LogError("Detector reference missing in ResultDisplay!");
        }
    }
}