using UnityEngine;
using System;

[System.Serializable]
public class CornQualityCalculator : MonoBehaviour
{
    [Header("Manual Input / Sensor")]
    [Tooltip("Since AI currently detects classification only, input this value manually or from a distance sensor.")]
    public float cornLengthCm = 18.0f; // Default example

    [Header("Environmental Constants")]
    // 3. Seed Resistance (Default: Strong)
    public SeedResistanceType seedResistance = SeedResistanceType.Strong;

    // 4. Soil Condition (Default: Moist)
    public SoilConditionType soilCondition = SoilConditionType.Moist;

    // Enums for easy selection in Inspector
    public enum SeedResistanceType { Strong, Medium, Weak }
    public enum SoilConditionType { Moist, Dry, VeryDry }

    [Header("Calculation Results")]
    public int physicalConditionScore;
    public int lengthScore;
    public int resistanceScore;
    public int soilScore;
    public float finalWeightedScore;

    /// <summary>
    /// Main function to calculate the total quality score based on weights.
    /// </summary>
    /// <param name="aiLabel">Label detected by CornTFLiteDetector (e.g., "Corn_Healthy")</param>
    public void CalculateQuality(string aiLabel)
    {
        // 1. Calculate Physical Condition Score (From AI)
        physicalConditionScore = GetConditionScore(aiLabel);

        // 2. Calculate Length Score
        lengthScore = GetLengthScore(cornLengthCm);

        // 3. Calculate Seed Resistance Score
        resistanceScore = GetResistanceScore(seedResistance);

        // 4. Calculate Soil Condition Score
        soilScore = GetSoilScore(soilCondition);

        // --- CALCULATE TOTAL WITH WEIGHTS ---
        // Condition: 30%, Length: 30%, Seed: 20%, Soil: 20%

        float weightCondition = 0.30f;
        float weightLength = 0.30f;
        float weightSeed = 0.20f;
        float weightSoil = 0.20f;

        finalWeightedScore = (physicalConditionScore * weightCondition) +
                             (lengthScore * weightLength) +
                             (resistanceScore * weightSeed) +
                             (soilScore * weightSoil);

        // --- LOGGING ---
        Debug.Log($"[CALCULATION] AI Label: {aiLabel} | Input Length: {cornLengthCm}cm");
        Debug.Log($"[RAW SCORES] Physical: {physicalConditionScore}, Length: {lengthScore}, Seed: {resistanceScore}, Soil: {soilScore}");
        Debug.Log($"[FORMULA] ({physicalConditionScore} * 0.3) + ({lengthScore} * 0.3) + ({resistanceScore} * 0.2) + ({soilScore} * 0.2)");
        Debug.Log($"[FINAL RESULT] Weighted Score: {finalWeightedScore}");
    }

    // 1. Corn Condition Logic (From AI)
    private int GetConditionScore(string label)
    {
        // Convert to lower case for case-insensitive comparison
        label = label.ToLower();

        // Checks for both English and Indonesian keywords for compatibility
        if (label.Contains("good") || label.Contains("healthy") || label.Contains("baik"))
            return 100;
        else if (label.Contains("damaged") || label.Contains("rusak"))
            return 50;
        else if (label.Contains("fungus") || label.Contains("disease") || label.Contains("jamur") || label.Contains("penyakit"))
            return 0;

        return 0; // Default if unknown
    }

    // 2. Corn Length Logic
    private int GetLengthScore(float cm)
    {
        if (cm > 20) return 100;
        else if (cm >= 15) return 75;  // 15 - 20
        else if (cm >= 12) return 50;  // 12 - 14
        else if (cm >= 7) return 25;   // 7 - 11
        else return 0;                 // 0 - 6
    }

    // 3. Seed Resistance Logic
    private int GetResistanceScore(SeedResistanceType type)
    {
        switch (type)
        {
            case SeedResistanceType.Strong: return 100;
            case SeedResistanceType.Medium: return 50;
            case SeedResistanceType.Weak: return 0;
            default: return 0;
        }
    }

    // 4. Soil Condition Logic
    private int GetSoilScore(SoilConditionType type)
    {
        switch (type)
        {
            case SoilConditionType.Moist: return 100;
            case SoilConditionType.Dry: return 50;
            case SoilConditionType.VeryDry: return 0;
            default: return 0;
        }
    }
}