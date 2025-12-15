using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class CornSizeWithCoin : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("Diameter koin asli dalam CM (Koin 1000 = 2.4, Koin 500 = 2.7)")]
    public float coinRealDiameterCm = 2.4f;

    [Tooltip("Sensitivitas perbedaan warna (0.1 - 0.9). Makin kecil makin sensitif.")]
    [Range(0.05f, 0.5f)]
    public float sensitivity = 0.15f;

    [Tooltip("Minimal pixel agar dianggap objek (cegah noise).")]
    public int minPixelArea = 400;

    // Untuk Debugging Visual (Opsional)
    private Texture2D debugTexture;

    public class DetectedObject
    {
        public Rect bounds;
        public int pixelCount;
        public float aspectRatio; // Lebar dibagi Tinggi
        public string type; // "Coin" atau "Corn"
    }

    /// <summary>
    /// Mengembalikan Texture Hitam Putih hasil deteksi (untuk ditampilkan di UI agar user paham)
    /// </summary>
    public Texture2D GetLastDebugTexture()
    {
        return debugTexture;
    }

    public float CalculateLength(Texture2D photo)
    {
        // 1. Pre-process & Find Objects
        List<DetectedObject> objects = FindObjectsImproved(photo);

        Debug.Log($"[SIZE V2] Ditemukan {objects.Count} kandidat objek.");

        if (objects.Count < 2)
        {
            Debug.LogWarning("Gagal: Kurang dari 2 objek terdeteksi. Cek lighting/background.");
            return 0f;
        }

        // 2. Identifikasi Koin vs Jagung berdasarkan BENTUK (Aspect Ratio)
        // Koin itu bulat, jadi Bounds Width & Height pasti mirip. Rasio mendekati 1.0
        // Jagung itu panjang. Rasio jauh dari 1.0

        // Urutkan objek dari yang paling mirip lingkaran (Rasio mendekati 1)
        var sortedBySquareness = objects.OrderBy(o => Mathf.Abs(1.0f - o.aspectRatio)).ToList();

        // Kandidat Koin adalah yang paling "Kotak/Bulat" DAN ukurannya masuk akal (bukan noise kecil)
        DetectedObject coin = sortedBySquareness.FirstOrDefault(o => o.pixelCount > minPixelArea && o.pixelCount < (photo.width * photo.height * 0.3f));

        if (coin == null)
        {
            Debug.LogWarning("Gagal: Tidak ada objek yang mirip koin.");
            return 0f;
        }

        // Kandidat Jagung adalah objek TERBESAR sisa selain koin
        DetectedObject corn = objects
            .Where(o => o != coin)
            .OrderByDescending(o => o.pixelCount)
            .FirstOrDefault();

        if (corn == null)
        {
            Debug.LogWarning("Gagal: Tidak ada objek jagung.");
            return 0f;
        }

        // 3. Hitung Ukuran
        // Gunakan rata-rata lebar & tinggi koin untuk mengurangi efek perspektif miring
        float coinSizePx = (coin.bounds.width + coin.bounds.height) / 2f;
        float pixelsPerCm = coinSizePx / coinRealDiameterCm;

        // Ambil sisi terpanjang jagung (diagonal bounds magnitude lebih akurat, tapi max side cukup)
        float cornSizePx = Mathf.Max(corn.bounds.width, corn.bounds.height);
        float cornRealCm = cornSizePx / pixelsPerCm;

        Debug.Log($"[SUCCESS] Coin Px: {coinSizePx} | Corn Px: {cornSizePx} | Result: {cornRealCm:F2} cm");

        return cornRealCm;
    }

    private List<DetectedObject> FindObjectsImproved(Texture2D tex)
    {
        int w = tex.width;
        int h = tex.height;
        Color32[] pixels = tex.GetPixels32();
        bool[] visited = new bool[pixels.Length];
        List<DetectedObject> foundObjects = new List<DetectedObject>();

        // Siapkan Debug Texture
        if (debugTexture != null) Destroy(debugTexture);
        debugTexture = new Texture2D(w, h);
        Color32[] debugPixels = new Color32[pixels.Length];

        // 1. Analisa Background (Sampling 4 pojok)
        Color32 bg1 = pixels[0];
        Color32 bg2 = pixels[w - 1];
        Color32 bg3 = pixels[(h - 1) * w];
        Color32 bg4 = pixels[pixels.Length - 1];

        // Ambil rata-rata warna background
        float bgR = (bg1.r + bg2.r + bg3.r + bg4.r) / 4f;
        float bgG = (bg1.g + bg2.g + bg3.g + bg4.g) / 4f;
        float bgB = (bg1.b + bg2.b + bg3.b + bg4.b) / 4f;

        // Step scan (4 pixel untuk performa)
        int step = 4;

        for (int y = step; y < h - step; y += step)
        {
            for (int x = step; x < w - step; x += step)
            {
                int idx = y * w + x;

                // Debug Visualization (Default Hitam)
                if (debugPixels[idx].a == 0) debugPixels[idx] = new Color32(0, 0, 0, 255);

                if (visited[idx]) continue;

                Color32 p = pixels[idx];

                // Cek perbedaan warna dengan background (Thresholding)
                float diff = (Mathf.Abs(p.r - bgR) + Mathf.Abs(p.g - bgG) + Mathf.Abs(p.b - bgB)) / (3f * 255f);

                if (diff > sensitivity)
                {
                    // Flood Fill
                    DetectedObject obj = FloodFill(pixels, visited, w, h, x, y, bgR, bgG, bgB, ref debugPixels);

                    // Hanya terima jika cukup besar
                    if (obj.pixelCount > minPixelArea)
                    {
                        foundObjects.Add(obj);
                    }
                }
            }
        }

        // Apply debug texture
        debugTexture.SetPixels32(debugPixels);
        debugTexture.Apply();

        return foundObjects;
    }

    private DetectedObject FloodFill(Color32[] pixels, bool[] visited, int w, int h, int startX, int startY, float bgR, float bgG, float bgB, ref Color32[] debugPixels)
    {
        int minX = startX, maxX = startX, minY = startY, maxY = startY;
        int count = 0;

        Queue<int> queue = new Queue<int>();
        queue.Enqueue(startY * w + startX);
        visited[startY * w + startX] = true;

        // Warna acak untuk debug tiap objek
        Color32 debugColor = new Color32((byte)Random.Range(50, 255), (byte)Random.Range(50, 255), (byte)Random.Range(50, 255), 255);

        while (queue.Count > 0)
        {
            int idx = queue.Dequeue();
            int cx = idx % w;
            int cy = idx / w;

            count++;

            // Visual Debugging
            debugPixels[idx] = debugColor;

            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;

            // Cek 4 arah
            int step = 2; // Lebih presisi saat fill
            int[] neighbors = { -step, step, -step * w, step * w };

            foreach (int offset in neighbors)
            {
                int nIdx = idx + offset;
                if (nIdx < 0 || nIdx >= pixels.Length) continue;

                int nx = nIdx % w;
                if (Mathf.Abs(nx - cx) > step * 2) continue; // Boundary wrap fix

                if (visited[nIdx]) continue;

                // Cek lagi threshold tetangga
                Color32 p = pixels[nIdx];
                float diff = (Mathf.Abs(p.r - bgR) + Mathf.Abs(p.g - bgG) + Mathf.Abs(p.b - bgB)) / (3f * 255f);

                if (diff > sensitivity)
                {
                    visited[nIdx] = true;
                    queue.Enqueue(nIdx);
                }
            }
        }

        float width = maxX - minX;
        float height = maxY - minY;

        return new DetectedObject
        {
            bounds = new Rect(minX, minY, width, height),
            pixelCount = count,
            aspectRatio = (height > 0) ? width / height : 0
        };
    }
}