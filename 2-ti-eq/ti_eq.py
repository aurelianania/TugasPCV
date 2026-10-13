import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_grayscale(bgr_img):
    # Konversi BGR ke Grayscale secara manual menggunakan bobot standar: 0.299*R + 0.587*G + 0.114*B
    b = bgr_img[:, :, 0].astype(float)
    g = bgr_img[:, :, 1].astype(float)
    r = bgr_img[:, :, 2].astype(float)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def manual_histogram(gray_img):
    # Menghitung frekuensi kemunculan setiap nilai intensitas (0-255) secara manual
    hist = np.zeros(256, dtype=int)
    for pixel in gray_img.ravel():
        hist[pixel] += 1
    return hist

def manual_histogram_equalization(gray_img):
    # 1. Hitung histogram
    hist = manual_histogram(gray_img)
    
    # 2. Hitung Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()
    
    # 3. Normalisasi CDF ke rentang 0 - 255 (Rumus Ekualisasi Histogram)
    cdf_min = cdf[cdf > 0].min() # Ambil nilai CDF non-nol pertama
    total_pixels = gray_img.size
    
    # Formula: H_eq(v) = round((cdf(v) - cdf_min) / (M*N - cdf_min) * 255)
    cdf_normalized = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)
    
    # 4. Petakan nilai piksel lama ke nilai piksel baru berdasarkan CDF ternormalisasi
    equalized_img = cdf_normalized[gray_img]
    return equalized_img, hist, cdf_normalized

def manual_intensity_transform(gray_img, gamma=1.5):
    # 1. Negatif Citra: s = L - 1 - r (dimana L = 256)
    negative_img = 255 - gray_img
    
    # 2. Transformasi Power-Law / Gamma Correction: s = c * r^gamma
    normalized = gray_img.astype(float) / 255.0
    gamma_img = np.array(255 * (normalized ** gamma), dtype=np.uint8)
    
    return negative_img, gamma_img

def process_image(image_path):
    # Load gambar
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print(f"Error: Gambar '{image_path}' tidak ditemukan! Pastikan file foto ada di folder yang sama.")
        return
    
    # Ubah ke Grayscale secara manual
    gray_img = manual_grayscale(bgr_img)
    
    # Terapkan Transformasi Intensitas (Negatif & Gamma Correction) secara manual
    neg_img, gamma_img = manual_intensity_transform(gray_img, gamma=1.8)
    
    # Terapkan Ekualisasi Histogram secara manual (tanpa cv2.equalizeHist)
    eq_img, hist_orig, cdf_mapping = manual_histogram_equalization(gray_img)
    hist_eq = manual_histogram(eq_img)
    
    # Visualisasi Hasil
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    # Baris 1: Grayscale & Negatif Citra
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title("1. Original Grayscale")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(neg_img, cmap='gray')
    axes[0, 1].set_title("2. Transformasi Negatif Citra")
    axes[0, 1].axis('off')
    
    # Baris 2: Gamma Correction & Hasil Ekualisasi Histogram
    axes[1, 0].imshow(gamma_img, cmap='gray')
    axes[1, 0].set_title("3. Transformasi Gamma (Gamma=1.8)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(eq_img, cmap='gray')
    axes[1, 1].set_title("4. Ekualisasi Histogram (Manual)")
    axes[1, 1].axis('off')
    
    # Baris 3: Grafik Perbandingan Histogram Asli vs Ekualisasi
    axes[2, 0].plot(hist_orig, color='black')
    axes[2, 0].set_title("Histogram Asli")
    axes[2, 0].set_xlim([0, 256])
    
    axes[2, 1].plot(hist_eq, color='blue')
    axes[2, 1].set_title("Histogram Setelah Ekualisasi")
    axes[2, 1].set_xlim([0, 256])
    
    plt.tight_layout()
    output_filename = 'intensity_equalization_output.png'
    plt.savefig(output_filename)
    print(f"Berhasil! Hasil disimpan sebagai '{output_filename}'.")
    plt.show()

if __name__ == '__main__':
    process_image('catpic.jpg')