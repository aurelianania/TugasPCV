import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_cmyk(rgb_image):
    # Normalisasi RGB ke rentang [0, 1]
    rgb_norm = rgb_image.astype(float) / 255.0
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    
    k = 1.0 - np.max(rgb_norm, axis=-1)
    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y = np.zeros_like(k)
    
    mask = k < 1.0
    c[mask] = (1.0 - r[mask] - k[mask]) / (1.0 - k[mask])
    m[mask] = (1.0 - g[mask] - k[mask]) / (1.0 - k[mask])
    y[mask] = (1.0 - b[mask] - k[mask]) / (1.0 - k[mask])
    
    cmyk = np.stack([c, m, y, k], axis=-1)
    return (cmyk * 255).astype(np.uint8)

def rgb_to_hsi(rgb_image):
    # Normalisasi RGB ke rentang [0, 1]
    rgb_norm = rgb_image.astype(float) / 255.0
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    
    # Intensity (I)
    i = np.mean(rgb_norm, axis=-1)
    
    # Saturation (S)
    min_rgb = np.min(rgb_norm, axis=-1)
    sum_rgb = np.sum(rgb_norm, axis=-1)
    
    s = np.zeros_like(i)
    mask = sum_rgb > 0
    s[mask] = 1.0 - (3.0 * min_rgb[mask] / sum_rgb[mask])
    
    # Hue (H)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b)*(g - b))
    
    theta = np.arccos(np.clip(num / (den + 1e-6), -1.0, 1.0))
    
    h = np.zeros_like(i)
    h[b > g] = 2 * np.pi - theta[b > g]
    h[b <= g] = theta[b <= g]
    h = h / (2 * np.pi)  # Normalisasi ke [0, 1]
    
    hsi = np.stack([h * 255, s * 255, i * 255], axis=-1)
    return hsi.astype(np.uint8)

def process_image(image_path):
    # Load gambar menggunakan OpenCV (Format BGR)
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print(f"Error: Gambar tidak ditemukan pada path {image_path}. Pastikan 'catpic.jpg' berada di folder yang sama!")
        return
    
    # Konversi BGR ke RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # 1. Konversi RGB ke HSV (bawaan OpenCV)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    
    # 2. Konversi RGB ke CMYK
    cmyk_img = rgb_to_cmyk(rgb_img)
    
    # 3. Konversi RGB ke HSI
    hsi_img = rgb_to_hsi(rgb_img)
    
    # Visualisasi hasil menggunakan Matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("1. Original RGB")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
    axes[0, 1].set_title("2. HSV Color Space")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cmyk_img[..., 0], cmap='gray')
    axes[1, 0].set_title("3. CMYK (Cyan Channel)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hsi_img)
    axes[1, 1].set_title("4. HSI Color Space")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('color_conversion_output.png')
    print("Berhasil! Hasil konversi disimpan sebagai 'color_conversion_output.png'.")
    plt.show()

if __name__ == '__main__':
    process_image('catpic.jpg')