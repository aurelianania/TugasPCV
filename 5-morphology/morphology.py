import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_morphology(image_path):
    # Load gambar dalam bentuk grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Gambar tidak ditemukan pada path {image_path}. Pastikan 'catpic.jpg' berada di folder yang sama!")
        return
    
    # Binarisasi gambar menggunakan thresholding agar struktur objek/tepi objek terlihat jelas
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Buat kernel / structuring element (matriks 5x5)
    kernel = np.ones((5, 5), np.uint8)
    
    # 1. Erosi (Erosion): Memperkecil/mengikis area objek putih
    erosion = cv2.erode(thresh_img, kernel, iterations=1)
    
    # 2. Dilasi (Dilation): Memperbesar/menebalkan area objek putih
    dilation = cv2.dilate(thresh_img, kernel, iterations=1)
    
    # 3. Opening (Erosi dilanjut Dilasi): Menghilangkan noise/titik putih kecil di luar objek
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    
    # 4. Closing (Dilasi dilanjut Erosi): Menutup lubang hitam kecil di dalam objek
    closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    
    # Visualisasi hasil menggunakan Matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(thresh_img, cmap='gray')
    axes[0, 1].set_title("Binary Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(erosion, cmap='gray')
    axes[0, 2].set_title("1. Erosi (Erosion)")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(dilation, cmap='gray')
    axes[1, 0].set_title("2. Dilasi (Dilation)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(opening, cmap='gray')
    axes[1, 1].set_title("3. Opening")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(closing, cmap='gray')
    axes[1, 2].set_title("4. Closing")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_filename = 'morphology_output.png'
    plt.savefig(output_filename)
    print(f"Berhasil! Hasil operasi morfologi disimpan sebagai '{output_filename}'.")
    plt.show()

if __name__ == '__main__':
    process_morphology('catpic.jpg')