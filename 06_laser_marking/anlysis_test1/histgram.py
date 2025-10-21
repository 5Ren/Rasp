import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# 画像フォルダのパスを指定
folder_path = './data'

# jpg画像を取得（名前順に）
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

# グレースケール画像を読み込む
gray_images = []
for fname in image_files:
    path = os.path.join(folder_path, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        gray_images.append(img)

if len(gray_images) == 0:
    raise ValueError("画像が見つかりません")

# 全体の最小・最大輝度
all_pixels = np.concatenate([img.flatten() for img in gray_images_resized])
global_min = np.min(all_pixels)
global_max = np.max(all_pixels)

# 正規化処理（0〜255へ）
normalized_images = [
    ((img - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    for img in gray_images_resized
]


# 正規化後のすべての画素値を1D配列にしてヒストグラム表示
all_norm_pixels = np.concatenate([img.flatten() for img in normalized_images])

plt.figure(figsize=(8, 4))
plt.hist(all_norm_pixels, bins=256, range=(0, 255), color='gray', edgecolor='black')
plt.title("Histogram of Normalized Pixel Intensities")
plt.xlabel("Pixel Value (0–255)")
plt.ylabel("Pixel Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
