import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

# サイズ統一（最初の画像に合わせる）
h, w = gray_images[0].shape
gray_images_resized = [cv2.resize(img, (w, h)) for img in gray_images]

## (1) グローバルmin/max取得（元画像）
all_pixels = np.concatenate([img.flatten() for img in gray_images_resized])
global_min = np.min(all_pixels)
global_max = np.max(all_pixels)  # ← 193

# (2) 正規化 (0〜255にスケーリング)
normalized_images = [
    np.round((img.astype(np.float32) - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    for img in gray_images_resized
]

# (3) カラーマップ (Jet)
color_mapped_images = [
    cv2.applyColorMap(img, cv2.COLORMAP_JET) for img in normalized_images
]

# 各段を横に連結
top_row = np.hstack(gray_images_resized)
middle_row = np.hstack(normalized_images)
bottom_row = np.hstack(color_mapped_images)

# グレースケール画像をカラーに変換（OpenCVではBGR）
top_row_color = cv2.cvtColor(top_row, cv2.COLOR_GRAY2BGR)
middle_row_color = cv2.cvtColor(middle_row, cv2.COLOR_GRAY2BGR)

# 3段を縦に連結（カラー3ch）
final_image_bgr = np.vstack([top_row_color, middle_row_color, bottom_row])
final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)  # matplotlib用にRGBへ

# 図を表示（カラーバーつき）
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(final_image_rgb)
ax.axis('off')
ax.set_title("Top: Original | Middle: Normalized | Bottom: Jet Colormap", fontsize=14)

# カラーバー（Jet, 0〜255）
norm = Normalize(vmin=0, vmax=255)
sm = ScalarMappable(cmap='jet', norm=norm)
sm.set_array([])  # 必要（matplotlibの仕様）
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8, pad=0.01)
cbar.set_label('Pixel Intensity (after normalization)', fontsize=12)

plt.tight_layout()
plt.show()

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

# 全画像での最大値確認
for i, img in enumerate(gray_images_resized):
    print(f"Image {i}: max = {img.max()}, min = {img.min()}")

# まとめて比較
print(f"\nGlobal min = {global_min}, Global max = {global_max}")

for i, img in enumerate(normalized_images):
    print(f"Normalized Image {i}: max = {img.max()}, min = {img.min()}")