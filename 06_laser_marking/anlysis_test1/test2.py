import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# フォルダパス
folder_path = './data'
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

gray_images = []
for fname in image_files:
    path = os.path.join(folder_path, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        gray_images.append(img)

if len(gray_images) == 0:
    raise ValueError("No images found")

# サイズ統一（最初の画像サイズに合わせる）
h, w = gray_images[0].shape
gray_images_resized = [cv2.resize(img, (w, h)) for img in gray_images]

# 各画像ごとにROI選択 → マスク生成 → 正規化 → カラーマップ
masked_images = []
normalized_images = []
color_mapped_images = []

for i, img in enumerate(gray_images_resized):
    # === ROI選択 ===
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Image {i+1}: Select ROI (click 4 points, then Enter)")
    pts = plt.ginput(4, timeout=0)
    plt.close()

    # === マスク生成 ===
    pts = np.array(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    path = Path(pts)
    inside = path.contains_points(coords).reshape(h, w)
    mask[inside] = 1

    # === ROI抽出 ===
    masked_img = np.where(mask == 1, img, 0)
    masked_images.append(masked_img)

    # === 正規化 ===
    roi_pixels = masked_img[mask == 1]
    roi_min = np.min(roi_pixels)
    roi_max = np.max(roi_pixels)
    img_float = masked_img.astype(np.float32)
    norm_img = np.zeros_like(img_float)
    norm_img[mask == 1] = (img_float[mask == 1] - roi_min) / (roi_max - roi_min) * 255
    norm_img = np.round(norm_img).astype(np.uint8)
    normalized_images.append(norm_img)

    # === カラーマップ ===
    color_map = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    color_mapped_images.append(color_map)

# === 各段を横に並べる ===
top_row = np.hstack(gray_images_resized)
middle_row = np.hstack(normalized_images)
bottom_row = np.hstack(color_mapped_images)
top_row_color = cv2.cvtColor(top_row, cv2.COLOR_GRAY2BGR)
middle_row_color = cv2.cvtColor(middle_row, cv2.COLOR_GRAY2BGR)
final_image_bgr = np.vstack([top_row_color, middle_row_color, bottom_row])
final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)

# === 画像表示 ===
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(final_image_rgb)
ax.axis('off')
ax.set_title("Top: Original | Middle: ROI-Normalized | Bottom: Jet-Colormap", fontsize=14)
norm = Normalize(vmin=0, vmax=255)
sm = ScalarMappable(cmap='jet', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8, pad=0.01)
cbar.set_label('Pixel Intensity (ROI normalized)', fontsize=12)
plt.tight_layout()
plt.show()

# === ヒストグラム（すべてのROIピクセル） ===
all_norm_pixels = np.concatenate([img[img > 0].flatten() for img in normalized_images])
plt.figure(figsize=(8, 4))
plt.hist(all_norm_pixels, bins=256, range=(0, 256), color='gray', edgecolor='black')
plt.title("Histogram of ROI-Normalized Pixels (All Images)")
plt.xlabel("Pixel Value (0–255)")
plt.ylabel("Pixel Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === 保存先フォルダの作成 ===
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# === 正規化画像を保存 ===
for fname, norm_img in zip(image_files, normalized_images):
    base_name = os.path.splitext(fname)[0]
    output_path = os.path.join(output_dir, f"{base_name}_rescaled.jpg")
    cv2.imwrite(output_path, norm_img)
    print(f"Saved: {output_path}")