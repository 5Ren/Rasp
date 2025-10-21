import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.path import Path
from matplotlib.patches import Polygon

# === フォルダの指定 ===
folder_path = './data1'
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

if len(image_files) == 0:
    raise ValueError("No .jpg images found in the folder.")

# === 画像ごとに処理 ===
for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # === 表示サイズ拡大（縦横ともに1.8倍相当）===
    scale = 1.2
    fig_w = (w / 100) * scale
    fig_h = (h / 100) * scale

    # === ROI選択 ===
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"{image_file}\nClick 4 points × 17 times (Enter after each 4 points)")

    rois = []
    centers = []

    for idx in range(17):
        pts = plt.ginput(4, timeout=0)
        pts = np.array(pts)
        rois.append(pts)

        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        centers.append((cx, cy))
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor='lime', linewidth=1.5))
        ax.text(cx, cy, str(idx), color='red', fontsize=12, ha='center', va='center')

    plt.title("Selected ROIs with indices")
    plt.show()

    # === ROIごとの輝度取得 ===
    results = []

    for idx, pts in enumerate(rois):
        mask = np.zeros((h, w), dtype=np.uint8)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)
        path = Path(pts)
        inside = path.contains_points(coords).reshape(h, w)
        mask[inside] = 1

        yx_coords = np.column_stack(np.where(mask == 1))
        if len(yx_coords) < 10:
            print(f"Warning: ROI {idx} has less than 10 pixels.")
            continue

        sampled_indices = np.random.choice(len(yx_coords), size=10, replace=False)
        sampled_coords = yx_coords[sampled_indices]
        values = [img[y, x] for y, x in sampled_coords]
        mean_val = np.mean(values)
        std_val = np.std(values)

        results.append([idx] + values + [round(mean_val, 2), round(std_val, 2)])

    # === 保存ファイル名作成 ===
    base_name = os.path.splitext(image_file)[0]
    output_csv = os.path.join(folder_path, f"{base_name}_counted.csv")

    header = ['index'] + [f'val{i+1}' for i in range(10)] + ['mean', 'std']

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"保存されました: {output_csv}")
