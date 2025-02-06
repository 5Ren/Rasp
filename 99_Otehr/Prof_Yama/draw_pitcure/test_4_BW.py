import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

import os

# 画像の読み込み
image_path = "image_files/25-1-27_cosun_black.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("画像の読み込みに失敗しました。")
    exit()

# ファイル名の最後のアンダーバー以降を取得
filename = os.path.basename(image_path)
file_suffix = filename.split("_")[-1].split(".")[0]  # 拡張子を除いた最後の部分

# ガウシアンフィルターの強度を設定（ファイル名に応じて変更）
if file_suffix.lower() == "black":
    gaussian_kernel_size = (9, 9)
    gaussian_sigma = 1.0
elif file_suffix.lower() == "white":
    gaussian_kernel_size = (5, 5)
    gaussian_sigma = 1.0
else:
    gaussian_kernel_size = (7, 7)  # デフォルト
    gaussian_sigma = 1.0

# 画像を縦横２倍にリサイズ（補間方法を変更）
image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# ガウシアンフィルターを適用
image_blurred = cv2.GaussianBlur(image_resized, gaussian_kernel_size, gaussian_sigma)

# 外側10画素分を消去（輪郭を削る）
image_blurred[:10, :] = 255  # 上部
image_blurred[-10:, :] = 255  # 下部
image_blurred[:, :10] = 255  # 左部
image_blurred[:, -10:] = 255  # 右部

# 二値化処理
_, binary_image = cv2.threshold(image_blurred, 127, 255, cv2.THRESH_BINARY)

# 元のピクセル単位の黒い部分の座標を取得
y_coords, x_coords = np.where(binary_image == 0)

# 縦の長さ（μm）を指定
y_length_microns = 10000  # 例として 10000μm
resolution_micron_per_pix = 10  # 例として 10μm/pix

# 新しい画像サイズを計算
y_new_size = int(y_length_microns / resolution_micron_per_pix)
x_new_size = int((y_new_size / binary_image.shape[0]) * binary_image.shape[1])

# 画像をリサイズして μm スケールに変換
image_scaled = cv2.resize(binary_image, (x_new_size, y_new_size), interpolation=cv2.INTER_NEAREST)

# μm スケールの黒い部分の座標を取得
y_coords_scaled, x_coords_scaled = np.where(image_scaled == 0)

# 画像の中心を原点 (0, 0) にする
x_centered = x_coords_scaled - x_new_size // 2
y_centered = -(y_coords_scaled - y_new_size // 2)  # Y軸の向きを反転

# エッジ検出のための始点・終点リスト
edge_segments = []
previous_color = None
for i in range(y_new_size):
    row = image_scaled[i, :]

    if i % 2 == 0:
        indices = range(len(row))  # 左→右
    else:
        indices = reversed(range(len(row)))  # 右→左

    prev_pixel = 255  # 初期状態は白
    for j in indices:
        if prev_pixel == 255 and row[j] == 0:
            start = (j - x_new_size // 2, -(i - y_new_size // 2))
        elif prev_pixel == 0 and row[j] == 255:
            end = (j - x_new_size // 2, -(i - y_new_size // 2))

            # ランダムな色を生成（前の色と異なるように）
            new_color = (random.random(), random.random(), random.random())
            while new_color == previous_color:
                new_color = (random.random(), random.random(), random.random())
            previous_color = new_color

            edge_segments.append((start, end, new_color))
        prev_pixel = row[j]

# 図を並べて表示
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

# 元のピクセル単位のプロット
axes[0].scatter(x_coords, -y_coords, s=2, c='black')
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
axes[0].set_title("Black Pixels Coordinates (Original)")
axes[0].set_xlim(0, binary_image.shape[1])
axes[0].set_ylim(-binary_image.shape[0], 0)
axes[0].set_aspect('equal')

# μm単位のプロット（中心を原点）
axes[1].scatter(x_centered * resolution_micron_per_pix, y_centered * resolution_micron_per_pix, s=2, c='black')
axes[1].set_xlabel("X (μm)")
axes[1].set_ylabel("Y (μm)")
axes[1].set_title("Black Pixels Coordinates (Centered at Origin)")
axes[1].set_xlim(-x_new_size * resolution_micron_per_pix // 2, x_new_size * resolution_micron_per_pix // 2)
axes[1].set_ylim(-y_new_size * resolution_micron_per_pix // 2, y_new_size * resolution_micron_per_pix // 2)
axes[1].set_aspect('equal')

# エッジ検出のプロット（異なる色を適用）
for start, end, color in edge_segments:
    axes[2].plot([start[0] * resolution_micron_per_pix, end[0] * resolution_micron_per_pix],
                 [start[1] * resolution_micron_per_pix, end[1] * resolution_micron_per_pix], c=color)
axes[2].set_xlabel("X (μm)")
axes[2].set_ylabel("Y (μm)")
axes[2].set_title("Edge Detection (Start & End Points with Random Colors)")
axes[2].set_xlim(-x_new_size * resolution_micron_per_pix // 2, x_new_size * resolution_micron_per_pix // 2)
axes[2].set_ylim(-y_new_size * resolution_micron_per_pix // 2, y_new_size * resolution_micron_per_pix // 2)
axes[2].set_aspect('equal')

plt.show()
