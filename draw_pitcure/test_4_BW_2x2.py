import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

import os

# 画像の読み込み
image_path = "./image_files/25-1-27_cosun_white.jpg"

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
y_length_microns = 20000  # 20000μm
resolution_micron_per_pix = 10  # 10μm/pix

# 新しい画像サイズを計算
y_new_size = int(y_length_microns / resolution_micron_per_pix)
x_new_size = int((y_new_size / binary_image.shape[0]) * binary_image.shape[1])

# 画像をリサイズして μm スケールに変換
image_scaled = cv2.resize(binary_image, (x_new_size, y_new_size), interpolation=cv2.INTER_NEAREST)

# μm スケールの黒い部分の座標を取得
y_coords_scaled, x_coords_scaled = np.where(image_scaled == 0)

# 画像を4つの領域に分割
mid_x, mid_y = x_new_size // 2, y_new_size // 2
quadrants = {
    "Top Left": image_scaled[:mid_y, :mid_x],
    "Top Right": image_scaled[:mid_y, mid_x:],
    "Bottom Left": image_scaled[mid_y:, :mid_x],
    "Bottom Right": image_scaled[mid_y:, mid_x:]
}

# 各領域ごとにエッジ検出（それぞれの中心を原点とする）
quadrant_segments = {}
for name, region in quadrants.items():
    segments = []
    q_x_offset = mid_x if "Right" in name else 0
    q_y_offset = mid_y if "Bottom" in name else 0
    center_x, center_y = region.shape[1] // 2, region.shape[0] // 2  # ローカル原点

    for i in range(region.shape[0]):
        row = region[i, :]
        prev_pixel = 255
        for j in range(region.shape[1]):
            if prev_pixel == 255 and row[j] == 0:
                start = (j - center_x, (center_y - i))  # 各領域の局所座標
            elif prev_pixel == 0 and row[j] == 255:
                end = (j - center_x, (center_y - i))
                # グローバル座標に変換し、適切な分割
                global_start_x = start[0] + q_x_offset - mid_x
                global_start_y = start[1] + q_y_offset - mid_y
                global_end_x = end[0] + q_x_offset - mid_x
                global_end_y = end[1] + q_y_offset - mid_y

                # 境界をまたぐ場合の処理
                if global_start_x * global_end_x < 0:  # x = 0 をまたぐ場合
                    split_x = 0
                    t = (split_x - global_start_x) / (global_end_x - global_start_x)
                    split_y = global_start_y + t * (global_end_y - global_start_y)
                    segments.append(((global_start_x, global_start_y), (split_x, split_y)))
                    segments.append(((split_x, split_y), (global_end_x, global_end_y)))
                elif global_start_y * global_end_y < 0:  # y = 0 をまたぐ場合
                    split_y = 0
                    t = (split_y - global_start_y) / (global_end_y - global_start_y)
                    split_x = global_start_x + t * (global_end_x - global_start_x)
                    segments.append(((global_start_x, global_start_y), (split_x, split_y)))
                    segments.append(((split_x, split_y), (global_end_x, global_end_y)))
                else:
                    segments.append(((global_start_x, global_start_y), (global_end_x, global_end_y)))

            prev_pixel = row[j]
    quadrant_segments[name] = segments

# 図を並べて表示
fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69), dpi=150)  # A4サイズ

# 元のピクセル単位のプロット
axes[0, 0].scatter(x_coords, -y_coords, s=2, c='black')
axes[0, 0].set_title("Black Pixels Coordinates (Original)")
axes[0, 0].set_aspect('equal')

# μm単位のプロット（全体）
axes[0, 1].scatter(x_coords_scaled - mid_x, -(y_coords_scaled - mid_y), s=2, c='black')
axes[0, 1].set_title("Black Pixels Coordinates (Centered at Origin)")
axes[0, 1].set_aspect('equal')

# 各領域のエッジ検出結果をプロット
for ax, (name, segments) in zip(axes[1:].flatten(), quadrant_segments.items()):
    for start, end in segments:
        ax.plot([start[0] * resolution_micron_per_pix, end[0] * resolution_micron_per_pix],
                [start[1] * resolution_micron_per_pix, end[1] * resolution_micron_per_pix], c=random.choice(['b', 'g', 'r', 'c', 'm', 'y']))
    ax.set_title(f"Edge Detection in {name}")
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    ax.set_aspect('equal')

plt.show()