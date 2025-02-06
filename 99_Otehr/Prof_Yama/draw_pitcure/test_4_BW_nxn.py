import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os


def perform_edge_detection(region_image, side_length):
    """
    各行を左から右（偶数行）または右から左（奇数行）に走査し、
    白から黒への遷移を始点、黒から白への遷移を終点としてエッジセグメントを検出します。
    始点と終点の座標は領域の中心を原点とした座標系で返します。
    """
    edge_segments = []
    height, width = region_image.shape

    for i in range(height):
        if i % 2 == 0:
            indices = range(width)  # 左→右
        else:
            indices = reversed(range(width))  # 右→左

        prev_pixel = 255  # 初期状態は白
        start = None

        for j in indices:
            current_pixel = region_image[i, j]
            if prev_pixel == 255 and current_pixel == 0:
                start = (j - side_length // 2, -(i - side_length // 2))
            elif prev_pixel == 0 and current_pixel == 255 and start is not None:
                end = (j - side_length // 2, -(i - side_length // 2))
                edge_segments.append((start, end))
                start = None
            prev_pixel = current_pixel

    return edge_segments


# -------------------------------------------------
# 画像の読み込み、前処理、および二値化
# -------------------------------------------------
image_path = "image_files/25-1-27_cosun_white.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("画像の読み込みに失敗しました。")
    exit()

# ファイル名からサフィックスを取得し、ガウシアンフィルターのパラメータを設定
filename = os.path.basename(image_path)
file_suffix = filename.split("_")[-1].split(".")[0]
if file_suffix.lower() == "black":
    gaussian_kernel_size = (9, 9)
    gaussian_sigma = 1.0
elif file_suffix.lower() == "white":
    gaussian_kernel_size = (5, 5)
    gaussian_sigma = 1.0
else:
    gaussian_kernel_size = (7, 7)
    gaussian_sigma = 1.0

# 画像リサイズ（縦横2倍）およびガウシアンフィルター適用
image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
image_blurred = cv2.GaussianBlur(image_resized, gaussian_kernel_size, gaussian_sigma)

# 外側10画素を白でパディング
image_blurred[:10, :] = 255
image_blurred[-10:, :] = 255
image_blurred[:, :10] = 255
image_blurred[:, -10:] = 255

# 二値化
_, binary_image = cv2.threshold(image_blurred, 127, 255, cv2.THRESH_BINARY)

# -------------------------------------------------
# ★ 新しいスケーリング・パディング処理 ★
# -------------------------------------------------
# 加工する領域のパラメータ（最終的に処理する領域の大きさ）
processing_area_height_microns = 30000  # 加工する領域の高さ（μm）
resolution_micron_per_pix = 10  # 解像度 (μm/pix)

# 加工する領域（正方形）のピクセルサイズ
processing_area_height_pix = int(processing_area_height_microns / resolution_micron_per_pix)
processing_area_width_pix = processing_area_height_pix  # 正方形

# 元画像を引き伸ばすための目標の高さ（μm）
target_stretch_height_microns = 20000  # 例: 20000 μm

# 対応するピクセル数を計算
target_height_pix = int(target_stretch_height_microns / resolution_micron_per_pix)
aspect_ratio = binary_image.shape[1] / binary_image.shape[0]
target_width_pix = int(target_height_pix * aspect_ratio)

# 元画像を引き伸ばす（アスペクト比を保ったままリサイズ）
image_stretched = cv2.resize(binary_image, (target_width_pix, target_height_pix), interpolation=cv2.INTER_NEAREST)

# 加工領域と引き伸ばした画像のサイズ差を計算し、不足分を白 (255) でパディングする
delta_w = processing_area_width_pix - target_width_pix
delta_h = processing_area_height_pix - target_height_pix

top = delta_h // 2 if delta_h > 0 else 0
bottom = delta_h - top if delta_h > 0 else 0
left = delta_w // 2 if delta_w > 0 else 0
right = delta_w - left if delta_w > 0 else 0

image_final = cv2.copyMakeBorder(image_stretched, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=255)
print(f"Final processed image size: {image_final.shape[1]} x {image_final.shape[0]} pixels")

# -------------------------------------------------
# 画像の分割（縦横 split_num 分割）
# -------------------------------------------------
split_num = 3  # 元画像全体は 3×3 分割
side_length_y = image_final.shape[0] // split_num
side_length_x = image_final.shape[1] // split_num
side_length = min(side_length_y, side_length_x)
print(f"One side length of each region: {side_length} pixels")

regions = {}
for row in range(split_num):
    for col in range(split_num):
        region_name = f"Region {row + 1}-{col + 1}"
        y_start = row * side_length
        y_end = y_start + side_length
        x_start = col * side_length
        x_end = x_start + side_length
        regions[region_name] = image_final[y_start:y_end, x_start:x_end]

# 表示する領域を "Region 1-1", "Region 1-2", "Region 2-1", "Region 2-2" に変更
selected_regions = {
    'Region 1-1': regions["Region 1-1"],
    'Region 1-2': regions["Region 1-2"],
    'Region 2-1': regions["Region 2-1"],
    'Region 2-2': regions["Region 2-2"]
}
print(f"Selected regions for plotting: {list(selected_regions.keys())}")

# 元画像（リサイズ前）の黒ピクセル座標
y_coords_orig, x_coords_orig = np.where(binary_image == 0)

# 加工後画像（image_final）の黒ピクセル中心シフト座標
y_coords_final, x_coords_final = np.where(image_final == 0)
x_centered = x_coords_final - image_final.shape[1] // 2
y_centered = -(y_coords_final - image_final.shape[0] // 2)

# -------------------------------------------------
# Figure 1: 上段（元画像と加工後画像の表示）
# -------------------------------------------------
fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

# 左: 元画像（リサイズ前）の黒ピクセル
axs1[0].scatter(x_coords_orig, -y_coords_orig, s=2, c='black')
axs1[0].set_xlabel("X (pixels)")
axs1[0].set_ylabel("Y (pixels)")
axs1[0].set_title("Black Pixels (Original)")
axs1[0].set_xlim(0, binary_image.shape[1])
axs1[0].set_ylim(-binary_image.shape[0], 0)
axs1[0].set_aspect('equal')

# 右: 加工後画像の中心シフト黒ピクセル（単位: μm）
axs1[1].scatter(x_centered * resolution_micron_per_pix,
                y_centered * resolution_micron_per_pix, s=2, c='black')
axs1[1].set_xlabel("X (μm)")
axs1[1].set_ylabel("Y (μm)")
axs1[1].set_title("Black Pixels (Processed)")
axs1[1].set_xlim(-image_final.shape[1] * resolution_micron_per_pix // 2,
                 image_final.shape[1] * resolution_micron_per_pix // 2)
axs1[1].set_ylim(-image_final.shape[0] * resolution_micron_per_pix // 2,
                 image_final.shape[0] * resolution_micron_per_pix // 2)
axs1[1].set_aspect('equal')

fig1.tight_layout()

# -------------------------------------------------
# Figure 2: 下段（選択した 4 領域の各種プロット）
# それぞれ、各領域について 3 種類のプロットを 3 行×4 列のグリッドで表示
# 1行目: 黒ピクセル, 2行目: 中心座標, 3行目: エッジ検出結果
# -------------------------------------------------
fig2, axs2 = plt.subplots(3, 4, figsize=(24, 18), dpi=150)

# 選択領域の順序（左上から右下）
region_order = ["Region 1-1", "Region 1-2", "Region 2-1", "Region 2-2"]

for col, region_name in enumerate(region_order):
    region_data = selected_regions[region_name]
    # (1) 黒ピクセルプロット
    y_reg, x_reg = np.where(region_data == 0)
    axs2[0, col].scatter(x_reg * resolution_micron_per_pix,
                         -y_reg * resolution_micron_per_pix, s=2, c='black')
    axs2[0, col].set_title(f"Black Pixels\n{region_name}")
    axs2[0, col].set_xlim(0, side_length * resolution_micron_per_pix)
    axs2[0, col].set_ylim(-side_length * resolution_micron_per_pix, 0)
    axs2[0, col].set_aspect('equal')

    # (2) 中心座標プロット
    x_reg_centered = x_reg - side_length // 2
    y_reg_centered = -(y_reg - side_length // 2)
    axs2[1, col].scatter(x_reg_centered * resolution_micron_per_pix,
                         y_reg_centered * resolution_micron_per_pix, s=2, c='black')
    axs2[1, col].set_title(f"Centered\n{region_name}")
    axs2[1, col].set_xlim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    axs2[1, col].set_ylim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    axs2[1, col].set_aspect('equal')

    # (3) エッジ検出プロット
    edges = perform_edge_detection(region_data, side_length)
    for start, end in edges:
        axs2[2, col].plot([start[0] * resolution_micron_per_pix, end[0] * resolution_micron_per_pix],
                          [start[1] * resolution_micron_per_pix, end[1] * resolution_micron_per_pix],
                          c=(random.random(), random.random(), random.random()), linewidth=0.5)
    axs2[2, col].set_title(f"Edge Detection\n{region_name}")
    axs2[2, col].set_xlim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    axs2[2, col].set_ylim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    axs2[2, col].set_aspect('equal')

fig2.tight_layout()

plt.show()
