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
y_length_microns = 20000  # 10000μmから20000μmに変更
resolution_micron_per_pix = 10  # 例として 10μm/pix

# 新しい画像サイズを計算
y_new_size = int(y_length_microns / resolution_micron_per_pix)
aspect_ratio = binary_image.shape[1] / binary_image.shape[0]
x_new_size = int(y_new_size * aspect_ratio)

# 画像をリサイズして μm スケールに変換
image_scaled = cv2.resize(binary_image, (x_new_size, y_new_size), interpolation=cv2.INTER_NEAREST)

# 正方形にするためにパディングを追加
if x_new_size < y_new_size:
    # 横方向が短い場合、左右に白を追加
    padding_total = y_new_size - x_new_size
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left
    image_scaled_padded = cv2.copyMakeBorder(image_scaled, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT,
                                             value=255)
    scaled_width = y_new_size
    scaled_height = y_new_size
    print(f"Padding added: left={padding_left} pix, right={padding_right} pix")
elif x_new_size > y_new_size:
    # 縦方向が短い場合、上下に白を追加
    padding_total = x_new_size - y_new_size
    padding_top = padding_total // 2
    padding_bottom = padding_total - padding_top
    image_scaled_padded = cv2.copyMakeBorder(image_scaled, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT,
                                             value=255)
    scaled_width = x_new_size
    scaled_height = x_new_size
    print(f"Padding added: top={padding_top} pix, bottom={padding_bottom} pix")
else:
    # 既に正方形の場合
    image_scaled_padded = image_scaled.copy()
    scaled_width, scaled_height = image_scaled_padded.shape[1], image_scaled_padded.shape[0]
    print("No padding needed; image is already square.")

print(f"Scaled image size: {scaled_width} x {scaled_height} pixels")

# μm スケールの黒い部分の座標を取得
y_coords_scaled, x_coords_scaled = np.where(image_scaled_padded == 0)

# 画像の中心を原点 (0, 0) にする
x_centered = x_coords_scaled - scaled_width // 2
y_centered = -(y_coords_scaled - scaled_height // 2)  # Y軸の向きを反転

# 図を並べて表示（3段目を追加）
fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=150)

# 1段目: 元のピクセル単位のプロット
axes[0, 0].scatter(x_coords, -y_coords, s=2, c='black')
axes[0, 0].set_xlabel("X (pixels)")
axes[0, 0].set_ylabel("Y (pixels)")
axes[0, 0].set_title("Black Pixels Coordinates (Original)")
axes[0, 0].set_xlim(0, binary_image.shape[1])
axes[0, 0].set_ylim(-binary_image.shape[0], 0)
axes[0, 0].set_aspect('equal')

# 1段目: μm単位のプロット（中心を原点）
axes[0, 1].scatter(x_centered * resolution_micron_per_pix, y_centered * resolution_micron_per_pix, s=2, c='black')
axes[0, 1].set_xlabel("X (μm)")
axes[0, 1].set_ylabel("Y (μm)")
axes[0, 1].set_title("Black Pixels Coordinates (Scaled)")
axes[0, 1].set_xlim(-scaled_width * resolution_micron_per_pix // 2, scaled_width * resolution_micron_per_pix // 2)
axes[0, 1].set_ylim(-scaled_height * resolution_micron_per_pix // 2, scaled_height * resolution_micron_per_pix // 2)
axes[0, 1].set_aspect('equal')

# 1段目の残りの2つのサブプロットを非表示にする
axes[0, 2].axis('off')
axes[0, 3].axis('off')

# リスケール後の画像を4領域に分割
half_y = scaled_height // 2
half_x = scaled_width // 2

# 正方形の一辺の長さ
side_length = min(half_y, half_x)
print(f"One side length of each quadrant: {side_length} pixels")

# 各領域の座標を取得（正方形を維持）
regions = {
    'Top Left': image_scaled_padded[0:side_length, 0:side_length],
    'Top Right': image_scaled_padded[0:side_length, scaled_width - side_length:scaled_width],
    'Bottom Left': image_scaled_padded[scaled_height - side_length:scaled_height, 0:side_length],
    'Bottom Right': image_scaled_padded[scaled_height - side_length:scaled_height,
                    scaled_width - side_length:scaled_width]
}

# 各領域をプロット（2段目）
for idx, (region_name, region_data) in enumerate(regions.items()):
    row = 1  # 2段目
    col = idx  # 0から3
    y_coords_region, x_coords_region = np.where(region_data == 0)
    axes[row, col].scatter(x_coords_region * resolution_micron_per_pix,
                           -y_coords_region * resolution_micron_per_pix,
                           s=2, c='black')
    axes[row, col].set_xlabel("X (μm)")
    axes[row, col].set_ylabel("Y (μm)")
    axes[row, col].set_title(f"Black Pixels {region_name}")
    axes[row, col].set_xlim(0, side_length * resolution_micron_per_pix)
    axes[row, col].set_ylim(-side_length * resolution_micron_per_pix, 0)
    axes[row, col].set_aspect('equal')

# 3段目: 各領域の中心を原点としたプロット
for idx, (region_name, region_data) in enumerate(regions.items()):
    row = 2  # 3段目
    col = idx  # 0から3
    y_coords_region, x_coords_region = np.where(region_data == 0)

    # 中心を原点にシフト
    x_center_region = x_coords_region - side_length // 2
    y_center_region = -(y_coords_region - side_length // 2)  # Y軸の向きを反転

    axes[row, col].scatter(x_center_region * resolution_micron_per_pix,
                           y_center_region * resolution_micron_per_pix,
                           s=2, c='black')
    axes[row, col].set_xlabel("X (μm)")
    axes[row, col].set_ylabel("Y (μm)")
    axes[row, col].set_title(f"Centered Black Pixels {region_name}")
    axes[row, col].set_xlim(-side_length // 2 * resolution_micron_per_pix, side_length // 2 * resolution_micron_per_pix)
    axes[row, col].set_ylim(-side_length // 2 * resolution_micron_per_pix, side_length // 2 * resolution_micron_per_pix)
    axes[row, col].set_aspect('equal')

plt.tight_layout()
plt.show()
