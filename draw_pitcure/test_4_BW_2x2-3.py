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

    Parameters:
    - region_image: 二値化された領域画像（numpy配列）
    - side_length: 領域の一辺の長さ（ピクセル）

    Returns:
    - edge_segments: 始点と終点のタプルのリスト（座標は中心を原点としたもの）
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
                # 白から黒への遷移: 始点
                start = (j - side_length // 2, -(i - side_length // 2))
            elif prev_pixel == 0 and current_pixel == 255 and start is not None:
                # 黒から白への遷移: 終点
                end = (j - side_length // 2, -(i - side_length // 2))
                edge_segments.append((start, end))
                start = None  # リセット
            prev_pixel = current_pixel

    return edge_segments


def plot_edge_segments(ax, edge_segments, resolution_micron_per_pix):
    """
    エッジセグメントをプロットします。

    Parameters:
    - ax: MatplotlibのAxesオブジェクト
    - edge_segments: エッジセグメントのリスト（座標は中心を原点としたもの）
    - resolution_micron_per_pix: 解像度（μm/pix）
    """
    for start, end in edge_segments:
        color = (random.random(), random.random(), random.random())
        ax.plot([start[0] * resolution_micron_per_pix, end[0] * resolution_micron_per_pix],
                [start[1] * resolution_micron_per_pix, end[1] * resolution_micron_per_pix],
                c=color, linewidth=0.5)
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_title("Edge Detection (Start & End Points with Random Colors)")
    ax.set_xlim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    ax.set_ylim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    ax.set_aspect('equal')


# 画像の読み込み
image_path = "./image_files/25-1-27_cosun_white.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("画像の読み込みに失敗しました。")
    exit()

# ファイル名のサフィックス取得
filename = os.path.basename(image_path)
file_suffix = filename.split("_")[-1].split(".")[0]

# ガウシアンフィルター設定
if file_suffix.lower() == "black":
    gaussian_kernel_size = (9, 9)
    gaussian_sigma = 1.0
elif file_suffix.lower() == "white":
    gaussian_kernel_size = (5, 5)
    gaussian_sigma = 1.0
else:
    gaussian_kernel_size = (7, 7)
    gaussian_sigma = 1.0

# 画像リサイズ（縦横2倍）
image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# ガウシアンフィルター適用
image_blurred = cv2.GaussianBlur(image_resized, gaussian_kernel_size, gaussian_sigma)

# 外側10画素を白でパディング
image_blurred[:10, :] = 255
image_blurred[-10:, :] = 255
image_blurred[:, :10] = 255
image_blurred[:, -10:] = 255

# 二値化
_, binary_image = cv2.threshold(image_blurred, 127, 255, cv2.THRESH_BINARY)

# 黒いピクセルの座標取得
y_coords, x_coords = np.where(binary_image == 0)

# スケーリングパラメータ
y_length_microns = 30000  # 20000μm
resolution_micron_per_pix = 10  # 10μm/pix

# 新しい画像サイズ計算
y_new_size = int(y_length_microns / resolution_micron_per_pix)  # 2000 pix
aspect_ratio = binary_image.shape[1] / binary_image.shape[0]
x_new_size = int(y_new_size * aspect_ratio)  # 2000 pix if aspect_ratio=1

# 画像リサイズ
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

# スケール後の黒いピクセルの座標取得
y_coords_scaled, x_coords_scaled = np.where(image_scaled_padded == 0)

# 中心を原点にシフト
x_centered = x_coords_scaled - scaled_width // 2
y_centered = -(y_coords_scaled - scaled_height // 2)

# プロット設定（4段目を追加）
fig, axes = plt.subplots(4, 4, figsize=(24, 24), dpi=150)

# 1段目: 元のピクセル座標
axes[0, 0].scatter(x_coords, -y_coords, s=2, c='black')
axes[0, 0].set_xlabel("X (pixels)")
axes[0, 0].set_ylabel("Y (pixels)")
axes[0, 0].set_title("Black Pixels Coordinates (Original)")
axes[0, 0].set_xlim(0, binary_image.shape[1])
axes[0, 0].set_ylim(-binary_image.shape[0], 0)
axes[0, 0].set_aspect('equal')

# 1段目: スケール後のμm座標
axes[0, 1].scatter(x_centered * resolution_micron_per_pix, y_centered * resolution_micron_per_pix, s=2, c='black')
axes[0, 1].set_xlabel("X (μm)")
axes[0, 1].set_ylabel("Y (μm)")
axes[0, 1].set_title("Black Pixels Coordinates (Scaled)")
axes[0, 1].set_xlim(-scaled_width * resolution_micron_per_pix // 2, scaled_width * resolution_micron_per_pix // 2)
axes[0, 1].set_ylim(-scaled_height * resolution_micron_per_pix // 2, scaled_height * resolution_micron_per_pix // 2)
axes[0, 1].set_aspect('equal')

# 1段目の残り2つのサブプロットを非表示にする
axes[0, 2].axis('off')
axes[0, 3].axis('off')

# 画像を4つの正方形領域に分割
half_y = scaled_height // 2
half_x = scaled_width // 2
side_length = min(half_y, half_x)  # 1000 pix
print(f"One side length of each quadrant: {side_length} pixels")

regions = {
    'Top Left': image_scaled_padded[0:side_length, 0:side_length],
    'Top Right': image_scaled_padded[0:side_length, scaled_width - side_length:scaled_width],
    'Bottom Left': image_scaled_padded[scaled_height - side_length:scaled_height, 0:side_length],
    'Bottom Right': image_scaled_padded[scaled_height - side_length:scaled_height,
                    scaled_width - side_length:scaled_width]
}

# 2段目: 各領域の黒いピクセル
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

# 3段目: 各領域の中心を原点とした黒いピクセル
for idx, (region_name, region_data) in enumerate(regions.items()):
    row = 2  # 3段目
    col = idx  # 0から3
    y_coords_region, x_coords_region = np.where(region_data == 0)

    # 中心を原点にシフト
    x_center_region = x_coords_region - side_length // 2
    y_center_region = -(y_coords_region - side_length // 2)

    # デバッグ用: 座標の範囲を確認
    print(f"{region_name} Centered Coordinates: X min={x_center_region.min()}, X max={x_center_region.max()}, "
          f"Y min={y_center_region.min()}, Y max={y_center_region.max()}")

    axes[row, col].scatter(x_center_region * resolution_micron_per_pix,
                           y_center_region * resolution_micron_per_pix,
                           s=2, c='black')
    axes[row, col].set_xlabel("X (μm)")
    axes[row, col].set_ylabel("Y (μm)")
    axes[row, col].set_title(f"Centered Black Pixels {region_name}")
    axes[row, col].set_xlim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    axes[row, col].set_ylim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    axes[row, col].set_aspect('equal')

# 4段目: 各領域のエッジ検出プロット
edge_segments_dict = {}  # 各領域のedge_segmentsを格納する辞書

for idx, (region_name, region_data) in enumerate(regions.items()):
    # エッジ検出を実施
    edge_segments = perform_edge_detection(region_data, side_length)
    edge_segments_dict[region_name] = edge_segments

    # プロット
    row = 3  # 4段目
    col = idx  # 0から3
    ax = axes[row, col]
    for start, end in edge_segments:
        ax.plot([start[0] * resolution_micron_per_pix, end[0] * resolution_micron_per_pix],
                [start[1] * resolution_micron_per_pix, end[1] * resolution_micron_per_pix],
                c=(random.random(), random.random(), random.random()), linewidth=0.5)
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_title(f"Edge Detection {region_name}")
    ax.set_xlim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    ax.set_ylim(-500 * resolution_micron_per_pix, 500 * resolution_micron_per_pix)
    ax.set_aspect('equal')

# 不要なサブプロットを非表示にする
for row in range(4):
    for col in range(4):
        if (row == 0 and col >= 2):
            axes[row, col].axis('off')

plt.tight_layout()

# プロットをファイルとして保存
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "edge_detection_results.png")
plt.savefig(plot_path)
print(f"\nPlot saved to {plot_path}")

plt.show()

# edge_segments のリストを表示（各領域ごと）
for region, segments in edge_segments_dict.items():
    print(f"\nEdge Segments for {region}:")
    for idx, (start, end) in enumerate(segments):
        print(f"  Segment {idx + 1}: Start {start}, End {end}")
