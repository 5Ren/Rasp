import cv2
import numpy as np
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

# 黒いピクセルの座標取得（デバッグ目的）
# y_coords, x_coords = np.where(binary_image == 0)

# スケーリングパラメータ
y_length_microns = 20000  # 20000μm
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
print(f"One side length of each quadrant: {scaled_width // 2} pixels")

# 画像を4つの正方形領域に分割
half_y = scaled_height // 2
half_x = scaled_width // 2
side_length = min(half_y, half_x)  # 1000 pix

regions = {
    'Top Left': image_scaled_padded[0:side_length, 0:side_length],
    'Top Right': image_scaled_padded[0:side_length, scaled_width - side_length:scaled_width],
    'Bottom Left': image_scaled_padded[scaled_height - side_length:scaled_height, 0:side_length],
    'Bottom Right': image_scaled_padded[scaled_height - side_length:scaled_height,
                    scaled_width - side_length:scaled_width]
}

# 各領域ごとにエッジ検出を実施
edge_segments_dict = {}  # 各領域のedge_segmentsを格納する辞書

for region_name, region_data in regions.items():
    edge_segments = perform_edge_detection(region_data, side_length)
    edge_segments_dict[region_name] = edge_segments

# エッジセグメントのリストを表示（各領域ごと）
for region, segments in edge_segments_dict.items():
    print(f"\nEdge Segments for {region}:")
    for idx, (start, end) in enumerate(segments):
        print(f"  Segment {idx + 1}: Start {start}, End {end}")
