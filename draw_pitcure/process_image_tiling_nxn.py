import cv2
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
                # 白から黒への遷移: 始点（中心を原点にシフト）
                start = (j - side_length // 2, -(i - side_length // 2))
            elif prev_pixel == 0 and current_pixel == 255 and start is not None:
                # 黒から白への遷移: 終点（中心を原点にシフト）
                end = (j - side_length // 2, -(i - side_length // 2))
                edge_segments.append((start, end))
                start = None  # リセット
            prev_pixel = current_pixel

    return edge_segments

# -------------------------------------------------
# 画像の読み込み、前処理、および二値化
# -------------------------------------------------
image_path = "./image_files/25-1-27_cosun_white.jpg"
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
# 元画像を引き伸ばすための目標のサイズ（長い方の辺の長さを target_stretch_height_microns に合わせる）
target_stretch_length_microns = 20000   # 例: 20000 μm

# 加工する領域のパラメータ（最終的に処理する領域の大きさ）
# １辺での分割数なので、タイリング数はsplit_num ^2 になる
split_num = 3  # 分割数は、target_stretch_length_microns が入る数にすること。

resolution_um_per_pix = 10  # 解像度 (μm/pix)
scanning_field_um = 10 * 10 ** 3  # レンズでスキャンできる領域
processing_area_length_um = split_num * scanning_field_um  # 加工する領域の高さ（μm）

x_correction_um = 300
y_correction_um = 130
# -------------------------------------------------

# ステージのタイリングのリスト生成
stage_tiling_list = []
corrected_tiling_x_um = scanning_field_um - x_correction_um
corrected_tiling_y_um = scanning_field_um - y_correction_um

for region_y in range(split_num):
    for region_x in range(split_num):
        if region_x == 0:
            if region_y == 0:
                x_posi_um = 0
                y_posi_um = 0
            else:
                x_posi_um = corrected_tiling_x_um * (split_num - 1)
                y_posi_um = corrected_tiling_y_um
        elif region_x < split_num:
            x_posi_um = -1 * corrected_tiling_x_um
            y_posi_um = 0

        stage_tiling_list.append([x_posi_um, y_posi_um])

print(stage_tiling_list)
# -------------------------------------------------
# 加工する領域（正方形）のピクセルサイズ
processing_area_height_pix = int(processing_area_length_um / resolution_um_per_pix)
processing_area_width_pix  = processing_area_height_pix  # 正方形

orig_h, orig_w = binary_image.shape  # 元画像の高さと幅（ピクセル）

if orig_h >= orig_w:
    # 縦が長い場合：高さを基準にする
    target_height_pix = int(target_stretch_length_microns / resolution_um_per_pix)
    target_width_pix  = int(target_height_pix * (orig_w / orig_h))
else:
    # 横が長い場合：幅を基準にする
    target_width_pix  = int(target_stretch_length_microns / resolution_um_per_pix)
    target_height_pix = int(target_width_pix * (orig_h / orig_w))

# 元画像を引き伸ばす（アスペクト比を保ったままリサイズ）
image_stretched = cv2.resize(binary_image, (target_width_pix, target_height_pix), interpolation=cv2.INTER_NEAREST)# 加工領域と引き伸ばした画像のサイズ差を計算し、不足分を白 (255) でパディングする
delta_w = processing_area_width_pix - target_width_pix
delta_h = processing_area_height_pix - target_height_pix

top    = delta_h // 2 if delta_h > 0 else 0
bottom = delta_h - top if delta_h > 0 else 0
left   = delta_w // 2 if delta_w > 0 else 0
right  = delta_w - left if delta_w > 0 else 0

image_final = cv2.copyMakeBorder(image_stretched, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=255)
print(f"Final processed image size: {image_final.shape[1]} x {image_final.shape[0]} pixels")

# -------------------------------------------------
# 画像の分割（縦横 split_num 分割）
# -------------------------------------------------
side_length_y = image_final.shape[0] // split_num
side_length_x = image_final.shape[1] // split_num
side_length = min(side_length_y, side_length_x)
print(f"One side length of each region: {side_length} pixels")

regions = {}
for row in range(split_num):
    for col in range(split_num):
        region_name = f"Region {row+1}-{col+1}"
        y_start = row * side_length
        y_end = y_start + side_length
        x_start = col * side_length
        x_end = x_start + side_length
        regions[region_name] = image_final[y_start:y_end, x_start:x_end]

# -------------------------------------------------
# 各 Region のエッジセグメント（ライン）の始点と終点を出力
# -------------------------------------------------
print("\n=== 各 Region のライン（エッジセグメント）の始点と終点 ===")
for region_name in sorted(regions.keys()):
    region_data = regions[region_name]
    segments = perform_edge_detection(region_data, side_length)
    print(f"{region_name}:")
    if segments:
        for idx, (start, end) in enumerate(segments):
            print(f"  Segment {idx+1}: Start {start}, End {end}")
    else:
        print("  エッジセグメントは検出されませんでした。")
