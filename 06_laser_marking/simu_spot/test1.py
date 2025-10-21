import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# パラメータ設定
D = 12  # 直径 [μm]
tau = 8.5  # ピッチ [μm]
radius = D / 2
num = 10  # 各方向に30個ずつ配置

# ファイル名生成用
filename_base = f"D{int(D)}_tau{int(tau)}"

# 解像度設定（1μmあたりのピクセル数）
resolution = 20  # pixels/μm
image_size_um = 2 * num * tau
image_size_px = int(image_size_um * resolution)
center = image_size_px // 2

# 空画像
image = np.zeros((image_size_px, image_size_px), dtype=np.float32)

# 円描画関数（高速版）
def draw_circle_fast(img, cx, cy, r_pix, value=1.0):
    x_min = max(cx - r_pix, 0)
    x_max = min(cx + r_pix + 1, img.shape[1])
    y_min = max(cy - r_pix, 0)
    y_max = min(cy + r_pix + 1, img.shape[0])
    y, x = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r_pix ** 2
    img[y_min:y_max, x_min:x_max][mask] += value

# ピクセル単位に変換
r_pix = int(radius * resolution)
pitch_pix = int(tau * resolution)

# 円を配置
for i in range(-num, num + 1):
    for j in range(-num, num + 1):
        cx = center + i * pitch_pix
        cy = center + j * pitch_pix
        draw_circle_fast(image, cx, cy, r_pix, value=1.0)

# 中心スポットをクロップ
crop_range = int(D * resolution // 2)
spot_crop = image[center - crop_range:center + crop_range,
                  center - crop_range:center + crop_range]

# 中心から radius 以内のピクセルを抽出
y, x = np.ogrid[-crop_range:crop_range, -crop_range:crop_range]
mask = (x**2 + y**2) <= (r_pix ** 2)
values_within_radius = spot_crop[mask]

# ヒストグラムデータ作成
max_val = int(np.max(values_within_radius))
counts, bins = np.histogram(values_within_radius, bins=np.arange(0, max_val + 2))
total_pixels = np.sum(mask)
percentages = 100 * counts / total_pixels

# データ保存（CSV）
df = pd.DataFrame({
    "Overlap Count": bins[:-1],
    "Percentage (%)": percentages
})
csv_path = f"{filename_base}.csv"
df.to_csv(csv_path, index=False)

# グレースケール反転画像
normalized_full = image / np.max(image)
inverted_full = 1.0 - normalized_full
normalized_spot = spot_crop / np.max(image)
inverted_spot = 1.0 - normalized_spot

# 描画（2段構成）
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 上段左：全体反転画像
axs[0, 0].imshow(inverted_full, cmap='gray', interpolation='nearest')
axs[0, 0].set_title("Full Pattern (Inverted)")
axs[0, 0].axis('off')

# 上段右：中心スポット反転画像
axs[0, 1].imshow(inverted_spot, cmap='gray', interpolation='nearest')
axs[0, 1].set_title("Single Spot Area (Inverted)")
axs[0, 1].axis('off')

# 下段左：ピクセル数ヒストグラム
axs[1, 0].bar(bins[:-1], counts, width=0.6, color='black', edgecolor='gray')
axs[1, 0].set_xticks(np.arange(0, max_val + 1, 1))
axs[1, 0].set_title("Histogram of Raw Overlap Values\n(Within Central Spot Radius)")
axs[1, 0].set_xlabel("Grayscale Value (Overlap Count)")
axs[1, 0].set_ylabel("Pixel Count")

# 下段右：割合ヒストグラム（％）
axs[1, 1].bar(bins[:-1], percentages, width=0.6, color='black', edgecolor='gray')
axs[1, 1].set_xticks(np.arange(0, max_val + 1, 1))
axs[1, 1].set_ylim(0, 100)
axs[1, 1].set_title("Percentage Histogram\n(Within Central Spot Radius)")
axs[1, 1].set_xlabel("Grayscale Value (Overlap Count)")
axs[1, 1].set_ylabel("Percentage of Pixels (%)")

plt.tight_layout()
png_path = f"{filename_base}.png"
plt.savefig(png_path, dpi=300)
plt.show()

(csv_path, png_path)
