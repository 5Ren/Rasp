import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 画像を読み込む
img = cv2.imread('../ITO-x10-test/ITO-100PER-x10K.bmp', 0)

# ピクセルを長さの次元に変換する比率
pixel_to_length_ratio = 1  # この値を実際の画像のスケールに基づいて調整してください

# 二値化
_, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 輪郭を見つける
contours, _ = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 粒子の数をカウント
particle_count = len(contours)
print(f"粒子の数: {particle_count}")

# 粒子径のリストを作成
particle_diameters = [cv2.contourArea(c)**0.5 * pixel_to_length_ratio for c in contours]

# ヒストグラムを作成
plt.hist(particle_diameters, bins=20, edgecolor='black')
plt.xlabel('Diameter of beds')
plt.ylabel('counts')
plt.show()

# 画像に輪郭を描く（赤色で）
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # グレースケール画像をカラー画像に変換
for contour in contours:
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(img_color, ellipse, (0,0,255), 2)

# 画像を表示する
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))  # BGRからRGBに変換
plt.title('Image with contours')
plt.show()
