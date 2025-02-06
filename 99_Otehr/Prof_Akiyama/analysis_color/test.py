import cv2
import numpy as np

# 画像を読み込む
image_path = "color.bmp"  # 画像のパス
image = cv2.imread(image_path)

if image is None:
    print("画像が見つかりません。パスを確認してください。")
else:
    # OpenCVのBGR形式をRGB形式に変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 階調の数を計算
    gradation_counts = {}
    colors = ('Red', 'Green', 'Blue')
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        unique_gradations = np.sum(histogram > 0)  # 頻度が0より大きい階調を数える
        gradation_counts[color] = unique_gradations

    # 結果を出力
    for color, count in gradation_counts.items():
        print(f"{color}の階調数: {int(count)}")
