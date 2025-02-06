import cv2
import numpy as np

# グローバル変数
points = []
circles = []
image_path = '../ITO-test/ITO-100PER-x50K.bmp'  # 画像のパスを指定してください

# マウスイベント時に処理を行う
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        points.append((x, y))
        if len(points) == 3:
            # 3点が選択されたら、それらを通る円を描画
            (x, y), radius = cv2.minEnclosingCircle(np.array(points))
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (0, 0, 255), 2)  # 円を描画
            circles.append((center, radius))  # 円の情報を保存
            points.clear()  # ポイントをクリア

# 画像の読み込み
original_img = cv2.imread(image_path)
img = original_img.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# 描画した全ての円の中心座標と直径を出力
for center, radius in circles:
    print(f"Center: {center}, Diameter: {2*radius}")