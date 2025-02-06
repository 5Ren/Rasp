import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# グローバル変数
points = []
circles = []
squares = []
scale_factor = 1000 // 200   # [nm/pix] ピクセルからナノメートルへのスケールファクター
square_side_length_nm = 2000
square_side_length_pix = square_side_length_nm // scale_factor  # 正方形の辺の長さ
directory_path = r'ITO-Old-x10'  # ディレクトリのパスを指定してください

# マウスイベント時に処理を行う
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        for i, square in enumerate(squares):
            if square[0] < x < square[0] + square_side_length_pix and square[1] < y < square[1] + square_side_length_pix:
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                points.append((x, y))
                if len(points) == 3:
                    # 3点が選択されたら、それらを通る円を描画
                    (x, y), radius = cv2.minEnclosingCircle(np.array(points))
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(img, center, radius, (0, 0, 255), 2)  # 円を描画
                    circles.append((i+1, center, radius))  # 円の情報を保存
                    points.clear()  # ポイントをクリア

# 指定したディレクトリ内の全ての画像ファイルでプログラムを実行
for filename in os.listdir(directory_path):
    if filename.endswith(".jpg"):  # 画像ファイルの拡張子を指定
        image_path = os.path.join(directory_path, filename)

        # 画像の読み込み
        original_img = cv2.imread(image_path)
        # original_img = cv2.resize(original_img, (original_img.shape[1] // 2, original_img.shape[0] // 2))  # 画像のサイズを半分にする
        original_img = cv2.resize(original_img, (original_img.shape[1] // 1, original_img.shape[0] // 1))  # 画像のサイズを半分にする
        img = original_img.copy()

        # 画像上に5つの重ならない正方形を描く
        for i in range(3):
            x = i * square_side_length_pix
            y = 0
            cv2.rectangle(img, (x, y), (x + square_side_length_pix, y + square_side_length_pix), (0, 255, 0), 2)
            squares.append((x, y))

        for i in range(2):
            x = i * square_side_length_pix
            y = square_side_length_pix
            cv2.rectangle(img, (x, y), (x + square_side_length_pix, y + square_side_length_pix), (0, 255, 0), 2)
            squares.append((x, y))

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # "q"キーを押した後、各正方形内の円に通し番号を付ける
        for i, square in enumerate(squares):
            square_circles = [(j+1, circle[1], circle[2]) for j, circle in enumerate(circles) if circle[0] == i + 1 and square[0] < circle[1][0] < square[0] + square_side_length_pix and square[1] < circle[1][1] < square[1] + square_side_length_pix]
            for circle in square_circles:
                cv2.putText(img, f"{i+1}-{circle[0]}", circle[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 各円に通し番号がついた状態の画像をJPEGで保存
        cv2.imwrite(f'{filename}_numbered.jpg', img)

        # 描画した全ての円の中心座標と直径をCSVファイルとして出力
        with open(f'{filename}_circles.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Square_number", "Circle_number", "Center_x", "Center_y", "Diameter_nm"])
            circles.sort()  # 正方形ごとに1-5の順番にソート
            for square_number, center, radius in circles:
                diameter_nm = 2 * radius * scale_factor
                writer.writerow([square_number, center[0], center[1], diameter_nm])

        # 変数をリセット
        points = []
        circles = []
        squares = []