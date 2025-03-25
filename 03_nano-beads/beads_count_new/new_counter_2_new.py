import cv2
import numpy as np
import csv
import os

# グローバル変数
points = []
circles = []
squares = []

# [nm/pix]（オリジナル画像） ピクセルからナノメートルへのスケールファクター
# 1000 nm を 200 pix 分として考えるなら 5 nm/pix という想定
scale_factor = 1000 // 200

square_side_length_nm = 4000
# オリジナル画像上での正方形のピクセル数（2000 nm → 400 pix）
square_side_length_pix_original = square_side_length_nm // scale_factor

directory_path = r'1001_x10'  # ディレクトリのパスを指定してください

def draw_circle(event, x, y, flags, param):
    """
    表示用にリサイズされた画像の座標系でマウスクリックを取得し、
    squares に登録された正方形内なら points に追加する。
    3点目が確定したら最小外接円を描画＆circles に記録する。
    """
    global points, circles, img
    if event == cv2.EVENT_LBUTTONUP:
        for i, square in enumerate(squares):
            # square: (x座標, y座標, 一辺の長さ) → 表示画像座標系
            x_top, y_top, side_len = square
            if x_top < x < x_top + side_len and y_top < y < y_top + side_len:
                # 小さな青点
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
                points.append((x, y))

                if len(points) == 3:
                    # 3点の最小外接円を計算（表示画像座標系で）
                    (x_c, y_c), radius = cv2.minEnclosingCircle(np.array(points))
                    center = (int(x_c), int(y_c))
                    radius = int(radius)

                    # 赤い円を描画
                    cv2.circle(img, center, radius, (0, 0, 255), 2)

                    # circles には (正方形番号, 中心, 半径) を保持
                    # 正方形番号は i+1 とする
                    circles.append((i + 1, center, radius))

                    # 次の円のために点をクリア
                    points.clear()
                break

# 画面サイズの取得（任意の固定値でOK）
screen_width = 1920
screen_height = 1080

# BMP拡張子のファイルを処理
for filename in os.listdir(directory_path):
    if filename.lower().endswith(".bmp"):
        image_path = os.path.join(directory_path, filename)

        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"画像の読み込みに失敗しました: {filename}")
            continue

        # オリジナル画像の幅・高さを取得
        img_height, img_width = original_img.shape[:2]

        # 表示用にリサイズする比率を求める
        scale = min(screen_width / img_width, screen_height / img_height)

        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        # 表示用画像を作成
        img = cv2.resize(original_img, (display_width, display_height))

        # 表示用に描画する正方形の一辺 (リサイズ後のサイズ)
        square_side_length_pix_display = int(square_side_length_pix_original * scale)

        # 画像上に5つの正方形を描く (表示画像座標系で)
        squares.clear()

        # 上段: 3つ
        for i in range(3):
            x = i * square_side_length_pix_display
            y = 0
            cv2.rectangle(img,
                          (x, y),
                          (x + square_side_length_pix_display, y + square_side_length_pix_display),
                          (0, 255, 0),
                          2)
            # squares には (x, y, 正方形の一辺の表示ピクセル) を保持
            squares.append((x, y, square_side_length_pix_display))

        # 下段: 2つ
        for i in range(2):
            x = i * square_side_length_pix_display
            y = square_side_length_pix_display
            cv2.rectangle(img,
                          (x, y),
                          (x + square_side_length_pix_display, y + square_side_length_pix_display),
                          (0, 255, 0),
                          2)
            squares.append((x, y, square_side_length_pix_display))

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', draw_circle)

        while True:
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # 正方形番号ごとに、円の番号を描画
        for i, square in enumerate(squares):
            # i+1 番目の square のみに該当する circle を絞り込み
            square_number = i + 1
            x_top, y_top, side_len = square

            # circles のうち、square_number が同じもの＆そこに中心が含まれるもの
            square_circles = [
                (circle_idx + 1, circle_data[1], circle_data[2])  # (通し番号, 中心, 半径)
                for circle_idx, circle_data in enumerate(circles)
                if (circle_data[0] == square_number
                    and x_top < circle_data[1][0] < x_top + side_len
                    and y_top < circle_data[1][1] < y_top + side_len)
            ]

            # 各円の上に "正方形番号-円通し番号" を描画
            for circle in square_circles:
                circle_id, center, radius = circle
                cv2.putText(img,
                            f"{square_number}-{circle_id}",
                            center,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # CSV保存
        csv_filename = f"{filename}_circles.csv"
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Square_number", "Circle_number", "Center_x_display", "Center_y_display", "Diameter_nm"])

            # circles は (正方形番号, (中心x, 中心y), 半径[表示画像px]) のリスト
            # 円の通し番号は出力時の並びで j+1 とする
            circles.sort(key=lambda c: (c[0]))  # 正方形番号順にソート

            for j, (square_number, center, radius_display) in enumerate(circles):
                # リサイズ後の半径 → リサイズ前の半径に戻す
                # オリジナル画像座標系での半径: radius_original = radius_display / scale
                radius_original = radius_display / scale

                # nm への変換: diameter_nm = 2 * radius_original * scale_factor
                diameter_nm = 2 * radius_original * scale_factor

                writer.writerow([
                    square_number,           # 正方形番号
                    j + 1,                   # 円の通し番号
                    center[0], center[1],   # 表示画像上の中心座標
                    diameter_nm              # 直径[nm]
                ])

        print(f"処理完了: {filename}, 結果を {csv_filename} に保存")

        # 次の画像を処理する前にリセット
        points.clear()
        circles.clear()
        squares.clear()
