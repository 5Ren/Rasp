import cv2


def threshold_image(image, lower_thresh, upper_thresh):
    # グレースケールに変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二値化
    _, binary_image = cv2.threshold(gray_image, lower_thresh, upper_thresh, cv2.THRESH_BINARY)
    return binary_image


if __name__ == "__main__":
    # 画像ファイルのパス
    image_path = '../ITO-test/ITO-100PER-x50K.bmp'

    # 画像を読み込む
    image = cv2.imread(image_path)

    # 初期のしきい値
    lower_thresh = 100
    upper_thresh = 255

    while True:
        # 二値化した画像を取得
        binary_image = threshold_image(image, lower_thresh, upper_thresh)

        # 画像をリサイズ
        resized_image = cv2.resize(binary_image, (0, 0), fx=0.4, fy=0.4)

        # 画像を表示
        cv2.imshow('Binary Image', resized_image)

        # キー入力を待つ
        key = cv2.waitKey(0)

        # 'q'が入力されたら終了
        if key == ord('q'):
            break

        # 上下矢印キーでしきい値を調整
        elif key == ord('w'):
            upper_thresh = min(upper_thresh + 1, 255)
        elif key == ord('s'):
            upper_thresh = max(upper_thresh - 1, lower_thresh)
        elif key == ord('a'):
            lower_thresh = max(lower_thresh - 1, 0)
        elif key == ord('d'):
            lower_thresh = min(lower_thresh + 1, upper_thresh)

    # ウィンドウを閉じる
    cv2.destroyAllWindows()
