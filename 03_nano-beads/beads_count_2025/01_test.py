import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

# ======== 設定ここだけ変えればOK ========

# 画像フォルダ
image_dir = Path(r"C:\Users\YamaLab-38\Downloads\D75_x10\10mJ_x30k")

# 出力フォルダ（なければ image_dir と同じでもOK）
output_dir = image_dir  # Path(r"...") に変えてもよい

# 丸っこさのしきい値（1.0 が完全な円）
CIRCULARITY_THRESHOLD = 0.7

# スケール（nm / pixel）を使いたい場合（なければ 1.0 のままでOK）
NM_PER_PIXEL = 1.6666667

# ========================================


def detect_round_particles(img_path: Path):
    """1枚の画像から丸っこい粒子を検出し、情報を DataFrame で返す。"""
    # 画像読み込み（グレースケール）
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Cannot read image: {img_path}")
        return pd.DataFrame()

    # ノイズを減らす（ガウシアンブラー）
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu で2値化（粒子が暗い前提：THRESH_BINARY_INV）
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 輪郭抽出
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    records = []

    for idx, cnt in enumerate(contours, start=1):
        # 面積と周囲長
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        # 円形度 (C = 4πA / P^2)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        if circularity < CIRCULARITY_THRESHOLD:
            # 丸っこくないものはスキップ
            continue

        # 重心（座標）
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 等価円半径 (A = πr² → r = sqrt(A/π))
        r_pixel = math.sqrt(area / math.pi)
        d_pixel = 2.0 * r_pixel

        # nm に変換（スケールを使いたい場合）
        r_nm = r_pixel * NM_PER_PIXEL
        d_nm = d_pixel * NM_PER_PIXEL

        records.append({
            "id": idx,
            "cx_pixel": cx,
            "cy_pixel": cy,
            "radius_pixel": r_pixel,
            "diameter_pixel": d_pixel,
            "radius_nm": r_nm,
            "diameter_nm": d_nm,
            "area_pixel": area,
            "circularity": circularity,
        })

    df = pd.DataFrame(records)
    return df


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    # 画像拡張子の候補
    exts = [".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue

        print(f"Processing: {img_path.name}")
        df = detect_round_particles(img_path)

        # 出力ファイル名は元画像名 + "_circles.csv"
        out_csv = output_dir / f"{img_path.stem}_circles.csv"

        df.to_csv(out_csv, index=False)
        print(f"  -> {len(df)} particles saved to {out_csv.name}")


if __name__ == "__main__":
    main()
