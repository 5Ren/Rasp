import csv
import pandas as pd

# ======== 定数 ========
IMG_W = 2560   # 画像幅
IMG_H = 2052   # 画像高さ
NM_PER_PIXEL = 1000 / 200  # = 5 nm/px

# square size
square_side = int(min(IMG_W / 3, IMG_H / 2))  # = 853 px

def square_number_from_xy(x, y):
    """ 座標(x,y) → Square_number へ変換 """
    if y < square_side:
        # 上段 (1〜3)
        if x < square_side:
            return 1
        elif x < square_side * 2:
            return 2
        else:
            return 3
    else:
        # 下段 (4〜5)
        if x < square_side:
            return 4
        elif x < square_side * 2:
            return 5
        else:
            return None   # 右端は判定外（本来存在しない）

def convert_results_csv(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    out_rows = []
    circle_id_counter = {sq: 1 for sq in range(1,6)}  # 各 square の通し番号

    for _, row in df.iterrows():

        x = float(row["X"])
        y = float(row["Y"])
        d_px = float(row["MinFeret"])
        d_nm = d_px * NM_PER_PIXEL

        sq = square_number_from_xy(x, y)
        if sq is None:
            continue  # 枠外

        circle_num = circle_id_counter[sq]
        circle_id_counter[sq] += 1

        out_rows.append([
            sq,
            circle_num,
            x,
            y,
            d_nm
        ])

    # CSV 出力
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Square_number",
            "Circle_number",
            "Center_x_display",
            "Center_y_display",
            "Diameter_nm"
        ])
        writer.writerows(out_rows)

    print(f"変換完了 → {output_csv}")


# ===== 実行例 =====
convert_results_csv(
    input_csv=r"C:\Users\YamaLab-38\PycharmProjects\Rasp\03_nano-beads\beads_count_2025\00_data_1shot\Results.csv",
    output_csv="converted_results.csv"
)
