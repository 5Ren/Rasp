import pandas as pd
import os
from glob import glob

# 親フォルダ（被験者ごとのフォルダがこの中にある）
parent_folder = r"E:\Takeout_data"

# 結果を格納するリスト
results = []

# 被験者ごとのDataFrameを保持（シート書き込み用）
subject_dfs = {}

# 各被験者フォルダを走査
for subject_folder in os.listdir(parent_folder):
    subject_path = os.path.join(parent_folder, subject_folder)

    if os.path.isdir(subject_path):
        # 被験者名（フォルダ名の中の括弧内を取得）
        if "(" in subject_folder and ")" in subject_folder:
            subject_name = subject_folder.split("(")[-1].replace(")", "").strip()
        else:
            subject_name = subject_folder

        # 日別のアクティビティ指標.csv のパス
        csv_path = os.path.join(subject_path, "Fit", "日別のアクティビティ指標", "日別のアクティビティ指標.csv")

        if os.path.exists(csv_path):
            # CSV読み込み
            df = pd.read_csv(csv_path)

            # 日付列をdatetime型に
            df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

            # 有効な日付データがあるか確認
            if df['日付'].notnull().any():
                start_date = df['日付'].min().strftime('%Y-%m-%d')
                end_date = df['日付'].max().strftime('%Y-%m-%d')
            else:
                start_date = ''
                end_date = ''

            # 平均を取るカラム（可能な限り指定通り）
            cols_to_avg = [
                "通常の運動（分）のカウント", "カロリー（kcal）", "距離（m）", "ハートポイント（強めの運動）",
                "強めの運動（分）", "平均速度（m/s）", "最高速度（m/s）", "最低速度（m/s）", "歩数",
                "平均体重（kg）", "最高体重（kg）", "最低体重（kg）", "「ウォーキング」の時間（ミリ秒）", "「ランニング」の時間（ミリ秒）"
            ]

            averages = {}
            for col in cols_to_avg:
                if col in df.columns:
                    averages[col] = df[col].mean()
                else:
                    averages[col] = None  # カラムがなければ空

            # 概要用のデータまとめ
            result = {
                "被験者": subject_name,
                "データ開始日": start_date,
                "データ終了日": end_date
            }
            result.update(averages)
            results.append(result)

            # 個別シート用のデータも保存
            # Excelのシート名は31文字制限、特殊記号もNGなので整形
            sheet_name = subject_name.replace("/", "_")[:31]
            subject_dfs[sheet_name] = df

# すべての被験者の結果をDataFrameに
summary_df = pd.DataFrame(results)

# 出力先のExcelファイル
output_excel = os.path.join(parent_folder, "被験者別_アクティビティ概要.xlsx")

# Excelファイルに書き込み
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    # 1シート目：概要
    summary_df.to_excel(writer, index=False, sheet_name="概要")

    # 2シート目以降：被験者ごと
    for name, df in subject_dfs.items():
        df.to_excel(writer, index=False, sheet_name=name)

print(f"Excelファイルに保存しました: {output_excel}")
