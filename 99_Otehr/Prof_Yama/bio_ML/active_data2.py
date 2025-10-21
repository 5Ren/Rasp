import pandas as pd
import os

# 親フォルダ（被験者フォルダがこの中にある）
parent_folder = r"E:\活動量計"

# 概要を格納するリスト
results = []

# 各被験者ごとのデータ（Excelシート用）
subject_dfs = {}

# 各被験者フォルダを走査
for subject_folder in os.listdir(parent_folder):
    subject_path = os.path.join(parent_folder, subject_folder)

    if os.path.isdir(subject_path):
        subject_id = subject_folder  # フォルダ名 = 被験者ID

        # CSVファイルのパス（Y-1.csvなど）
        csv_path = os.path.join(subject_path, f"{subject_id}.csv")

        if os.path.exists(csv_path):
            # CSV読み込み（2行目ヘッダー）
            df = pd.read_csv(csv_path, header=1, encoding="cp932")

            # 日付列をdatetime型に
            df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

            # 有効な日付データがあるか確認
            if df['日付'].notnull().any():
                start_date = df['日付'].min().strftime('%Y-%m-%d')
                end_date = df['日付'].max().strftime('%Y-%m-%d')
            else:
                start_date = ''
                end_date = ''

            # 平均を取るカラム
            cols_to_avg = [
                "無効フラグ", "歩行カロリー合計(kcal)", "生活活動カロリー合計(kcal)", "カロリー合計(kcal)",
                "総カロリー合計(kcal)", "歩行エクササイズ合計(Ex)", "生活活動エクササイズ合計(Ex)",
                "エクササイズ合計(Ex)",
                "歩数合計(歩)", "歩行時間(分)", "活動時間１(分)", "活動時間２(分)", "活動時間３(分)", "活動時間４(分)",
                "活動時間５(分)", "活動時間６(分)", "活動時間７(分)", "活動時間８(分)", "装着時間(分)", "身長(cm)",
                "体重(kg)", "基礎代謝(kcal)"
            ]

            averages = {}
            for col in cols_to_avg:
                if col in df.columns:
                    averages[col] = df[col].mean()
                else:
                    averages[col] = None

            # メモと活動量計シリアルIDは「平均」ではなく最頻値（mode）をとる
            for col in ["メモ", "活動量計シリアルID"]:
                if col in df.columns and df[col].notnull().any():
                    try:
                        averages[col] = df[col].mode().iloc[0]
                    except:
                        averages[col] = None
                else:
                    averages[col] = None

            # 概要データまとめ
            result = {
                "被験者ID": subject_id,
                "データ開始日": start_date,
                "データ終了日": end_date
            }
            result.update(averages)
            results.append(result)

            # Excel用（個別シート）
            sheet_name = subject_id[:31]  # Excelのシート名制限
            subject_dfs[sheet_name] = df

# 概要のDataFrame
summary_df = pd.DataFrame(results)

# 出力先Excel
output_excel = os.path.join(parent_folder, "被験者別_活動量計_概要.xlsx")

# Excel書き込み
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    # 1シート目：概要
    summary_df.to_excel(writer, index=False, sheet_name="概要")

    # 2シート目以降：被験者ごと
    for subject_id, df in subject_dfs.items():
        df.to_excel(writer, index=False, sheet_name=subject_id)

print(f"Excelファイルに保存しました: {output_excel}")
