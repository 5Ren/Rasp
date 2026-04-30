import pandas as pd
from pathlib import Path

# ===== 設定ここだけ変えればOK =====
# CSV が入っているフォルダパス
folder = Path(r"C:\Users\YamaLab-38\Downloads\D75_x10\10mJ\defalt_dark")

# 最小直径 [nm]
min_diameter_nm = 20.0
# =================================


def main():
    # フォルダ内の *.csv をすべて走査
    for csv_path in folder.glob("*.csv"):
        print(f"Processing: {csv_path.name}")

        # ImageJ の Results.csv はタブ区切りのことが多いので、区切り文字を自動判定
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
        except Exception as e:
            print(f"  [WARN] Failed to read {csv_path.name}: {e}")
            continue

        # MinFeret 列があるか確認
        if "MinFeret" not in df.columns:
            print(f"  [WARN] 'MinFeret' column not found in {csv_path.name}. Skipped.")
            continue

        # MinFeret が min_diameter_nm 以上の行だけ抽出
        df_filtered = df[df["MinFeret"] >= min_diameter_nm].copy()

        # 出力ファイル名：拡張子だけ .xlsx に変更
        out_path = csv_path.with_suffix(".xlsx")

        try:
            df_filtered.to_excel(out_path, index=False)
            print(f"  [OK] Saved filtered data to: {out_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to save {out_path.name}: {e}")


if __name__ == "__main__":
    main()
