import pandas as pd
from pathlib import Path
import re

# ====== ここだけ自分のパスに変更 ======
BASE = Path("/Users/ren/Downloads/AGF_00h_Iinioi/AGF_06h_Iinioi")
# ======================================

# TIME自動抽出（例：00h_01 → 00h）
folder_names = [p.name for p in BASE.iterdir() if p.is_dir()]
m = re.match(r"^(.+?)_\d+$", folder_names[0])
TIME = m.group(1)
print(f"[INFO] TIME = {TIME}")

# replicateフォルダ自動取得
rep_folders = sorted(
    [p for p in BASE.iterdir() if p.is_dir() and p.name.startswith(TIME)],
    key=lambda x: int(x.name.split("_")[1])
)
print(f"[INFO] Replica folders: {[p.name for p in rep_folders]}")

# graphファイル数を検出
sample_folder = rep_folders[0]
graph_files = sorted(sample_folder.glob("graph_*.csv"))
N_GRAPH = len(graph_files)
print(f"[INFO] N_GRAPH = {N_GRAPH}")

# ===== マージ処理 =====
for g in range(1, N_GRAPH + 1):
    graph_name = f"graph_{g:02}.csv"
    merged_df = None

    for folder in rep_folders:
        fpath = folder / graph_name
        if not fpath.exists():
            print(f"[WARN] missing → {fpath}")
            continue

        df = pd.read_csv(fpath)

        # 必須列チェック
        if "Time[s]" not in df.columns or "Y" not in df.columns:
            print(f"[ERROR] 必須列(Time[s], Y)が見つかりません → {fpath}")
            continue

        df = df[["Time[s]", "Y"]].rename(columns={"Y": folder.name})

        # 初回のみTimeごと取得
        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = merged_df.merge(df, on="Time[s]", how="outer")

    if merged_df is not None:
        out_path = BASE / f"{TIME}_graph_{g:02}.csv"
        merged_df.sort_values("Time[s]").to_csv(out_path, index=False)
        print(f"[OK] {out_path}")

print("\n✅ All merged successfully!")
