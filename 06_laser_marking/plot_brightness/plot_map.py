import pandas as pd
import matplotlib.pyplot as plt
import os

# CSVファイルの読み込み（このセルではファイル名を仮定）
csv_path = './02_white_nylon.csv'

# CSVを読み込む
df = pd.read_csv(csv_path)

# ピボットテーブルに変換（行=F, 列=OR, 値=mean）
pivot = df.pivot(index="F", columns="OR", values="mean")

# プロット作成
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('jet')  # 赤が高輝度、青が低輝度
im = plt.imshow(pivot.values, cmap=cmap, origin='lower',
                extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                aspect='auto', vmin=0, vmax=255)  # 凡例スケールを0–255に固定

# 軸とカラーバー
plt.colorbar(im, label='Mean Brightness (0–255)')
plt.xlabel('Overlap Ratio (OR)')
plt.ylabel('Fluence (F)')
plt.title('Fluence vs Overlap Ratio Brightness Map')
plt.xticks(pivot.columns)
plt.yticks(pivot.index)
plt.tight_layout()

# SVG保存パスの作成と保存
svg_output_path = './02_map.svg'
plt.savefig(svg_output_path, format='svg')
