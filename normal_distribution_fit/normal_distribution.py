import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# xlsxファイルを読み込む
df = pd.read_excel('./010PER.xlsx', engine='openpyxl')

# データをプロットする
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], label='Data')

# 正規分布にフィットさせる
mu, std = norm.fit(df.iloc[:, 0])

# フィッティングカーブをプロットする
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'Fit: $\mu$ = {mu:.2f}, $\sigma$ = {std:.2f}')

# ピーク値をプロットする
peak = norm.pdf(mu, mu, std)
plt.plot(mu, peak, 'ro', label=f'Peak: x = {mu:.2f}, y = {peak:.2f}')

# グラフのタイトルとラベルを設定する
plt.title('Particle Size Distribution')
plt.xlabel('Particle Diameter')
plt.ylabel('Frequency')
plt.legend()

# グラフをPNGファイルとして保存する
plt.savefig('distribution.png')

# グラフを表示する
plt.show()
