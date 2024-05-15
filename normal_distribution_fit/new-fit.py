import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import os

# ディレクトリ名を指定する
directory_name = r'./ITO-beads'

# 指定されたディレクトリ内のxlsxファイルのリストを取得する
xlsx_files = [f for f in os.listdir(directory_name) if f.endswith('.xlsx')]

# 各xlsxファイルに対して処理を実行する
for file in xlsx_files:
    # ファイルパスを作成する
    file_path = os.path.join(directory_name, file)

    # xlsxファイルを読み込む
    df = pd.read_excel(file_path, engine='openpyxl')


    # 正規分布の関数を定義する
    def normal_dist(x, mean, sd):
        return norm.pdf(x, mean, sd) * len(df.iloc[:, 1]) * sd * np.sqrt(2 * np.pi)


    # データにフィットさせる
    params, _ = curve_fit(normal_dist, df.iloc[:, 0], df.iloc[:, 1], p0=[np.mean(df.iloc[:, 0]), np.std(df.iloc[:, 0])])

    # フィッティングカーブをプロットする
    x = np.linspace(min(df.iloc[:, 0]), max(df.iloc[:, 0]), 100)
    plt.plot(x, normal_dist(x, *params), 'r-', label=f'Fit: $\mu$ = {params[0]:.2f}, $\sigma$ = {params[1]:.2f}')

    # ピーク値を計算する
    peak_y = max(normal_dist(x, *params))
    peak_x = x[list(normal_dist(x, *params)).index(peak_y)]

    # ピーク値をプロットする
    plt.plot(peak_x, peak_y, 'ro', label=f'Peak: x = {peak_x:.2f}, y = {peak_y:.2f}')

    # データをプロットする
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], label='Data')

    # グラフのタイトルとラベルを設定する
    plt.title('Particle Size Distribution with Normal Fit and Peak')
    plt.xlabel('Particle Diameter')
    plt.ylabel('Frequency')
    plt.legend()

    # グラフをPNGファイルとして保存する
    # 処理したxlsxファイル名をPNGファイル名に追加
    plt.savefig(os.path.join(directory_name, f'{os.path.splitext(file)[0]}_distribution_with_fit_and_peak.png'))

    # グラフを表示する
    plt.show()
    # 次のファイルのグラフをクリアする
    plt.clf()
