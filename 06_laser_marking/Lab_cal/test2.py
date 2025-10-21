import numpy as np
import pandas as pd
import os
from glob import glob

# ---------- 読み込み関数 ----------
def load_sample_txt(filepath):
    # ヘッダーなしで2列を読み込み（wavelength, reflectance）
    df = pd.read_csv(filepath, header=None, names=['wavelength', 'reflectance'], encoding='cp932')

    # 数値に強制変換（型不一致や文字列混入への保険）
    df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
    df['reflectance'] = pd.to_numeric(df['reflectance'], errors='coerce')
    df.dropna(inplace=True)

    # 明示的なfloat変換
    df['wavelength'] = df['wavelength'].astype(float)
    df['reflectance'] = df['reflectance'].astype(float)

    # ★ 5nm刻みに間引く（小数誤差に注意してroundで対応）
    df = df[np.round(df['wavelength'] % 5, 3) == 0]

    # R% → 0–1 に変換
    df['reflectance'] = df['reflectance'] / 100.0

    return df



# ---------- f(t), XYZ, Lab変換関数 ----------
def f(t):
    delta = 6/29
    return t**(1/3) if t > delta**3 else t/(3*delta**2) + 4/29

def xyz_to_lab(X, Y, Z):
    Xn, Yn, Zn = 95.047, 100.000, 108.883
    L = 116 * f(Y/Yn) - 16
    a = 500 * (f(X/Xn) - f(Y/Yn))
    b = 200 * (f(Y/Yn) - f(Z/Zn))
    return L, a, b

def compute_xyz(reflectance, d65, cmf):
    cie_y_sum = np.sum(d65['intensity'] * cmf['Y'])
    X = np.sum(reflectance * d65['intensity'] * cmf['X']) / cie_y_sum
    Y = np.sum(reflectance * d65['intensity'] * cmf['Y']) / cie_y_sum
    Z = np.sum(reflectance * d65['intensity'] * cmf['Z']) / cie_y_sum
    return X, Y, Z

# ---------- パス設定 ----------
input_folder = './ref_data'  # 必要に応じて変更
cmf_file = './color_matching_functions.csv'
d65_file = './D65_distribution.csv'

# ---------- 標準データの読み込み ----------
cmf = pd.read_csv(cmf_file, names=['wavelength', 'X', 'Y', 'Z'])
d65 = pd.read_csv(d65_file, names=['wavelength', 'intensity'])

# CIE 等色関数を float に変換
for col in cmf.columns:
    cmf[col] = pd.to_numeric(cmf[col], errors='coerce')

# D65 分光強度も float に変換 ←★ここが必要！
for col in d65.columns:
    d65[col] = pd.to_numeric(d65[col], errors='coerce')

# NaN 行を削除
cmf.dropna(inplace=True)
d65.dropna(inplace=True)

# ---------- 各ファイルを処理 ----------
file_paths = glob(os.path.join(input_folder, '*.txt'))
results = []

for path in file_paths:
    name = os.path.basename(path)
    try:
        sample = load_sample_txt(path)

        print(f"--- DEBUG: {name} ---")
        print("sample reflectance dtype :", sample['reflectance'].dtype)
        print("d65 intensity dtype       :", d65['intensity'].dtype)
        print("cmf X dtype               :", cmf['X'].dtype)
        print("type(reflectance):", type(sample['reflectance'].values))


        # 波長に合わせて補間し、CIEとD65と合成
        ref_interp = pd.merge(cmf[['wavelength']], sample, on='wavelength', how='left')
        ref_interp['reflectance'] = ref_interp['reflectance'].interpolate(method='linear')

        X, Y, Z = compute_xyz(ref_interp['reflectance'].values, d65, cmf)
        L, a, b = xyz_to_lab(X, Y, Z)

        results.append({'filename': name, 'L*': L, 'a*': a, 'b*': b})
        print(f"{name}: L*={L:.2f}, a*={a:.2f}, b*={b:.2f}")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

# ---------- 結果の保存 ----------
result_df = pd.DataFrame(results)
result_df.to_csv('lab_results2.csv', index=False)
print("✔ Lab結果を lab_results.csv に保存しました。")
