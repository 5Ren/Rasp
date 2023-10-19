import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# CSVファイルの読み込み
data = np.genfromtxt('test1.py', delimiter=',', encoding='utf-8')

print(f'{data=}')
# データをxとyに分割
x = data[0, :]
y = data[1, :]

# データのプロット
plt.scatter(x, y, label='データ')
plt.xlabel('x軸の値')
plt.ylabel('y軸の値')

# フィット関数の提案とフィッティング
# 以下の例では、指数関数 y = a * exp(b * x) を提案してフィットしてみます
def exponential_fit(x, a, b):
    return a * np.exp(b * x)

# パラメータの初期値を設定
initial_guess = [1.0, 1.0]

# 最適なパラメータをフィット
params, covariance = curve_fit(exponential_fit, x, y, p0=initial_guess)

# フィッティングされたパラメータを取得
a, b = params

# フィッティングされた式をプロット
x_fit = np.linspace(min(x), max(x), 100)  # フィッティングされた式の描画用データ
y_fit = exponential_fit(x_fit, a, b)
plt.plot(x_fit, y_fit, 'r-', label=f'フィット: y = {a:.2f} * exp({b:.2f} * x)')

plt.legend()
plt.grid(True)
plt.show()

# フィットした式を表示
print(f'フィットした式: y = {a:.2f} * exp({b:.2f} * x)')
