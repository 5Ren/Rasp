import numpy as np
from scipy.optimize import least_squares

# データ (x, y, z, k)
# 未加工
# data = np.array([
#     [2.073085714, 0.12, 10.6812, 320.436],
#     [2.993142857, 0.12, 10.6812, 534.06],
#     [3.454285714, 0.12, 10.6812, 747.684]
# ])

# 化学処理
# data = np.array([
#     [2.389028571, 0.12, 10.6812, 320.436],
#     [3.354814286, 0.12, 10.6812, 534.06],
#     [3.972514286, 0.12, 10.6812, 747.684]
# ])

# 階層構造
data = np.array([
    [2.763942857, 0.12, 10.6812, 320.436],
    [3.868628571, 0.12, 10.6812, 534.06],
    [4.690028571, 0.12, 10.6812, 747.684]
])



x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
k = data[:, 3]

# モデル: Ax - ABy + zB
def model(params, x, y, z):
    A, B = params
    return A * x - A * B * y + z * B

# 残差関数（モデル出力と実測kの差）
def residuals(params, x, y, z, k):
    return model(params, x, y, z) - k

# 初期推定値（適当に）
initial_guess = [1.0, 1.0]

# 最小二乗フィッティング
result = least_squares(residuals, initial_guess, args=(x, y, z, k))

# 結果表示
A_fit, B_fit = result.x
print(f"推定された係数: A = {A_fit}, B = {B_fit}")
