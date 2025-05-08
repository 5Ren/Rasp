import numpy as np
from scipy.optimize import fsolve

# 与えられた値
V_uL = 2  # μL
V = V_uL * 1e-3  # mm^3 に変換 (1 μL = 1 mm^3)
theta_deg = 92.1
theta_rad = np.deg2rad(theta_deg)

# 球キャップの高さ h を R の関数として表現
def h_from_R(R, theta):
    return R * (1 - np.cos(theta))

# 球キャップの体積 V の式
def volume_equation(R, V, theta):
    h = h_from_R(R, theta)
    return (np.pi * h**2 / 3) * (3 * R - h) - V

# R を数値的に解く
R_initial_guess = 1.0  # 初期推定
R_solution = fsolve(volume_equation, R_initial_guess, args=(V, theta_rad))[0]

# 接触面半径 a を求める
h = h_from_R(R_solution, theta_rad)
a = np.sqrt(2 * R_solution * h - h**2)
A = np.pi * a**2  # 接触面積

print(f'{R_solution}, \n{h}, \n{a}, \n{A}')
