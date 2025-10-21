import numpy as np
import pandas as pd
import ace_tools as tools
# 再掲：ITOに対するハマカー定数の推定

# 定数
k_B = 1.38e-23     # ボルツマン定数 [J/K]
T = 300            # 温度 [K]
kT = k_B * T

# 誘電率の代表値（静的近似）
epsilon_1 = 3.9     # 基板（SiO2）
epsilon_2 = 10.0    # 膜（ITO）
epsilon_3 = 1.0     # 上部媒質（空気）

# ハマカー関数部分：φ = ((ε1 - ε2)/(ε1 + ε2)) * ((ε3 - ε2)/(ε3 + ε2))
phi = ((epsilon_1 - epsilon_2) / (epsilon_1 + epsilon_2)) * ((epsilon_3 - epsilon_2) / (epsilon_3 + epsilon_2))

# ハマカー定数の近似式（A ≈ (3/4) kT * φ）
A = (3/4) * kT * phi  # [J]

# 表面張力（単位を N/m = J/m² に）
gamma = 25e-3  # 25 mN/m

# 膜厚リスト [m]（50, 150, 300 nm）
h_list_nm = np.array([50, 150, 300])
h_list_m = h_list_nm * 1e-9

# スピノーダル脱湿における特徴長スケール Λ(h) の計算式
# Λ(h) = sqrt(16 * π^3 * γ / A) * h^2
factor = np.sqrt(16 * np.pi**3 * gamma / A)
Lambda = factor * h_list_m**2  # 単位は [m]

# 結果をnm単位に変換
Lambda_nm = Lambda * 1e9


df = pd.DataFrame({
    "膜厚 h [nm]": h_list_nm,
    "Λ(h) [nm]": Lambda_nm.round(1)
})

tools.display_dataframe_to_user(name="スピノーダル脱湿に基づく Λ(h) の理論値（ITO）", dataframe=df)
