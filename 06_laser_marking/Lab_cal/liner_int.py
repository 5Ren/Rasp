import pandas as pd
import numpy as np

# 元のCSV読み込み
cmf = pd.read_csv('./color_matching_functions.csv')
d65 = pd.read_csv('./D65_distribution.csv')

# 新しい波長軸（0.5nm刻み）
wavelengths_interp = np.arange(380, 780.5, 0.5)

# 線形補間
cmf_interp = pd.DataFrame({
    'wavelength': wavelengths_interp,
    'X': np.interp(wavelengths_interp, cmf['wavelength'], cmf['X']),
    'Y': np.interp(wavelengths_interp, cmf['wavelength'], cmf['Y']),
    'Z': np.interp(wavelengths_interp, cmf['wavelength'], cmf['Z']),
})

d65_interp = pd.DataFrame({
    'wavelength': wavelengths_interp,
    'intensity': np.interp(wavelengths_interp, d65['wavelength'], d65['intensity'])
})

# 保存
cmf_interp_path = './color_matching_functions_interp.csv'
d65_interp_path = './D65_distribution_interp.csv'

cmf_interp.to_csv(cmf_interp_path, index=False)
d65_interp.to_csv(d65_interp_path, index=False)

cmf_interp_path, d65_interp_path
