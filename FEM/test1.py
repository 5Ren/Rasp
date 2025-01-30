import numpy as np
from sfepy.discrete import Problem
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete import Variables
from sfepy.discrete.conditions import EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term
from sfepy.discrete.integrals import Integral
from sfepy.discrete.variables import FieldVariable
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.base.conf import ProblemConf, get_standard_keywords
from sfepy.base.base import IndexedStruct
from sfepy.postprocess.viewer import Viewer
import matplotlib.pyplot as plt

# 1. メッシュの定義
mesh = Mesh.from_file('meshes/rectangle.mesh')  # 2Dの矩形メッシュ

# 2. ドメインの作成
domain = FEDomain('domain', mesh)
omega = domain.create_region('Omega', 'all')

# 3. フィールドの定義 (速度場と圧力場)
field_velocity = Field.from_args('velocity', np.float64, 'vector', omega, approx_order=1)
field_pressure = Field.from_args('pressure', np.float64, 'scalar', omega, approx_order=1)

# 4. 変数の定義
v = FieldVariable('v', 'unknown', field_velocity)  # 速度場
p = FieldVariable('p', 'unknown', field_pressure)  # 圧力場
u = FieldVariable('u', 'test', field_velocity)     # 試験関数 (速度)
q = FieldVariable('q', 'test', field_pressure)     # 試験関数 (圧力)

# 5. 積分の設定
integral = Integral('i', order=2)

# 6. Navier-Stokes方程式の定義 (弱形式)
nu = 0.001  # 粘性係数 (流体の粘度)
f = np.array([0.0, 0.0])  # 外力

# 弱形式の各項 (粘性、圧力、外力)
term1 = Term.new('dw_div_grad(v, u)', integral, omega, v=v, u=u, coef=nu)
term2 = Term.new('dw_stokes(p, u)', integral, omega, v=v, p=p)
term3 = Term.new('dw_mass_vector(v, u)', integral, omega, v=v, u=f)

# 全体の方程式
equations = {
    'balance_of_momentum': term1 + term2 + term3
}

# 7. 境界条件の設定
inlet_velocity = EssentialBC('inlet', domain, {'v.0': 1.0, 'v.1': 0.0}, region='near(x[0], 0)')
wall_velocity = EssentialBC('walls', domain, {'v.all': 0.0}, region='near(x[1], 0) || near(x[1], 1)')
outlet_pressure = EssentialBC('outlet', domain, {'p.all': 0.0}, region='near(x[0], 1)')

# 境界条件の適用
bcs = [inlet_velocity, wall_velocity, outlet_pressure]

# 8. 問題の作成
pb = Problem('navier_stokes', equations=equations, nls=Newton({'i_max': 10}), ls=ScipyDirect({}), bc=bcs)

# 9. ソルバーの実行
status = IndexedStruct()
pb.time_update()
pb.solve()

# 10. 結果の可視化
viewer = Viewer(pb.get_output_name())
viewer()

# プロット
v_data = v.get_data().reshape(-1, 2)
plt.quiver(v_data[:, 0], v_data[:, 1], angles='xy', scale_units='xy', scale=1)
plt.title('Velocity Field')
plt.show()
