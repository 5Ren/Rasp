"""
単純支持ばりのたわみ・曲げ応力・せん断応力の計算と可視化

前提
----
- 梁長さ L
- 矩形断面：幅 b、高さ h
- 線形弾性、小変形
- Euler-Bernoulli 梁理論
- 荷重形式：
    1. 中央集中荷重
    2. 任意位置の集中荷重
    3. 全長にわたる等分布荷重

単位は SI 単位系で入力します。
長さ: m
荷重: N
分布荷重: N/m
ヤング率: Pa
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Beam:
    length: float          # 梁長さ L [m]
    width: float           # 断面幅 b [m]
    height: float          # 断面高さ h [m]
    young_modulus: float   # ヤング率 E [Pa]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def second_moment(self) -> float:
        # 矩形断面の断面二次モーメント I = b h^3 / 12
        return self.width * self.height**3 / 12.0


def point_load_response(
    beam: Beam,
    load: float,
    load_position: float,
    n_points: int = 1001
):
    """
    単純支持ばりに任意位置 x=a で集中荷重 P が作用する場合。

    戻り値
    -------
    x          : 梁軸方向位置 [m]
    deflection : たわみ [m]（下向きを負）
    moment     : 曲げモーメント [N m]
    shear      : せん断力 [N]
    """
    L = beam.length
    E = beam.young_modulus
    I = beam.second_moment
    P = load
    a = load_position
    b_span = L - a

    if not 0.0 < a < L:
        raise ValueError("集中荷重位置は 0 < load_position < L としてください。")

    x = np.linspace(0.0, L, n_points)

    # 支点反力
    reaction_left = P * b_span / L
    reaction_right = P * a / L

    # せん断力・曲げモーメント
    shear = np.where(x < a, reaction_left, -reaction_right)
    moment = np.where(
        x <= a,
        reaction_left * x,
        reaction_right * (L - x)
    )

    # 任意位置集中荷重によるたわみ
    deflection = np.empty_like(x)

    left = x <= a
    right = ~left

    deflection[left] = (
        -P * b_span * x[left]
        / (6.0 * L * E * I)
        * (L**2 - b_span**2 - x[left]**2)
    )

    deflection[right] = (
        -P * a * (L - x[right])
        / (6.0 * L * E * I)
        * (L**2 - a**2 - (L - x[right])**2)
    )

    return x, deflection, moment, shear, reaction_left, reaction_right


def uniform_load_response(
    beam: Beam,
    distributed_load: float,
    n_points: int = 1001
):
    """
    単純支持ばりの全長に等分布荷重 w が作用する場合。
    """
    L = beam.length
    E = beam.young_modulus
    I = beam.second_moment
    w = distributed_load

    x = np.linspace(0.0, L, n_points)

    reaction_left = w * L / 2.0
    reaction_right = w * L / 2.0

    shear = reaction_left - w * x
    moment = reaction_left * x - w * x**2 / 2.0

    # y(x) = -w x (L^3 - 2Lx^2 + x^3)/(24EI)
    deflection = (
        -w * x * (L**3 - 2.0 * L * x**2 + x**3)
        / (24.0 * E * I)
    )

    return x, deflection, moment, shear, reaction_left, reaction_right


def calculate_stresses(beam: Beam, moment: np.ndarray, shear: np.ndarray):
    """
    各断面における最大曲げ応力と最大せん断応力を計算。

    曲げ応力:
        sigma_max = M(h/2)/I

    矩形断面の最大せん断応力:
        tau_max = 3V/(2A)
    """
    I = beam.second_moment
    A = beam.area
    c = beam.height / 2.0

    bending_stress_top = -moment * c / I
    bending_stress_bottom = moment * c / I
    max_shear_stress = 1.5 * shear / A

    return bending_stress_top, bending_stress_bottom, max_shear_stress


def shear_stress_over_height(beam: Beam, shear_force: float, n_points: int = 301):
    """
    矩形断面内のせん断応力分布。

    tau(y) = 3V/(2A) * (1 - (2y/h)^2)

    y=0 が中立軸、y=±h/2 が上下面。
    """
    h = beam.height
    A = beam.area

    y = np.linspace(-h / 2.0, h / 2.0, n_points)
    tau = 1.5 * shear_force / A * (1.0 - (2.0 * y / h)**2)

    return y, tau


def plot_results(
    beam: Beam,
    x: np.ndarray,
    deflection: np.ndarray,
    moment: np.ndarray,
    shear: np.ndarray,
    bending_top: np.ndarray,
    bending_bottom: np.ndarray,
    max_shear_stress: np.ndarray,
    title: str
):
    """
    計算結果を可視化する。
    """
    # 最大絶対せん断力となる断面の断面内分布
    critical_index = int(np.argmax(np.abs(shear)))
    critical_x = x[critical_index]
    critical_shear = shear[critical_index]
    y_section, tau_section = shear_stress_over_height(
        beam,
        critical_shear
    )

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    # 1. 変形形状
    axes[0, 0].plot(x * 1e3, deflection * 1e3)
    axes[0, 0].axhline(0.0, linewidth=0.8)
    axes[0, 0].set_title("Deflection")
    axes[0, 0].set_xlabel("Position x [mm]")
    axes[0, 0].set_ylabel("Deflection [mm]")
    axes[0, 0].grid(True)

    # 2. せん断力線図
    axes[0, 1].plot(x * 1e3, shear)
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].set_title("Shear force diagram")
    axes[0, 1].set_xlabel("Position x [mm]")
    axes[0, 1].set_ylabel("Shear force V [N]")
    axes[0, 1].grid(True)

    # 3. 曲げモーメント線図
    axes[1, 0].plot(x * 1e3, moment)
    axes[1, 0].axhline(0.0, linewidth=0.8)
    axes[1, 0].set_title("Bending moment diagram")
    axes[1, 0].set_xlabel("Position x [mm]")
    axes[1, 0].set_ylabel("Bending moment M [N m]")
    axes[1, 0].grid(True)

    # 4. 曲げ応力
    axes[1, 1].plot(
        x * 1e3,
        bending_top / 1e6,
        label="Top surface"
    )
    axes[1, 1].plot(
        x * 1e3,
        bending_bottom / 1e6,
        label="Bottom surface"
    )
    axes[1, 1].axhline(0.0, linewidth=0.8)
    axes[1, 1].set_title("Bending stress at outer surfaces")
    axes[1, 1].set_xlabel("Position x [mm]")
    axes[1, 1].set_ylabel("Bending stress [MPa]")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 5. 梁軸方向の最大せん断応力
    axes[2, 0].plot(x * 1e3, max_shear_stress / 1e6)
    axes[2, 0].axhline(0.0, linewidth=0.8)
    axes[2, 0].set_title("Maximum shear stress in each cross-section")
    axes[2, 0].set_xlabel("Position x [mm]")
    axes[2, 0].set_ylabel("Maximum shear stress [MPa]")
    axes[2, 0].grid(True)

    # 6. 断面高さ方向のせん断応力分布
    axes[2, 1].plot(tau_section / 1e6, y_section * 1e3)
    axes[2, 1].axvline(0.0, linewidth=0.8)
    axes[2, 1].set_title(
        f"Shear stress over height at x = {critical_x * 1e3:.1f} mm"
    )
    axes[2, 1].set_xlabel("Shear stress [MPa]")
    axes[2, 1].set_ylabel("Height coordinate y [mm]")
    axes[2, 1].grid(True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()


def print_summary(
    beam: Beam,
    x: np.ndarray,
    deflection: np.ndarray,
    moment: np.ndarray,
    shear: np.ndarray,
    bending_top: np.ndarray,
    bending_bottom: np.ndarray,
    max_shear_stress: np.ndarray,
    reaction_left: float,
    reaction_right: float
):
    max_deflection_index = int(np.argmax(np.abs(deflection)))
    max_moment_index = int(np.argmax(np.abs(moment)))
    max_shear_index = int(np.argmax(np.abs(shear)))

    max_bending = max(
        np.max(np.abs(bending_top)),
        np.max(np.abs(bending_bottom))
    )
    max_tau = np.max(np.abs(max_shear_stress))

    print("=" * 58)
    print("Calculation summary")
    print("=" * 58)
    print(f"Cross-sectional area A       : {beam.area:.6e} m^2")
    print(f"Second moment of area I      : {beam.second_moment:.6e} m^4")
    print(f"Left support reaction        : {reaction_left:.3f} N")
    print(f"Right support reaction       : {reaction_right:.3f} N")
    print()
    print(
        f"Maximum deflection           : "
        f"{deflection[max_deflection_index] * 1e3:.6f} mm "
        f"at x = {x[max_deflection_index] * 1e3:.3f} mm"
    )
    print(
        f"Maximum bending moment       : "
        f"{moment[max_moment_index]:.6f} N m "
        f"at x = {x[max_moment_index] * 1e3:.3f} mm"
    )
    print(
        f"Maximum absolute shear force : "
        f"{abs(shear[max_shear_index]):.6f} N"
    )
    print(f"Maximum bending stress       : {max_bending / 1e6:.6f} MPa")
    print(f"Maximum shear stress         : {max_tau / 1e6:.6f} MPa")
    print("=" * 58)


def main():
    # ==========================================================
    # 入力条件
    # ==========================================================
    beam = Beam(
        length=0.300,            # 梁長さ [m]
        width=0.020,             # 断面幅 [m]
        height=0.010,            # 断面高さ [m]
        young_modulus=205e9      # ヤング率 [Pa]（例：鋼）
    )

    # "point" または "uniform"
    load_type = "point"

    # 集中荷重の条件
    point_load = 1000.0              # 荷重 P [N]
    point_load_position = 0.150      # 荷重位置 a [m]

    # 等分布荷重の条件
    uniform_load = 3000.0            # 分布荷重 w [N/m]

    # ==========================================================
    # 計算
    # ==========================================================
    if load_type == "point":
        (
            x,
            deflection,
            moment,
            shear,
            reaction_left,
            reaction_right
        ) = point_load_response(
            beam=beam,
            load=point_load,
            load_position=point_load_position
        )

        title = (
            f"Simply supported rectangular beam: "
            f"point load P = {point_load:.1f} N"
        )

    elif load_type == "uniform":
        (
            x,
            deflection,
            moment,
            shear,
            reaction_left,
            reaction_right
        ) = uniform_load_response(
            beam=beam,
            distributed_load=uniform_load
        )

        title = (
            f"Simply supported rectangular beam: "
            f"uniform load w = {uniform_load:.1f} N/m"
        )

    else:
        raise ValueError('load_type は "point" または "uniform" としてください。')

    bending_top, bending_bottom, max_shear_stress = calculate_stresses(
        beam,
        moment,
        shear
    )

    print_summary(
        beam,
        x,
        deflection,
        moment,
        shear,
        bending_top,
        bending_bottom,
        max_shear_stress,
        reaction_left,
        reaction_right
    )

    plot_results(
        beam,
        x,
        deflection,
        moment,
        shear,
        bending_top,
        bending_bottom,
        max_shear_stress,
        title
    )


if __name__ == "__main__":
    main()
