"""
単純支持ばりの疑似FEMコンター表示

解析方法:
- Euler-Bernoulli 梁理論による解析解
- 梁長手方向 x と断面高さ方向 y に展開
- 曲げ応力 sigma_x
- せん断応力 tau_xy
- von Mises 相当応力
- 変形後形状

注意:
これは2次元連続体FEMではなく、梁理論の結果をFEM風に可視化したものです。
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Beam:
    length: float
    width: float
    height: float
    young_modulus: float

    @property
    def area(self):
        return self.width * self.height

    @property
    def second_moment(self):
        return self.width * self.height**3 / 12.0


def point_load_response(beam, load, load_position, nx=501):
    L = beam.length
    E = beam.young_modulus
    I = beam.second_moment
    P = load
    a = load_position
    b = L - a

    if not 0 < a < L:
        raise ValueError("load_position は 0 < a < L としてください。")

    x = np.linspace(0.0, L, nx)

    R_left = P * b / L
    R_right = P * a / L

    shear = np.where(x < a, R_left, -R_right)

    moment = np.where(
        x <= a,
        R_left * x,
        R_right * (L - x)
    )

    deflection = np.empty_like(x)

    left = x <= a
    right = ~left

    deflection[left] = (
        -P * b * x[left]
        / (6.0 * L * E * I)
        * (L**2 - b**2 - x[left]**2)
    )

    deflection[right] = (
        -P * a * (L - x[right])
        / (6.0 * L * E * I)
        * (L**2 - a**2 - (L - x[right])**2)
    )

    return x, deflection, moment, shear, R_left, R_right


def uniform_load_response(beam, distributed_load, nx=501):
    L = beam.length
    E = beam.young_modulus
    I = beam.second_moment
    w = distributed_load

    x = np.linspace(0.0, L, nx)

    R_left = w * L / 2.0
    R_right = R_left

    shear = R_left - w * x
    moment = R_left * x - w * x**2 / 2.0

    deflection = (
        -w * x * (L**3 - 2.0 * L * x**2 + x**3)
        / (24.0 * E * I)
    )

    return x, deflection, moment, shear, R_left, R_right


def create_stress_fields(beam, x, moment, shear, ny=101):
    """
    梁長手方向 x、断面高さ方向 y の応力場を作成する。

    曲げ応力:
        sigma_x = -M y / I

    矩形断面のせん断応力:
        tau_xy = 3V/(2A) * (1 - (2y/h)^2)
    """
    y = np.linspace(-beam.height / 2.0, beam.height / 2.0, ny)
    X, Y = np.meshgrid(x, y)

    M = moment[np.newaxis, :]
    V = shear[np.newaxis, :]

    sigma_x = -M * Y / beam.second_moment

    tau_xy = (
        1.5 * V / beam.area
        * (1.0 - (2.0 * Y / beam.height)**2)
    )

    # 平面応力状態で sigma_y = 0 とした von Mises 応力
    von_mises = np.sqrt(sigma_x**2 + 3.0 * tau_xy**2)

    return X, Y, sigma_x, tau_xy, von_mises


def deformed_coordinates(X, Y, x, deflection, scale):
    """
    梁の断面は回転せず、そのまま鉛直方向へ移動すると仮定した簡易表示。
    """
    W = np.tile(deflection, (Y.shape[0], 1))
    X_def = X
    Y_def = Y + scale * W
    return X_def, Y_def


def symmetric_limits(field):
    value = np.max(np.abs(field))
    return -value, value


def plot_contours(
    beam,
    X,
    Y,
    sigma_x,
    tau_xy,
    von_mises,
    x,
    deflection,
    moment,
    shear,
    deformation_scale=20.0
):
    X_mm = X * 1e3
    Y_mm = Y * 1e3

    X_def, Y_def = deformed_coordinates(
        X, Y, x, deflection, deformation_scale
    )

    X_def_mm = X_def * 1e3
    Y_def_mm = Y_def * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 曲げ応力
    vmin, vmax = symmetric_limits(sigma_x / 1e6)
    pcm1 = axes[0, 0].pcolormesh(
        X_mm,
        Y_mm,
        sigma_x / 1e6,
        shading="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax
    )
    axes[0, 0].set_title("Bending normal stress $\\sigma_x$")
    axes[0, 0].set_xlabel("Longitudinal position x [mm]")
    axes[0, 0].set_ylabel("Height y [mm]")
    axes[0, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(pcm1, ax=axes[0, 0], label="Stress [MPa]")

    # せん断応力
    vmin, vmax = symmetric_limits(tau_xy / 1e6)
    pcm2 = axes[0, 1].pcolormesh(
        X_mm,
        Y_mm,
        tau_xy / 1e6,
        shading="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax
    )
    axes[0, 1].set_title("Shear stress $\\tau_{xy}$")
    axes[0, 1].set_xlabel("Longitudinal position x [mm]")
    axes[0, 1].set_ylabel("Height y [mm]")
    axes[0, 1].set_aspect("equal", adjustable="box")
    fig.colorbar(pcm2, ax=axes[0, 1], label="Stress [MPa]")

    # von Mises
    pcm3 = axes[1, 0].pcolormesh(
        X_mm,
        Y_mm,
        von_mises / 1e6,
        shading="auto",
        cmap="viridis"
    )
    axes[1, 0].set_title("von Mises equivalent stress")
    axes[1, 0].set_xlabel("Longitudinal position x [mm]")
    axes[1, 0].set_ylabel("Height y [mm]")
    axes[1, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(pcm3, ax=axes[1, 0], label="Equivalent stress [MPa]")

    # 変形後形状を von Mises 応力で着色
    pcm4 = axes[1, 1].pcolormesh(
        X_def_mm,
        Y_def_mm,
        von_mises / 1e6,
        shading="auto",
        cmap="viridis"
    )

    axes[1, 1].plot(
        x * 1e3,
        deflection * deformation_scale * 1e3,
        linewidth=1.5,
        label="Neutral axis"
    )

    axes[1, 1].set_title(
        f"Deformed shape, deformation scale = {deformation_scale:.1f}"
    )
    axes[1, 1].set_xlabel("Longitudinal position x [mm]")
    axes[1, 1].set_ylabel("Deformed height [mm]")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].legend()
    fig.colorbar(pcm4, ax=axes[1, 1], label="von Mises stress [MPa]")

    fig.tight_layout()
    plt.show()

    # 追加の長手方向グラフ
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes2[0].plot(x * 1e3, deflection * 1e3)
    axes2[0].set_ylabel("Deflection [mm]")
    axes2[0].grid(True)

    axes2[1].plot(x * 1e3, moment)
    axes2[1].set_ylabel("Moment [N m]")
    axes2[1].grid(True)

    axes2[2].plot(x * 1e3, shear)
    axes2[2].set_xlabel("Longitudinal position x [mm]")
    axes2[2].set_ylabel("Shear force [N]")
    axes2[2].grid(True)

    fig2.tight_layout()
    plt.show()


def print_summary(
    beam,
    x,
    deflection,
    sigma_x,
    tau_xy,
    von_mises,
    R_left,
    R_right
):
    max_def_index = np.argmax(np.abs(deflection))
    max_sigma_index = np.unravel_index(
        np.argmax(np.abs(sigma_x)),
        sigma_x.shape
    )
    max_tau_index = np.unravel_index(
        np.argmax(np.abs(tau_xy)),
        tau_xy.shape
    )
    max_vm_index = np.unravel_index(
        np.argmax(von_mises),
        von_mises.shape
    )

    print("=" * 60)
    print("Calculation summary")
    print("=" * 60)
    print(f"Area                    : {beam.area:.6e} m^2")
    print(f"Second moment of area   : {beam.second_moment:.6e} m^4")
    print(f"Left reaction           : {R_left:.3f} N")
    print(f"Right reaction          : {R_right:.3f} N")
    print(
        f"Maximum deflection      : "
        f"{deflection[max_def_index] * 1e3:.6f} mm "
        f"at x = {x[max_def_index] * 1e3:.3f} mm"
    )
    print(
        f"Maximum bending stress  : "
        f"{np.abs(sigma_x[max_sigma_index]) / 1e6:.6f} MPa"
    )
    print(
        f"Maximum shear stress    : "
        f"{np.abs(tau_xy[max_tau_index]) / 1e6:.6f} MPa"
    )
    print(
        f"Maximum von Mises stress: "
        f"{von_mises[max_vm_index] / 1e6:.6f} MPa"
    )
    print("=" * 60)


def main():
    # ==========================================================
    # 梁条件
    # ==========================================================
    beam = Beam(
        length=0.150,  # 150 mm
        width=240e-6,  # 240 µm
        height=60e-6,  # 60 µm
        young_modulus=68e9  # Al: 68 GPa
    )

    # "point" または "uniform"
    load_type = "uniform"

    # 集中荷重
    point_load = 1000.0
    point_load_position = 0.150

    # 等分布荷重
    uniform_load = 49

    # 変形表示倍率
    deformation_scale = 30.0

    # ==========================================================
    # 計算
    # ==========================================================
    if load_type == "point":
        x, deflection, moment, shear, R_left, R_right = point_load_response(
            beam,
            point_load,
            point_load_position
        )

    elif load_type == "uniform":
        x, deflection, moment, shear, R_left, R_right = uniform_load_response(
            beam,
            uniform_load
        )

    else:
        raise ValueError('load_type は "point" または "uniform" としてください。')

    X, Y, sigma_x, tau_xy, von_mises = create_stress_fields(
        beam,
        x,
        moment,
        shear
    )

    print_summary(
        beam,
        x,
        deflection,
        sigma_x,
        tau_xy,
        von_mises,
        R_left,
        R_right
    )

    plot_contours(
        beam,
        X,
        Y,
        sigma_x,
        tau_xy,
        von_mises,
        x,
        deflection,
        moment,
        shear,
        deformation_scale
    )


if __name__ == "__main__":
    main()
