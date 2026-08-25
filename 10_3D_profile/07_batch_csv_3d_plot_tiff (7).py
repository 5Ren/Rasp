#!/usr/bin/env python3
"""CSV height data in a selected folder -> consistently formatted 3D TIFFs.

PyCharm から引数なしで実行するとフォルダー選択画面が開きます。
コマンドラインでは、入力フォルダーを第1引数に指定することもできます。

入力CSVの仕様:
    1024 x 1024 = 1,048,576個の数値が、カンマ区切りで格納されていること。
    1行に平坦化されたCSVと、1024行 x 1024列のCSVの両方に対応します。

出力:
    選択フォルダー内に 3d_plot_YYYYMMDD_HHMMSS フォルダーを作成し、
    各CSVを「元のファイル名_3d_plot.tif」として保存します。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

# 画像だけを作るバックエンドを明示し、PyCharmの表示環境に依存しないようにする。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np


# -----------------------------------------------------------------------------
# 作図設定
# -----------------------------------------------------------------------------
GRID_SIZE = 1024
EXPECTED_VALUE_COUNT = GRID_SIZE * GRID_SIZE

DPI = 800
FIGURE_WIDTH_MM = 170.0
FIGURE_HEIGHT_MM = 110.0

# 1024 x 1024の全点を使って描画する。描画後に800 dpiで保存する。
SURFACE_STRIDE = 1

# 下側の青は維持し、上側だけ余裕を増やして先端の色差を残す。
COLOR_RANGE_LOWER_MARGIN_FRACTION = 0.02
COLOR_RANGE_UPPER_MARGIN_FRACTION = 0.02

# すべての画像で視点を固定する。
VIEW_ELEVATION_DEG = 35.0
VIEW_AZIMUTH_DEG = -52.0

# レーザー顕微鏡の表示スケールに近い配色。
# 下側に設定した色範囲の余白により、実データの最小値は黒ではなく濃い紫になる。
HEIGHT_COLORMAP = LinearSegmentedColormap.from_list(
    "microscope_height",
    (
        (0.00, "#19001F"),  # 黒紫
        (0.06, "#66007F"),  # 濃い紫
        (0.13, "#8B00B5"),  # 紫
        (0.20, "#3F00CC"),  # 青紫
        (0.28, "#001FEF"),  # 濃い青
        (0.38, "#0076FF"),  # 青
        (0.48, "#00C9E8"),  # 水色
        (0.56, "#00D45A"),  # 緑
        (0.66, "#39EA00"),  # 黄緑
        (0.77, "#F4F000"),  # 黄色
        (0.86, "#FF8C00"),  # オレンジ
        (0.94, "#FF2F00"),  # 赤
        (1.00, "#D90000"),  # 濃い赤
    ),
    N=256,
)
AXIS_LINE_WIDTH_PT = 0.5

# X:Y:Z の見かけの比率を固定する。
Z_BOX_ASPECT = 0.30


class Arrow3D(FancyArrowPatch):
    """2点の3D座標を結ぶ、矢尻が明瞭な矢印。"""

    def __init__(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        **kwargs: object,
    ) -> None:
        super().__init__((0.0, 0.0), (0.0, 0.0), **kwargs)
        self._vertices_3d = (start, end)

    def _update_projected_positions(self) -> np.ndarray:
        """現在の3D表示行列で始点と終点を2Dへ投影する。"""

        start, end = self._vertices_3d
        x_3d = (start[0], end[0])
        y_3d = (start[1], end[1])
        z_3d = (start[2], end[2])
        x_2d, y_2d, z_2d = proj3d.proj_transform(
            x_3d,
            y_3d,
            z_3d,
            self.axes.get_proj(),
        )
        self.set_positions((x_2d[0], y_2d[0]), (x_2d[1], y_2d[1]))
        return z_2d

    def draw(self, renderer: object) -> None:
        self._update_projected_positions()
        super().draw(renderer)

    def do_3d_projection(self, renderer: object | None = None) -> float:
        z_2d = self._update_projected_positions()
        return float(np.min(z_2d))


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(
        description="フォルダー内の1024 x 1024 CSVを3D TIFFへ一括変換します。"
    )
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        help="CSVファイルが入っているフォルダー。省略すると選択画面を表示します。",
    )
    return parser.parse_args(argv)


def select_input_directory() -> Path | None:
    """GUIで入力フォルダーを選択する。キャンセル時はNoneを返す。"""

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        parent=root,
        title="CSVファイルが入っているフォルダーを選択してください",
        mustexist=True,
    )
    root.destroy()

    return Path(selected) if selected else None


def show_result_dialog(title: str, message: str, *, warning: bool = False) -> None:
    """GUI起動時に処理結果をダイアログで知らせる。"""

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if warning:
        messagebox.showwarning(title, message, parent=root)
    else:
        messagebox.showinfo(title, message, parent=root)
    root.destroy()


def create_timestamped_output_directory(input_directory: Path) -> Path:
    """日時を含む重複しない出力フォルダーを作成する。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"3d_plot_{timestamp}"
    output_directory = input_directory / base_name

    suffix = 1
    while output_directory.exists():
        output_directory = input_directory / f"{base_name}_{suffix:02d}"
        suffix += 1

    output_directory.mkdir(parents=False, exist_ok=False)
    return output_directory


def load_height_data(csv_path: Path) -> np.ndarray:
    """CSVの数値を読み込み、1024 x 1024の高さ配列へ変換する。"""

    # 改行区切りもカンマ区切りとして扱うため、1行の平坦化CSVと
    # 通常の1024行 x 1024列CSVを同じ方法で読み込む。
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    normalized_text = (
        csv_text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", ",")
        .strip(" \t,")
    )
    values = np.fromstring(normalized_text, dtype=np.float64, sep=",")

    if values.size != EXPECTED_VALUE_COUNT:
        raise ValueError(
            f"要素数が不正です。期待値={EXPECTED_VALUE_COUNT:,}, "
            f"実測値={values.size:,}"
        )

    height = values.reshape(GRID_SIZE, GRID_SIZE)
    if not np.isfinite(height).any():
        raise ValueError("有限の数値が1つもありません。")

    return height


def add_arrow_axes(
    axis: matplotlib.axes.Axes,
    *,
    x_max: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """目盛りやラベルを持たない黒色の矢印軸を描く。"""

    z_range = max(z_max - z_min, np.finfo(float).eps)
    z_base = z_min - 0.045 * z_range

    # 矢尻を画素数に依存しない大きさで描き、800 dpiでも明瞭にする。
    origin = (0.0, 0.0, z_base)
    # Y軸は右外周へ置き、表面を横切る黒線が生じないようにする。
    arrow_segments = (
        (origin, (x_max, 0.0, z_base)),
        ((x_max, 0.0, z_base), (x_max, y_max, z_base)),
        (origin, (0.0, 0.0, z_base + 1.08 * z_range)),
    )
    for startpoint, endpoint in arrow_segments:
        arrow = Arrow3D(
            startpoint,
            endpoint,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=AXIS_LINE_WIDTH_PT,
            color="black",
            shrinkA=0.0,
            shrinkB=0.0,
            clip_on=False,
            zorder=1000,
        )
        axis.add_artist(arrow)

    axis.set_zlim(z_base, z_max + 0.12 * z_range)


def save_3d_tiff(height: np.ndarray, output_path: Path) -> None:
    """高さ配列を指定フォーマットの800 dpi TIFFとして保存する。"""

    finite_values = height[np.isfinite(height)]
    z_min = float(np.min(finite_values))
    z_max = float(np.max(finite_values))

    # 全点が同じ値でもNormalizeが成立するように、色範囲だけ微小に広げる。
    if np.isclose(z_min, z_max):
        color_margin = max(abs(z_min) * 1.0e-9, 1.0e-9)
        norm = Normalize(vmin=z_min - color_margin, vmax=z_max + color_margin)
    else:
        color_range = z_max - z_min
        lower_margin = color_range * COLOR_RANGE_LOWER_MARGIN_FRACTION
        upper_margin = color_range * COLOR_RANGE_UPPER_MARGIN_FRACTION
        norm = Normalize(vmin=z_min - lower_margin, vmax=z_max + upper_margin)

    x = np.arange(0, GRID_SIZE, SURFACE_STRIDE, dtype=np.float64)
    y = np.arange(0, GRID_SIZE, SURFACE_STRIDE, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    zz = height[::SURFACE_STRIDE, ::SURFACE_STRIDE]

    figure_size_inches = (
        FIGURE_WIDTH_MM / 25.4,
        FIGURE_HEIGHT_MM / 25.4,
    )
    figure = plt.figure(figsize=figure_size_inches, facecolor="white")

    try:
        axis = figure.add_subplot(111, projection="3d")
        axis.set_facecolor("white")
        axis.set_axis_off()
        axis.set_proj_type("persp", focal_length=1.0)

        axis.plot_surface(
            xx,
            yy,
            zz,
            cmap=HEIGHT_COLORMAP,
            norm=norm,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )

        axis.set_xlim(0.0, GRID_SIZE - 1.0)
        axis.set_ylim(0.0, GRID_SIZE - 1.0)
        axis.view_init(elev=VIEW_ELEVATION_DEG, azim=VIEW_AZIMUTH_DEG)
        axis.set_box_aspect((1.0, 1.0, Z_BOX_ASPECT), zoom=1.00)

        add_arrow_axes(
            axis,
            x_max=GRID_SIZE - 1.0,
            y_max=GRID_SIZE - 1.0,
            z_min=z_min,
            z_max=z_max,
        )

        # 余白を一定にし、すべての画像で面の大きさと位置をそろえる。
        axis.set_position((0.015, 0.015, 0.970, 0.970))

        figure.savefig(
            output_path,
            format="tiff",
            dpi=DPI,
            facecolor="white",
            edgecolor="none",
            transparent=False,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    finally:
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    """フォルダー内のCSVを一括処理する。"""

    args = parse_arguments(argv)
    used_dialog = args.input_directory is None
    input_directory = args.input_directory

    if input_directory is None:
        input_directory = select_input_directory()
        if input_directory is None:
            print("フォルダー選択がキャンセルされました。")
            return 0

    input_directory = input_directory.expanduser().resolve()
    if not input_directory.is_dir():
        print(f"入力フォルダーが見つかりません: {input_directory}", file=sys.stderr)
        return 1

    # 選択フォルダー直下のCSVのみを対象にする。
    csv_files = sorted(input_directory.glob("*.csv"), key=lambda path: path.name.lower())
    if not csv_files:
        message = f"CSVファイルが見つかりませんでした。\n\n{input_directory}"
        print(message)
        if used_dialog:
            show_result_dialog("CSVがありません", message, warning=True)
        return 1

    output_directory = create_timestamped_output_directory(input_directory)
    print(f"入力フォルダー : {input_directory}")
    print(f"出力フォルダー : {output_directory}")
    print(f"対象CSV数       : {len(csv_files)}")
    print()

    succeeded = 0
    failures: list[str] = []

    for index, csv_path in enumerate(csv_files, start=1):
        output_path = output_directory / f"{csv_path.stem}_3d_plot.tif"
        print(f"[{index}/{len(csv_files)}] {csv_path.name}")

        try:
            height = load_height_data(csv_path)
            finite_values = height[np.isfinite(height)]
            print(
                f"    shape={height.shape}, "
                f"min={np.min(finite_values):.6g}, "
                f"max={np.max(finite_values):.6g}"
            )
            save_3d_tiff(height, output_path)
            print(f"    保存完了: {output_path.name}")
            succeeded += 1
        except Exception as error:  # 1ファイルの失敗で全体を止めない。
            failure = f"{csv_path.name}: {error}"
            failures.append(failure)
            print(f"    エラー: {error}", file=sys.stderr)

    print()
    print("処理終了")
    print(f"成功: {succeeded} / {len(csv_files)}")
    print(f"失敗: {len(failures)} / {len(csv_files)}")
    print(f"保存先: {output_directory}")

    result_message = (
        f"処理が完了しました。\n\n"
        f"成功: {succeeded} / {len(csv_files)}\n"
        f"失敗: {len(failures)} / {len(csv_files)}\n\n"
        f"保存先:\n{output_directory}"
    )
    if failures:
        result_message += "\n\nエラー:\n" + "\n".join(failures[:8])
        if len(failures) > 8:
            result_message += f"\nほか {len(failures) - 8} 件"

    if used_dialog:
        show_result_dialog(
            "3D TIFF変換完了",
            result_message,
            warning=bool(failures),
        )

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
