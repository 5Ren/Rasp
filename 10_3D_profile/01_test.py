from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from matplotlib.widgets import RectangleSelector
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# True にすると，最初の1枚で選んだ ROI を全画像に共通適用します。
USE_COMMON_ROI = False

# 3Dプロットを描くときの間引き間隔です。大きいほど軽くなります。
DOWNSAMPLE_STEP = 4

# ROI選択時に，何も選ばず Enter を押した場合は画像全体を使います。
USE_FULL_IMAGE_IF_NO_ROI = True


def choose_folders() -> tuple[Path, Path]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    input_dir = filedialog.askdirectory(title="入力画像フォルダーを選択してください")
    if not input_dir:
        raise SystemExit("入力フォルダーが選択されませんでした。")

    output_dir = filedialog.askdirectory(title="出力先フォルダーを選択してください")
    if not output_dir:
        raise SystemExit("出力フォルダーが選択されませんでした。")

    return Path(input_dir), Path(output_dir)


def collect_image_paths(input_dir: Path) -> list[Path]:
    paths = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not paths:
        raise SystemExit("入力フォルダー内に対応画像がありません。")
    return paths


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def show_overview(image_paths: list[Path]) -> None:
    n = len(image_paths)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, path in zip(axes, image_paths):
        rgb = load_rgb_image(path)
        ax.imshow(rgb)
        ax.set_title(path.name, fontsize=9)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle("入力画像一覧（確認用）", fontsize=14)
    plt.tight_layout()
    plt.show()


def select_roi_interactively(
    rgb: np.ndarray,
    title: str,
) -> tuple[int, int, int, int] | None:
    """画像上で正方形のROIを選択する。"""

    height, width = rgb.shape[:2]

    state: dict[str, tuple[int, int, int, int] | bool | None] = {
        "roi": None,
        "skip": False,
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(
        title
        + "\nドラッグ: 正方形ROIを選択"
        + "  |  Enter: 決定"
        + "  |  A: 中央の最大正方形"
        + "  |  S: スキップ"
        + "  |  Esc: 終了"
    )

    def get_center_square_roi() -> tuple[int, int, int, int]:
        """画像中央から取得可能な最大の正方形ROIを返す。"""

        side = min(width, height)

        left = (width - side) // 2
        top = (height - side) // 2
        right = left + side
        bottom = top + side

        return left, top, right, bottom

    def on_select(eclick, erelease):
        """ドラッグ範囲を短辺基準の正方形に変換する。"""

        if eclick.xdata is None or eclick.ydata is None:
            return

        if erelease.xdata is None or erelease.ydata is None:
            return

        start_x = int(np.clip(round(eclick.xdata), 0, width - 1))
        start_y = int(np.clip(round(eclick.ydata), 0, height - 1))

        end_x = int(np.clip(round(erelease.xdata), 0, width - 1))
        end_y = int(np.clip(round(erelease.ydata), 0, height - 1))

        delta_x = end_x - start_x
        delta_y = end_y - start_y

        # 短い辺を正方形の一辺として使う
        side = min(abs(delta_x), abs(delta_y))

        if side < 5:
            print("選択範囲が小さすぎます。もう一度選択してください。")
            return

        # ドラッグ方向を維持したまま正方形化
        square_end_x = start_x + side if delta_x >= 0 else start_x - side
        square_end_y = start_y + side if delta_y >= 0 else start_y - side

        square_end_x = int(np.clip(square_end_x, 0, width))
        square_end_y = int(np.clip(square_end_y, 0, height))

        left = min(start_x, square_end_x)
        right = max(start_x, square_end_x)
        top = min(start_y, square_end_y)
        bottom = max(start_y, square_end_y)

        # 画像端でクリップされた場合にも、最終的に正方形にする
        final_side = min(right - left, bottom - top)

        if delta_x >= 0:
            right = left + final_side
        else:
            left = right - final_side

        if delta_y >= 0:
            bottom = top + final_side
        else:
            top = bottom - final_side

        left = int(np.clip(left, 0, width - final_side))
        top = int(np.clip(top, 0, height - final_side))
        right = left + final_side
        bottom = top + final_side

        state["roi"] = (left, top, right, bottom)

        # 表示されている選択枠も正方形に補正
        selector.extents = (left, right, top, bottom)

        print("----------------------------------------")
        print(f"正方形ROI: x={left}:{right}, y={top}:{bottom}")
        print(f"一辺      : {final_side} pixel")
        print(f"出力形状  : {final_side} × {final_side}")
        print("----------------------------------------")

        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        drag_from_anywhere=True,
    )

    def on_key(event):
        if event.key == "enter":
            if state["roi"] is None:
                if USE_FULL_IMAGE_IF_NO_ROI:
                    state["roi"] = get_center_square_roi()

                    left, top, right, bottom = state["roi"]
                    selector.extents = (left, right, top, bottom)

                    print("ROI未選択のため、中央の最大正方形を使用します。")
                    print(f"正方形ROI: x={left}:{right}, y={top}:{bottom}")

                else:
                    print("先に正方形ROIを選択してください。")
                    return

            plt.close(fig)

        elif event.key in {"a", "A"}:
            state["roi"] = get_center_square_roi()

            left, top, right, bottom = state["roi"]
            selector.extents = (left, right, top, bottom)

            print("中央の最大正方形を選択しました。")
            print(f"正方形ROI: x={left}:{right}, y={top}:{bottom}")
            print(f"一辺      : {right - left} pixel")
            print("Enterで確定してください。")

            fig.canvas.draw_idle()

        elif event.key in {"s", "S"}:
            state["skip"] = True
            plt.close(fig)

        elif event.key == "escape":
            plt.close("all")
            raise SystemExit("ユーザーが処理を終了しました。")

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    selector.set_active(False)

    if state["skip"]:
        return None

    return state["roi"]


def pseudocolor_to_height(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb_f)

    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    # 有彩色らしい画素だけをアンカー探索に使う
    valid = (sat > 0.25) & (val > 0.15)
    if not np.any(valid):
        gray = rgb_f.mean(axis=2)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-12)
        return gray

    flat_rgb = rgb_f.reshape(-1, 3)
    flat_hue = hue.reshape(-1)
    flat_valid = valid.reshape(-1)

    # 最も赤い画素と最も青い画素を自動検出
    red_score = flat_rgb[:, 0] - 0.5 * flat_rgb[:, 1] - 0.5 * flat_rgb[:, 2]
    blue_score = flat_rgb[:, 2] - 0.5 * flat_rgb[:, 0] - 0.5 * flat_rgb[:, 1]

    red_score[~flat_valid] = -np.inf
    blue_score[~flat_valid] = -np.inf

    red_idx = int(np.argmax(red_score))
    blue_idx = int(np.argmax(blue_score))

    red_hue = float(flat_hue[red_idx])
    blue_hue = float(flat_hue[blue_idx])

    # 青 -> 緑 -> 黄 -> 赤 になる向きで正規化する
    denom = (blue_hue - red_hue) % 1.0
    if denom < 1e-6:
        denom = 2.0 / 3.0  # だいたい blue(240°) -> red(0°)
        red_hue = 0.0
        blue_hue = 2.0 / 3.0

    height = ((blue_hue - hue) % 1.0) / denom
    height = np.clip(height, 0.0, 1.0)

    # 非常に暗い画素は底側へ寄せる（背景の黒ずみ対策）
    dark_mask = val < 0.08
    height[dark_mask] = 0.0

    # 念のため，画像内で 0～1 に再正規化
    hmin = float(np.min(height))
    hmax = float(np.max(height))
    if hmax - hmin > 1e-12:
        height = (height - hmin) / (hmax - hmin)
    else:
        height = np.zeros_like(height)

    return height.astype(np.float32)


def save_3d_surface_svg(
    height: np.ndarray,
    save_path: Path,
    title: str,
    step: int = 4,
) -> None:
    """高さ配列を測定ソフト風の3D表面図としてSVG保存する。"""

    z = height[::step, ::step]

    y = np.arange(z.shape[0])
    x = np.arange(z.shape[1])
    xx, yy = np.meshgrid(x, y)

    # Z方向を表示上だけ強調する
    z_exaggeration = 40.0
    z_display = z * z_exaggeration

    fig = plt.figure(
        figsize=(10, 10),
        facecolor="black",
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
        facecolor="black",
    )

    surface = ax.plot_surface(
        xx,
        yy,
        z_display,
        cmap="jet",
        vmin=0.0,
        vmax=z_exaggeration,
        linewidth=0,
        antialiased=False,
        shade=True,
        rcount=z.shape[0],
        ccount=z.shape[1],
    )

    # 視点を低くする
    ax.view_init(
        elev=28,
        azim=-55,
    )

    # 透視投影
    ax.set_proj_type("persp")

    # 表示範囲
    ax.set_xlim(0, z.shape[1] - 1)
    ax.set_ylim(z.shape[0] - 1, 0)
    ax.set_zlim(0, z_exaggeration)

    # X・Y・Zの見た目の比率
    ax.set_box_aspect(
        (
            z.shape[1],
            z.shape[0],
            min(z.shape) * 0.35,
        )
    )

    ax.set_xlabel(
        "X pixel",
        color="limegreen",
        labelpad=10,
    )
    ax.set_ylabel(
        "Y pixel",
        color="limegreen",
        labelpad=10,
    )
    ax.set_zlabel(
        "Normalized height",
        color="limegreen",
        labelpad=10,
    )

    ax.set_title(
        title,
        color="white",
        pad=18,
    )

    # 目盛りの色
    ax.tick_params(
        axis="x",
        colors="limegreen",
    )
    ax.tick_params(
        axis="y",
        colors="limegreen",
    )
    ax.tick_params(
        axis="z",
        colors="limegreen",
    )

    # 3つの面を黒くする
    ax.xaxis.pane.set_facecolor((0, 0, 0, 1))
    ax.yaxis.pane.set_facecolor((0, 0, 0, 1))
    ax.zaxis.pane.set_facecolor((0, 0, 0, 1))

    ax.xaxis.pane.set_edgecolor("limegreen")
    ax.yaxis.pane.set_edgecolor("limegreen")
    ax.zaxis.pane.set_edgecolor("limegreen")

    # グリッドを緑色にする
    ax.xaxis._axinfo["grid"]["color"] = (0.0, 0.5, 0.0, 0.65)
    ax.yaxis._axinfo["grid"]["color"] = (0.0, 0.5, 0.0, 0.65)
    ax.zaxis._axinfo["grid"]["color"] = (0.0, 0.5, 0.0, 0.65)

    colorbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.48,
        pad=0.03,
        aspect=20,
    )

    colorbar.set_label(
        "Normalized height",
        color="limegreen",
    )

    colorbar.ax.tick_params(
        colors="limegreen",
    )

    colorbar.outline.set_edgecolor("limegreen")

    plt.tight_layout()

    fig.savefig(
        save_path,
        format="svg",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    input_dir, output_dir = choose_folders()
    image_paths = collect_image_paths(input_dir)

    svg_dir = output_dir / "3d_svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    show_overview(image_paths)

    common_roi = None
    if USE_COMMON_ROI:
        first_rgb = load_rgb_image(image_paths[0])
        common_roi = select_roi_interactively(first_rgb, f"共通ROI選択: {image_paths[0].name}")
        if common_roi is None:
            raise SystemExit("共通ROIが選ばれなかったため終了します。")

    for idx, path in enumerate(image_paths, start=1):
        print(f"処理中 {idx}/{len(image_paths)}: {path.name}")
        rgb = load_rgb_image(path)

        if USE_COMMON_ROI:
            roi = common_roi
        else:
            roi = select_roi_interactively(rgb, f"ROI選択: {path.name}")

        if roi is None:
            print(f"スキップしました: {path.name}")
            continue

        left, top, right, bottom = roi
        cropped = rgb[top:bottom, left:right]
        if cropped.size == 0:
            print(f"ROIが空だったためスキップしました: {path.name}")
            continue

        height = pseudocolor_to_height(cropped)

        save_name = path.stem + "_3d.svg"
        save_path = svg_dir / save_name
        save_3d_surface_svg(
            height=height,
            save_path=save_path,
            title=f"{path.name}  |  ROI x={left}:{right}, y={top}:{bottom}",
            step=DOWNSAMPLE_STEP,
        )

        print(f"保存しました: {save_path}")

    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()