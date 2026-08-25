#!/usr/bin/env python3
"""黒背景・緑軸でCSV高さデータを3D TIFFへ一括変換する。

このファイルと batch_csv_3d_plot_tiff.py を同じフォルダーに置き、
PyCharmからこのファイルを実行してください。
描画条件、フォルダー選択、出力名などは白背景版と共通です。
"""

from __future__ import annotations

import batch_csv_3d_plot_tiff as plotter


# 元のレーザー顕微鏡画像から取得した、暗めの緑色。
plotter.BACKGROUND_COLOR = "black"
plotter.AXIS_COLOR = "#0E6D1B"


if __name__ == "__main__":
    raise SystemExit(plotter.main())
