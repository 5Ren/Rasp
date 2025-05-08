import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os

from pharos import DeviceManager
from pharos import ProcessingWrapper

# クラスオブジェクト生成
manager = DeviceManager(simulation_mode=False)
manager.connect_devices()
stage = manager.get_stage_obj()
scanner = manager.get_scanner_obj()

processing = ProcessingWrapper(stage_obj=stage, scanner_obj=scanner)

"""
画像をそのまま描画するプログラム (タイリングなし)
出てくるグラフのばつをクリックすると、加工が始まります

最初の漏れ光は加工領域の左下に合わせる。
自動でステージが加工カ所の中心へ移動

動作モード
スキャナのスイッチ: PWM
Burst packet size: 1
"""

# パラメータ
#############################################
# 画像の読み込み
image_path = "image_files/QR_668348(dot).png"

# 加工パターン
shot_pitch_um = 1  # ピッチ
line_pitch_um = 20  # ピッチ
processing_height_mm = 1  # 加工縦長さ (画像がどんなにあらくてもこの長さに引き延ばされる)

shot_numbers = 1  # 重ね照射回数

# 照射周波数
frequency_hz = 50 * 1000 / 1

# ステージのZ
stage_z_um = 4600

# ステージ速度
stage_speed_s = 75
stage_speed_f = 750
stage_speed_r = 100

scanner_jump_speed = 300  # スキャナジャンプスピード

gate_on_delay = 200
gate_off_delay = 380

# スキャナの補正パターン
scanner.set_calibration(file_name='20221125_shg_f56.TXT')
#############################################

# 加工条件 +++++++++++++++++++++++++++++++++++++++++++
y_length_microns = processing_height_mm * 1000
resolution_micron_per_pix = line_pitch_um
# ++++++++++++++++++++++++++++++++++++++++++++++++++

# ステージ関連 ++++++++++++++++++++++++++++++++++++++++
# ステージのZ移動
stage.move_position_absolute(z_posi_um=stage_z_um)
# ステージ速度設定
stage.set_stage_speed(speed_s=stage_speed_s,
                      speed_f=stage_speed_f,
                      speed_r=stage_speed_r)
# ++++++++++++++++++++++++++++++++++++++++++++++++++

# スキャナ関連 +++++++++++++++++++++++++++++++++++++++++
# スキャナのJumpSpeed設定 (ジャンプ速度)
scanner.set_jump_speed(scanner_jump_speed)
# スキャナのMoveSpeed設定 (加工速度)
scanner_move_speed = int(shot_pitch_um * frequency_hz * 10 ** -3)
# スキャナからレーザーパルスの作成
scanner.generate_pp_pwm(frequency_hz=frequency_hz)
scanner.set_move_speed(scanner_move_speed)
# スキャナのgateディレイ設定
scanner.set_laser_times(gate_on_delay=gate_on_delay,
                        gate_off_delay=gate_off_delay)
# ++++++++++++++++++++++++++++++++++++++++++++++++++


# 画像から座標求める +++++++++++++++++++++++++++++++++++
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("画像の読み込みに失敗しました。")
    exit()

# 外側10画素分を消去（輪郭を削る）
image[:10, :] = 255  # 上部
image[-10:, :] = 255  # 下部
image[:, :10] = 255  # 左部
image[:, -10:] = 255  # 右部

# 二値化処理
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 元のピクセル単位の黒い部分の座標を取得
y_coords, x_coords = np.where(binary_image == 0)


# 新しい画像サイズを計算
y_new_size = int(y_length_microns / resolution_micron_per_pix)
x_new_size = int((y_new_size / binary_image.shape[0]) * binary_image.shape[1])

# 画像をリサイズして μm スケールに変換
image_scaled = cv2.resize(binary_image, (x_new_size, y_new_size), interpolation=cv2.INTER_NEAREST)

# μm スケールの黒い部分の座標を取得
y_coords_scaled, x_coords_scaled = np.where(image_scaled == 0)

# 画像の中心を原点 (0, 0) にする
x_centered = x_coords_scaled - x_new_size // 2
y_centered = -(y_coords_scaled - y_new_size // 2)  # Y軸の向きを反転

# エッジ検出のための始点・終点リスト
edge_segments = []
previous_color = None
for i in range(y_new_size):
    row = image_scaled[i, :]

    if i % 2 == 0:
        indices = range(len(row))  # 左→右
    else:
        indices = reversed(range(len(row)))  # 右→左

    prev_pixel = 255  # 初期状態は白
    for j in indices:
        if prev_pixel == 255 and row[j] == 0:
            start = (j - x_new_size // 2, -(i - y_new_size // 2))
        elif prev_pixel == 0 and row[j] == 255:
            end = (j - x_new_size // 2, -(i - y_new_size // 2))

            # ランダムな色を生成（前の色と異なるように）
            new_color = (random.random(), random.random(), random.random())
            while new_color == previous_color:
                new_color = (random.random(), random.random(), random.random())
            previous_color = new_color

            edge_segments.append((start, end, new_color))
        prev_pixel = row[j]

# 図を並べて表示
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

# 元のピクセル単位のプロット
axes[0].scatter(x_coords, -y_coords, s=2, c='black')
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
axes[0].set_title("Black Pixels Coordinates (Original)")
axes[0].set_xlim(0, binary_image.shape[1])
axes[0].set_ylim(-binary_image.shape[0], 0)
axes[0].set_aspect('equal')

# μm単位のプロット（中心を原点）
axes[1].scatter(x_centered * resolution_micron_per_pix, y_centered * resolution_micron_per_pix, s=2, c='black')
axes[1].set_xlabel("X (μm)")
axes[1].set_ylabel("Y (μm)")
axes[1].set_title("Black Pixels Coordinates (Centered at Origin)")
axes[1].set_xlim(-x_new_size * resolution_micron_per_pix // 2, x_new_size * resolution_micron_per_pix // 2)
axes[1].set_ylim(-y_new_size * resolution_micron_per_pix // 2, y_new_size * resolution_micron_per_pix // 2)
axes[1].set_aspect('equal')

# エッジ検出のプロット（異なる色を適用）
for start, end, color in edge_segments:
    axes[2].plot([start[0] * resolution_micron_per_pix, end[0] * resolution_micron_per_pix],
                 [start[1] * resolution_micron_per_pix, end[1] * resolution_micron_per_pix], c=color)
axes[2].set_xlabel("X (μm)")
axes[2].set_ylabel("Y (μm)")
axes[2].set_title("Edge Detection (Start & End Points with Random Colors)")
axes[2].set_xlim(-x_new_size * resolution_micron_per_pix // 2, x_new_size * resolution_micron_per_pix // 2)
axes[2].set_ylim(-y_new_size * resolution_micron_per_pix // 2, y_new_size * resolution_micron_per_pix // 2)
axes[2].set_aspect('equal')

plt.show()
# ++++++++++++++++++++++++++++++++++++++++++++++++++

def algorithm():
    # 加工アルゴリズム ++++++++++++++++++++++++++++++++++++

    # 照射位置の真ん中にステージ移動
    stage.move_position_relative(x_posi_um=int(y_length_microns / 2),
                                 y_posi_um=int(y_length_microns / 2))

    # スキャナ加工の命令の流し入れスタート
    # list_close()までの間にステージの命令は書いてはいけない！
    scanner.list_open()

    # 重ね打ち
    for loop in range(shot_numbers):

        # エッジ検出のプロット（異なる色を適用）
        for start, end, color in edge_segments:
            scanner.jump_to_absolute(x_um=start[0] * resolution_micron_per_pix,
                                     y_um=start[1] * resolution_micron_per_pix)

            scanner.line_to_absolute(x_um=end[0] * resolution_micron_per_pix,
                                     y_um=end[1] * resolution_micron_per_pix)

    # リストクローズ
    scanner.list_close()
    # open - closeまでの間に入れられた分の加工スタート
    scanner.wait_for_complete()

    stage.move_first_position_xy()


# 加工スタート
processing.run_processing(algorithm)
plt.show()
# 機器の切断
manager.disconnect_devices()

plt.show()