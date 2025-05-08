import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
import qrcode

from pharos import DeviceManager
from pharos import ProcessingWrapper

# クラスオブジェクト生成
manager = DeviceManager(simulation_mode=False)
manager.connect_devices()
stage = manager.get_stage_obj()
scanner = manager.get_scanner_obj()

processing = ProcessingWrapper(stage_obj=stage, scanner_obj=scanner)


"""
千鳥配置をパーカッションで行うプログラム
配置は60°千鳥

動作モード
スキャナのスイッチ: パーカッション
Burst packet size: 0

※少ないショット数(1-100程度?)で加工を行う際には, 
　Burst packet sizeにショット数を入れるといい
"""

# パラメータ
#############################################
# QRコードに入れたいデータ
data = "https://example.com"

# 加工パターン
shot_pitch_um = 10  # ピッチ
shot_numbers = 5  # 照射回数

# 照射周波数　発振周波数と同一にする
frequency_hz = 200000

# ステージ設定
# ステージのZ
stage_z_um = 2100
# ステージ速度
stage_speed_s = 75
stage_speed_f = 750
stage_speed_r = 100

# スキャナ設定
# スキャナジャンプスピード
scanner_jump_speed = 300
# set laser time
gate_on_delay = 200
gate_off_delay = 380
# mark delay jump delay
jump_delay = 400
mark_delay = 400

# スキャナの補正パターン
# scanner.set_calibration(file_name='230705_f67.TXT')
#############################################


def algorithm():
    # 加工条件 +++++++++++++++++++++++++++++++++++++++++++
    shot_sec = shot_numbers / frequency_hz
    shot_usec = int(shot_sec * 10 ** 6)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    # QR作ろ ++++++++++++++++++++++++++++++++++++++++++++
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # True/False のマトリクスを取得
    matrix = qr.get_matrix()
    height = len(matrix)  # 行数（Y方向）
    width = len(matrix[0])  # 列数（X方向）

    # 座標を取得（y,x の順）
    black_cells = []
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell:  # 黒セル
                black_cells.append((x, y))

    processing_width_um = width * shot_pitch_um  # 加工幅
    processing_height_um = height * shot_pitch_um  # 加工高さ
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

    # # 加工時間計算
    # total_time = x_point_numbers * y_point_numbers * shot_sec
    # print(f'\n推定加工時間: {total_time}sec ({total_time / 60: .1f}min)')

    # 加工アルゴリズム ++++++++++++++++++++++++++++++++++++

    # ステージ初期移動 (ステージを中心から第一象限の真ん中に移動)
    stage.move_position_relative(x_posi_um=processing_width_um // 2,
                                 y_posi_um=processing_height_um // 2)

    # スキャナ移動位置作成
    scanner_shift_x_um = -int(processing_width_um / 2)
    scanner_shift_y_um = -int(processing_height_um / 2)

    # ---------------- 可視化 ----------------------
    plt.figure(figsize=(6, 6), dpi=100)
    for coord in black_cells:
        scanner_x_position_um = coord[0] * shot_pitch_um
        scanner_y_position_um = coord[1] * shot_pitch_um

        plt.scatter(scanner_x_position_um,
                    scanner_y_position_um,
                    s=(shot_pitch_um ** 2),  # 点の大きさ (面積なので^2)
                    color='black')

    plt.xlim(0, processing_width_um)
    plt.ylim(0, processing_height_um)
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title('加工予定ショット配置')

    plt.gca().set_aspect('equal')

    plt.grid(True)

    plt.show()  # ウィンドウを閉じるまで停止
    # ---------------------------------------------

    # 一つのタイル加工
    scanner.list_open()

    for coord in black_cells:
        print(type(coord))

        scanner_x_position_um = coord[0] * shot_pitch_um
        scanner_y_position_um = coord[1] * shot_pitch_um


        scanner.jump_to_absolute(x_um=scanner_x_position_um,
                                 y_um=scanner_y_position_um)

        # 照射
        scanner.rt_sleep(jump_delay)
        scanner.burst(shot_usec)
        scanner.rt_sleep(mark_delay)

    scanner.list_close()
    scanner.wait_for_complete()

# 加工アルゴリズム
processing.run_processing(algorithm)

# 機器の切断
manager.disconnect_devices()