import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft
import warnings

import analyze_funcs

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# 計測，解析条件
dt = 0.01  # サンプル周期[s]

# Cloggerで書き出したcsvを入力に与える
file_path = r'./Pharos出力測定/NL_IR/NL_IR_20230211.csv'
# ファイル名
base_name = os.path.basename(file_path).split('.')[0]
# データの読み込み
import_signal_freq = np.loadtxt(file_path, dtype='int64', skiprows=6)


# 表示データ数
min_list = [60, 20, 5]

# 表示データのループ
for min_range in min_list:
    window_size = 6000 * min_range

    # データの数でクロップ
    signal_freq = import_signal_freq[:window_size]
    # 信号データからサンプル数取得
    N = signal_freq.shape[0]

    # グラフの設定
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["figure.dpi"] = 300

    # 時間軸作成
    t = np.arange(0, N * dt, dt)  # 時間

    ir_or_green = ''

    # 電圧をパワーに変換
    if 'IR' in file_path or 'ir' in file_path:
        print('IR!!')
        signal_freq = signal_freq / 20000
        plot_title = 'Power changes (IR)'
        ir_or_green = 'IR'
    elif 'SHG' in file_path or 'shg' in file_path:
        print('SHG!!')
        signal_freq = signal_freq / 20000
        plot_title = 'Power changes (SHG)'
        ir_or_green = 'SHG'

    # 移動平均
    # ---------------------------------------------------------
    moving_range1 = 10
    filtered_signal_freq1 = analyze_funcs.moving_average(
        signal_freq, moving_range1)[moving_range1:-moving_range1]

    moving_range2 = 50
    filtered_signal_freq2 = analyze_funcs.moving_average(
        signal_freq, moving_range2)[moving_range2:-moving_range2]
    # ---------------------------------------------------------

    # グラフを表示
    # ---------------------------------------------------------
    plot_numbers = 3
    plot_list = [signal_freq, filtered_signal_freq1, filtered_signal_freq2]
    title_list = [f'RAW {ir_or_green} Signal',
                  f'Moving average (range: {moving_range1})',
                  f'Moving average (range: {moving_range2})']

    fig1, ax = plt.subplots(nrows=plot_numbers, figsize=(10, 12))
    fig1.suptitle(f'Signal ({base_name})', fontsize=16)
    plt.subplots_adjust(top=0.92, bottom=0.05)

    # グラフの設定
    for i in range(plot_numbers):
        ax[i].plot(plot_list[i])
        ax[i].set_title(title_list[i], loc='left')
    # 軸の設定
    [ax[i].set_ylabel('Power [W]') for i in range(3)]
    ax[2].set_xlabel("time [sec]")

    # 平均
    average = np.average(signal_freq)
    # 平均，上限のバーをひく
    bar_range = 0.005

    y_max = average + bar_range
    y_min = average - bar_range
    print(f'{y_max - y_min = }')
    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(y_min, y_max)
    ax[2].set_ylim(y_min, y_max)


    ax[0].axhline(y=average, color='g')
    ax[0].axhline(y=average + bar_range/2, color='r')
    ax[0].axhline(y=average - bar_range/2, color='r')
    ax[1].axhline(y=average, color='g')
    ax[1].axhline(y=average + bar_range/2, color='r')
    ax[1].axhline(y=average - bar_range/2, color='r')
    ax[2].axhline(y=average, color='g')
    ax[2].axhline(y=average + bar_range/2, color='r')
    ax[2].axhline(y=average - bar_range/2, color='r')
    plt.show()
    # ---------------------------------------------------------


    # # # fft用の2の累乗
    # fft_data_size = int(2**math.floor(math.log2(window_size)))
    # print(f'{fft_data_size=}')
    # signal_freq = signal_freq[:fft_data_size]
    # print(f'do_fft.py {signal_freq.shape=}')
    # # do_fft.py
    # # ---------------------------------------------------------
    # sample_numbers = signal_freq.shape[0]
    #
    # large_f = np.do_fft.py.do_fft.py(signal_freq)
    # fft_freq = np.do_fft.py.fftfreq(sample_numbers, d=dt)
    #
    # fig2, ax = plt.subplots(nrows=3, figsize=(8, 12))
    # fig2.suptitle(f'FFT ({base_name})', fontsize=16)
    # plt.subplots_adjust(top=0.92, bottom=0.05)
    #
    # ax[0].plot(large_f.real)
    # ax[0].set_title('Real part', loc='left')
    #
    # ax[1].plot(large_f.imag)
    # ax[1].set_title('Imaginary part', loc='left')
    #
    # ax[2].plot(fft_freq)
    # ax[2].set_title('Frequency', loc='left')
    # ax[2].set_xlabel("time [sec]")
    #
    # ax[1].axhline(y=0.5, color='r')
    # ax[1].axhline(y=-0.5, color='r')
    # plt.show()
    #
    # Amp = np.abs(large_f / (sample_numbers / 2))  # 振幅
    #
    # fig3, ax = plt.subplots()
    # fig3.suptitle(f'Amplitude ({base_name})')
    # ax.plot(fft_freq[1:int(sample_numbers / 2)], Amp[1:int(sample_numbers / 2)])
    # ax.set_xlabel("Freqency [Hz]")
    # ax.set_ylabel("Amplitude")
    # ax.grid()
    # ax.axhline(y=0.0001, color='r')
    # plt.show()
    # # ---------------------------------------------------------
