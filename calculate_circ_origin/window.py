import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import winsound
from tkinter import messagebox
import threading

font_of_button = ("Helve", "12", "bold")
font_of_label = ("Helve", "15", "bold")


def calculate_circ_origin(coordinate_list) -> list:
    """
    3点の座標から、円の中心を求める関数

    Parameters
    ----------
    coordinate_list
        2次元配列 リスト
        [[x1, y1], [x2, y2], [x3, y3]]
        ※ Unit: なんでも
    Returns
    -------
        座標情報のリスト
        [x座標, y座標, 半径]
    """

    if sum(len(v) for v in coordinate_list) != 6:
        print("要素数が足りません")
        return [0, 0, 0]

    alpha = coordinate_list[0][0] - coordinate_list[1][0]  # x1 - x2
    beta = coordinate_list[0][1] - coordinate_list[1][1]  # y1 - y2
    gamma = coordinate_list[1][0] - coordinate_list[2][0]  # x2 - x3
    delta = coordinate_list[1][1] - coordinate_list[2][1]  # y2 - y3
    bi_ad_bg = 2 * (alpha * delta - beta * gamma)

    X_1 = coordinate_list[0][0] ** 2 + coordinate_list[0][1] ** 2  # x1^2 + y1^2
    X_2 = coordinate_list[1][0] ** 2 + coordinate_list[1][1] ** 2  # x2^2 + y2^2
    X_3 = coordinate_list[2][0] ** 2 + coordinate_list[2][1] ** 2  # x3^2 + y2^2

    origin_x = (delta * (X_1 - X_2) - beta * (X_2 - X_3)) / bi_ad_bg
    origin_y = (-1 * gamma * (X_1 - X_2) + alpha * (X_2 - X_3)) / bi_ad_bg
    radius = np.sqrt((coordinate_list[0][0] - origin_x) ** 2 + (coordinate_list[0][1] - origin_y) ** 2)
    return [origin_x, origin_y, radius]



class CalcuCenterWindow:
    # 機器のインスタンス生成
    # interface = ProcessingInterface(laser_mode='newson', program_name=__file__)
    # stage1 = interface.get_stage_obj()
    # scanner1 = interface.get_scanner_obj()
    laser_flg: bool = False

    # レーザーオンオフフラグ
    # laser_on_sound = r'./pharos/libs/music/eva (mp3cut.net).wav'
    one_pulse = 2

    # ステージの1パルス移動量[um]
    font_of_button = ("Helve", "15", "bold")

    # フォント
    font_of_label = ("Helve", "12", "bold")
    frame_laser = None

    # フレーム
    frame_coordinate_input = None
    frame_graph = None
    first_point_x_entry = None

    # エントリー
    first_point_y_entry = None
    second_point_x_entry = None
    second_point_y_entry = None
    third_point_x_entry = None
    third_point_y_entry = None

    # グラフ描画
    ax = None
    fig_canvas = None
    toolbar = None

    def __init__(self):
        # ステージの速度設定
        # stage_speed_s = 75
        # stage_speed_f = 750
        # stage_speed_r = 100
        # self.stage1.set_stage_speed(
        #     stage_speed_s, stage_speed_f, stage_speed_r
        # )

        self.create_window()

    def laser_on(self):
        yes_or_no = messagebox.askyesno('安全確認',
                                        '【警告】\n'
                                        '周囲の安全を確認してください。\n'
                                        'レーザーが照射されます！！！')
        if yes_or_no:
            self.frame_top.configure(bg='#000fff000')
            # 照射処理
            thread1 = threading.Thread(target=self.laser_burst_loop)
            thread1.start()

    def laser_burst_loop(self):
        self.laser_flg = True

        # # 30秒照射を繰り返す
        burst_time_usec = 30 * 10 ** 6
        print('burst start')

        while 1:
            # 警告音再生
            # winsound.PlaySound(self.laser_on_sound,
            #                    winsound.SND_ASYNC)
            # 照射
            # self.scanner1.burst_standalone(burst_time_usec=burst_time_usec)

            if not self.laser_flg:
                print('break laser on loop')
                break

    def laser_off(self):
        print('laser flg off')
        self.laser_flg = False

        print('scanner format flash')
        # self.scanner1.format_flash()
        # self.scanner1.wait_for_complete()

        self.frame_top.configure(bg='#f2f2f2')
        # winsound.PlaySound(None, winsound.SND_PURGE)

    def delete_window(self, root):
        self.laser_off()
        # self.scanner1.scanner_reset()
        root.destroy()
        print('> Delete Window')

    def button_click(self):
        # 表示するデータの作成
        x = np.arange(-np.pi, np.pi, 0.1)
        y = np.sin(x)
        # グラフの描画
        self.ax.plot(x, y)
        # 表示
        self.fig_canvas.draw()

    def create_window(self):
        # rootメインウィンドウの設定 ################################################
        root = tk.Tk()
        root.title("Frame")
        root.geometry("600x500")
        font_of_button = self.font_of_button
        font_of_label = self.font_of_label
        #########################################################################

        # LASER Frame ##########################################################
        frame_laser = tk.Frame(root, pady=5, padx=5, relief=tk.GROOVE, bd=2)

        # ラベルとボタンの配置
        laser_label = tk.Label(frame_laser, text='LASER: ', font=font_of_label)
        laser_on_button = tk.Button(frame_laser, text='ON', font=font_of_button,
                                    command=lambda: self.laser_on())
        laser_off_button = tk.Button(frame_laser, text='OFF', font=font_of_button,
                                     command=lambda: self.laser_off())

        # ラベルとボタンの配置
        laser_label.grid(column=0, row=0, ipadx=0, padx=5, ipady=0, pady=0)
        laser_on_button.grid(column=1, row=0, ipadx=24, padx=5, ipady=0, pady=5)
        laser_off_button.grid(column=2, row=0, ipadx=22, padx=5, ipady=0, pady=5)
        #########################################################################

        # 座標入力 Frame ################################################################
        frame_coordinate_input = tk.Frame(root, pady=5, padx=5, relief=tk.GROOVE, bd=2)

        # ラベルとボタンの配置
        coordinate_input_label = tk.Label(frame_coordinate_input, text='3-points coordinate input:                      ',
                                          anchor=tk.E, font=font_of_label)

        first_point_label = tk.Label(frame_coordinate_input, text='(1)', font=font_of_button)
        second_point_label = tk.Label(frame_coordinate_input, text='(2)', font=font_of_button)
        third_point_label = tk.Label(frame_coordinate_input, text='(3)', font=font_of_button)

        x_label = tk.Label(frame_coordinate_input, text='X', font=font_of_button)
        y_label = tk.Label(frame_coordinate_input, text='Y', font=font_of_button)

        first_point_x_entry = tk.Entry(frame_coordinate_input, width=10, justify=tk.RIGHT, font=font_of_button)
        first_point_y_entry = tk.Entry(frame_coordinate_input, width=10, justify=tk.RIGHT, font=font_of_button)
        second_point_x_entry = tk.Entry(frame_coordinate_input, width=10, justify=tk.RIGHT, font=font_of_button)
        second_point_y_entry = tk.Entry(frame_coordinate_input, width=10, justify=tk.RIGHT, font=font_of_button)
        third_point_x_entry = tk.Entry(frame_coordinate_input, width=10, justify=tk.RIGHT, font=font_of_button)
        third_point_y_entry = tk.Entry(frame_coordinate_input, width=10, justify=tk.RIGHT, font=font_of_button)

        calculate_button = tk.Button(frame_coordinate_input, text='CALCULATE', font=font_of_button)
        move_stage_button = tk.Button(frame_coordinate_input, text='MOVE STAGE', font=font_of_button)

        # ラベルとボタンの配置
        coordinate_input_label.grid(column=0, row=0, columnspan=3)

        x_label.grid(column=1, row=1)
        y_label.grid(column=2, row=1)

        first_point_label.grid(column=0, row=2)
        first_point_x_entry.grid(column=1, row=2)
        first_point_y_entry.grid(column=2, row=2)

        second_point_label.grid(column=0, row=3)
        second_point_x_entry.grid(column=1, row=3)
        second_point_y_entry.grid(column=2, row=3)

        third_point_label.grid(column=0, row=4)
        third_point_x_entry.grid(column=1, row=4)
        third_point_y_entry.grid(column=2, row=4)
        calculate_button.grid(column=1, row=6, columnspan=2, ipadx=24, padx=5, ipady=0, pady=5)
        move_stage_button.grid(column=1, row=7, columnspan=2, ipadx=24, padx=5, ipady=0, pady=5)
        #########################################################################

        # グラフ描画 Frame ################################################################
        frame_graph = tk.Frame(root, pady=5, padx=5, relief=tk.GROOVE)

        # matplotlibの描画領域の作成
        fig = Figure()
        # 座標軸の作成
        self.ax = fig.add_subplot(1, 1, 1)
        # matplotlibの描画領域とウィジェット(Frame)の関連付け
        self.fig_canvas = FigureCanvasTkAgg(fig, frame_graph)
        # matplotlibのツールバーを作成
        self.toolbar = NavigationToolbar2Tk(self.fig_canvas, frame_graph)
        # matplotlibのグラフをフレームに配置
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ボタンの作成
        button = tk.Button(root, text="Draw Graph", command=self.button_click)
        # 配置
        button.pack(side=tk.BOTTOM)
        #########################################################################

        # ウィジェットの配置 #######################################################
        # フレームの配置
        frame_laser.pack(fill=tk.X)
        frame_coordinate_input.pack(side=tk.LEFT, fill=tk.Y)
        frame_graph.pack(side=tk.LEFT, fill=tk.Y)
        #########################################################################

        root.mainloop()


if __name__ == '__main__':
    Calculate_Center_Window = CalcuCenterWindow()
