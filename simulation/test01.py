import time

import tqdm
import playsound

from pharos_simu.laser import Laser
from pharos_simu.stage import Stage
from pharos_simu.scanner import Scanner


def mysleep_sec(sleep_time_sec):
    pass


class LaserProcessor:
    CONTEC_BOARD_NAME = 'AIO000'
    CONTEC_LASER_CHANNEL = 0
    CONTEC_RA_STATUS_CHANNEL = 1
    STAGE_PORT = 'COM3'
    SCANNER_DEVICE_NAME = 'USB'

    initial_stage_posi_x = None
    initial_stage_posi_y = None
    stage_speed_S = None
    stage_speed_F = None
    stage_speed_R = None
    laser_freq_kHz = None
    total_shot_numbers = None
    focus_z_um = None
    tile_size_mm = None
    pitch_um = None

    def __init__(self):
        self.temp_pitch = None
        self.inner_f1_pitch = None
        self.f2 = None
        self.f1 = None
        self.laser_spot_diameter = None
        self.pitch = None
        self.set_config()
        self.__setup()

    def __setup(self):
        print('\n------preparation------')
        # 機器のインスタンス生成
        self.stage = Stage(port=self.STAGE_PORT)
        self.laser = Laser(device_name=self.CONTEC_BOARD_NAME,
                           laser_channel=self.CONTEC_LASER_CHANNEL,
                           ra_channel=self.CONTEC_RA_STATUS_CHANNEL)
        self.scanner = Scanner(device_name=self.SCANNER_DEVICE_NAME)

        # ステージの速度設定
        self.stage.set_stage_speed(
            self.stage_speed_S,
            self.stage_speed_F,
            self.stage_speed_R)
        print(f'set stage SFR (S:{self.stage_speed_S}, F:{self.stage_speed_F}, '
              f'R{self.stage_speed_R})')

        # 現在のステージ位置を取得
        current_stage_position = self.stage.get_current_positon()
        self.initial_stage_posi_x = current_stage_position[0]
        self.initial_stage_posi_y = current_stage_position[1]
        print(f'current stage position (x:{self.initial_stage_posi_x * 2}um,'
              f' y:{self.initial_stage_posi_y * 2}um)')

        # move z
        self.focus_z_pluses = int(self.focus_z_um / 2)
        self.stage.move_position_absolute(axis=3, position=int(self.focus_z_pluses))
        print(f'moved stage z to {self.stage.get_current_positon()[2] * 2}um')

        # check instrument connection
        # if connection fail, call clean_up method before program exit

    def clean_up(self):
        print('\n------finish------')
        self.laser.laser_off()
        print('laser off')
        self.laser.aio_exit()
        print('aio exit')

        # playsound.playsound(r'./phaors/noritz.mp3')
        self.stage.move_position_absolute(1, self.initial_stage_posi_x)
        self.stage.move_position_absolute(2, self.initial_stage_posi_y)
        self.stage.move_position_absolute(3, self.focus_z_pluses)
        print('stage reset')

        self.scanner.move_to_zero()
        print('scanner reset')
        self.scanner.print_position()

    def set_config(self):
        # レーザー設定
        #################################################
        # ステージ速度
        self.stage_speed_S = 75
        self.stage_speed_F = 750
        self.stage_speed_R = 100

        # 照射設定
        self.laser_freq_kHz = 50 / 1
        # 1か所の合計照射回数
        # 下の照射回数指定後に、pharos control APPの
        # 「Configure　PP」→「Burst packet size」にも同じ値を入れること
        self.total_shot_numbers = 1

        # 焦点位置
        self.focus_z_um = 4190  # zスタート地点

        # 照射範囲
        self.tile_size_mm = 0.5

        # 大きいスポットのピッチ
        self.pitch = 25
        # 大きいスポットの大きさ
        self.f1 = 17
        # レーザースポット径
        self.laser_spot_diameter = 13

        #################################################

        self.f2 = self.pitch - self.f1
        self.inner_f1_pitch = self.f1 - self.laser_spot_diameter
        self.temp_pitch = self.f1 + self.f2

    def run(self):
        print('\n------start------')

        laser1 = self.laser
        stage1 = self.stage
        scanner1 = self.scanner

        pitch_um = self.pitch_um

        shot_sec = self.total_shot_numbers / (self.laser_freq_kHz * 1000)

        # total hole numbers x, y
        x_point_numbers = int(self.tile_size_mm * 1000 / (self.f1 + self.f2))
        y_point_numbers = int(self.tile_size_mm * 1000 / (self.f1 + self.f2))

        # move scanner 0 to pattern center
        scanner_x_shift_um = int(self.tile_size_mm / 2) * 1000
        scanner_y_shift_um = int(self.tile_size_mm / 2) * 1000

        for y in tqdm.tqdm(range(y_point_numbers)):
            for x in range(x_point_numbers):
                # スキャナ移動
                scanner_x_position_um = x * self.temp_pitch - scanner_x_shift_um
                scanner_y_position_um = y * self.temp_pitch - scanner_y_shift_um

                scanner1.move_position_absolute(scanner_x_position_um,
                                                scanner_y_position_um)

                laser1.laser_on()
                mysleep_sec(shot_sec)
                laser1.laser_off()

                scanner1.move_position_relative(self.inner_f1_pitch, 0)
                laser1.laser_on()
                mysleep_sec(shot_sec)
                laser1.laser_off()

                scanner1.move_position_relative(0, -self.inner_f1_pitch)
                laser1.laser_on()
                mysleep_sec(shot_sec)
                laser1.laser_off()

                scanner1.move_position_relative(-self.inner_f1_pitch, 0)
                laser1.laser_on()
                mysleep_sec(shot_sec)
                laser1.laser_off()



if __name__ == '__main__':
    laser_processor = LaserProcessor()

    try:
        laser_processor.run()
    except KeyboardInterrupt:
        print('STOP!')
    finally:
        laser_processor.clean_up()
