from pharos_simu.laser import Laser
from pharos_simu.scanner import Scanner
from pharos_simu.stage import Stage

import tqdm

if __name__ == '__main__':
    laser1 = Laser(device_name='hoge', laser_channel=0, ra_channel=0, pp_channel=0)
    scanner1 = Scanner(device_name='hogehoge')
    stage1 = Stage('pom')

    #############################################
    # 加工パターン
    pitch_um = 30  # 大きいスポットのピッチ
    innner_pitch = 8  # 小さいスポットのピッチ
    square_size_mm = 1  # 加工の大きさ
    #############################################

    num = int(square_size_mm * 1000 / pitch_um)

    # scanner1.move_position_absolute(int(-square_size_mm / 2), int(square_size_mm / 2))

    scan_list = [[0, 0],
                 [innner_pitch, 0]]

    for clo in tqdm.tqdm(range(len(scan_list))):
        scanner1.move_position_absolute(int(square_size_mm * -1000 / 2 + scan_list[clo][0]), int(square_size_mm * 1000 / 2 + scan_list[clo][1]))
        for y in range(num * 2):
            for x in range(num):
                laser1.laser_on()

                if y % 2 == 0 and x < num - 1:
                    scanner1.move_position_relative(pitch_um, 0)
                elif y % 2 == 1 and x < num - 1:
                    scanner1.move_position_relative(-pitch_um, 0)

            if y % 2 == 0:
                scanner1.move_position_relative(0, -innner_pitch)

            elif y % 2 == 1 and y < num * 2 - 1:
                scanner1.move_position_relative(0, -(pitch_um - innner_pitch))

    # scanner1.move_position_absolute(innner_pitch, 0)
    #
    # for y in range(num * 2 + 2):
    #     for x in range(num):
    #         laser1.laser_on()
    #
    #         if y % 2 == 0 and x < num - 1:
    #             scanner1.move_position_relative(pitch_um, 0)
    #         elif y % 2 == 1 and x < num - 1:
    #             scanner1.move_position_relative(-pitch_um, 0)
    #
    #     if y % 2 == 0 and y < num + 1:
    #         scanner1.move_position_relative(0, -innner_pitch)
    #
    #     elif y % 2 == 1 and y < num + 1:
    #         scanner1.move_position_relative(0, -(pitch_um - innner_pitch))
