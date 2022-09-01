import csv
import os
import serial

class Stage(serial.Serial):
    def __init__(self, port: str):
        self.speed_S = 75
        self.speed_F = 750
        self.speed_R = 100

        print('Stage: __init__')

        with open('simu.csv', 'r+') as f:
            f.truncate(0)

        #
        # rec_path = os.path.dirname(os.path.abspath(__file__)) + '/simulation_csv/'
        # os.makedirs(rec_path, exist_ok=True)
        


        # l = ['x', 'y']
        # with open('simulation_csv/sample_writer.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(l)

    def __del__(self):
        pass

    def communicate_rs232c(self, cmd) -> bool:
        pass

    def reset_all_axis(self) -> bool:
        pass

    def reset_any_axis(self, axis: int) -> bool:
        pass

    def move_position_relative(self, axis: int, position: int) -> bool:
        pass

    def move_position_absolute(self, axis: int, position: int) -> bool:
        pass

    def move_position_absolute_jog(self, axis: int, position: int) -> bool:
        pass

    def get_current_positon(self) -> tuple[int, int, int]:
        return [0, 0, 0]

    def set_stage_speed(self, speed_S: int, speed_F: int, speed_R: int) -> bool:
        pass


