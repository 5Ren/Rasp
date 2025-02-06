import csv
import os
import serial
from csv import writer

class Stage(serial.Serial):
    def __init__(self, port: str):
        self.speed_S = 75
        self.speed_F = 750
        self.speed_R = 100

        print('Stage: __init__')
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_csv', 'simulation_log.csv')
        with open(self.path, 'r+') as f:
            f.truncate(0)
            f.close()

        self.reset_all_axis()
        self.reset_all_axis()

    def __del__(self):
        print('Stage: __del__')

    def reset_all_axis(self) -> bool:
        self.write_csv(0, 0, 0, 0, 0, 0)
    def reset_any_axis(self, axis: int) -> bool:
        if axis == 1:
            data = self.tail()
            x_stage = int(data[0])
            y_stage = int(data[1])
            x_scanner = int(data[2]) - x_stage
            y_scanner = int(data[3]) - y_stage
            x_laser = 0
            y_laser = y_stage
            self.write_csv(0, y_stage, x_scanner, y_scanner, x_laser, y_laser)
        elif axis == 2:
            data = self.tail()
            x_stage = int(data[0])
            y_stage = int(data[1])
            x_scanner = int(data[2]) - x_stage
            y_scanner = int(data[3]) - y_stage
            x_laser = x_stage
            y_laser = 0
            self.write_csv(x_stage, 0, x_scanner, y_scanner, x_laser, y_laser)
        else:
            pass

    def move_position_relative(self, axis: int, position: int) -> bool:
        if axis == 1:
            data = self.tail()
            x_stage = position + int(data[0])
            y_stage = int(data[1])
            x_scanner = position + int(data[2])
            y_scanner = int(data[3])
            x_laser = x_stage
            y_laser = y_stage
            self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)

        elif axis == 2:
            data = self.tail()
            x_stage = int(data[0])
            y_stage = position + int(data[1])
            x_scanner = int(data[2])
            y_scanner = position + int(data[3])
            x_laser = x_stage
            y_laser = y_stage
            self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)
        else:
            pass

    def move_position_absolute(self, axis: int, position: int) -> bool:
        if axis == 1:
            data = self.tail()
            x_stage = position
            y_stage = int(data[1])
            x_scanner = int(data[2] - data[0]) + position
            y_scanner = int(data[3])
            x_laser = position
            y_laser = y_stage
            self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)
        elif axis == 2:
            data = self.tail()
            x_stage = int(data[0])
            y_stage = position
            x_scanner = int(data[2])
            y_scanner = int(data[3] - data[1]) + position
            x_laser = x_stage
            y_laser = position
            self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)
        else:
            pass

    def move_position_absolute_jog(self, axis: int, position: int) -> bool:
        pass

    def get_current_positon(self) -> tuple[int, int, int]:
        # data = self.tail()
        ret = (0, 0, 0)
        return ret

    def set_stage_speed(self, speed_S: int, speed_F: int, speed_R: int) -> bool:
        pass

    def write_csv(self, x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser):
        list_data = [x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser]
        with open(self.path, 'a', newline='') as f_object:
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(list_data)
            # Close the file object
            f_object.close()

    def tail(self):
        with open(self.path) as f:
            # ファイルオブジェクトをcsvリーダーに変換
            reader = csv.reader(f)
            # ヘッダーを捨てる
            next(reader)
            # 全行読む
            rows = [row for row in reader]
            final = [list(map(float, row)) for row in rows[-1:]]
            ret = final[0]

        # 最後のn行だけfloatにして返す
        return ret



