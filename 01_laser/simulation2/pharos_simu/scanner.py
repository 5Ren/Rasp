import csv
import os
from csv import writer

class Scanner:
    def __init__(self, device_name):
        self.DEVICE_NAME = device_name
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_csv', 'simulation_log.csv')

    def set_speed(self, speed: int) -> int:
        pass

    def move_to_zero(self) -> int:
        data = self.tail()
        x_stage = int(data[0])
        y_stage = int(data[1])
        x_scanner = x_stage
        y_scanner = y_stage
        x_laser = x_stage
        y_laser = y_stage
        self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)
    def move_position_relative(self, x: int, y: int) -> int:
        data = self.tail()
        x_stage = int(data[0])
        y_stage = int(data[1])
        x_scanner = x + int(data[2])
        y_scanner = y + int(data[3])
        x_laser = x_scanner
        y_laser = y_scanner
        self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)
    def move_position_absolute(self, x: int, y: int) -> int:
        data = self.tail()
        x_stage = int(data[0])
        y_stage = int(data[1])
        x_scanner = x + x_stage
        y_scanner = y + y_stage
        x_laser = x_scanner
        y_laser = y_scanner
        self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)

    def get_position(self) -> tuple[int, int, int]:
        pass

    def print_position(self) -> int:
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