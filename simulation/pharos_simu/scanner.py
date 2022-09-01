import csv
import os
from csv import writer

class Scanner:
    def __init__(self, device_name):
        self.DEVICE_NAME = device_name
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_csv', 'simu.csv')

    def set_speed(self, speed: int) -> int:
        pass

    def move_to_zero(self) -> int:
        pass

    def move_position_relative(self, x: int, y: int) -> int:
        pass

    def move_position_absolute(self, x: int, y: int) -> int:
        pass

    def get_position(self) -> tuple[int, int, int]:
        pass

    def print_position(self) -> int:
        pass

    def write_csv(self, x, y):
        list_data = [x, y]
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