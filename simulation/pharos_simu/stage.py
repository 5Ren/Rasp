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
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_csv', 'simu.csv')
        with open(self.path, 'r+') as f:
            f.truncate(0)
            f.close()

        list_data = ['x', 'y']
        with open(self.path, 'a', newline='') as f_object:
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(list_data)
            # Close the file object
            f_object.close()

        self.reset_all_axis()

    def __del__(self):
        print('Stage: __del__')

    def reset_all_axis(self) -> bool:
        self.write_csv(0, 0)
        print('Stage: reset all axis')
    def reset_any_axis(self, axis: int) -> bool:
        if axis == 1:
            data = self.tail()
            y = int(data[1])
            self.write_csv(0, y)
            print('Stage: reset x axis')
        elif axis == 2:
            data = self.tail()
            x = int(data[0])
            self.write_csv(x, 0)
            print('Stage: reset y axis')
        else:
            print('Stage: reset z axis')

    def move_position_relative(self, axis: int, position: int) -> bool:
        if axis == 1:
            data = self.tail()
            x = position + int(data[0])
            y = int(data[1])
            self.write_csv(x, y)
        elif axis == 2:
            data = self.tail()
            x = int(data[0])
            y = position + int(data[1])
            self.write_csv(x, y)
        else:
            pass

    def move_position_absolute(self, axis: int, position: int) -> bool:
        if axis == 1:
            data = self.tail()
            y = int(data[1])
            self.write_csv(position, y)
        elif axis == 2:
            data = self.tail()
            x = int(data[0])
            self.write_csv(x, position)
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



