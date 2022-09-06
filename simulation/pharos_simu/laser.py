import csv
import os
import numpy as np
from csv import writer
class Laser:
    def __init__(self, device_name: str, laser_channel: int, ra_channel: int):
        self.device_name = device_name
        self.laser_channel = laser_channel
        self.ra_channel = ra_channel

        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_csv', 'simu.csv')
        self.spot_d = 6


    def laser_on(self) -> None:
        data = self.tail()
        x_tail = data[0]
        y_tail = data[1]
        for i in range(11):
            x = self.spot_d / 2 * np.cos((2 * np.pi) / 10 * i)
            y = self.spot_d / 2 * np.sin((2 * np.pi) / 10 * i)
            x = float(x_tail + x)
            y = float(y_tail + y)
            self.write_csv(x, y)
        self.write_csv(int(x_tail), int(y_tail))


    def laser_off(self) -> None:
        pass

    def get_ra_status(self) -> str:
        pass

    def aio_exit(self):
        pass

    def aio_reset(self):
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


