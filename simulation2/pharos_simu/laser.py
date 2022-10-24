import csv
import os
import numpy as np
from csv import writer
class Laser:
    def __init__(self, device_name: str, laser_channel: int, ra_channel: int, pp_channel: int):
        self.device_name = device_name
        self.laser_channel = laser_channel
        self.ra_channel = ra_channel
        self.pp_channel = pp_channel

        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_csv', 'simulation_log.csv')
        self.spot_d = 12


    def laser_on(self) -> None:
        data = self.tail()
        x_stage = int(data[0])
        y_stage = int(data[1])
        x_scanner = int(data[2])
        y_scanner = int(data[3])
        x_laser = int(data[4])
        y_laser = int(data[5])
        for i in range(21):
            x = self.spot_d / 2 * np.cos((2 * np.pi) / 20 * i)
            y = self.spot_d / 2 * np.sin((2 * np.pi) / 20 * i)
            x = float(x_laser + x)
            y = float(y_laser + y)
            self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x, y)
        # self.write_csv(x_stage, y_stage, x_scanner, y_scanner, x_laser, y_laser)


    def laser_off(self) -> None:
        pass

    def get_ra_status(self) -> float:
        return float(5.0)

    def get_pp_status(self) -> float:
        return float(3.0)

    def aio_exit(self):
        pass

    def aio_reset(self):
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


