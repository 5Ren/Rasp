import math
import time
import serial


class Motor(serial.Serial):
    dino = None
    sm_resolution: float = None
    resolution_dec = {'0': 0.72, '1': 0.36, '2': 0.288, '3': 0.18,
                      '4': 0.144, '5': 0.09, '6': 0.072, '7': 0.036,
                      '8': 0.0288, '9': 0.018, 'A': 0.0144, 'B': 0.009,
                      'C': 0.0072, 'D': 0.00576, 'E': 0.0036, 'F': 0.00288}
    ss1_deg: float = None
    ss2_deg: float = None
    ss1pulse: int = None
    ss2pulse: int = None

    def __init__(self, port: str, resolution1: str = '0', resolution2: str = 'B'):
        baudrate = 19200
        super().__init__(port, baudrate)
        self.ss1_deg = self.resolution_dec[resolution1]
        self.ss2_deg = self.resolution_dec[resolution2]
        time.sleep(2)
        # print(f'{self.ss1_deg=} deg, {self.ss2_deg=}deg')

    def __del__(self):
        self.close()

    def rotate_cw(self, cw_deg: float) -> bool:
        if cw_deg > 0.0:
            self.deg_set(degree=cw_deg)
            self.run_rotate(mode='a', pulse=self.ss1pulse)
            self.run_rotate(mode='c', pulse=self.ss2pulse)
            return True
        elif cw_deg < 0.0:
            self.deg_set(degree=abs(cw_deg))
            self.run_rotate(mode='b', pulse=self.ss1pulse)
            self.run_rotate(mode='d', pulse=self.ss2pulse)
            return True
        else:
            print('引数が無効です。')
            return False

    def rotate_ccw(self, cw_deg: float):
        self.rotate_cw(cw_deg=-cw_deg)

    def deg_set(self, degree: float):
        self.ss1pulse, mod = divmod(degree, self.ss1_deg)
        self.ss2pulse = int(mod / self.ss2_deg)
        print(f'in: {degree}, out: {self.ss1pulse * self.ss1_deg + self.ss2pulse * self.ss2_deg}')
        print(f'{self.ss1pulse=}, {self.ss2pulse=}')
    def run_rotate(self, mode: str, pulse: int) -> bool:
        sm_pulse = mode + str(pulse) + ';'
        self.write(sm_pulse.encode('utf-8'))
        result_str = self.readline().decode()
        return True


if __name__ == '__main__':
    motor = Motor('/dev/tty.usbmodem1101', )

    motor.rotate_cw(100)
    motor.rotate_ccw(180)
    motor.rotate_cw(80)
