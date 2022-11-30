import time
import serial


class Moter(serial.Serial):
    dino = None
    sm_resolution = float
    sm_pulse = str

    def __init__(self, port: str, resolution_deg: float):
        self.dino = serial.Serial(port, 9600)
        self.sm_resolution = resolution_deg

    def __del__(self):
        self.close()

    def rotate_cw(self, degree: float):
        self.sm_pulse = 'w' + str(int(degree / self.sm_resolution)) + ';'
        self.dino.write(self.sm_pulse.encode())
        print(f'ステッピングモーターは　CW方向に {degree} deg, {int(degree / self.sm_resolution)} pulse 回転しろ')

    def rotate_ccw(self, degree: float):
        self.sm_pulse = 'c' + str(int(degree / self.sm_resolution)) + ';'
        self.dino.write(self.sm_pulse.encode())
        print(f'ステッピングモーターは　CCW方向に {degree} deg, {int(degree / self.sm_resolution)} pulse 回転しろ')
