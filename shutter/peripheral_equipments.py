import serial
import time

class Periphaerals(serial.Serial):
    def __init__(self, port: str):
        baudrate = 19200
        try:
            super().__init__(port, baudrate)
            time.sleep(2)
        except:
            print("エラー：ポートが開けませんでした。")

    def __del__(self):
        self.close()

    def shutter_on(self):
        self.run_command(mode='a', command='1')

    def shutter_off(self):
        self.run_command(mode='a', command='0')


    def run_command(self, mode: str, command: str) -> bool:
        data = mode + command + ';'
        self.write(data.encode('utf-8'))

    def read_command(self):
        return self.readline().decode()


if __name__ == '__main__':
    shutter1 = Periphaerals('/dev/tty.usbmodem1101')


