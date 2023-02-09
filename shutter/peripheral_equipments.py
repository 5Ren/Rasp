import serial
import time

class Periphaerals(serial.Serial):
    def __init__(self, port: str) -> None:
        baudrate = 19200
        try:
            super().__init__(port, baudrate)
            time.sleep(2)
        except:
            print("エラー：ポートが開けませんでした。")

    def __del__(self) -> None:
        self.close()

    def shutter_on(self) -> bool:
        shutter_mode = 'a'
        open_command = '1'
        self.__run_command(mode=shutter_mode, command=open_command)
        print(self.__read_message())
        if self.__read_message() == 'OPEN':
            return True

    def shutter_off(self) -> bool:
        shutter_mode = 'a'
        close_command = '0'
        self.__run_command(mode=shutter_mode, command=close_command)
        print(self.__read_message())
        if self.__read_message() == 'CLOSE':
            return True

    def __run_command(self, mode: str, command: str) -> None:
        data = mode + command + ';'
        self.write(data.encode('utf-8'))

    def __read_message(self) -> str:
        return self.readline().decode()

    def get_status(self):
        return self.__read_message()


if __name__ == '__main__':
    com_name = '/dev/tty.usbmodem101'
    shutter1 = Periphaerals(port=com_name)

    print(shutter1.shutter_on())
    time.sleep(1)
    print(shutter1.shutter_off())
    # print(shutter1.get_status())