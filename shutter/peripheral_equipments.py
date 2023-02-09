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
        self.shutter_close()
        self.close()

    def shutter_open(self) -> bool:
        print('shutter open > ', end='')
        shutter_mode = 'a'
        open_command = '1'
        result = self.__run_command(mode=shutter_mode, command=open_command)
        if result == 'OPEN':
            print('SUCCESS')
            return True
        else:
            print('ERROR')
            return False

    def shutter_close(self) -> bool:
        print('shutter close > ', end='')
        shutter_mode = 'a'
        close_command = '0'
        result = self.__run_command(mode=shutter_mode, command=close_command)
        if result == 'CLOSE':
            print('SUCCESS')
            return True
        else:
            print('ERROR')
            return False

    def __communicate_device(self, data: str) -> str:
        self.write(data.encode('utf-8'))
        return_message = self.readline().decode()[:-2]
        return return_message

    def __run_command(self, mode: str, command: str) -> str:
        data = mode + command + ';'
        return_message = self.__communicate_device(data)
        return return_message


if __name__ == '__main__':
    com_name = '/dev/tty.usbmodem101'
    shutter1 = Periphaerals(port=com_name)

    shutter1.shutter_open()
    shutter1.shutter_close()