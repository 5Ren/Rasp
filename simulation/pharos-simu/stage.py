import numpy
import serial

class Stage(serial.Serial):
    def __init__(self, port: str):
        super().__init__(port=port)
        self.speed_S = 75
        self.speed_F = 750
        self.speed_R = 100

    def __del__(self):
        pass

    def communicate_rs232c(self, cmd) -> bool:
        pass

    def reset_all_axis(self) -> bool:
        pass

    def reset_any_axis(self, axis: int) -> bool:
        pass

    def move_position_relative(self, axis: int, position: int) -> bool:
        pass

    def move_position_absolute(self, axis: int, position: int) -> bool:
        pass

    def move_position_absolute_jog(self, axis: int, position: int) -> bool:
        pass

    def get_current_positon(self) -> tuple[int, int, int]:
        pass

    def set_stage_speed(self, speed_S: int, speed_F: int, speed_R: int) -> bool:
        pass


