import numpy

class Scanner:
    def __init__(self, device_name):
        self.DEVICE_NAME = device_name

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
