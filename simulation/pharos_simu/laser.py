import csv
import os

class Laser:
    def __init__(self, device_name: str, laser_channel: int, ra_channel: int):
        self.device_name = device_name
        self.laser_channel = laser_channel
        self.ra_channel = ra_channel


    def laser_on(self) -> None:


        pass

    def laser_off(self) -> None:
        pass

    def get_ra_status(self) -> str:
        pass

    def aio_exit(self):
        pass

    def aio_reset(self):
        pass


