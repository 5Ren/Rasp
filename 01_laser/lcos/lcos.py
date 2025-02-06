import numpy as np
import cv2
import os


class Lcos:
    beam_bundle_number = None

    screen_width = None
    screen_height = None
    grating_pixel = None

    grating1 = None
    grating2 = None
    grating3 = None
    grating4 = None
    grating5 = None

    def __init__(self, beam_number: int, pixel_number: int, lcos_screen_width: int = 1272,
                 lcos_screen_height: int = 1024):
        self.screen_height = lcos_screen_height
        self.screen_width = lcos_screen_width

        self.beam_bundle_number = beam_number
        self.grating_pixel = pixel_number

        for i in range(1, beam_number):
            exec_command = 'print(str_' + str(i) + ')'
            exec(exec_command)


if __name__ == '__main__':
    lcos1 = Lcos(beam_number=2, pixel_number=2)
