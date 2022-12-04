import time

from dino import Motor

motor1 = Motor(port='/dev/tty.usbmodem1101')

for i in range(12):
    motor1.rotate_cw(30.0)