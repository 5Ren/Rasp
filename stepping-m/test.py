import time

from dino import Motor

motor1 = Motor(port='/dev/tty.usbmodem1101')

for i in range(36):
    motor1.rotate_cw(10)
    time.sleep(1)

# motor1.rotate_zero()