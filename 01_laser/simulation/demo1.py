from pharos_simu.laser import Laser
from pharos_simu.stage import Stage

STAGE_PORT = 'COM3'
CONTEC_BOARD_NAME = 'AIO000'
CONTEC_LASER_CHANNEL = 0
CONTEC_RA_STATUS_CHANNEL = 1

stage1 = Stage(port=STAGE_PORT)
laser1 = Laser(device_name=CONTEC_BOARD_NAME,
                           laser_channel=CONTEC_LASER_CHANNEL,
                           ra_channel=CONTEC_RA_STATUS_CHANNEL)
stage1.reset_all_axis()
stage1.reset_any_axis(1)

laser1.laser_on()