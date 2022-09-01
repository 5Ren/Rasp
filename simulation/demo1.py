from pharos_simu.laser import Laser
from pharos_simu.stage import Stage

STAGE_PORT = 'COM3'
stage1 = Stage(port=STAGE_PORT)

stage1.reset_all_axis()
stage1.reset_any_axis(1)