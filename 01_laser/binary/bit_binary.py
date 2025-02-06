import ctypes

prog_no = int(25)

# prog_no_bi = format(prog_no, 'b')
# prog_no_bi_1st = format((prog_no // 64), 'b')
# prog_no_bi_2nd = format(((prog_no % 64) // 32), 'b')
#
#
# print(f'1st: {prog_no_bi_1st}')
# print(f'2nd: {prog_no_bi_2nd}')

prog_no_bi = '{:07b}'.format(prog_no)
print(prog_no_bi)
print(f'1st: {int(prog_no_bi[-1])}')
print(f'2nd: {int(prog_no_bi[-2])}')
print(f'3rd: {int(prog_no_bi[-3])}')
print(f'4th: {int(prog_no_bi[-4])}')
print(f'5th: {int(prog_no_bi[-5])}')
print(f'6th: {int(prog_no_bi[-6])}')
print(f'7th: {int(prog_no_bi[-7])}')

a = ctypes.c_ubyte(int(prog_no_bi[-1]))
print(a)
