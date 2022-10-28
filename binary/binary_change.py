prog_no = int(128)

print(f'{prog_no=}')
print(f'DN: {int(prog_no)}')
print(f'FEX: {hex(prog_no)}')
print(f'Binary: {bin(prog_no)}\n')

print(f'bi2: {format(prog_no, "b")}')
print(f'Fex2: {format(prog_no, "x")}')

a = '{:07b}'.format(prog_no)
print(a)
print(type(a))
