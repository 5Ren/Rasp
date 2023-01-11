sudoku_list = []

print('Please question!')
for i in range(9):
    sudoku_list.append(input('').split(','))

print(sudoku_list)
sudoku_list = sum(sudoku_list, [])

for i in range(len(sudoku_list)):
    num = sudoku_list[i]
    if num == '':
        num = 0
    sudoku_list[i] = int(num)

print(sudoku_list)