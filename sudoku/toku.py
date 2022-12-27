def print_tile(tile_data, label='現在のマス'):
    print(f'== {label} ======================')
    print('   a  b  c | d  e  f | g  h  i\n   ````````````````````````````\n1: ', end='')
    for x in range(len(tile_data)):
        if x % 27 == 26 and x < 9 * 7:
            print(f'{tile_data[x]}\n   ---------------------------\n{x // 9 + 2}: ', end='')
        elif x % 9 == 8:
            print(f'{tile_data[x]}\n{x // 9 + 2}: ', end='')
        elif x % 3 == 2:
            print(f'{tile_data[x]} | ', end='')
        else:
            print(f'{tile_data[x]}  ', end='')
    print('\r ')


def error_check(tile_data):
    print('+++++++++++++++++++++++++++++++++++++\n'
          '行のチェック！')
    row_list = split_row(tile_data=tile_data)
    for x in range(9):
        check_list = [values for values in row_list[x] if values != 0]
        # print(check_list)
        if len(check_list) != len(set(check_list)):
            print(f'{x + 1}行に重複があります。残念！')
    print('行のチェック終了')

    print('\n次、列のチェック！')
    column_list = split_column(tile_data=tile_data)
    for y in range(9):
        check_list = [values for values in column_list[y] if values != 0]
        # print(check_list)
        if len(check_list) != len(set(check_list)):
            print(f'{y + 1}列に重複があります。残念！')
    print('列のチェック終了')

    print('\n次、ボックスのチェック！')
    square_list = split_square(tile_data=tile_data)
    for z in range(9):
        check_list = [values for values in square_list[z] if values != 0]
        # print(check_list)
        if len(check_list) != len(set(check_list)):
            print(f'{z + 1}番目のボックスに重複があります。残念！')
    print('ボックスのチェック終了\n++++++++++++++++++++++++++++++++++++++')


def split_row(tile_data):
    row_list = [[0 for i in range(9)] for j in range(9)]
    for y in range(9):
        for x in range(9):
            row_list[y][x] = tile_data[y * 9 + x]
    return row_list


def split_column(tile_data):
    column_list = [[0 for i in range(9)] for j in range(9)]
    for y in range(9):
        for x in range(9):
            column_list[y][x] = tile_data[y + x * 9]
    return column_list


def split_square(tile_data):
    square_list = [[0 for i in range(9)] for j in range(9)]
    row_list = split_row(tile_data)
    for z in range(3):
        for y in range(3):
            for x in range(9):
                square_list[y + z * 3][x] = row_list[x // 3 + z * 3][y * 3 + x % 3]
    return square_list


def judgment_square(no):
    if no < 3 ** 3:
        if no % 9 < 3:
            square_no = 0
        elif no % 9 < 6:
            square_no = 1
        else:
            square_no = 2
    elif no < 2 * 3 ** 3:
        if no % 9 < 3:
            square_no = 3
        elif no % 9 < 6:
            square_no = 4
        else:
            square_no = 5
    else:
        if no % 9 < 3:
            square_no = 6
        elif no % 9 < 6:
            square_no = 7
        else:
            square_no = 8
    return square_no



if __name__ == '__main__':
    sudoku_list = [0] * 9 ** 2
    coordinate_dict = {}
    coordinate_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    basic_list = [i + 1 for i in range(9)]
    total = 0

    # 初期化 いったんゼロにする
    for tile in range(len(sudoku_list)):
        key = coordinate_list[(tile % 9)] + str(tile // 9 + 1)
        coordinate_dict[key] = tile

    # print_tile(tile_data=sudoku_list, label='初期化した')

    # 既知のマスを設定
    known_values = {'b1': 5, 'd1': 1, 'f1': 8, 'h1': 9,
                    'c2': 3, 'e2': 4, 'g2': 8, 'i2': 1,
                    'd3': 6, 'f3': 7, 'h3': 3,
                    'a4': 3, 'e4': 5, 'g4': 1, 'i4': 7,
                    'b5': 8, 'd5': 4, 'f5': 9, 'h5': 5,
                    'a6': 5, 'c6': 2, 'e6': 6, 'i6': 8,
                    'b7': 4, 'd7': 7, 'f7': 6,
                    'a8': 7, 'c8': 9, 'e8': 8, 'g8': 5,
                    'b9': 3, 'd9': 5, 'f9': 4, 'h9': 2}

    # 値をいれるよ
    for key, value in known_values.items():
        sudoku_list[coordinate_dict[key]] = value
    print_tile(tile_data=sudoku_list, label='値が入った状態')

    total = sum(sudoku_list)
    print(f'Total: {total} 出発 !')

    # 既知マスの重複チェック
    # error_check(tile_data=sudoku_list)

    while total < 45 * 9:
        for i in range(len(sudoku_list)):
            # 要素が0のものに入る候補を決めていくよ
            if sudoku_list[i] == 0:
                # 横軸で入りうるもの
                row_data = split_row(tile_data=sudoku_list)
                candidate_row_value = set(row_data[i // 9]) ^ set(basic_list)
                # 縦軸で入りうるもの
                column_data = split_column(tile_data=sudoku_list)
                candidate_column_value = set(column_data[i % 9]) ^ set(basic_list)
                # ボックスで入りうるもの
                square_data = split_square(tile_data=sudoku_list)
                candidate_square_value = set(square_data[judgment_square(no=i)]) ^ set(basic_list)
                # すべての候補のマージ
                candidate_and_value = list(set(candidate_row_value) & set(candidate_column_value) & set(candidate_square_value))
                # 絞られた候補を代入
                if len(candidate_and_value) == 2:
                    sudoku_list[i] = candidate_and_value[1]
                    key = [j for j, v in coordinate_dict.items() if v == i]
                    print(f'マスの{key}に、{candidate_and_value[1]}入れます。')
                    print_tile(tile_data=sudoku_list, label=str(i))

        total = sum(sudoku_list)
        print(f'{i}番目にして、Total {total}。')

    error_check(tile_data=sudoku_list)
    print('終わった。ご苦労さま。')
