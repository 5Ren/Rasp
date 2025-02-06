def print_tile(tile_data, label='現在のマス'):
    print(f'== {label} ========================')
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
    print('+++++++++++++++++++++++++++++++\n'
          '行のチェック！: ', end='')
    row_list = split_row(tile_data=tile_data)
    for x in range(9):
        check_list = [values for values in row_list[x] if values != 0]
        if len(check_list) != len(set(check_list)):
            print(f'{x + 1}行に重複があります。残念！')
    print('✅')

    print('次、列のチェック！: ', end='')
    column_list = split_column(tile_data=tile_data)
    for y in range(9):
        check_list = [values for values in column_list[y] if values != 0]
        if len(check_list) != len(set(check_list)):
            print(f'{y + 1}列に重複があります。残念！')
    print('✅')

    print('次、ボックスのチェック！: ', end='')
    square_list = split_square(tile_data=tile_data)
    for z in range(9):
        check_list = [values for values in square_list[z] if values != 0]
        # print(check_list)
        if len(check_list) != len(set(check_list)):
            print(f'{z + 1}番目のボックスに重複があります。残念！')
    print('✅\n+++++++++++++++++++++++++++++++')


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


def attack_row_and_column(tile_data):
    for i in range(len(tile_data)):
        # 要素が0のものに入る候補を決めていくよ
        if tile_data[i] == 0:
            # 横軸で入りうるもの
            row_data = split_row(tile_data=tile_data)
            candidate_row_value = set(row_data[i // 9]) ^ set(basic_list)
            # 縦軸で入りうるもの
            column_data = split_column(tile_data=tile_data)
            candidate_column_value = set(column_data[i % 9]) ^ set(basic_list)
            # ボックスで入りうるもの
            square_data = split_square(tile_data=tile_data)
            candidate_square_value = set(square_data[judgment_square(no=i)]) ^ set(basic_list)
            # すべての候補のマージ
            candidate_and_value = list(set(candidate_row_value) & set(candidate_column_value)
                                       & set(candidate_square_value))
            # 絞られた候補を代入
            if len(candidate_and_value) == 2:
                tile_data[i] = candidate_and_value[1]
    return tile_data


def attack_box(tile_data):
    row_data = split_row(tile_data=tile_data)
    column_data = split_column(tile_data=tile_data)
    square_data = split_square(tile_data=tile_data)
    for num in range(9):
        if row_data[num]

if __name__ == '__main__':
    # 初期化
    sudoku_list = []
    coordinate_dict = {}
    coordinate_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    basic_list = [i + 1 for i in range(9)]
    total = 0

    # ディクショナリのKeyとリストの値を紐付け
    for address_value in range(len(sudoku_list)):
        key = coordinate_list[(address_value % 9)] + str(address_value // 9 + 1)
        coordinate_dict[key] = address_value

    # 問題をとってくる
    print('Please question!')
    for i in range(9):
        sudoku_list.append(input('').split(','))
    print('Thank you!')

    # 2次元リストから1次元リストへ (わざわざ) 直す
    sudoku_list = sum(sudoku_list, [])

    # 空白に0を入れる
    for i in range(len(sudoku_list)):
        num = sudoku_list[i]
        if num == '':
            num = 0
        sudoku_list[i] = int(num)

    # 問題を出力
    print_tile(tile_data=sudoku_list, label='問題')

    while total < 45 * 9:
        # 縦、横、ボックス内の重複する候補を入れる
        sudoku_list = attack_row_and_column(tile_data=sudoku_list)

        # 値の合計を計算
        total = sum(sudoku_list)

    print_tile(tile_data=sudoku_list, label='答え')
    error_check(tile_data=sudoku_list)
    print('終わった。ご苦労さま。')
