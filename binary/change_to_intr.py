light_path_bit = [0] * 3
light_path_no = [1, 1, 1, 1, 1, 1, 1]
no = 0

for num in range(len(light_path_no)):
    no += int(light_path_no[num] * 2 ** num)

    print(no)