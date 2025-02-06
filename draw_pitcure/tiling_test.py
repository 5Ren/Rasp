split_num = 3  # 分割数は、target_stretch_length_microns が入る数にすること。

scanning_field_um = 10 * 10 ** 3  # レンズでスキャンできる領域
processing_area_length_um = split_num * scanning_field_um  # 加工する領域の高さ（μm）
x_correction_um = 300
y_correction_um = 130


stage_tiling_list = []
corrected_tiling_x_um = scanning_field_um - x_correction_um
corrected_tiling_y_um = scanning_field_um - y_correction_um

for region_y in range(split_num):
    for region_x in range(split_num):
        if region_x == 0:
            if region_y == 0:
                x_posi_um = 0
                y_posi_um = 0
            else:
                x_posi_um = corrected_tiling_x_um * (split_num - 1)
                y_posi_um = corrected_tiling_y_um
        elif region_x < split_num:
            x_posi_um = -1 * corrected_tiling_x_um
            y_posi_um = 0

        stage_tiling_list.append([x_posi_um, y_posi_um])

print(stage_tiling_list)