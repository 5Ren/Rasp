import os
import pandas as pd


def txt_to_excel(directory):
    # ディレクトリの名前をExcelファイル名に使用
    output_filename = os.path.basename(directory.rstrip('/')) + '.xlsx'

    # 指定ディレクトリ内のすべてのTXTファイルを取得
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    if not txt_files:
        print("No TXT files found in the directory.")
        return

    # 最初のTXTファイルを開いて、x軸データを取得
    first_file_path = os.path.join(directory, txt_files[0])
    with open(first_file_path, 'r', encoding='CP932') as file:
        lines = file.readlines()
        x_data = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 1:
                x_data.append(parts[0])

    # データフレームの初期化
    df = pd.DataFrame()
    df['X'] = x_data

    # 各TXTファイルのy軸データを取得してデータフレームに追加
    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)
        with open(file_path, 'r', encoding='CP932') as file:
            lines = file.readlines()
            y_data = []
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    y_data.append(parts[1])
                else:
                    y_data.append(None)  # データが不足している場合はNoneを追加

            # x_dataとy_dataの長さが一致しない場合、y_dataの長さをx_dataに合わせる
            if len(y_data) < len(x_data):
                y_data.extend([None] * (len(x_data) - len(y_data)))
            elif len(y_data) > len(x_data):
                y_data = y_data[:len(x_data)]

            df[txt_file] = y_data

    # 1行目にファイル名を設定
    df.columns = ['X'] + [f'{txt_file}' for txt_file in txt_files]

    # データフレームをExcelファイルに出力
    df.to_excel(output_filename, index=False)
    print(f'Successfully saved to {output_filename}')


# 使用例
directory_path = r'ITO_slit5nm'  # ここに対象ディレクトリのパスを入力
txt_to_excel(directory_path)
