import os
from glob import glob

# 対象フォルダ
input_folder = './ref_data'
file_paths = glob(os.path.join(input_folder, '*.txt'))

for path in file_paths:
    try:
        # cp932 (Shift_JIS拡張) で読み込み
        with open(path, 'r', encoding='cp932') as f:
            lines = f.readlines()

        cleaned_lines = lines[2:]  # 上2行をスキップ

        # 同じエンコーディングで上書き
        with open(path, 'w', encoding='cp932') as f:
            f.writelines(cleaned_lines)

        print(f"✔ 処理完了: {os.path.basename(path)}")

    except Exception as e:
        print(f"[ERROR] {os.path.basename(path)}: {e}")
