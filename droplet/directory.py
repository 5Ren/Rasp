import os

def list_avi_files(directory_path):
    """
    指定されたディレクトリパス内にあるAVIファイル名のリストを返します。
    """
    avi_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".avi"):
            avi_files.append(filename)
    return avi_files

if __name__ == '__main__':
    avi_files_list = list_avi_files("D://High_speed_camera//0502_after3day_2uL_h40mm//1-9")
    print(avi_files_list)