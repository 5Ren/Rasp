import ffmpeg

# Blu-rayディスク内のファイルを指定して変換
input_file = r"C:\Users\YamaLab-38\Downloads\BD\BDAV\STREAM\00001.m2ts"  # Blu-rayの映像ファイル
output_file = "output.mp4"

# ffmpegで変換処理
try:
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec="libx264", acodec="aac")
        .run()
    )
    print("変換が完了しました:", output_file)
except ffmpeg.Error as e:
    print("エラーが発生しました:", e.stderr.decode())  # 詳細なエラー出力