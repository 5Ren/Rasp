import qrcode

data = "https://example.com"

qr = qrcode.QRCode(
    version=2,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
)
qr.add_data(data)
qr.make(fit=True)

# True/False のマトリクスを取得
matrix = qr.get_matrix()

# 座標を取得（y,x の順）
black_cells = []
for y, row in enumerate(matrix):
    for x, cell in enumerate(row):
        if cell:  # 黒セル
            black_cells.append( (x, y) )

# 結果を表示
print("黒セルの座標リスト：")
for coord in black_cells:
    print(type(coord))
