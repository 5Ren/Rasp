import cv2

# 画像を読み込み
image = cv2.imread('/Users/ren/Downloads/DSCF3161.JPG')

# ノイズ除去 (Non-local Means Denoising)
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 結果を保存
cv2.imwrite('denoised_image.jpg', denoised_image)
