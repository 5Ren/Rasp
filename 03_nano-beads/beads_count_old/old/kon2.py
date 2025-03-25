import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

img_raw = cv2.imread('../ITO-test/ITO-100PER-x50K.bmp', 1)
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

#画像の高さ, 幅を取得
h, w = img.shape

#画像の前処理(拡大)
mag = 3
img = cv2.resize(img, (w*mag, h*mag))

#画像の前処理(ぼかし)
img_blur = cv2.GaussianBlur(img,(5,5),0)

#2値画像を取得
ret,th = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#モルフォロジー変換(膨張)
kernel = np.ones((3,3),np.uint8)
th = cv2.dilate(th,kernel,iterations = 1)

#画像を保存
cv2.imwrite('thresholds.png', th)

#Fill Holes処理
th_fill = ndimage.binary_fill_holes(th).astype(int) * 255
cv2.imwrite('thresholds_fill.png', th_fill)

#境界検出と描画
cnt, __ = cv2.findContours(th_fill.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_raw = cv2.resize(img_raw, (w*mag, h*mag))
img_cnt = cv2.drawContours(img_raw, cnt, -1, (0,255,255), 1)
cv2.imwrite('cnt.png', img_cnt)

Areas = []
Circularities = []
Eq_diameters = []

for i in cnt:
    #面積(px*px)
    area = cv2.contourArea(i)
    Areas.append(area)

    #円形度
    arc = cv2.arcLength(i, True)
    circularity = 4 * np.pi * area / (arc * arc)
    Circularities.append(circularity)

    #等価直径(px)
    eq_diameter = np.sqrt(4*area/np.pi)
    Eq_diameters.append(eq_diameter)

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.title("Areas (px^2)")
plt.hist(Areas, bins=25, range=(0, 150), rwidth=0.7)
plt.subplot(2, 2, 2)
plt.title("Circularity")
plt.hist(Circularities, bins=25, range=(0.5, 1), rwidth=0.7)
plt.subplot(2, 2, 3)
plt.title("Equal Diameters (px)")
plt.hist(Eq_diameters, bins=25, range=(3.0, 15.0), rwidth=0.7)
plt.show()
