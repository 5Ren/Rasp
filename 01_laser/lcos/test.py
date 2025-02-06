import numpy as np
import cv2
import os

width = 1272
height = 1272

# 折り返し数
j = 150

#諧調値?
step = 255

#歪み補正パターン読み込み
# hosei = cv2.imread('CAL_751_515nm.bmp',0)

# 初期化 ##############################################################
target1 = np.zeros((width, height), np.float64)
target2 = np.zeros((width, height), np.float64)
target3 = np.zeros((width, height), np.float64)
target4 = np.zeros((width, height), np.float64)
hologram = np.zeros((width, height), np.complex128)
x = np.linspace(0,2*np.pi,width)
# 初期化-end ###########################################################

# 関数 #################################################################
def normalization(origin):
    maxi = np.max(origin)
    mini = np.min(origin)
    norm = ((origin - mini) / (maxi - mini))
    return norm

def inamp(r):
    mask = np.where(r <= br, 1, 0)
    peakAmp = np.sqrt((bp**2)/(np.pi * br**2))
    in_amp = peakAmp * np.exp((-2 * r**2)/br**2)
    return in_amp
# 関数-end ############################################################

# CGH表示 #############################################################
# cv2.namedWindow('window', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.moveWindow("window", 1921, 0)
# CGH表示-end ##########################################################

#ブレーズド回折格子作成
for i in range(width):                              # targetのタネ
    target1[i] = np.fmod(-1 * j * x[i], 2 * np.pi)
    a = width - i -1
    target2[i] = np.fmod(-1 * j * x[a], 2 * np.pi)

target1 = np.uint8(normalization(target1) * 255)    # target1
target3 = target1.T                                 # target3
target2 = np.uint8(normalization(target2) * 255)    # target2
target4 = target2.T                                 # target4

# ホログラムを足す（前処理：位相に変換）
phase1 = target1 / np.max(target1) * 2 * np.pi
phase2 = target2 / np.max(target2) * 2 * np.pi
phase3 = target3 / np.max(target3) * 2 * np.pi
phase4 = target4 / np.max(target4) * 2 * np.pi

# 複素表示にして足す
hologram = np.exp(1j*phase3)+np.exp(1j*phase4)#+np.exp(1j*phase1)+np.exp(1j*phase2)      # カエルとしたらココ！
hologram = np.angle(hologram)
hologram = np.where(hologram < 0, hologram + np.pi * 2, hologram)
hologram = np.fmod(hologram, 2 * np.pi)
hologram = np.uint8(normalization(hologram) * 255)

# ホログラムを補正データ用にスライス
holo0 = hologram[int((1272-1024)/2):int((1272-1024)/2)+1024, :]

# ホログラムと補正データの足し合わせ
# z0 = holo0 + hosei      # カエルとしたらココ！
z1 = np.fmod(z0,256)*step/255
z2 = z1.astype("uint8")

# CGH表示
cv2.imshow('', z2)
cv2.imshow('', z2)
cv2.imwrite('beam4.bmp', z2)
cv2.waitKey(0)


# ガウスビーム作成
uh = np.zeros((width,height),np.complex128)

bp = 1
gb = 1
br = 600/1024
xx,yy = np.meshgrid(np.linspace(-1.0,1.0,width),np.linspace(-1.0,1.0,))
r = np.sqrt((xx-0)**2+(yy-0)**2)
beam = inamp(r)

# ビーム照射
holo1 = hologram/np.max(hologram) * 2 * np.pi
uh.real = beam * np.cos(holo1)
uh.imag = beam * np.sin(holo1)

# レンズを通る（集光）
fft = np.fft.fftshift(uh)
fft = np.fft.fft2(fft)/width
fft = np.fft.fftshift(fft)

# rec保存(path)
rec_path = os.path.dirname(os.path.abspath(__file__))+'/rec/'
os.makedirs(rec_path,exist_ok =True)

# 強度変換
intensity = np.abs(fft)**2
intensity = np.uint8(normalization(intensity)*255)
cv2.imshow('Intensity',intensity)
cv2.imwrite('test'+'.bmp',intensity)
cv2.waitKey(0)
