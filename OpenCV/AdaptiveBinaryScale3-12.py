# 適応的二値化処理（光や影の影響を受けにくい）
# cv2.adaptiveThresholdメソッドを使えば１行で実装可能
# dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# src -> 入力画像
# maxValue -> 閾値を満たす画素に与える画素値
# adaptiveMethod -> 閾値の計算方法（cv2.ADAPTIVE_THRESH_MEAN_C なら近傍画素値の平均値、
#                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C ならガウシアンの重みつき平均値）
# thresholdType -> 閾値の種類(THRESH_BINARY or THRESH_BINARY_INV)
# blockSize -> 近傍領域のサイズ（３なら８近傍）
# C -> 定数（計算した閾値からCを引いた値が閾値になる）

# 方法１（Numpy実装）
import cv2
import numpy as np


def threshold(src, ksize=3, c=2):
    
    # 局所領域の幅
    d = int((ksize-1)/2)

    # 画像の高さと幅
    h, w = src.shape[0], src.shape[1]
    
    # 出力画像用の配列（要素は全て255）
    dst = np.empty((h, w))
    dst.fill(255)
    
    # 局所領域の画素数
    N = ksize**2

    for y in range(0, h):
        for x in range(0, w):
            # 局所領域内の画素値の平均を計算し、閾値に設定
            t = np.sum(src[y-d:y+d+1, x-d:x+d+1]) / N

            # 求めた閾値で二値化処理
            if(src[y][x] < t - c): 
                dst[y][x] = 0
            else:
                dst[y][x] = 255

    return dst

# 入力画像を読み込み
img = cv2.imread("Photos/kokoro.jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 方法1
dst = threshold(gray, ksize=11, c=13)
# 方法2       
# dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

# 結果を出力
cv2.imwrite("Photos/output.jpg", dst)
