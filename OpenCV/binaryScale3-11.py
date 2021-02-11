import cv2 as cv
import numpy as py

# 方法１　Numpyでアルゴリズムを書いて実装

# 閾値
threshold_val = 127

# 入力画像の読み込み
img = cv.imread("Photos/kokoro.jpg")

# グレースケール変換
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# 出力画像用の配列生成
threshold_img = gray.copy()

# Numpyで実装
threshold_img[gray < threshold_val] = 0
threshold_img[gray >= threshold_val] = 255

# 結果を出力
cv.imwrite("Photos/test.png", threshold_img)

# 方法２ cv2.thresholdで簡単に実装
# ret, dst = cv.threshold(src, thresh, maxval, threshold_type, dst)
# src -> 入力画像(グレースケール)
# threshold -> 閾値
# maxval -> 二値化した時の最大値（255なら真っ白)
# threshold_type -> 単純な二値化ならcv2.THRESH_BINARYを使用
# dst -> 出力画像
