import cv2 as cv
import numpy as np


# 大津の手法
def threshold_otsu(gray, min_value=0, max_value=255):

    # ヒストグラムの算出
    hist = [np.sum(gray == i) for i in range(256)]

    s_max = (0, -10)

    for th in range(256):

        # 領域１と領域２の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])

        # 領域１と領域２の画素値の平均を計算
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0, th)]) / n1
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # 領域間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # 領域間分散の分子が最大の時、領域間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)

        # クラス間分散が最大の時の閾値を取得（分離度が最大となる時）
        t = s_max[0]

        # 算出化した閾値で二値化処理
        gray[gray < t] = min_value
        gray[gray >= t] = max_value
    
    return gray


# 方法２cv.thresholdで簡単に実装
# ret, dst = cv2.threshold(src,threshold, max_value, threshold_type)
# src -> 入力画像
# threshold -> 大津の時は０
# maxValue -> 閾値を満たす画素に与える画素値
# thresholdType -> 閾値の種類(大津式の場合はcv2.THRESH_OTSU)


# 入力画像の読み込み
img = cv.imread("Photos/kokoro.jpg")

# グレースケール変換
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# 方法１
# th = threshold_otsu(gray)
# 方法２
ret, th = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

# 結果を出力
cv.imwrite("Photos/output.png", th)