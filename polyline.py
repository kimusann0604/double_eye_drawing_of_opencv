import cv2
import sys
import numpy as np

# imread : 画像ファイルを読み込んで、多次元配列(numpy.ndarray)にする。
# imreadについて : https://kuroro.blog/python/wqh9VIEmRXS4ZAA7C4wd/
# 第一引数 : 画像のファイルパス
# 戻り値 : 行 x 列 x 色の三次元配列(numpy.ndarray)が返される。
img = cv2.imread('bbc6c9066fa41d8de797b46e34d91a39.jpg')

# 画像ファイルが正常に読み込めなかった場合、プログラムを終了する。
if img is None:
    sys.exit("Could not read the image.")

# 第一引数(必須) : 多次元配列(numpy.ndarray)
# 第二引数(必須) : 複数の座標。numpy.ndarray型。
# 第三引数(必須) : 複数の座標(第二引数)の始点と終点を結ぶのか指定する。
# 第四引数(必須) : 折れ線の色を指定する。B(Blue)G(Green)R(Red)形式で指定する。tuple型。
# thickness : 折れ線の太さ(px)を指定する。
cv2.polylines(img, [np.array([(200, 200), (210, 230), (300, 260), (350, 300)])], False, (255, 255, 0), thickness=3)

# imwrite : 画像の保存を行う関数
# 第一引数 : 保存先の画像ファイル名
# 第二引数 : 多次元配列(numpy.ndarray)
# <第二引数の例>
# [
# [
# [234 237 228]
# ...
# [202 209 194]
# ]
# [
# [10 27 16]
# ...
# [36 67 46]
# ]
# [
# [34 51 40]
# ...
# [50 81 60]
# ]
# ]
# imwriteについて : https://kuroro.blog/python/i0tNE1Mp8aEz8Z7n6Ggg/
cv2.imwrite('output.jpg', img)