import cv2
import numpy as np

# 画像を作成（背景が白）
image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# 縁を描くための座標（四角形）
points = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])

# 輪郭（縁）を描画、Trueにすると閉じた図形になる
cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=3)

# 画像を表示
cv2.imshow('Image with edge', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
