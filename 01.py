import cv2
import numpy as np

# 顔画像を読み込む
image = cv2.imread('face_image.jpg')

# シフト量を定義 (画像全体の高さの5%分下げる)
y_offset = int(image.shape[0] * 0.05)

# 検出された目のランドマーク (例: 左目の上下)
left_eye_up = (int(0.4364071190357208 * image.shape[1]), int(0.33588847517967224 * image.shape[0]) + y_offset)
left_eye_down = (int(0.43726876378059387 * image.shape[1]), int(0.3522436022758484 * image.shape[0]) + y_offset)

# 検出された右目のランドマーク (例: 右目の上下)
right_eye_up = (int(0.5925089120864868 * image.shape[1]), int(0.33434101939201355 * image.shape[0]) + y_offset)
right_eye_down = (int(0.5912229418754578 * image.shape[1]), int(0.35067176818847656 * image.shape[0]) + y_offset)

# 二重ラインを描画する (目の上下にラインを描画)
cv2.line(image, left_eye_up, left_eye_down, (0, 255, 0), 2)
cv2.line(image, right_eye_up, right_eye_down, (0, 255, 0), 2)

# 結果を表示
cv2.imshow('Double Eyelid Simulation with Adjusted Position', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# このコードで座標を取り組んで# Draw an ellipse
x1=392
y1=290
x2=240
y2=280
img = cv2.ellipse(
    img,
 (x1,y1),(30, 40),angle=90,startAngle=230,endAngle=95,color=(68,92,135),thickness=1,)
img = cv2.ellipse(
    img,
    (x2,y2),(30, 40),angle=93,startAngle=270,endAngle=130,color=(68,92,135),thickness=1,)
#angleが向きstartAngle,endAngleが線の始まりと線の終わりthicknessが太さ
cv2.imshow("CarrotCake",img)
cv2.waitKey(0)

def draw_double_eyelid(img, eye_center, size, angle, start_angle, end_angle, color=(68, 92, 135), thickness=1):
    cv2.ellipse(
        img,
        eye_center,
        size,
        angle,
        start_angle,
        end_angle,
        color,
        thickness
    )

# 左目の二重まぶたを描画
left_eye_center = ((left_eye_up[0] + left_eye_down[0]) // 2, (left_eye_up[1] + left_eye_down[1]) // 2)
draw_double_eyelid(image, left_eye_center, (30, 10), 90, 230, 95)

# 右目の二重まぶたを描画
right_eye_center = ((right_eye_up[0] + right_eye_down[0]) // 2, (right_eye_up[1] + right_eye_down[1]) // 2)
draw_double_eyelid(image, right_eye_center, (30, 10), 93, 270, 130)

# 結果を表示
cv2.imshow('Double Eyelid Simulation with Adjusted Position', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
