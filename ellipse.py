import cv2
import numpy as np


img = np.zeros((300, 300, 3), dtype=np.uint8)

cv2.ellipse(
    img,
    (150, 150),
    (50, 70),
    angle=90,
    startAngle=250,
    endAngle=100,
    color=(255, 0, 0),
    thickness=2,
)
cv2.imshow("Ellipse", img)

cv2.waitKey(0)
cv2.destroyAllWindows()  
