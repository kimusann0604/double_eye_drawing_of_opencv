import boto3
import cv2
import numpy as np

# AWSのrekognitionのAPI読み込み
rekognition_client = boto3.client(
    'rekognition',
    aws_access_key_id='Aa55209eb-e8e8-4496-9e15-1c09407977f5',
    aws_secret_access_key='JVWKEfMKoSJtLeqrK1P4ezeTwlHo5iNOeXsxCmdk',
    region_name='ap-northeast-1'
)

# 顔のランドマークの値を表示させる
with open('201908060474 (1).jpg', 'rb') as image_file:
    image_bytes = image_file.read()

response = rekognition_client.detect_faces(
    Image={'Bytes': image_bytes},
    Attributes=['ALL']
)

for face_detail in response['FaceDetails']:
    print("顔のランドマーク:")
    for landmark in face_detail['Landmarks']:
        print(f"{landmark['Type']} - X: {landmark['X']}, Y: {landmark['Y']}")
        


# 目の上の場所特定
image = cv2.imread('image_path')

y_offset = int(image.shape[0] * 0.05)

left_eye_up = (int(0.4364071190357208 * image.shape[1]), int(0.33588847517967224 * image.shape[0]) + y_offset)
left_eye_down = (int(0.43726876378059387 * image.shape[1]), int(0.3522436022758484 * image.shape[0]) + y_offset)

right_eye_up = (int(0.5925089120864868 * image.shape[1]), int(0.33434101939201355 * image.shape[0]) + y_offset)
right_eye_down = (int(0.5912229418754578 * image.shape[1]), int(0.35067176818847656 * image.shape[0]) + y_offset)

# 二重ラインを描画する (目の上下にラインを描画)
x1=392
y1=290
x2=240
y2=280
#opencvの描画機能で二重のラインを描く
image = cv2.ellipse(
    image,
    (x1,y1),(30, 40),angle=90,startAngle=230,endAngle=95,color=(68,92,135),thickness=1,)
image = cv2.ellipse(
    image,
    (x2,y2),(30, 40),angle=93,startAngle=270,endAngle=130,color=(68,92,135),thickness=1,)
#angleが向きstartAngle,endAngleが線の始まりと線の終わりthicknessが太さ
cv2.imshow("CarrotCake",image)
cv2.waitKey(0)
# 結果を表示
cv2.destroyAllWindows()