import boto3
import cv2
import os

class FaceLandmarkProcessor:
    
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-1'):
        '''キーの設定'''
        
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    def detect_faces_landmark(self, image_bytes):
        '''顔認証開始'''
        
        response = self.rekognition_client.detect_faces(
            Image={'Bytes': image_bytes}, 
            Attributes=['ALL']
        )
        return response

    def draw_double_eye_ellipse(self, img, eye_center, size, angle, start_angle, end_angle, color=(68, 92, 135), thickness=1):
        '''弧を描く関数'''
        
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

    def left_eye_point(self, landmarks):
        '''左目の座標'''
        
        left_eye_left = landmarks['leftEyeLeft']
        left_eye_right = landmarks['leftEyeRight']
        left_eye_up = landmarks['leftEyeUp']
        left_eye_down = landmarks['leftEyeDown']
        return left_eye_left,left_eye_right,left_eye_up,left_eye_down
    
    def right_eye_point(self, landmarks):
        '''右目の座標'''
        
        right_eye_left = landmarks['rightEyeLeft']
        right_eye_right = landmarks['rightEyeRight']
        right_eye_up = landmarks['rightEyeUp']
        right_eye_down = landmarks['rightEyeDown']
        return right_eye_left,right_eye_right,right_eye_up,right_eye_down
    
    def calculate_Eye_Position_Draw(self, image, landmarks, roll_angle, height_factor=0.99):
        
        #左目、右目の座標を取得
        left_eye_left, left_eye_right, left_eye_up, left_eye_down = self.left_eye_point(landmarks)
        right_eye_left, right_eye_right, right_eye_up, right_eye_down = self.right_eye_point(landmarks)
        
        #左と右のpointから中心
        left_eye_center = ((left_eye_left[0] + left_eye_right[0]) // 2, 
                           (left_eye_up[1] + left_eye_down[1]) // 2)
        
        right_eye_center = ((right_eye_left[0] + right_eye_right[0]) // 2, 
                            (right_eye_up[1] + right_eye_down[1]) // 2)
        
        #目の長さ、高さ
        left_eye_width = left_eye_right[0] - left_eye_left[0]
        left_eye_height = left_eye_down[1] - left_eye_up[1]
        
        right_eye_width = right_eye_right[0] - right_eye_left[0]
        right_eye_height = right_eye_down[1] - right_eye_up[1]

        #二重の描く位置
        right_eyelid_height = int(right_eye_up[1] * height_factor)
        left_eyelid_height = int(left_eye_up[1] * height_factor)
        
        #中心の座標の調整
        right_double_eyelid_center = (right_eye_center[0] + 8, right_eyelid_height + 12)
        left_double_eyelid_center = (left_eye_center[0] - 4, left_eyelid_height + 13)
        
        #二重の長さの調整
        double_eye_width = 16
        
        #二重の丸みを調整
        double_eye_height = 9
        
        #二重線の弧の形を調整
        start_angle = 190
        end_angle = 360
        
        #二重線を描画
        self.draw_double_eye_ellipse(image, left_double_eyelid_center, 
                                    (left_eye_width - double_eye_width, left_eye_height + double_eye_height),
                                    roll_angle, start_angle, end_angle, color=(68, 92, 135), thickness=1)
        
        self.draw_double_eye_ellipse(image, right_double_eyelid_center, 
                                    (right_eye_width - double_eye_width, right_eye_height + double_eye_height),
                                    roll_angle, end_angle, end_angle, color=(68, 92, 135), thickness=1)
        
    def draw_landmarks(self, image, face_details, height, width):
        """ランドマークを取得し、二重ラインを描画"""
        for face_detail in face_details:
            landmarks = {landmark['Type']: (int(landmark['X'] * width), int(landmark['Y'] * height)) 
                        for landmark in face_detail['Landmarks']}
        
        #首の角度から二重の線を調整
        pose = face_detail['Pose']
        roll_angle = pose['Roll']
        
        self.calculate_Eye_Position_Draw(image, landmarks, roll_angle)


class ImageProcessor:
    """画像の読み込みや表示、保存などの処理を担当するクラス"""

    @staticmethod
    def load_image(image_path):
        """画像ファイルを読み込む"""
        
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def show_image(window_name, image):
        """画像をウィンドウに表示する"""
        
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def get_image_bytes(image_path):
        """画像ファイルをバイト形式で読み込む"""
        
        with open(image_path, 'rb') as image_file:
            return image_file.read()

    @staticmethod
    def get_image_dimensions(image):
        """画像の高さと幅を取得する"""
        
        height, width = image.shape[:2]
        return height, width

def main(image_path):
    
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    if not aws_access_key_id or not aws_secret_access_key:
        raise EnvironmentError('AWSアクセスキーが設定されていません。環境変数から読み込んでください。')
    
    face_processor = FaceLandmarkProcessor(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    image_processor = ImageProcessor()

    # 画像の読み込み
    image = image_processor.load_image(image_path)
    height, width = image_processor.get_image_dimensions(image)
    
    # 画像のバイトデータを取得して顔のランドマークを検出
    image_bytes = image_processor.get_image_bytes(image_path)
    response = face_processor.detect_faces_landmark(image_bytes)
    
    # ランドマークを処理して二重まぶたを描画
    face_processor.draw_landmarks(image, response['FaceDetails'], height, width)
    
    # 結果を表示
    image_processor.show_image('Processed Image', image)


# 実行
main('image_path')
