import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# 姿势地标模型
MODEL_PATH = "./models/pose_landmarker_full.task"
BASE_OPTIONS = python.BaseOptions(model_asset_path=MODEL_PATH)
# 是否生成带关键点的可视化图
DRAW_ANNOTATED = False
ANNOTATED_DIR = "./features/images"
FEATURES_DIR = "./features"


def extract_features_from_image(image_path):
    # 默认running_mode是单张图片输入
    options = vision.PoseLandmarkerOptions(base_options=BASE_OPTIONS)
    detector = vision.PoseLandmarker.create_from_options(options)
    # 输入图片/Numpy数组需要转换为mediapipe.Image对象
    mp_image = mp.Image.create_from_file(image_path)
    result = detector.detect(mp_image)
    if not result.pose_landmarks:
        print(f"未检测出人体:{image_path}")
        return None
    # 针对我的数据集一张图片只有一个人物，所以只取第一个就可以
    landmarks = result.pose_landmarks[0]
    # 33*4=132维向量
    feature_vector = []
    for landmark in landmarks:
        feature_vector.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    feature_vector = np.array(feature_vector, dtype=np.float32).flatten()

    # 生成带关键点的可视化图
    if DRAW_ANNOTATED:
        annotated = draw_landmarks_on_image(mp_image.numpy_view(), result)
        save_path = os.path.join(
            ANNOTATED_DIR,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_annotated.jpg",
        )
        cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"已保存可视化图: {save_path}")
    return feature_vector


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    return annotated_image


def process_images(image_path):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    if DRAW_ANNOTATED:
        os.makedirs(ANNOTATED_DIR, exist_ok=True)

    if os.path.isfile(image_path):
        files = [image_path]
    elif os.path.isdir(image_path):
        files = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    else:
        print("路径不存在")
        return

    for file in tqdm(files):
        vector = extract_features_from_image(file)
        if vector is not None:
            save_name = f"{os.path.splitext(os.path.basename(file))[0]}.npy"
            save_path = os.path.join(FEATURES_DIR, save_name)
            np.save(save_path, vector)
    print("特征向量保存完成")


if __name__ == "__main__":
    process_images("./datasets/action_frames")
