import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_VERBOSITY"] = "3"
warnings.filterwarnings("ignore")
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
TARGET_FEATURES_DIR = "./target_features"
TARGET_ANNOTATED_DIR = "./features/images"


def extract_features_from_image(image_path):
    """从图像中提取132维特征向量"""
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

    # 可选择生成带关键点的可视化图
    if DRAW_ANNOTATED:
        annotated = draw_landmarks_on_image(mp_image.numpy_view(), result)
        save_path = os.path.join(
            ANNOTATED_DIR,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_annotated.jpg",
        )
        cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"已保存可视化图: {save_path}")
    return feature_vector


def extract_features_from_video(video_path):
    """从视频中逐帧提取132维特征向量"""
    if not os.path.exists(TARGET_FEATURES_DIR):
        os.makedirs(TARGET_FEATURES_DIR)
    if DRAW_ANNOTATED:
        if not os.path.exists(TARGET_ANNOTATED_DIR):
            os.makedirs(TARGET_ANNOTATED_DIR)

    options = vision.PoseLandmarkerOptions(
        base_options=BASE_OPTIONS,
        running_mode=vision.RunningMode.VIDEO,  # VIDEO 模式
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    pbar = tqdm(desc="Processing frames", unit="frame")

    # 创建目标视频的特征向量存储文件夹
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(TARGET_FEATURES_DIR, video_name)
    os.makedirs(save_path, exist_ok=True)

    # 读取目标视频
    print(f"开始处理视频：{video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转为mp_image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # video模式必须传入时间戳
        timestamp_ms = int(frame_count * (1000 / fps))

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            print(f"未检测出人体:{timestamp_ms}")
            continue
        landmarks = result.pose_landmarks[0]
        feature_vector = []
        for landmark in landmarks:
            feature_vector.append(
                [landmark.x, landmark.y, landmark.z, landmark.visibility]
            )
        feature_vector = np.array(feature_vector, dtype=np.float32).flatten()
        npy_path = os.path.join(save_path, f"frame_{frame_count:06d}.npy")
        np.save(npy_path, feature_vector)

        if DRAW_ANNOTATED:
            annotated = draw_landmarks_on_image(mp_image.numpy_view(), result)
            jpg_path = os.path.join(
                TARGET_ANNOTATED_DIR, f"frame_{frame_count:06d}_annotated.jpg"
            )
            cv2.imwrite(jpg_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        frame_count += 1
        pbar.update(1)
        pbar.set_description(f"Frame {frame_count:06d}")  # 动态更新描述

    pbar.close()
    cap.release()
    return save_path


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
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    if DRAW_ANNOTATED:
        if not os.path.exists(ANNOTATED_DIR):
            os.makedirs(ANNOTATED_DIR)

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
    process_images("./datasets/action_frames/action5_815.jpg")
