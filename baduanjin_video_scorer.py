from glob import glob
import os
import numpy as np
from collections import deque, Counter
from baduanjin_feature_extractor import extract_features_from_video
from baduanjin_chroma_ingest import query_similar_poses

# "./datasets/baduanjian_action_videos/test_action1&2-1.mp4"


class BaduanjinScorer:
    def __init__(
        self,
        video_path=None,
        feature_path=None,
        window_size=30,
        confidence_threshold=0.85,
    ):
        # 验证参数
        if video_path is None and feature_path is None:
            raise ValueError("必须提供 video_path 或 feature_path 中的一项")

        if feature_path:
            self.feature_path = feature_path
        elif video_path:
            self.feature_path = BaduanjinScorer.extract_features(video_path)

        self.WINDOW_SIZE = window_size
        self.CONFIDENCE_THRESHOLD = confidence_threshold

    @staticmethod
    def extract_features(video_path):
        feature_path = extract_features_from_video(video_path)
        return feature_path

    def extract_score(self):
        """读取目标视频的特征向量并返回评分"""
        feature_path = self.feature_path
        if not os.path.exists(feature_path):
            print("目标视频的特征向量文件夹不存在！")
            return None

        npy_files = sorted(glob(os.path.join(feature_path, "frame_*.npy")))
        if not npy_files:
            print("目标视频的特征向量文件不存在！")
            return None

        window = deque(maxlen=self.WINDOW_SIZE)
        current_pose = None
        segment_start_idx = 0
        segment_similarities = []
        segments = []

        for i, npy_path in enumerate(npy_files):
            pose_name, similarity = query_similar_poses(npy_path, top_k=3)
            # 只有高置信度帧参与投票
            if similarity > self.CONFIDENCE_THRESHOLD:
                window.append((pose_name, similarity))
            else:
                window.append((None, similarity))

            # 滑动窗口多数投票
            if len(window) == self.WINDOW_SIZE:
                valid_poses = [p for p, s in window if p is not None]
                if valid_poses:
                    # most_common(1)[0]:出现最多的1个元素（元素, 次数）
                    voted_pose = Counter(valid_poses).most_common(1)[0][0]
                    voted_count = Counter(valid_poses).most_common(1)[0][1]

                # 如果最高票数不过半则不认为是有效动作，维持上一个动作状态
                if voted_count >= self.WINDOW_SIZE // 2:
                    # 动作切换
                    if current_pose and current_pose != voted_pose:
                        avg_score = round(
                            sum(segment_similarities) / len(segment_similarities) * 100,
                            1,
                        )
                        end_idx = i - self.WINDOW_SIZE
                        segments.append(
                            {
                                "pose": current_pose,
                                "start_frame": segment_start_idx,
                                "end_frame": end_idx,
                                "score": avg_score,
                            }
                        )
                        print(f" 完成 {current_pose} | 得分 {avg_score}分")

                    if current_pose != voted_pose:
                        current_pose = voted_pose
                        segment_start_idx = i - self.WINDOW_SIZE + 1
                        segment_similarities = [similarity]
                    else:
                        segment_similarities.append(similarity)

        # 处理最后一个动作段
        if current_pose and segment_similarities:
            avg_score = round(
                sum(segment_similarities) / len(segment_similarities) * 100, 1
            )
            segments.append(
                {
                    "pose": current_pose,
                    "start_frame": segment_start_idx,
                    "end_frame": len(npy_files) - 1,
                    "score": avg_score,
                }
            )

        self._print_report(segments, npy_files)
        return segments

    def _print_report(self, segments: list, npy_files: list):
        print("\n" + "=" * 70)
        print("八段锦评分报告：")
        print("=" * 70)
        total = 0
        for seg in segments:
            duration_frames = seg["end_frame"] - seg["start_frame"]
            print(
                f"{seg['pose']:6}  |  帧 {seg['start_frame']:5} ~ {seg['end_frame']:5}  "
                f"|  得分 {seg['score']:6.1f}分  |  帧数 {duration_frames:4}"
            )
            total += seg["score"]
        if segments:
            print(f"\n 整体平均得分：{round(total / len(segments), 1)}分")


if __name__ == "__main__":
    scorer = BaduanjinScorer(feature_path="./target_features/test_action1&2-1")
    scorer.extract_score()
