# Code by AkinoAlice@TyrantRey
# 更新至 YOLOv11 姿態估計模型


from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
from ultralytics.engine.results import Results
from pathlib import Path
from uuid import uuid4
import numpy as np


# YOLOv11 Pose 模型選項
YOLO11_POSE_MODELS = {
    'nano': 'yolo11n-pose.pt',      # 最快，適合即時應用
    'small': 'yolo11s-pose.pt',     # 平衡速度與準確度
    'medium': 'yolo11m-pose.pt',    # 推薦 - 較好的準確度
    'large': 'yolo11l-pose.pt',     # 高準確度
    'xlarge': 'yolo11x-pose.pt',    # 最高準確度
}

# 預設使用 YOLOv11 medium pose 模型
DEFAULT_POSE_MODEL = "./model/yolo11m-pose.pt"


# only processing frame to landmark
class ModelLoader:
    def __init__(self, model_path: str = DEFAULT_POSE_MODEL) -> None:
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.predict: list[Results] = []

        # 顯示模型資訊
        model_name = Path(model_path).stem
        print(f"已載入模型: {model_name}")

    def detect_video(self, video_path: str | Path) -> tuple[Path, Path]:
        self.path = Path(video_path)
        self.predict = self.model.predict(
            video_path, show_boxes=False, save=True, project="./result"
        )

        self.uuid = str(uuid4())
        saved_predicted_path = sum(1 for _ in Path("./result").rglob("*") if _.is_dir())
        predicted_video_path = Path(
            f"./result/predict{str('' if saved_predicted_path == 1 else saved_predicted_path)}"
        ) / (self.uuid + ".avi")

        predicted_npy_path = Path(
            f"./result/predict{str('' if saved_predicted_path == 1 else saved_predicted_path)}"
        ) / (self.uuid + ".npy")

        yolo_output_path = Path(
            f"./result/predict{str('' if saved_predicted_path == 1 else saved_predicted_path)}"
        ) / (self.path.stem + ".avi")

        # rename to uuid format
        yolo_output_path.rename(predicted_video_path)

        return predicted_video_path, predicted_npy_path

    def save_npy(self, save_path: str | Path) -> np.ndarray:
        if self.predict is None:
            raise RuntimeError("Must call detect_video() before save_npy()")

        all_keypoints: list[np.ndarray] = []

        for result in self.predict:
            keypoints = result.keypoints
            if keypoints is None:
                continue

            if keypoints.shape[0] < 1:
                continue

            xyn = keypoints.xyn[0].cpu().numpy()
            all_keypoints.append(xyn)

        if len(all_keypoints) == 0:
            raise ValueError("No valid keypoints detected in video")

        keypoints_array = np.array(all_keypoints)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(path)
        np.save(path, keypoints_array)

        return keypoints_array


hand_model = ModelLoader("./model/hand_model.pt")
# 使用 YOLOv11 pose 模型 (預設使用 medium 版本)
pose_model = ModelLoader(DEFAULT_POSE_MODEL)


def get_pose_model(model_size: str = 'medium') -> ModelLoader:
    """
    取得指定大小的 YOLOv11 pose 模型

    參數:
        model_size: 模型大小 ('nano', 'small', 'medium', 'large', 'xlarge')

    回傳:
        ModelLoader 實例
    """
    if model_size not in YOLO11_POSE_MODELS:
        raise ValueError(f"無效的模型大小: {model_size}。可用選項: {list(YOLO11_POSE_MODELS.keys())}")

    model_filename = YOLO11_POSE_MODELS[model_size]
    model_path = f"./model/{model_filename}"

    # 如果模型不存在，自動下載
    if not Path(model_path).exists():
        print(f"正在下載 YOLOv11 {model_size} pose 模型...")
        temp_model = YOLO(model_filename)  # 這會自動下載模型
        import shutil
        shutil.move(model_filename, model_path)
        print(f"模型已儲存至: {model_path}")

    return ModelLoader(model_path)


if __name__ == "__main__":
    print("=" * 50)
    print("YOLOv11 姿態估計模型測試")
    print("=" * 50)

    # 使用 YOLOv11 medium pose 模型
    pose_model = ModelLoader(DEFAULT_POSE_MODEL)
    print(f"\n模型路徑: {pose_model.model_path}")
    print(f"模型任務: {pose_model.model.task}")

    # 測試視頻偵測 (如果有測試視頻)
    test_video = Path("./video/金手 - Trim.mp4")
    if test_video.exists():
        print(f"\n正在處理視頻: {test_video}")
        predicted_video_path, predicted_npy_path = pose_model.detect_video(str(test_video))
        pose_model.save_npy(predicted_npy_path)
        print(f"處理完成！")
    else:
        print(f"\n測試視頻不存在: {test_video}")
        print("跳過視頻測試")

    print("\n" + "=" * 50)
    print("測試完成！")
