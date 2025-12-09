# Code by AkinoAlice@TyrantRey
# 功夫動作分類器 - 使用 PyTorch 深度學習模型進行動作辨識

import joblib
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn


class KungfuClassifierDNN(nn.Module):
    """
    功夫動作分類深度神經網路

    架構:
    - 輸入層: 6 個特徵 (關節角度)
    - 隱藏層 1: 64 神經元 + BatchNorm + ReLU + Dropout
    - 隱藏層 2: 128 神經元 + BatchNorm + ReLU + Dropout
    - 隱藏層 3: 64 神經元 + BatchNorm + ReLU + Dropout
    - 隱藏層 4: 32 神經元 + BatchNorm + ReLU
    - 輸出層: 4 個類別
    """

    def __init__(self, input_size: int = 6, num_classes: int = 4, dropout_rate: float = 0.3):
        super(KungfuClassifierDNN, self).__init__()

        self.network = nn.Sequential(
            # 隱藏層 1
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 隱藏層 2
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 隱藏層 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 隱藏層 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # 輸出層
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class KungfuClassifier:
    """功夫動作分類器 (支援 PyTorch 深度學習模型)"""

    # 動作代碼對應中文名稱
    ACTION_NAMES = {
        'act1': '握拳式 (Fist)',
        'act2': '出拳式 (Punch)',
        'act3': '踢腿式 (Kick)',
        'act4': '提膝式 (Knee)'
    }

    def __init__(self, model_dir: str | Path = './model') -> None:
        """
        初始化分類器

        參數:
            model_dir: 模型檔案目錄
        """
        model_dir = Path(model_dir)

        # 設定裝置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 載入模型資訊
        self.info = joblib.load(model_dir / 'model_info.joblib')
        self.feature_columns = self.info['feature_columns']
        self.classes = self.info['classes']
        self.num_classes = len(self.classes)

        # 載入 PyTorch 模型
        self.model = self._load_model(model_dir)

        # 載入 scaler 和 encoder
        self.scaler = joblib.load(model_dir / 'scaler.joblib')
        self.encoder = joblib.load(model_dir / 'label_encoder.joblib')

    def _load_model(self, model_dir: Path) -> nn.Module:
        """載入 PyTorch 模型"""
        weights_path = model_dir / 'kungfu_dnn_best.pth'
        full_model_path = model_dir / 'kungfu_dnn_full.pth'

        # 優先使用權重檔案載入 (更安全且相容性更好)
        if weights_path.exists():
            # 建立模型架構並載入權重
            model = KungfuClassifierDNN(
                input_size=len(self.feature_columns),
                num_classes=self.num_classes
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        elif full_model_path.exists():
            # 載入完整模型 (PyTorch 2.6+ 需要 weights_only=False)
            model = torch.load(full_model_path, map_location=self.device, weights_only=False)
        else:
            raise FileNotFoundError(
                f"找不到模型檔案。請先執行 train_kungfu_deep_learning.ipynb 訓練模型。\n"
                f"預期路徑: {weights_path} 或 {full_model_path}"
            )

        model.to(self.device)
        model.eval()
        return model

    def predict(self, angles_dict: dict[str, float]) -> tuple[str, str, dict[str, float]]:
        """
        預測動作類型

        參數:
            angles_dict: 包含 6 個角度的字典
                - R_Elbow_Angle: 右手肘角度
                - L_Elbow_Angle: 左手肘角度
                - R_Knee_Angle: 右膝角度
                - L_Knee_Angle: 左膝角度
                - R_Hip_Angle: 右臀角度
                - L_Hip_Angle: 左臀角度

        回傳:
            (動作代碼, 動作名稱, 各類別機率字典)
        """
        # 按照訓練時的特徵順序排列
        X = np.array([[angles_dict[col] for col in self.feature_columns]])

        # 標準化
        X_scaled = self.scaler.transform(X)

        # 轉換為 PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # 預測
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = outputs.argmax(dim=1).cpu().numpy()[0]

        # 解碼結果
        action_code = self.encoder.inverse_transform([pred_idx])[0]
        action_name = self.ACTION_NAMES.get(action_code, action_code)

        # 建立機率字典
        prob_dict = dict(zip(self.encoder.classes_, probabilities))

        return action_code, action_name, prob_dict

    def predict_from_array(self, angles: list[float] | np.ndarray) -> tuple[str, str, dict[str, float]]:
        """
        從角度陣列預測動作類型

        參數:
            angles: 角度陣列 [R_Elbow, L_Elbow, R_Knee, L_Knee, R_Hip, L_Hip]

        回傳:
            (動作代碼, 動作名稱, 各類別機率字典)
        """
        angles_dict = dict(zip(self.feature_columns, angles))
        return self.predict(angles_dict)

    def get_confidence(self, prob_dict: dict[str, float]) -> float:
        """
        取得最高機率（信心度）

        參數:
            prob_dict: 機率字典

        回傳:
            最高機率值
        """
        return float(max(prob_dict.values()))

    def get_action_name(self, action_code: str) -> str:
        """
        取得動作中文名稱

        參數:
            action_code: 動作代碼 (act1, act2, act3, act4)

        回傳:
            動作名稱
        """
        return self.ACTION_NAMES.get(action_code, action_code)

    def get_model_info(self) -> dict:
        """取得模型資訊"""
        return {
            'model_type': self.info.get('model_type', 'PyTorch DNN'),
            'device': str(self.device),
            'input_size': len(self.feature_columns),
            'num_classes': self.num_classes,
            'classes': self.classes,
            'test_accuracy': self.info.get('test_accuracy', 'N/A')
        }


# 建立全域分類器實例（延遲載入）
_kungfu_classifier: KungfuClassifier | None = None


def get_kungfu_classifier() -> KungfuClassifier:
    """取得功夫分類器實例（單例模式）"""
    global _kungfu_classifier
    if _kungfu_classifier is None:
        _kungfu_classifier = KungfuClassifier()
    return _kungfu_classifier


if __name__ == "__main__":
    # 測試分類器
    print("載入功夫動作分類器 (PyTorch 深度學習模型)...")

    try:
        classifier = KungfuClassifier()

        # 顯示模型資訊
        info = classifier.get_model_info()
        print(f"\n模型資訊:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 測試範例 - 踢腿式的典型角度
        test_angles = {
            'R_Elbow_Angle': 155.0,
            'L_Elbow_Angle': 40.0,
            'R_Knee_Angle': 164.0,
            'L_Knee_Angle': 168.0,
            'R_Hip_Angle': 176.0,
            'L_Hip_Angle': 125.0
        }

        action_code, action_name, probs = classifier.predict(test_angles)

        print(f"\n預測結果: {action_name} ({action_code})")
        print(f"信心度: {classifier.get_confidence(probs):.2%}")
        print("\n各類別機率:")
        for code, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            name = classifier.get_action_name(code)
            bar = "█" * int(prob * 30)
            print(f"  {name}: {prob:.2%} {bar}")

    except FileNotFoundError as e:
        print(f"\n錯誤: {e}")
        print("\n請先執行 train_kungfu_deep_learning.ipynb 來訓練深度學習模型！")
