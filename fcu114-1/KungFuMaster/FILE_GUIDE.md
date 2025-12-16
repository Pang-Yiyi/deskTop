# 功夫訓練系統 - 檔案結構說明

本文檔詳細說明專案中各個檔案的用途與功能。

---

## 目錄結構總覽

```
KungFu-master/
├── main_new.py                 # 主程式 (PyQt6 GUI)
├── test_camera.py              # 攝影機測試程式
├── generate_report_figures.py  # 報告圖表生成器
├── run_model_comparison.py     # 分類模型比較腳本
├── helper/                     # 核心模組目錄
│   ├── model.py               # YOLOv11 姿態估計模型
│   ├── kungfu_classifier.py   # DNN 動作分類器
│   ├── angle_calculator.py    # 關節角度計算
│   └── database.py            # SQLite 資料庫操作
├── model/                      # 模型檔案目錄
├── dataset/                    # 資料集目錄
├── display/                    # 動作示範圖片
├── *.ipynb                     # Jupyter Notebook 訓練/比較
├── REPORT.md                   # 技術報告
└── README.md                   # 專案說明
```

---

## 1. 主程式檔案

### 1.1 `main_new.py` - 主應用程式

**用途：** 功夫訓練系統的主要圖形化介面

**功能：**
- PyQt6 現代化 GUI 介面
- 四種功夫動作選擇卡片
- 即時攝影機動作辨識
- 相似度環形進度條顯示
- 關節角度即時顯示
- 動作正確計數功能

**主要類別：**
| 類別名稱 | 功能描述 |
|----------|----------|
| `MainWindow` | 主視窗，管理頁面切換 |
| `MainPage` | 主選單頁面，展示動作卡片 |
| `ActionPracticePage` | 動作練習頁面，整合攝影機與回饋 |
| `ActionCard` | 動作示範卡片元件 |
| `VideoDisplay` | 影片顯示元件，支援骨架繪製 |
| `CircularProgress` | 環形進度條元件 |
| `AngleBar` | 角度進度條元件 |

**執行方式：**
```bash
python main_new.py
```

---

### 1.2 `test_camera.py` - 攝影機測試程式

**用途：** 終端機模式的即時動作辨識測試

**功能：**
- 攝影機即時影像偵測
- YOLOv11 姿態估計
- DNN 動作分類
- 鍵盤控制參數調整
- 截圖保存功能

**鍵盤控制：**
| 按鍵 | 功能 |
|------|------|
| `q` / `ESC` | 退出程式 |
| `s` | 截圖保存 |
| `0-9` | 切換攝影機 |
| `+` / `-` | 調整信心度門檻 |
| `[` / `]` | 調整站立角度閾值 |
| `,` / `.` | 調整身體偵測置信度 |

**執行方式：**
```bash
python test_camera.py
```

---

## 2. Helper 模組目錄

### 2.1 `helper/model.py` - YOLOv11 姿態估計模型

**用途：** 載入和使用 YOLOv11 姿態估計模型

**主要類別/函數：**

| 名稱 | 類型 | 功能 |
|------|------|------|
| `ModelLoader` | 類別 | YOLO 模型載入器 |
| `pose_model` | 實例 | 預載入的姿態模型 (YOLOv11m) |
| `get_pose_model()` | 函數 | 取得指定大小的模型 |
| `YOLO11_POSE_MODELS` | 字典 | 可用模型列表 |

**支援的模型大小：**
```python
YOLO11_POSE_MODELS = {
    'nano': 'yolo11n-pose.pt',    # 最快
    'small': 'yolo11s-pose.pt',   # 平衡
    'medium': 'yolo11m-pose.pt',  # 推薦 (預設)
    'large': 'yolo11l-pose.pt',   # 高準確
    'xlarge': 'yolo11x-pose.pt',  # 最高準確
}
```

---

### 2.2 `helper/kungfu_classifier.py` - 動作分類器

**用途：** PyTorch DNN 動作分類器

**主要類別/函數：**

| 名稱 | 類型 | 功能 |
|------|------|------|
| `KungfuClassifierDNN` | 類別 | DNN 網路架構定義 |
| `KungfuClassifier` | 類別 | 分類器封裝，提供預測介面 |
| `get_kungfu_classifier()` | 函數 | 取得分類器單例 |

**DNN 網路架構：**
```
輸入層 (6 特徵)
  → Linear(64) + BatchNorm + ReLU + Dropout
  → Linear(128) + BatchNorm + ReLU + Dropout
  → Linear(64) + BatchNorm + ReLU + Dropout
  → Linear(32) + BatchNorm + ReLU
  → 輸出層 (4 類別)
```

**使用範例：**
```python
from helper.kungfu_classifier import get_kungfu_classifier

classifier = get_kungfu_classifier()
action_code, action_name, probs = classifier.predict(angles_dict)
confidence = classifier.get_confidence(probs)
```

---

### 2.3 `helper/angle_calculator.py` - 角度計算器

**用途：** 從 YOLO 關鍵點計算關節角度

**主要函數：**

| 函數名稱 | 功能 |
|----------|------|
| `calculate_angle(p1, p2, p3)` | 計算三點間角度 (p2 為頂點) |
| `extract_angles_from_keypoints(keypoints)` | 從 17 關鍵點提取 6 個角度 |
| `is_valid_keypoints(keypoints)` | 檢查關鍵點有效性 |

**輸出的 6 個角度：**
```python
{
    'R_Elbow_Angle': 右肘角度,  # 右肩→右肘→右腕
    'L_Elbow_Angle': 左肘角度,  # 左肩→左肘→左腕
    'R_Knee_Angle': 右膝角度,   # 右臀→右膝→右踝
    'L_Knee_Angle': 左膝角度,   # 左臀→左膝→左踝
    'R_Hip_Angle': 右臀角度,    # 右肩→右臀→右膝
    'L_Hip_Angle': 左臀角度,    # 左肩→左臀→左膝
}
```

---

### 2.4 `helper/database.py` - 資料庫操作

**用途：** SQLite3 資料庫操作封裝

**主要類別/函數：**

| 名稱 | 功能 |
|------|------|
| `Database` | 資料庫操作類別 |
| `sqlite3_database` | 預載入的資料庫實例 |

**資料表結構：**
```sql
-- posture 表：儲存教師示範動作
CREATE TABLE posture (
    id INTEGER PRIMARY KEY,
    posture_name TEXT,      -- 動作名稱
    video_path TEXT,        -- 影片路徑
    npy_path TEXT           -- 關鍵點資料路徑
);

-- score 表：儲存練習分數
CREATE TABLE score (
    id INTEGER PRIMARY KEY,
    time INTEGER,           -- 時間戳記
    score INTEGER,          -- 分數
    video_path TEXT         -- 影片路徑
);
```

---

## 3. 輔助腳本

### 3.1 `generate_report_figures.py` - 報告圖表生成器

**用途：** 生成技術報告所需的圖表

**輸出圖表：**
| 檔案名稱 | 內容 |
|----------|------|
| `system_architecture.png` | 系統架構圖 |
| `three_modes_flowchart.png` | 三種模式流程圖 |
| `yolo_pose_pipeline.png` | YOLOv11 姿態偵測流程 |
| `fastdtw_explanation.png` | FastDTW 演算法說明 |
| `model_comparison.png` | 模型效能比較 |
| `challenges_solutions.png` | 挑戰與解決方案 |

**執行方式：**
```bash
python generate_report_figures.py
```

---

### 3.2 `run_model_comparison.py` - 分類模型比較

**用途：** 比較多種機器學習/深度學習分類模型

**比較的模型：**
- Logistic Regression
- K-Nearest Neighbors
- SVM (RBF / Linear)
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost (若已安裝)
- DNN (PyTorch)

**輸出檔案：**
- `model/classification_models_comparison.csv` - 結果表格
- `model/classification_models_comparison.png` - 比較圖
- `model/confusion_matrices_all_models.png` - 混淆矩陣
- `model/class_performance_comparison.png` - 各類別表現
- `model/model_radar_chart.png` - 雷達圖

---

## 4. Jupyter Notebook 檔案

### 4.1 `train_kungfu_classifier.ipynb`

**用途：** 訓練傳統機器學習分類器

**內容：**
- 資料載入與預處理
- 特徵工程
- 模型訓練
- 超參數調整
- 模型評估

---

### 4.2 `train_kungfu_deep_learning.ipynb`

**用途：** 訓練 PyTorch DNN 分類器

**內容：**
- DNN 架構設計
- 訓練迴圈實作
- Early Stopping
- 模型儲存
- 訓練曲線視覺化

---

### 4.3 `compare_deep_learning_models.ipynb`

**用途：** 比較不同深度學習架構

**比較項目：**
- 不同層數的 DNN
- 不同神經元數量
- 不同 Dropout 率
- 不同優化器

---

### 4.4 `compare_pose_models.ipynb`

**用途：** 比較不同姿態估計模型

**比較項目：**
- YOLOv11 不同大小 (n/s/m/l/x)
- 推論速度
- 準確度
- 關鍵點置信度

---

### 4.5 `compare_classification_models.ipynb`

**用途：** 互動式分類模型比較 (Notebook 版)

---

## 5. Model 目錄

### 5.1 姿態估計模型

| 檔案名稱 | 說明 |
|----------|------|
| `yolo11m-pose.pt` | YOLOv11 medium 姿態估計模型 (主要使用) |
| `pose_model.pt` | 舊版姿態模型 (備用) |
| `hand_model.pt` | 手部偵測模型 (備用) |

---

### 5.2 動作分類模型

| 檔案名稱 | 說明 |
|----------|------|
| `kungfu_dnn_best.pth` | DNN 最佳權重檔 (推薦載入) |
| `kungfu_dnn_full.pth` | DNN 完整模型檔 |
| `kungfu_dnn_weights.pth` | DNN 權重檔 (備份) |
| `scaler.joblib` | 特徵標準化器 (StandardScaler) |
| `label_encoder.joblib` | 標籤編碼器 (LabelEncoder) |
| `model_info.joblib` | 模型元資訊 |

---

### 5.3 結果檔案

| 檔案名稱 | 說明 |
|----------|------|
| `training_curves.png` | DNN 訓練曲線 |
| `confusion_matrix.png` | 混淆矩陣 |
| `model_comparison.png` | 模型比較圖 |
| `classification_models_comparison.csv` | 分類模型比較結果 |
| `pose_model_comparison_results.csv` | 姿態模型比較結果 |

---

## 6. Dataset 目錄

### 6.1 資料檔案

| 檔案名稱 | 說明 |
|----------|------|
| `pose_angles_summary_actual.csv` | 角度特徵資料集 (訓練用) |

**資料格式：**
```csv
Image_Path,R_Elbow_Angle,L_Elbow_Angle,R_Knee_Angle,L_Knee_Angle,R_Hip_Angle,L_Hip_Angle,Action_Type
```

---

### 6.2 分析結果圖片

| 命名規則 | 說明 |
|----------|------|
| `analyzed_act{1-4}_{action}_p{1-2}_{n}.jpg` | 姿態分析結果圖 |

**動作代碼：**
- `act1` / `fist` - 馬步站拳
- `act2` / `punch` - 弓步出拳
- `act3` / `kick` - 側踢腿
- `act4` / `knee` - 提膝

---

## 7. Display 目錄

**用途：** 存放動作示範圖片 (主程式卡片用)

| 檔案名稱 | 動作 |
|----------|------|
| `act1_fist.jpg` | 馬步站拳示範圖 |
| `act2_punch.jpg` | 弓步出拳示範圖 |
| `act3_kick.jpg` | 側踢腿示範圖 |
| `act4_knee.jpg` | 提膝示範圖 |

---

## 8. 文檔檔案

| 檔案名稱 | 說明 |
|----------|------|
| `README.md` | 專案說明文件 |
| `REPORT.md` | 技術報告 |
| `FILE_GUIDE.md` | 本檔案 - 檔案結構說明 |

---

## 9. 檔案依賴關係圖

```
main_new.py
    ├── helper/model.py
    │       └── model/yolo11m-pose.pt
    ├── helper/kungfu_classifier.py
    │       ├── model/kungfu_dnn_best.pth
    │       ├── model/scaler.joblib
    │       ├── model/label_encoder.joblib
    │       └── model/model_info.joblib
    ├── helper/angle_calculator.py
    └── display/act{1-4}_*.jpg

test_camera.py
    ├── helper/model.py
    ├── helper/kungfu_classifier.py
    └── helper/angle_calculator.py

train_kungfu_deep_learning.ipynb
    └── dataset/pose_angles_summary_actual.csv
        → 輸出: model/kungfu_dnn_*.pth

run_model_comparison.py
    └── dataset/pose_angles_summary_actual.csv
        → 輸出: model/*_comparison.{csv,png}
```

---

## 10. 快速開始

### 執行主程式
```bash
python main_new.py
```

### 執行攝影機測試
```bash
python test_camera.py
```

### 訓練模型
1. 開啟 `train_kungfu_deep_learning.ipynb`
2. 執行所有儲存格

### 比較模型
```bash
python run_model_comparison.py
```

### 生成報告圖表
```bash
python generate_report_figures.py
```

---

*文檔更新日期：2024*
