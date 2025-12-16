# Code by AkinoAlice@TyrantRey
# 角度計算工具 - 將 YOLO 關鍵點轉換為關節角度

import numpy as np


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    計算三點間的角度 (p2 為頂點)

    參數:
        p1: 第一個點座標 [x, y]
        p2: 頂點座標 [x, y] (角度的頂點)
        p3: 第三個點座標 [x, y]

    回傳:
        角度 (度)
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product < 1e-6:
        return 0.0

    cos_angle = np.dot(v1, v2) / norm_product
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def extract_angles_from_keypoints(keypoints: np.ndarray) -> dict[str, float]:
    """
    從 YOLO 關鍵點提取 6 個關節角度

    YOLO COCO 17 關鍵點索引:
    0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳
    5: 左肩, 6: 右肩
    7: 左肘, 8: 右肘
    9: 左腕, 10: 右腕
    11: 左臀, 12: 右臀
    13: 左膝, 14: 右膝
    15: 左踝, 16: 右踝

    參數:
        keypoints: YOLO 輸出的關鍵點陣列，形狀為 (17, 2) 或 (17, 3)

    回傳:
        包含 6 個角度的字典
    """
    # 確保只取 x, y 座標
    if keypoints.shape[-1] == 3:
        keypoints = keypoints[:, :2]

    # 右手肘角度 (右肩-右肘-右腕)
    r_elbow = calculate_angle(keypoints[6], keypoints[8], keypoints[10])

    # 左手肘角度 (左肩-左肘-左腕)
    l_elbow = calculate_angle(keypoints[5], keypoints[7], keypoints[9])

    # 右膝角度 (右臀-右膝-右踝)
    r_knee = calculate_angle(keypoints[12], keypoints[14], keypoints[16])

    # 左膝角度 (左臀-左膝-左踝)
    l_knee = calculate_angle(keypoints[11], keypoints[13], keypoints[15])

    # 右臀角度 (右肩-右臀-右膝)
    r_hip = calculate_angle(keypoints[6], keypoints[12], keypoints[14])

    # 左臀角度 (左肩-左臀-左膝)
    l_hip = calculate_angle(keypoints[5], keypoints[11], keypoints[13])

    return {
        'R_Elbow_Angle': r_elbow,
        'L_Elbow_Angle': l_elbow,
        'R_Knee_Angle': r_knee,
        'L_Knee_Angle': l_knee,
        'R_Hip_Angle': r_hip,
        'L_Hip_Angle': l_hip
    }


def is_valid_keypoints(keypoints: np.ndarray, confidence_threshold: float = 0.3) -> bool:
    """
    檢查關鍵點是否有效（用於有置信度的情況）

    參數:
        keypoints: 關鍵點陣列
        confidence_threshold: 置信度閾值

    回傳:
        是否為有效的關鍵點
    """
    if keypoints is None or len(keypoints) < 17:
        return False

    # 檢查關鍵身體部位是否被偵測到（非零座標）
    important_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 肩膀到腳踝

    for idx in important_indices:
        if keypoints[idx][0] == 0 and keypoints[idx][1] == 0:
            return False

    return True


if __name__ == "__main__":
    # 測試範例
    test_keypoints = np.random.rand(17, 2) * 640  # 模擬 640x640 影像的關鍵點

    angles = extract_angles_from_keypoints(test_keypoints)
    print("提取的角度:")
    for name, angle in angles.items():
        print(f"  {name}: {angle:.2f}°")
