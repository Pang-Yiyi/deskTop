# Code by AkinoAlice@TyrantRey
# 攝影機即時動作辨識測試程式

import cv2
import numpy as np
import sys

from helper.model import pose_model
from helper.kungfu_classifier import get_kungfu_classifier
from helper.angle_calculator import extract_angles_from_keypoints, is_valid_keypoints


def is_full_body_detected(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, min_confidence: float = 0.5) -> tuple:
    """
    檢查是否偵測到完整身體（從肩膀到腳踝）
    使用置信度分數來判斷關鍵點是否真正被偵測到

    參數:
        keypoints_xy: 關鍵點座標陣列 (17, 2)
        keypoints_conf: 關鍵點置信度陣列 (17,)
        min_confidence: 最小置信度閾值

    回傳:
        (是否偵測到完整身體, 缺少的部位列表)
    """
    # 關鍵點名稱對應
    keypoint_names = {
        5: "L_Shoulder", 6: "R_Shoulder",
        11: "L_Hip", 12: "R_Hip",
        13: "L_Knee", 14: "R_Knee",
        15: "L_Ankle", 16: "R_Ankle"
    }

    if keypoints_xy is None or len(keypoints_xy) < 17:
        return False, ["keypoints_xy invalid"]
    if keypoints_conf is None or len(keypoints_conf) < 17:
        return False, ["keypoints_conf invalid"]

    # 必須偵測到的關鍵點索引（肩膀、臀部、膝蓋、腳踝）
    required_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    missing_parts = []

    for idx in required_indices:
        # 檢查置信度是否足夠高
        conf = keypoints_conf[idx]
        if conf < min_confidence:
            missing_parts.append(f"{keypoint_names[idx]}({conf:.0%})")
            continue

        # 同時檢查座標是否有效
        x, y = keypoints_xy[idx][0], keypoints_xy[idx][1]
        if x <= 0 or y <= 0:
            missing_parts.append(f"{keypoint_names[idx]}(pos)")

    return len(missing_parts) == 0, missing_parts


def is_standing_pose(angles: dict, threshold: float = 160.0) -> bool:
    """
    判斷是否為站立姿態

    站立特徵：
    - 雙膝角度接近 180° (腿伸直)
    - 雙臀角度接近 180° (身體直立)

    參數:
        angles: 角度字典
        threshold: 角度閾值，大於此值視為伸直 (預設 160°)

    回傳:
        是否為站立姿態
    """
    r_knee = angles.get('R_Knee_Angle', 0)
    l_knee = angles.get('L_Knee_Angle', 0)
    r_hip = angles.get('R_Hip_Angle', 0)
    l_hip = angles.get('L_Hip_Angle', 0)

    # 雙腿伸直且身體直立
    knees_straight = r_knee > threshold and l_knee > threshold
    hips_straight = r_hip > threshold and l_hip > threshold

    return knees_straight and hips_straight


def main():
    print("=" * 50)
    print("功夫動作即時辨識測試程式")
    print("=" * 50)

    # 載入模型
    print("\n正在載入模型...")
    posture_detector = pose_model
    kungfu_classifier = get_kungfu_classifier()

    # 顯示模型資訊
    info = kungfu_classifier.get_model_info()
    print(f"模型類型: {info['model_type']}")
    print(f"裝置: {info['device']}")
    print(f"動作類別: {info['classes']}")
    print(f"測試準確率: {info.get('test_accuracy', 'N/A')}")

    # 信心度門檻設定
    CONFIDENCE_THRESHOLD = 0.85  # 信心度低於 85% 視為「無動作」
    STANDING_THRESHOLD = 150.0   # 站立判斷的角度閾值
    BODY_CONF_THRESHOLD = 0.5    # 完整身體偵測的置信度閾值
    print(f"動作信心度門檻: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"站立角度閾值: {STANDING_THRESHOLD}°")
    print(f"身體偵測置信度: {BODY_CONF_THRESHOLD:.0%}")

    # 動作顏色對應
    action_colors = {
        'act1': (107, 107, 255),   # 握拳式 - 紅色 (BGR)
        'act2': (196, 205, 78),    # 出拳式 - 青色
        'act3': (209, 183, 69),    # 踢腿式 - 藍色
        'act4': (180, 206, 150),   # 提膝式 - 綠色
    }

    # 開啟攝影機
    camera_id = 1  # 預設攝影機 ID，如果無法開啟會嘗試 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"無法開啟攝影機 {camera_id}，嘗試攝影機 0...")
        camera_id = 0
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("錯誤: 無法開啟任何攝影機！")
        return

    print(f"\n成功開啟攝影機 {camera_id}")
    print("\n操作說明:")
    print("  - 按 'q' 或 ESC 退出程式")
    print("  - 按 's' 截圖保存")
    print("  - 按 '0-9' 切換攝影機")
    print("  - 按 '+' 或 '=' 提高動作信心度門檻 (+5%)")
    print("  - 按 '-' 降低動作信心度門檻 (-5%)")
    print("  - 按 '[' 降低站立角度閾值 (-5°)")
    print("  - 按 ']' 提高站立角度閾值 (+5°)")
    print("  - 按 ',' 降低身體偵測置信度 (-5%)")
    print("  - 按 '.' 提高身體偵測置信度 (+5%)")
    print("\n開始即時辨識...\n")

    confidence_threshold = CONFIDENCE_THRESHOLD  # 使用可調整的變數
    standing_threshold = STANDING_THRESHOLD      # 站立判斷閾值
    body_conf_threshold = BODY_CONF_THRESHOLD    # 身體偵測置信度閾值

    # 設定視窗
    window_name = "Kungfu Action Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_count = 0
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影格")
            break

        frame_count += 1
        display_frame = frame.copy()

        # 使用 YOLO 姿態偵測
        results = posture_detector.model.predict(frame, verbose=False)

        action_text = "No Detection"
        confidence_text = ""
        color = (128, 128, 128)  # 灰色

        if results and len(results[0]) > 0:
            # 繪製骨架
            annotated_frame = results[0].plot()
            display_frame = annotated_frame

            # 取得關鍵點座標和置信度
            keypoints_xy = results[0][0].keypoints.xy[0].cpu().numpy()
            keypoints_conf = results[0][0].keypoints.conf[0].cpu().numpy()

            # 先檢查是否偵測到完整身體（使用置信度判斷）
            is_full_body, missing_parts = is_full_body_detected(keypoints_xy, keypoints_conf, min_confidence=body_conf_threshold)
            if not is_full_body:
                action_text = "Waiting for full body..."
                # 顯示缺少的部位
                if missing_parts:
                    confidence_text = f"Missing: {', '.join(missing_parts[:3])}"
                    if len(missing_parts) > 3:
                        confidence_text += f" +{len(missing_parts)-3} more"
                color = (128, 128, 128)
            elif is_valid_keypoints(keypoints_xy):
                try:
                    # 提取角度
                    angles = extract_angles_from_keypoints(keypoints_xy)

                    # 先判斷是否為站立姿態
                    if is_standing_pose(angles, standing_threshold):
                        action_text = "Standing"
                        confidence_text = "(Detected by joint angles)"
                        color = (128, 128, 128)  # 灰色

                        if frame_count % 60 == 0:
                            print(f"[Frame {frame_count}] Standing - knees/hips > {standing_threshold}°")
                    else:
                        # 非站立狀態，進行動作分類
                        action_code, action_name, probs = kungfu_classifier.predict(angles)
                        confidence = kungfu_classifier.get_confidence(probs)

                        # 信心度門檻過濾
                        if confidence >= confidence_threshold:
                            action_text = action_name
                            confidence_text = f"Confidence: {confidence:.1%}"
                            color = action_colors.get(action_code, (128, 128, 128))

                            # 每 30 幀輸出一次到終端機
                            if frame_count % 30 == 0:
                                print(f"[Frame {frame_count}] {action_name} ({confidence:.1%})")
                        else:
                            # 信心度太低，視為無明確動作
                            action_text = "Preparing..."
                            confidence_text = f"(Best: {action_name} {confidence:.1%})"
                            color = (128, 128, 128)

                            if frame_count % 60 == 0:
                                print(f"[Frame {frame_count}] Preparing - {action_name} ({confidence:.1%}) < {confidence_threshold:.0%}")

                except Exception as e:
                    action_text = "Detection Error"
                    if frame_count % 60 == 0:
                        print(f"分類錯誤: {e}")
            else:
                action_text = "Incomplete Pose"

        # 繪製結果面板
        panel_height = 120
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], panel_height), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], panel_height), color, 3)

        # 繪製動作文字
        cv2.putText(
            display_frame,
            action_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )

        # 繪製信心度
        if confidence_text:
            cv2.putText(
                display_frame,
                confidence_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        # 繪製操作提示和門檻值
        cv2.putText(
            display_frame,
            f"Conf: {confidence_threshold:.0%} | Stand: {standing_threshold:.0f} | Body: {body_conf_threshold:.0%} | 'q' quit",
            (display_frame.shape[1] - 580, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # 顯示畫面
        cv2.imshow(window_name, display_frame)

        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q 或 ESC
            print("\n退出程式...")
            break
        elif key == ord('s'):  # 截圖
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.png"
            cv2.imwrite(filename, display_frame)
            print(f"已保存截圖: {filename}")
        elif ord('0') <= key <= ord('9'):  # 切換攝影機
            new_camera_id = key - ord('0')
            print(f"嘗試切換到攝影機 {new_camera_id}...")
            cap.release()
            cap = cv2.VideoCapture(new_camera_id)
            if cap.isOpened():
                camera_id = new_camera_id
                print(f"成功切換到攝影機 {camera_id}")
            else:
                print(f"無法開啟攝影機 {new_camera_id}，恢復使用攝影機 {camera_id}")
                cap = cv2.VideoCapture(camera_id)
        elif key == ord('+') or key == ord('='):  # 提高信心度門檻
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"信心度門檻調整為: {confidence_threshold:.0%}")
        elif key == ord('-'):  # 降低信心度門檻
            confidence_threshold = max(0.30, confidence_threshold - 0.05)
            print(f"信心度門檻調整為: {confidence_threshold:.0%}")
        elif key == ord(']'):  # 提高站立角度閾值
            standing_threshold = min(175.0, standing_threshold + 5.0)
            print(f"站立角度閾值調整為: {standing_threshold:.0f}°")
        elif key == ord('['):  # 降低站立角度閾值
            standing_threshold = max(140.0, standing_threshold - 5.0)
            print(f"站立角度閾值調整為: {standing_threshold:.0f}°")
        elif key == ord('.'):  # 提高身體偵測置信度
            body_conf_threshold = min(0.9, body_conf_threshold + 0.05)
            print(f"身體偵測置信度調整為: {body_conf_threshold:.0%}")
        elif key == ord(','):  # 降低身體偵測置信度
            body_conf_threshold = max(0.1, body_conf_threshold - 0.05)
            print(f"身體偵測置信度調整為: {body_conf_threshold:.0%}")

    # 清理資源
    cap.release()
    cv2.destroyAllWindows()
    print("\n程式結束")


if __name__ == "__main__":
    main()
