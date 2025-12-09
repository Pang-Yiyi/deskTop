# Code by AkinoAlice@TyrantRey
# 攝影機即時動作辨識測試程式

import cv2
import numpy as np
import sys

from helper.model import pose_model
from helper.kungfu_classifier import get_kungfu_classifier
from helper.angle_calculator import extract_angles_from_keypoints, is_valid_keypoints


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
    print("\n開始即時辨識...\n")

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

            # 取得關鍵點
            keypoints_xy = results[0][0].keypoints.xy[0].cpu().numpy()

            if is_valid_keypoints(keypoints_xy):
                try:
                    # 提取角度
                    angles = extract_angles_from_keypoints(keypoints_xy)

                    # 進行動作分類
                    action_code, action_name, probs = kungfu_classifier.predict(angles)
                    confidence = kungfu_classifier.get_confidence(probs)

                    action_text = action_name
                    confidence_text = f"Confidence: {confidence:.1%}"
                    color = action_colors.get(action_code, (128, 128, 128))

                    # 每 30 幀輸出一次到終端機
                    if frame_count % 30 == 0:
                        print(f"[Frame {frame_count}] {action_name} ({confidence:.1%})")

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

        # 繪製操作提示
        cv2.putText(
            display_frame,
            "Press 'q' to quit | 's' to screenshot",
            (display_frame.shape[1] - 400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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

    # 清理資源
    cap.release()
    cv2.destroyAllWindows()
    print("\n程式結束")


if __name__ == "__main__":
    main()
