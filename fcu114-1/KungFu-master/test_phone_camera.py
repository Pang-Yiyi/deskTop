# Code by AkinoAlice@TyrantRey
# 手機攝影機即時動作辨識測試程式 (支援 EpocCam / DroidCam / IP Webcam)

import cv2
import numpy as np
import sys

from helper.model import pose_model
from helper.kungfu_classifier import get_kungfu_classifier
from helper.angle_calculator import extract_angles_from_keypoints, is_valid_keypoints


def list_available_cameras(max_cameras: int = 10) -> list[int]:
    """掃描可用的攝影機"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def connect_ip_camera(url: str) -> cv2.VideoCapture | None:
    """連接 IP 攝影機 (用於 IP Webcam App)"""
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        return cap
    return None


def main():
    print("=" * 60)
    print("功夫動作即時辨識 - 手機攝影機版")
    print("=" * 60)

    # 載入模型
    print("\n正在載入模型...")
    posture_detector = pose_model
    kungfu_classifier = get_kungfu_classifier()

    # 顯示模型資訊
    info = kungfu_classifier.get_model_info()
    print(f"模型類型: {info['model_type']}")
    print(f"裝置: {info['device']}")
    print(f"動作類別: {info['classes']}")

    # 動作顏色對應
    action_colors = {
        'act1': (107, 107, 255),   # 握拳式 - 紅色 (BGR)
        'act2': (196, 205, 78),    # 出拳式 - 青色
        'act3': (209, 183, 69),    # 踢腿式 - 藍色
        'act4': (180, 206, 150),   # 提膝式 - 綠色
    }

    # 掃描可用攝影機
    print("\n正在掃描可用攝影機...")
    available_cameras = list_available_cameras()

    if available_cameras:
        print(f"找到 {len(available_cameras)} 個攝影機: {available_cameras}")
        print("\n攝影機列表:")
        for cam_id in available_cameras:
            cap = cv2.VideoCapture(cam_id)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"  [{cam_id}] 解析度: {width}x{height}")
    else:
        print("未找到任何攝影機！")

    # 選擇攝影機
    print("\n" + "=" * 60)
    print("連接方式:")
    print("  1. 輸入攝影機編號 (0, 1, 2, ...)")
    print("  2. 輸入 IP Webcam URL (例如: http://192.168.1.100:8080/video)")
    print("  3. 按 Enter 使用預設 (最後一個攝影機，通常是 EpocCam)")
    print("=" * 60)

    user_input = input("\n請輸入選擇: ").strip()

    cap = None

    if user_input == "":
        # 使用最後一個攝影機 (通常是虛擬攝影機如 EpocCam)
        if available_cameras:
            camera_id = available_cameras[-1]
            print(f"\n使用攝影機 {camera_id}")
            cap = cv2.VideoCapture(camera_id)
        else:
            print("錯誤: 沒有可用的攝影機！")
            return

    elif user_input.startswith("http"):
        # IP Webcam URL
        print(f"\n正在連接 IP 攝影機: {user_input}")
        cap = connect_ip_camera(user_input)
        if cap is None:
            print("錯誤: 無法連接到 IP 攝影機！")
            print("請確認:")
            print("  1. 手機和電腦在同一個 WiFi 網路")
            print("  2. IP Webcam App 正在運行")
            print("  3. URL 正確 (通常是 http://手機IP:8080/video)")
            return

    else:
        # 攝影機編號
        try:
            camera_id = int(user_input)
            print(f"\n使用攝影機 {camera_id}")
            cap = cv2.VideoCapture(camera_id)
        except ValueError:
            print("錯誤: 無效的輸入！")
            return

    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機！")
        return

    # 取得攝影機資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"攝影機解析度: {width}x{height}")

    print("\n操作說明:")
    print("  - 按 'q' 或 ESC 退出程式")
    print("  - 按 's' 截圖保存")
    print("  - 按 'f' 切換全螢幕")
    print("  - 按 'm' 切換鏡像模式")
    print("\n開始即時辨識...\n")

    # 設定視窗
    window_name = "Kungfu Action Recognition - Phone Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_count = 0
    screenshot_count = 0
    mirror_mode = True  # 預設開啟鏡像模式（自拍視角）
    fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影格，嘗試重新連接...")
            continue

        frame_count += 1

        # 鏡像模式
        if mirror_mode:
            frame = cv2.flip(frame, 1)

        display_frame = frame.copy()

        # 使用 YOLO 姿態偵測
        results = posture_detector.model.predict(frame, verbose=False)

        action_text = "No Detection"
        confidence_text = ""
        color = (128, 128, 128)  # 灰色

        if results and len(results[0]) > 0:
            # 繪製骨架
            annotated_frame = results[0].plot()
            if mirror_mode:
                display_frame = annotated_frame
            else:
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

        # 繪製狀態提示
        status_text = f"Mirror: {'ON' if mirror_mode else 'OFF'} | 'q' quit | 's' screenshot | 'm' mirror | 'f' fullscreen"
        cv2.putText(
            display_frame,
            status_text,
            (10, display_frame.shape[0] - 10),
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
            filename = f"phone_screenshot_{screenshot_count}.png"
            cv2.imwrite(filename, display_frame)
            print(f"已保存截圖: {filename}")
        elif key == ord('m'):  # 切換鏡像
            mirror_mode = not mirror_mode
            print(f"鏡像模式: {'開啟' if mirror_mode else '關閉'}")
        elif key == ord('f'):  # 切換全螢幕
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # 清理資源
    cap.release()
    cv2.destroyAllWindows()
    print("\n程式結束")


if __name__ == "__main__":
    main()
