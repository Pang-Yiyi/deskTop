# Code by AkinoAlice@TyrantRey
# 訓練資料收集工具 - 使用攝影機收集新的動作資料

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from helper.model import pose_model
from helper.angle_calculator import extract_angles_from_keypoints, is_valid_keypoints


def main():
    print("=" * 60)
    print("功夫動作訓練資料收集工具")
    print("=" * 60)

    # 載入姿態偵測模型
    print("\n正在載入姿態偵測模型...")
    posture_detector = pose_model

    # 讀取現有資料集
    csv_path = Path('./dataset/pose_angles_summary_actual.csv')
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        print(f"\n現有資料集統計:")
        print(existing_df['Action_Type'].value_counts())
        print(f"總計: {len(existing_df)} 筆")
    else:
        existing_df = pd.DataFrame()
        print("\n尚無現有資料集，將建立新檔案")

    # 動作類別選項
    action_options = {
        '1': ('act1', '握拳式 (Fist)'),
        '2': ('act2', '出拳式 (Punch)'),
        '3': ('act3', '踢腿式 (Kick)'),
        '4': ('act4', '提膝式 (Knee)'),
        '5': ('act5', '站立/無動作 (Standby)'),  # 新類別
    }

    print("\n" + "=" * 60)
    print("動作類別選項:")
    for key, (code, name) in action_options.items():
        print(f"  [{key}] {name}")
    print("=" * 60)

    # 選擇要收集的動作
    while True:
        choice = input("\n請選擇要收集的動作類別 (1-5): ").strip()
        if choice in action_options:
            action_code, action_name = action_options[choice]
            break
        print("無效選擇，請重新輸入")

    print(f"\n將收集: {action_name} ({action_code})")

    # 開啟攝影機
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機！")
        return

    print(f"攝影機已開啟 (ID: {camera_id})")

    print("\n" + "=" * 60)
    print("操作說明:")
    print("  - 按 [空白鍵] 擷取當前姿態作為訓練資料")
    print("  - 按 [s] 儲存所有收集的資料")
    print("  - 按 [q] 或 [ESC] 退出 (會詢問是否儲存)")
    print("  - 按 [c] 切換動作類別")
    print("=" * 60)

    # 設定視窗
    window_name = "Training Data Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # 收集的資料
    collected_data = []
    frame_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影格")
            break

        frame_count += 1
        display_frame = frame.copy()

        # 姿態偵測
        results = posture_detector.model.predict(frame, verbose=False)

        current_angles = None
        status_text = "No Pose Detected"
        status_color = (128, 128, 128)

        if results and len(results[0]) > 0:
            # 繪製骨架
            display_frame = results[0].plot()

            # 取得關鍵點
            keypoints_xy = results[0][0].keypoints.xy[0].cpu().numpy()

            if is_valid_keypoints(keypoints_xy):
                try:
                    current_angles = extract_angles_from_keypoints(keypoints_xy)
                    status_text = "Pose Ready - Press SPACE to capture"
                    status_color = (0, 255, 0)
                except Exception:
                    status_text = "Angle Extraction Error"
                    status_color = (0, 0, 255)
            else:
                status_text = "Incomplete Pose"
                status_color = (0, 165, 255)

        # 繪製資訊面板
        panel_height = 150
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], panel_height), (0, 0, 0), -1)

        # 動作類別
        cv2.putText(
            display_frame,
            f"Collecting: {action_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        # 狀態
        cv2.putText(
            display_frame,
            status_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        # 已收集數量
        cv2.putText(
            display_frame,
            f"Collected: {len(collected_data)} samples",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # 操作提示
        cv2.putText(
            display_frame,
            "[SPACE] Capture | [S] Save | [C] Change Action | [Q] Quit",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # 顯示當前角度（如果有）
        if current_angles:
            y_offset = panel_height + 30
            for i, (name, angle) in enumerate(current_angles.items()):
                cv2.putText(
                    display_frame,
                    f"{name}: {angle:.1f}",
                    (20, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1
                )

        cv2.imshow(window_name, display_frame)

        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # 空白鍵 - 擷取
            if current_angles:
                # 建立檔案名稱
                sample_num = len(collected_data) + 1
                file_name = f"{action_code}_collect_{timestamp}_{sample_num}.jpg"

                # 儲存資料
                data_row = {
                    'File_Name': file_name,
                    **current_angles,
                    'Action_Type': action_code
                }
                collected_data.append(data_row)

                # 儲存圖片
                img_path = Path('./dataset') / file_name
                cv2.imwrite(str(img_path), frame)

                print(f"[{len(collected_data)}] 已擷取: {file_name}")
                print(f"    角度: R_Elbow={current_angles['R_Elbow_Angle']:.1f}, "
                      f"L_Elbow={current_angles['L_Elbow_Angle']:.1f}, "
                      f"R_Knee={current_angles['R_Knee_Angle']:.1f}")
            else:
                print("無法擷取: 未偵測到有效姿態")

        elif key == ord('s'):  # 儲存
            if collected_data:
                save_data(csv_path, existing_df, collected_data)
                print(f"\n已儲存 {len(collected_data)} 筆資料！")
                collected_data = []  # 清空已儲存的
            else:
                print("沒有待儲存的資料")

        elif key == ord('c'):  # 切換動作
            print("\n動作類別選項:")
            for k, (code, name) in action_options.items():
                print(f"  [{k}] {name}")
            new_choice = input("請選擇新的動作類別 (1-5): ").strip()
            if new_choice in action_options:
                action_code, action_name = action_options[new_choice]
                print(f"已切換到: {action_name}")
            else:
                print("無效選擇，維持原類別")

        elif key == ord('q') or key == 27:  # 退出
            if collected_data:
                save_choice = input(f"\n有 {len(collected_data)} 筆未儲存的資料，是否儲存？(y/n): ").strip().lower()
                if save_choice == 'y':
                    save_data(csv_path, existing_df, collected_data)
                    print(f"已儲存 {len(collected_data)} 筆資料！")
            print("\n退出程式...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n資料收集完成！")
    print(f"請重新執行 train_kungfu_deep_learning.ipynb 來訓練更新後的模型。")


def save_data(csv_path: Path, existing_df: pd.DataFrame, new_data: list):
    """儲存收集的資料到 CSV"""
    new_df = pd.DataFrame(new_data)

    if len(existing_df) > 0:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False)

    print(f"\n更新後的資料集統計:")
    print(combined_df['Action_Type'].value_counts())
    print(f"總計: {len(combined_df)} 筆")


if __name__ == "__main__":
    main()
