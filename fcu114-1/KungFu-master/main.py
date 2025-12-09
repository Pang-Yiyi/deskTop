# Code by AkinoAlice@TyrantRey

import torch  # noqa: F401
import sys
import cv2
import logging

import numpy as np

from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QStackedWidget,
    QMessageBox,
    QSlider,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from typing import Callable
from fastdtw import fastdtw  # type: ignore[import-untyped]
from scipy.spatial.distance import euclidean  # type: ignore[import-untyped]
from ultralytics.engine.results import Results

from helper.model import hand_model, pose_model
from helper.database import sqlite3_database
from helper.kungfu_classifier import get_kungfu_classifier
from helper.angle_calculator import extract_angles_from_keypoints, is_valid_keypoints

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(filename="log.log", filemode="w+", level=logging.DEBUG)


class AppState:
    def __init__(self) -> None:
        self.recorded_videos: list[str] = []


class MainPage(QWidget):
    def __init__(self, switch_page_callback) -> None:
        super().__init__()
        self.switch_page = switch_page_callback
        self.showMaximized()

        layout = QVBoxLayout()
        layout.addStretch()

        for name, page_idx in [("錄製", 1), ("測試", 2), ("指導", 3)]:
            btn = QPushButton(name)
            btn.setMinimumHeight(60)
            btn.clicked.connect(lambda checked, idx=page_idx: self.switch_page(idx))
            layout.addWidget(btn)

        layout.addStretch()
        self.setLayout(layout)


class VideoWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self.label = QLabel("未載入影片")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("border: 2px solid #ccc; background: #000;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def load_video(self, path):
        self.stop()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.label.setText(f"載入失敗: {path}")
            return False
        self.timer.start(30)
        return True

    def load_camera(self):
        self.stop()
        # 預設使用攝影機 ID 1
        self.cap = cv2.VideoCapture(1)  # type: ignore[assignment]
        if not self.cap.isOpened():  # type: ignore[attr-defined]
            self.label.setText("攝影機不可用")
            return False
        self.timer.start(30)
        return True

    def _update_frame(self) -> None:
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        scaled = qt_image.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(QPixmap.fromImage(scaled))

    def stop(self) -> None:
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.label.setText("已停止")


class RecordingPage(QWidget):
    def __init__(self, app_state, back_callback: Callable) -> None:
        super().__init__()
        self.app_state = app_state
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.hand_detector = hand_model
        self.sqlite3_database = sqlite3_database

        # 使用者介面
        self.video_widget = VideoWidget()

        btn_load = QPushButton("載入影片")
        btn_load.clicked.connect(self._load_video)

        btn_confirm = QPushButton("確認並偵測姿態")
        btn_confirm.clicked.connect(self._on_confirm)
        btn_confirm.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_confirm)
        btn_layout.addWidget(btn_back)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>錄製頁面</h2>"))
        layout.addWidget(self.video_widget)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影片", "", "影片檔案 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return

        self.path = path
        if not self.video_widget.load_video(path):
            return

        if path not in self.app_state.recorded_videos:
            self.app_state.recorded_videos.append(path)

    def _on_confirm(self) -> None:
        if self.path == "":
            QMessageBox.warning(self, "無影片", "請先載入影片。")
            return

        reply = QMessageBox.question(
            self,
            "確認偵測",
            f"開始姿態偵測:\n{Path(self.path).name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            predicted_video_path, predicted_npy_path = (
                self.posture_detector.detect_video(self.path)
            )
            self.posture_detector.save_npy(predicted_npy_path)
            # 注意: 訓練後手部偵測效果不佳
            # predicted_hand_video_path, predicted_hand_npy_path = (
            #     self.hand_detector.detect_video(self.path)
            # )
            # ...

            self.sqlite3_database.insert_posture(
                posture_name=Path(self.path).stem,
                video_path=str(predicted_video_path),
                npy_path=str(predicted_npy_path),
            )

            if predicted_video_path.exists():
                self.video_widget.load_video(str(predicted_video_path))
            else:
                raise FileNotFoundError(
                    f"Predicted video not found: {predicted_video_path}"
                )

            QMessageBox.information(
                self,
                "成功",
                f"姿態偵測完成！\n結果儲存至:\n{predicted_video_path}",
            )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"偵測失敗:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _on_back(self):
        self.video_widget.stop()
        self.back_callback()


class TestingPage(QWidget):
    def __init__(self, app_state: AppState, back_callback: Callable):
        super().__init__()
        self.app_state = app_state
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.hand_detector = hand_model
        self.sqlite3_database = sqlite3_database
        self.kungfu_classifier = get_kungfu_classifier()

        self.left_npy_path: Path | str | None = None
        self.right_npy_path: Path | str | None = None
        self.dtw_path: dict[int, tuple] = {}

        # 學生影片的關鍵點資料 (用於動作分類)
        self.student_keypoints: list[np.ndarray] = []

        self.video_left = VideoWidget()
        self.video_right = VideoWidget()

        btn_load_student = QPushButton("載入學生影片")
        btn_load_student.clicked.connect(self._load_student)

        self.combo_recorded = QComboBox()
        btn_load_teacher = QPushButton("載入教師示範")
        btn_load_teacher.clicked.connect(self._load_teacher)

        btn_compare = QPushButton("比較姿態")
        btn_compare.clicked.connect(self._compare_postures)
        btn_compare.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold;"
        )

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)

        self.similarity_label = QLabel("相似度: 無")
        self.similarity_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #4CAF50;"
        )

        # 動作分類結果標籤
        self.action_label = QLabel("動作辨識: 等待載入影片...")
        self.action_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2196F3; "
            "border: 2px solid #2196F3; border-radius: 8px; padding: 8px;"
        )

        # 動作統計標籤
        self.action_stats_label = QLabel("")
        self.action_stats_label.setStyleSheet(
            "font-size: 12px; color: #666;"
        )

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.valueChanged.connect(self._on_slider_changed)

        self.frame_label = QLabel("影格: 0 / 0")

        left_controls = QVBoxLayout()
        left_controls.addWidget(QLabel("<b>學生影片</b>"))
        left_controls.addWidget(btn_load_student)

        right_controls = QVBoxLayout()
        right_controls.addWidget(QLabel("<b>教師示範</b>"))
        right_controls.addWidget(self.combo_recorded)
        right_controls.addWidget(btn_load_teacher)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_left)
        video_layout.addWidget(self.video_right)

        controls_layout = QHBoxLayout()
        controls_layout.addLayout(left_controls)
        controls_layout.addLayout(right_controls)
        controls_layout.addWidget(btn_compare)
        controls_layout.addWidget(btn_back)

        result_layout = QVBoxLayout()
        result_layout.addWidget(self.similarity_label)
        result_layout.addWidget(self.action_label)
        result_layout.addWidget(self.action_stats_label)
        result_layout.addWidget(QLabel("播放同步:"))
        result_layout.addWidget(self.progress_slider)
        result_layout.addWidget(self.frame_label)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>測試頁面</h2>"))
        layout.addLayout(video_layout)
        layout.addLayout(controls_layout)
        layout.addLayout(result_layout)
        self.setLayout(layout)

    def showEvent(self, a0):
        self.combo_recorded.clear()
        postures = self.sqlite3_database.fetch_all_postures()
        for posture in postures:
            self.combo_recorded.addItem(
                posture["posture_name"],
                {"video_path": posture["video_path"], "npy_path": posture["npy_path"]},
            )
        super().showEvent(a0)

    def _load_student(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇學生影片", "", "影片檔案 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            uploaded_video_path = Path(path)
            predicted_video_path, predicted_npy_path = self.posture_detector.detect_video(
                uploaded_video_path
            )

            self.posture_detector.save_npy(predicted_npy_path)
            self.video_left.load_video(predicted_video_path)
            self.left_npy_path = predicted_npy_path

            # 從原始影片擷取關鍵點進行動作分類
            self._analyze_student_actions(path)

            QMessageBox.information(self, "成功", "學生影片已載入並分析完成！")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"分析學生影片失敗:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _analyze_student_actions(self, video_path: str) -> None:
        """分析學生影片中的動作類型"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        self.student_keypoints = []
        action_counts: dict[str, int] = {}
        total_valid_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.posture_detector.model.predict(frame_rgb, verbose=False)

            if results and len(results[0]) > 0:
                keypoints_xy = results[0][0].keypoints.xy[0].cpu().numpy()
                self.student_keypoints.append(keypoints_xy)

                if is_valid_keypoints(keypoints_xy):
                    try:
                        angles = extract_angles_from_keypoints(keypoints_xy)
                        action_code, action_name, probs = self.kungfu_classifier.predict(angles)
                        confidence = self.kungfu_classifier.get_confidence(probs)

                        # 只有信心度 >= 50% 才計入統計
                        if confidence >= 0.5:
                            action_counts[action_name] = action_counts.get(action_name, 0) + 1
                            total_valid_frames += 1
                    except Exception:
                        pass

        cap.release()

        # 更新動作分類結果顯示
        if total_valid_frames > 0:
            # 找出主要動作（出現次數最多的）
            main_action = max(action_counts.items(), key=lambda x: x[1])
            main_action_name = main_action[0]
            main_action_ratio = main_action[1] / total_valid_frames * 100

            self.action_label.setText(f"主要動作: {main_action_name} ({main_action_ratio:.1f}%)")

            # 根據動作類型設定顏色
            action_colors = {
                '握拳式 (Fist)': '#FF6B6B',
                '出拳式 (Punch)': '#4ECDC4',
                '踢腿式 (Kick)': '#45B7D1',
                '提膝式 (Knee)': '#96CEB4',
            }
            color = action_colors.get(main_action_name, '#2196F3')
            self.action_label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: white; "
                f"background-color: {color}; border: 2px solid {color}; "
                f"border-radius: 8px; padding: 8px;"
            )

            # 顯示各動作統計
            stats_text = "動作分佈: " + " | ".join(
                f"{name}: {count}幀 ({count/total_valid_frames*100:.1f}%)"
                for name, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
            )
            self.action_stats_label.setText(stats_text)
        else:
            self.action_label.setText("動作辨識: 無法辨識有效動作")
            self.action_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #888; "
                "border: 2px solid #888; border-radius: 8px; padding: 8px;"
            )
            self.action_stats_label.setText("")

    def _load_teacher(self) -> None:
        data = self.combo_recorded.currentData()
        if not data:
            return

        video_path = data.get("video_path")
        self.right_npy_path = data.get("npy_path")
        
        if not self.right_npy_path:
            return

        if not Path(self.right_npy_path).exists():
            QMessageBox.warning(self, "錯誤", f"NPY 檔案未找到: {self.right_npy_path}")
            return

        self.video_right.load_video(video_path)

    def normalize_keypoints(self, kpts: np.ndarray) -> np.ndarray:
        kpts = np.array(kpts, dtype=np.float32)
        if kpts.ndim == 3:
            kpts = kpts[:, :2]
        elif kpts.shape[-1] == 3:
            kpts = kpts[:, :2]
        center = np.mean(kpts, axis=0)
        scale = np.linalg.norm(kpts - center)
        return (kpts - center) / scale

    def compute_similarity(self, seq_a, seq_b):
        seq_a = [self.normalize_keypoints(pose).flatten() for pose in seq_a]
        seq_b = [self.normalize_keypoints(pose).flatten() for pose in seq_b]

        distance, path = fastdtw(seq_a, seq_b, dist=euclidean)
        avg_distance = distance / max(len(seq_a), len(seq_b))

        similarity = np.exp(-5 * avg_distance) * 100
        similarity = max(0.0, min(100.0, similarity))
        return similarity, avg_distance, distance, path

    def _compare_postures(self) -> None:
        if not self.left_npy_path or not self.right_npy_path:
            QMessageBox.warning(self, "錯誤", "請先載入兩個影片！")
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            left_poses = np.load(self.left_npy_path)
            right_poses = np.load(self.right_npy_path)

            similarity, avg_distance, total_distance, path = self.compute_similarity(
                left_poses, right_poses
            )

            self.similarity_label.setText(f"相似度: {similarity:.2f}%")

            # 設定進度顯示
            self.progress_slider.setEnabled(True)
            self.progress_slider.setMaximum(len(path) - 1)
            self.progress_slider.setValue(0)
            self.frame_label.setText(f"影格: 0 / {len(path)}")

            self.dtw_path = path
            self.left_total_frames = left_poses.shape[0]
            self.right_total_frames = right_poses.shape[0]

            QMessageBox.information(
                self,
                "比較完成",
                f"相似度: {similarity:.2f}%\n"
                f"平均距離: {avg_distance:.4f}\n"
                f"總 DTW 距離: {total_distance:.2f}\n\n"
                f"學生影格數: {self.left_total_frames}\n"
                f"教師影格數: {self.right_total_frames}\n"
                f"DTW 路徑長度: {len(path)}",
            )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"比較失敗:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _on_slider_changed(self, value: int) -> None:
        if not hasattr(self, "dtw_path"):
            return

        left_frame, right_frame = self.dtw_path[value]
        self.frame_label.setText(
            f"影格: {value} / {len(self.dtw_path)} | "
            f"學生: {left_frame} | 教師: {right_frame}"
        )

        if self.video_left.cap:
            self.video_left.cap.set(cv2.CAP_PROP_POS_FRAMES, left_frame)
        if self.video_right.cap:
            self.video_right.cap.set(cv2.CAP_PROP_POS_FRAMES, right_frame)

    def _on_back(self) -> None:
        self.video_left.stop()
        self.video_right.stop()
        # 重置動作分類顯示
        self.action_label.setText("動作辨識: 等待載入影片...")
        self.action_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2196F3; "
            "border: 2px solid #2196F3; border-radius: 8px; padding: 8px;"
        )
        self.action_stats_label.setText("")
        self.student_keypoints = []
        self.back_callback()


class GuidingPage(QWidget):
    def __init__(self, back_callback: Callable) -> None:
        super().__init__()
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.sqlite3_database = sqlite3_database
        self.kungfu_classifier = get_kungfu_classifier()

        self.teacher_frames: list[np.ndarray] = []
        self.teacher_poses = None
        self.current_frame_idx = 0
        self.finished_times = 0
        self.is_running = False
        self.camera_cap = None
        self.selected_camera_id: int = 0

        self.frame_label = QLabel("選擇影片並開始")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(320, 240)
        self.frame_label.setStyleSheet("border: 2px solid #4CAF50; background: #000;")

        self.combo_videos = QComboBox()
        self.combo_videos.currentIndexChanged.connect(self._on_video_selected)

        btn_load = QPushButton("載入教師影片")
        btn_load.clicked.connect(self._load_teacher_video)

        self.similarity_label = QLabel("相似度: 無")
        self.similarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.similarity_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #888; "
            "border: 3px solid #888; border-radius: 10px; padding: 20px;"
        )
        self.similarity_label.setMinimumHeight(100)

        # 動作分類結果標籤
        self.action_label = QLabel("動作: 等待偵測...")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2196F3; "
            "border: 2px solid #2196F3; border-radius: 8px; padding: 10px;"
        )
        self.action_label.setMinimumHeight(60)

        # 信心度標籤
        self.confidence_label = QLabel("信心度: --")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(
            "font-size: 14px; color: #666;"
        )

        self.progress_label = QLabel("影格: 0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.finished_label = QLabel("已完成: 0 次")
        self.finished_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.camera_widget = VideoWidget()

        btn_start = QPushButton("開始練習")
        btn_start.clicked.connect(self._start_practice)
        btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )

        btn_stop = QPushButton("停止")
        btn_stop.clicked.connect(self._stop_practice)
        btn_stop.setStyleSheet("background-color: #f44336; color: white;")

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)

        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._process_frame)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>教師示範</b>"))
        left_layout.addWidget(self.frame_label, 3)
        left_layout.addWidget(QLabel("選擇影片:"))
        left_layout.addWidget(self.combo_videos)
        left_layout.addWidget(btn_load)

        middle_layout = QVBoxLayout()
        middle_layout.addStretch()
        middle_layout.addWidget(self.similarity_label)
        middle_layout.addWidget(self.action_label)
        middle_layout.addWidget(self.confidence_label)
        middle_layout.addWidget(self.progress_label)
        middle_layout.addWidget(self.finished_label)
        middle_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>您的攝影機</b>"))
        right_layout.addWidget(self.camera_widget, 3)
        right_layout.addWidget(btn_start)
        right_layout.addWidget(btn_stop)

        content_layout = QHBoxLayout()
        content_layout.addLayout(left_layout, 2)
        content_layout.addLayout(middle_layout, 1)
        content_layout.addLayout(right_layout, 2)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>互動式姿態指導</h2>"))
        layout.addLayout(content_layout)
        layout.addWidget(btn_back)
        self.setLayout(layout)

    def showEvent(self, a0):
        self.combo_videos.clear()
        postures = self.sqlite3_database.fetch_all_postures()
        for posture in postures:
            self.combo_videos.addItem(
                posture["posture_name"],
                {"video_path": posture["video_path"], "npy_path": posture["npy_path"]},
            )
        super().showEvent(a0)

    def _on_video_selected(self, index: int):
        self._stop_practice()
        self.teacher_frames = []
        self.teacher_poses = None
        self.current_frame_idx = 0
        self.frame_label.setText("點擊 '載入教師影片'")

    def _load_teacher_video(self):
        data = self.combo_videos.currentData()
        if not data:
            QMessageBox.warning(self, "錯誤", "請先選擇影片！")
            return

        video_path = data["video_path"]
        npy_path = data["npy_path"]

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            if not Path(npy_path).exists():
                raise FileNotFoundError(f"NPY file not found: {npy_path}")

            self.teacher_poses = np.load(npy_path)  # (frames, 21, 2)

            cap = cv2.VideoCapture(video_path)
            self.teacher_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.teacher_frames.append(frame_rgb)

            cap.release()

            if len(self.teacher_frames) == 0:
                raise ValueError("No frames extracted from video")

            self.current_frame_idx = 0
            self._display_current_frame()

            QMessageBox.information(
                self,
                "成功",
                f"已載入 {len(self.teacher_frames)} 個影格！\n準備練習。",
            )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入影片失敗:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _display_current_frame(self):
        if self.current_frame_idx >= len(self.teacher_frames):
            self.current_frame_idx = 0
            self.finished_times += 1
            self.finished_label.setText(f"已完成: {self.finished_times + 1} 次")

        frame = self.teacher_frames[self.current_frame_idx]
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        scaled = qt_image.scaled(
            self.frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.frame_label.setPixmap(QPixmap.fromImage(scaled))

        self.progress_label.setText(
            f"影格: {self.current_frame_idx + 1} / {len(self.teacher_frames)}"
        )

    def _start_practice(self):
        if not self.teacher_frames or self.teacher_poses is None:
            QMessageBox.warning(self, "錯誤", "請先載入教師影片！")
            return

        self.current_frame_idx = 0
        self._display_current_frame()

        if self.camera_cap is None or not self.camera_cap.isOpened():
            self.camera_cap = cv2.VideoCapture(1)  # 預設使用攝影機 ID 1
            if not self.camera_cap.isOpened():
                QMessageBox.critical(self, "錯誤", "無法開啟攝影機！")
                return

        self.is_running = True
        self.detection_timer.start(60)

    def _stop_practice(self):
        self.is_running = False
        self.detection_timer.stop()

        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None

        self.camera_widget.stop()
        self.similarity_label.setText("相似度: 無")
        self.similarity_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #888; "
            "border: 3px solid #888; border-radius: 10px; padding: 20px;"
        )
        # 重置動作分類標籤
        self.action_label.setText("動作: 等待偵測...")
        self.action_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2196F3; "
            "border: 2px solid #2196F3; border-radius: 8px; padding: 10px;"
        )
        self.confidence_label.setText("信心度: --")
        self.confidence_label.setStyleSheet("font-size: 14px; color: #666;")

    def _process_frame(self):
        if not self.is_running or not self.camera_cap:
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        scaled = qt_image.scaled(
            self.camera_widget.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))

        student_pose = self.posture_detector.model.predict(frame_rgb, verbose=False)
        if student_pose is None:
            self._update_similarity(0)
            self._update_action_label(None, None, None)
            return

        if len(student_pose[0]) < 1:
            self._update_similarity(0)
            self._update_action_label(None, None, None)
            return

        # 取得關鍵點座標 (用於動作分類)
        keypoints_xy = student_pose[0][0].keypoints.xy[0].cpu().numpy()  # type: ignore[reportOptionalMemberAccess]

        # 進行動作分類
        if is_valid_keypoints(keypoints_xy):
            try:
                angles = extract_angles_from_keypoints(keypoints_xy)
                action_code, action_name, probs = self.kungfu_classifier.predict(angles)
                confidence = self.kungfu_classifier.get_confidence(probs)
                self._update_action_label(action_code, action_name, confidence)
            except Exception as e:
                logger.warning(f"動作分類失敗: {e}")
                self._update_action_label(None, None, None)
        else:
            self._update_action_label(None, None, None)

        student_pose_normalized = student_pose[0][0].keypoints.xyn[0].cpu().numpy().T[0]  # type: ignore[reportOptionalMemberAccess]
        if self.teacher_poses is None:
            self._update_similarity(0)
            return

        teacher_pose = self.teacher_poses[self.current_frame_idx].T[0]

        similarity = np.linalg.norm(teacher_pose - student_pose_normalized).mean() * 100

        self._update_similarity(float(similarity))

        if similarity >= 75:
            self.current_frame_idx += 1
            self._display_current_frame()

            if self.current_frame_idx >= len(self.teacher_frames):
                QMessageBox.information(self, "恭喜！", "恭喜你完成了！")
                self._stop_practice()

    def _extract_pose_from_results(self, results: Results) -> np.ndarray | None:
        keypoints = results[0].keypoints

        if keypoints is None:
            return None

        return keypoints.xyn[0].cpu().numpy()

    def _update_similarity(self, similarity: float):
        if similarity > 100:
            similarity = 100
        self.similarity_label.setText(f"{similarity:.1f}%")

        if similarity >= 75:
            self.similarity_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: white; "
                "background-color: #4CAF50; border: 3px solid #4CAF50; "
                "border-radius: 10px; padding: 20px;"
            )
        else:
            self.similarity_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: white; "
                "background-color: #f44336; border: 3px solid #f44336; "
                "border-radius: 10px; padding: 20px;"
            )

    def _update_action_label(
        self,
        action_code: str | None,
        action_name: str | None,
        confidence: float | None
    ):
        """更新動作分類結果顯示"""
        if action_code is None or action_name is None:
            self.action_label.setText("動作: 偵測中...")
            self.action_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #888; "
                "border: 2px solid #888; border-radius: 8px; padding: 10px;"
            )
            self.confidence_label.setText("信心度: --")
            return

        self.action_label.setText(f"動作: {action_name}")

        # 根據動作類型設定不同顏色
        action_colors = {
            'act1': '#FF6B6B',  # 握拳式 - 紅色
            'act2': '#4ECDC4',  # 出拳式 - 青色
            'act3': '#45B7D1',  # 踢腿式 - 藍色
            'act4': '#96CEB4',  # 提膝式 - 綠色
        }
        color = action_colors.get(action_code, '#2196F3')

        self.action_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: white; "
            f"background-color: {color}; border: 2px solid {color}; "
            f"border-radius: 8px; padding: 10px;"
        )

        if confidence is not None:
            self.confidence_label.setText(f"信心度: {confidence:.1%}")
            if confidence >= 0.8:
                self.confidence_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
            elif confidence >= 0.5:
                self.confidence_label.setStyleSheet("font-size: 14px; color: #FF9800;")
            else:
                self.confidence_label.setStyleSheet("font-size: 14px; color: #f44336;")

    def _on_back(self):
        self._stop_practice()
        self.back_callback()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("影片訓練應用程式")
        self.setGeometry(100, 100, 1200, 800)

        self.app_state = AppState()

        self.stack = QStackedWidget()
        self.stack.addWidget(MainPage(self.stack.setCurrentIndex))
        self.stack.addWidget(
            RecordingPage(self.app_state, lambda: self.stack.setCurrentIndex(0))
        )
        self.stack.addWidget(
            TestingPage(self.app_state, lambda: self.stack.setCurrentIndex(0))
        )
        self.stack.addWidget(GuidingPage(lambda: self.stack.setCurrentIndex(0)))

        self.setCentralWidget(self.stack)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
