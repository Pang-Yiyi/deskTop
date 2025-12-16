# Code by AkinoAlice@TyrantRey
# 重新設計的功夫訓練介面 - 現代化 UI

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
    QStackedWidget,
    QMessageBox,
    QProgressBar,
    QFrame,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen
from typing import Callable

from helper.model import pose_model
from helper.kungfu_classifier import get_kungfu_classifier
from helper.angle_calculator import extract_angles_from_keypoints, is_valid_keypoints


# ============================================
# 動作偵測輔助函數 (與 test_camera.py 一致)
# ============================================
def is_full_body_detected(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, min_confidence: float = 0.5) -> tuple:
    """
    檢查是否偵測到完整身體（從肩膀到腳踝）
    使用置信度分數來判斷關鍵點是否真正被偵測到
    """
    keypoint_names = {
        5: "左肩", 6: "右肩",
        11: "左臀", 12: "右臀",
        13: "左膝", 14: "右膝",
        15: "左腳踝", 16: "右腳踝"
    }

    if keypoints_xy is None or len(keypoints_xy) < 17:
        return False, ["關鍵點無效"]
    if keypoints_conf is None or len(keypoints_conf) < 17:
        return False, ["置信度無效"]

    required_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    missing_parts = []

    for idx in required_indices:
        conf = keypoints_conf[idx]
        if conf < min_confidence:
            missing_parts.append(f"{keypoint_names[idx]}")
            continue
        x, y = keypoints_xy[idx][0], keypoints_xy[idx][1]
        if x <= 0 or y <= 0:
            missing_parts.append(f"{keypoint_names[idx]}")

    return len(missing_parts) == 0, missing_parts


def is_standing_pose(angles: dict, threshold: float = 150.0) -> bool:
    """
    判斷是否為站立姿態
    站立特徵：雙膝角度接近 180° (腿伸直)，雙臀角度接近 180° (身體直立)
    """
    r_knee = angles.get('R_Knee_Angle', 0)
    l_knee = angles.get('L_Knee_Angle', 0)
    r_hip = angles.get('R_Hip_Angle', 0)
    l_hip = angles.get('L_Hip_Angle', 0)

    knees_straight = r_knee > threshold and l_knee > threshold
    hips_straight = r_hip > threshold and l_hip > threshold

    return knees_straight and hips_straight

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(filename="log.log", filemode="w+", level=logging.DEBUG)


# ============================================
# 樣式定義
# ============================================
STYLE_SHEET = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #eee;
    font-family: "Microsoft JhengHei", "Arial";
}

QPushButton {
    background-color: #4a4a6a;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #5a5a7a;
}

QPushButton:pressed {
    background-color: #3a3a5a;
}

QPushButton#startBtn {
    background-color: #4CAF50;
}

QPushButton#startBtn:hover {
    background-color: #5CBF60;
}

QPushButton#stopBtn {
    background-color: #f44336;
}

QPushButton#stopBtn:hover {
    background-color: #ff5349;
}

QPushButton#backBtn {
    background-color: #607D8B;
}

QPushButton#modeBtn {
    background-color: #3a3a5a;
    border: 2px solid #4a4a6a;
    padding: 20px 40px;
    font-size: 18px;
}

QPushButton#modeBtn:hover {
    background-color: #4a4a6a;
    border: 2px solid #6a6a8a;
}

QLabel#titleLabel {
    font-size: 28px;
    font-weight: bold;
    color: #fff;
    padding: 10px;
}

QLabel#videoLabel {
    background-color: #0f0f1a;
    border: 3px solid #3a3a5a;
    border-radius: 10px;
}

QLabel#similarityLabel {
    font-size: 48px;
    font-weight: bold;
    color: #4CAF50;
    padding: 20px;
}

QLabel#actionLabel {
    font-size: 24px;
    font-weight: bold;
    padding: 15px;
    border-radius: 10px;
}

QProgressBar {
    background-color: #2a2a4a;
    border: none;
    border-radius: 5px;
    height: 20px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #4CAF50;
    border-radius: 5px;
}

QFrame#infoPanel {
    background-color: #2a2a4a;
    border-radius: 10px;
    padding: 15px;
}

QFrame#anglePanel {
    background-color: #2a2a4a;
    border-radius: 8px;
    padding: 10px;
}
"""


# ============================================
# 自定義元件
# ============================================
class CircularProgress(QWidget):
    """環形進度條元件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.setMinimumSize(150, 150)
        self.setMaximumSize(150, 150)

    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景圓
        pen = QPen(QColor(60, 60, 90), 12)
        painter.setPen(pen)
        painter.drawArc(15, 15, 120, 120, 0, 360 * 16)

        # 進度圓
        if self.value >= 75:
            color = QColor(76, 175, 80)  # 綠色
        elif self.value >= 50:
            color = QColor(255, 193, 7)  # 黃色
        else:
            color = QColor(244, 67, 54)  # 紅色

        pen = QPen(color, 12)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        span = int(self.value * 360 / 100 * 16)
        painter.drawArc(15, 15, 120, 120, 90 * 16, -span)

        # 中心文字
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.value:.0f}%")


class AngleBar(QWidget):
    """角度進度條元件"""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.value = 0
        self.setMinimumHeight(30)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        self.name_label = QLabel(name)
        self.name_label.setFixedWidth(80)
        self.name_label.setStyleSheet("color: #aaa; font-size: 12px;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 180)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(15)

        self.value_label = QLabel("0°")
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setStyleSheet("color: #fff; font-size: 12px; font-weight: bold;")

        layout.addWidget(self.name_label)
        layout.addWidget(self.progress, 1)
        layout.addWidget(self.value_label)

    def setValue(self, value):
        self.value = value
        self.progress.setValue(int(value))
        self.value_label.setText(f"{value:.0f}°")

        # 根據角度設定顏色
        if value > 150:
            self.progress.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        elif value > 90:
            self.progress.setStyleSheet("QProgressBar::chunk { background-color: #FFC107; }")
        else:
            self.progress.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")


class VideoDisplay(QLabel):
    """影片顯示元件，支援骨架疊加"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setObjectName("videoLabel")
        self.setText(f"{title}\n\n等待載入...")

        # 骨架連接定義 (COCO 17 keypoints)
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 頭部
            (5, 6),  # 肩膀
            (5, 7), (7, 9),  # 左臂
            (6, 8), (8, 10),  # 右臂
            (5, 11), (6, 12),  # 軀幹
            (11, 12),  # 臀部
            (11, 13), (13, 15),  # 左腿
            (12, 14), (14, 16),  # 右腿
        ]

    def updateFrame(self, frame, keypoints=None, show_skeleton=True):
        """更新畫面，可選擇性疊加骨架"""
        if frame is None:
            return

        display_frame = frame.copy()

        # 繪製骨架
        if show_skeleton and keypoints is not None and len(keypoints) >= 17:
            display_frame = self._draw_skeleton(display_frame, keypoints)

        # 轉換為 QPixmap
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        scaled = qt_image.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(QPixmap.fromImage(scaled))

    def _draw_skeleton(self, frame, keypoints, confidence=None):
        """在影像上繪製骨架"""
        h, w = frame.shape[:2]

        # 繪製骨架線條
        for start_idx, end_idx in self.skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start = keypoints[start_idx]
                end = keypoints[end_idx]

                # 檢查座標是否有效
                if start[0] > 0 and start[1] > 0 and end[0] > 0 and end[1] > 0:
                    pt1 = (int(start[0]), int(start[1]))
                    pt2 = (int(end[0]), int(end[1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

        # 繪製關鍵點
        for i, kp in enumerate(keypoints):
            if kp[0] > 0 and kp[1] > 0:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, pt, 6, (255, 100, 100), -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 2)

        return frame


# ============================================
# 動作示範資料
# ============================================
ACTION_DATA = {
    'act1': {
        'name': '馬步站拳',
        'english': 'Horse Stance Fist',
        'image': 'display/act1_fist.jpg',
        'color': '#FF6B6B',
        'description': '雙腳張開與肩同寬，雙拳握緊置於腰間'
    },
    'act2': {
        'name': '弓步出拳',
        'english': 'Bow Stance Punch',
        'image': 'display/act2_punch.jpg',
        'color': '#4ECDC4',
        'description': '前腳弓步，後腳微彎，出拳向前'
    },
    'act3': {
        'name': '側踢腿',
        'english': 'Side Kick',
        'image': 'display/act3_kick.jpg',
        'color': '#45B7D1',
        'description': '單腳站立，另一腳向側邊踢出'
    },
    'act4': {
        'name': '提膝',
        'english': 'Knee Strike',
        'image': 'display/act4_knee.jpg',
        'color': '#96CEB4',
        'description': '單腳站立，另一腳提膝向上'
    }
}


# ============================================
# 動作卡片元件
# ============================================
class ActionCard(QFrame):
    """動作示範卡片"""
    clicked = None  # 將在 __init__ 中設置

    def __init__(self, action_code, action_info, parent=None):
        super().__init__(parent)
        self.action_code = action_code
        self.action_info = action_info
        self.callback = None

        self.setObjectName("actionCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(260, 340)

        color = action_info['color']
        self.setStyleSheet(f"""
            QFrame#actionCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a4a, stop:1 #1f1f35);
                border: 2px solid {color}88;
                border-radius: 18px;
            }}
            QFrame#actionCard:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a5a, stop:1 #2a2a4a);
                border: 2px solid {color};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # 圖片容器
        self.image_label = QLabel()
        self.image_label.setFixedSize(236, 180)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            background-color: #0f0f1a;
            border: 2px solid {color}66;
            border-radius: 12px;
        """)

        # 載入圖片
        image_path = Path(action_info['image'])
        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            scaled = pixmap.scaled(
                232, 176,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText("圖片載入中...")

        # 動作名稱
        name_label = QLabel(action_info['name'])
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(f"""
            font-size: 17px;
            font-weight: bold;
            color: {color};
            padding-top: 5px;
        """)

        # 英文名稱
        eng_label = QLabel(action_info['english'])
        eng_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        eng_label.setStyleSheet("font-size: 11px; color: #777;")

        # 開始練習按鈕
        btn_practice = QPushButton("開始練習")
        btn_practice.setMinimumHeight(38)
        btn_practice.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color}, stop:1 {color}cc);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color}ee, stop:1 {color});
            }}
        """)
        btn_practice.clicked.connect(self._on_click)

        layout.addWidget(self.image_label)
        layout.addWidget(name_label)
        layout.addWidget(eng_label)
        layout.addStretch()
        layout.addWidget(btn_practice)

    def set_callback(self, callback):
        self.callback = callback

    def _on_click(self):
        if self.callback:
            self.callback(self.action_code)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click()
        super().mousePressEvent(event)


# ============================================
# 主頁面
# ============================================
class MainPage(QWidget):
    def __init__(self, start_practice_callback):
        super().__init__()
        self.start_practice = start_practice_callback

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(40, 30, 40, 40)

        # ===== 標題區域 =====
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2a2a4a, stop:0.5 #3a3a5a, stop:1 #2a2a4a);
                border-radius: 20px;
                padding: 20px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(8)
        header_layout.setContentsMargins(30, 25, 30, 25)

        # 主標題
        title = QLabel("功夫訓練系統")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 42px;
            font-weight: bold;
            color: #ffffff;
            letter-spacing: 8px;
        """)

        # 英文標題
        title_eng = QLabel("KUNG FU TRAINING SYSTEM")
        title_eng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_eng.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4CAF50;
            letter-spacing: 4px;
        """)

        # 分隔線
        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet("background-color: #4a4a6a;")

        # 說明文字
        instruction = QLabel("選擇下方動作卡片開始練習")
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction.setStyleSheet("font-size: 16px; color: #aaa; margin-top: 10px;")

        header_layout.addWidget(title)
        header_layout.addWidget(title_eng)
        header_layout.addSpacing(10)
        header_layout.addWidget(divider)
        header_layout.addWidget(instruction)

        # ===== 卡片區域 =====
        cards_container = QFrame()
        cards_container.setStyleSheet("""
            QFrame {
                background-color: transparent;
            }
        """)
        cards_outer_layout = QHBoxLayout(cards_container)
        cards_outer_layout.setContentsMargins(0, 0, 0, 0)

        # 卡片水平佈局（置中）
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(25)

        self.action_cards = {}
        for action_code, action_info in ACTION_DATA.items():
            card = ActionCard(action_code, action_info)
            card.set_callback(self._on_action_selected)
            self.action_cards[action_code] = card
            cards_layout.addWidget(card)

        cards_outer_layout.addStretch()
        cards_outer_layout.addLayout(cards_layout)
        cards_outer_layout.addStretch()

        # ===== 底部提示 =====
        footer = QLabel("點擊卡片或「開始練習」按鈕進入練習模式")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("font-size: 13px; color: #666;")

        # 組合佈局
        layout.addWidget(header_frame)
        layout.addSpacing(35)
        layout.addWidget(cards_container, 1)
        layout.addSpacing(20)
        layout.addWidget(footer)

        self.setLayout(layout)

    def _on_action_selected(self, action_code):
        """當選擇動作時的處理"""
        if self.start_practice:
            self.start_practice(action_code)


# ============================================
# 動作練習頁面 (新增)
# ============================================
class ActionPracticePage(QWidget):
    """動作練習頁面 - 左側示範圖片，右側攝影機比對"""

    def __init__(self, back_callback: Callable):
        super().__init__()
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.kungfu_classifier = get_kungfu_classifier()

        self.current_action = None
        self.is_running = False
        self.camera_cap = None
        self.correct_count = 0
        self.total_attempts = 0

        self._setup_ui()

        # 偵測計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(20, 15, 20, 20)

        # ===== 標題列 =====
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a4a;
                border-radius: 12px;
            }
        """)
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(20, 12, 20, 12)

        self.title_label = QLabel("動作練習")
        self.title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
        """)

        btn_back = QPushButton("← 返回選單")
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #78909C;
            }
        """)
        btn_back.clicked.connect(self._on_back)

        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(btn_back)

        # ===== 主要內容區域 =====
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(0, 15, 0, 0)

        # ========== 左側：示範圖片 ==========
        left_frame = QFrame()
        left_frame.setStyleSheet("""
            QFrame {
                background-color: #1f1f35;
                border-radius: 15px;
            }
        """)
        left_panel = QVBoxLayout(left_frame)
        left_panel.setSpacing(12)
        left_panel.setContentsMargins(15, 15, 15, 15)

        left_title = QLabel("標準示範動作")
        left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            padding: 5px;
        """)

        # 示範圖片顯示區
        self.demo_image_label = QLabel()
        self.demo_image_label.setMinimumSize(420, 340)
        self.demo_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.demo_image_label.setStyleSheet("""
            background-color: #0f0f1a;
            border: 3px solid #4CAF50;
            border-radius: 12px;
        """)
        self.demo_image_label.setText("請選擇動作")

        # 動作說明
        self.action_desc_label = QLabel("選擇一個動作開始練習")
        self.action_desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_desc_label.setWordWrap(True)
        self.action_desc_label.setStyleSheet("""
            font-size: 13px;
            color: #aaa;
            padding: 12px;
            background-color: #2a2a4a;
            border-radius: 8px;
        """)

        left_panel.addWidget(left_title)
        left_panel.addWidget(self.demo_image_label, 1)
        left_panel.addWidget(self.action_desc_label)

        # ========== 中間：結果顯示 ==========
        middle_frame = QFrame()
        middle_frame.setStyleSheet("""
            QFrame {
                background-color: #1f1f35;
                border-radius: 15px;
            }
        """)
        middle_panel = QVBoxLayout(middle_frame)
        middle_panel.setSpacing(12)
        middle_panel.setContentsMargins(15, 20, 15, 20)

        # 相似度環形進度條
        self.similarity_circle = CircularProgress()
        similarity_container = QHBoxLayout()
        similarity_container.addStretch()
        similarity_container.addWidget(self.similarity_circle)
        similarity_container.addStretch()

        # 狀態提示
        self.status_label = QLabel("準備就緒")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #888;
            padding: 10px;
        """)

        # 辨識結果
        self.detected_action_label = QLabel("偵測動作: --")
        self.detected_action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detected_action_label.setStyleSheet("""
            font-size: 14px;
            color: #2196F3;
            background-color: #2196F322;
            border: 2px solid #2196F3;
            border-radius: 8px;
            padding: 10px;
        """)

        # 信心度
        self.confidence_label = QLabel("信心度: --")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 13px; color: #888;")

        # 計數器
        counter_frame = QFrame()
        counter_frame.setStyleSheet("""
            background-color: #2a2a4a;
            border-radius: 10px;
        """)
        counter_layout = QVBoxLayout(counter_frame)
        counter_layout.setContentsMargins(10, 12, 10, 12)
        counter_layout.setSpacing(5)

        self.correct_label = QLabel("正確次數: 0")
        self.correct_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.correct_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")

        self.accuracy_label = QLabel("準確率: --")
        self.accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accuracy_label.setStyleSheet("font-size: 14px; color: #FFC107;")

        counter_layout.addWidget(self.correct_label)
        counter_layout.addWidget(self.accuracy_label)

        # 角度面板
        angle_frame = QFrame()
        angle_frame.setStyleSheet("""
            background-color: #2a2a4a;
            border-radius: 10px;
        """)
        angle_layout = QVBoxLayout(angle_frame)
        angle_layout.setSpacing(3)
        angle_layout.setContentsMargins(10, 10, 10, 10)

        angle_title = QLabel("關節角度")
        angle_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #aaa;")
        angle_layout.addWidget(angle_title)

        self.angle_bars = {}
        for name in ["右肘", "左肘", "右膝", "左膝", "右臀", "左臀"]:
            bar = AngleBar(name)
            self.angle_bars[name] = bar
            angle_layout.addWidget(bar)

        middle_panel.addLayout(similarity_container)
        middle_panel.addWidget(self.status_label)
        middle_panel.addWidget(self.detected_action_label)
        middle_panel.addWidget(self.confidence_label)
        middle_panel.addSpacing(5)
        middle_panel.addWidget(counter_frame)
        middle_panel.addSpacing(5)
        middle_panel.addWidget(angle_frame)
        middle_panel.addStretch()

        # ========== 右側：即時攝影機 ==========
        right_frame = QFrame()
        right_frame.setStyleSheet("""
            QFrame {
                background-color: #1f1f35;
                border-radius: 15px;
            }
        """)
        right_panel = QVBoxLayout(right_frame)
        right_panel.setSpacing(12)
        right_panel.setContentsMargins(15, 15, 15, 15)

        right_title = QLabel("即時攝影機")
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FF9800;
            padding: 5px;
        """)

        self.camera_display = VideoDisplay("即時攝影機")
        self.camera_display.setMinimumSize(420, 340)

        # 控制按鈕
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_start = QPushButton("開啟攝影機")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5CBF60;
            }
        """)
        self.btn_start.clicked.connect(self._start_camera)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setMinimumHeight(45)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff5349;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.btn_stop.clicked.connect(self._stop_camera)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.setMinimumHeight(45)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #78909C;
            }
        """)
        self.btn_reset.clicked.connect(self._reset_counter)

        btn_layout.addWidget(self.btn_start, 2)
        btn_layout.addWidget(self.btn_stop, 1)
        btn_layout.addWidget(self.btn_reset, 1)

        right_panel.addWidget(right_title)
        right_panel.addWidget(self.camera_display, 1)
        right_panel.addLayout(btn_layout)

        # 組合佈局 - 調整比例
        content_layout.addWidget(left_frame, 5)
        content_layout.addWidget(middle_frame, 3)
        content_layout.addWidget(right_frame, 5)

        main_layout.addWidget(header_frame)
        main_layout.addLayout(content_layout, 1)

        self.setLayout(main_layout)

    def set_action(self, action_code):
        """設定要練習的動作"""
        if action_code not in ACTION_DATA:
            return

        self.current_action = action_code
        action_info = ACTION_DATA[action_code]

        # 更新標題
        self.title_label.setText(f"動作練習 - {action_info['name']}")

        # 載入示範圖片
        image_path = Path(action_info['image'])
        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            scaled = pixmap.scaled(
                480, 380,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.demo_image_label.setPixmap(scaled)
        else:
            self.demo_image_label.setText(f"無法載入圖片:\n{image_path}")

        # 更新說明
        self.action_desc_label.setText(f"{action_info['description']}")

        # 更新邊框顏色
        color = action_info['color']
        self.demo_image_label.setStyleSheet(f"""
            background-color: #0f0f1a;
            border: 3px solid {color};
            border-radius: 15px;
        """)

        # 重置計數器
        self._reset_counter()

    def _start_camera(self):
        """開啟攝影機"""
        if not self.current_action:
            QMessageBox.warning(self, "錯誤", "請先選擇動作！")
            return

        # 嘗試開啟攝影機
        self.camera_cap = cv2.VideoCapture(1)
        if not self.camera_cap.isOpened():
            self.camera_cap = cv2.VideoCapture(0)

        if not self.camera_cap.isOpened():
            QMessageBox.critical(self, "錯誤", "無法開啟攝影機！\n請確認攝影機已正確連接。")
            return

        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("偵測中...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3;")
        self.timer.start(50)  # 20 FPS

    def _stop_camera(self):
        """停止攝影機"""
        self.is_running = False
        self.timer.stop()

        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("已停止")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #888;")
        self.camera_display.setText("即時攝影機\n\n已停止")

    def _reset_counter(self):
        """重置計數器"""
        self.correct_count = 0
        self.total_attempts = 0
        self.correct_label.setText("正確次數: 0")
        self.accuracy_label.setText("準確率: --")

    def _process_frame(self):
        """處理每一幀 (與 test_camera.py 使用相同的偵測邏輯)"""
        if not self.is_running or not self.camera_cap:
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLOv11 姿態偵測
        results = self.posture_detector.model.predict(frame_rgb, verbose=False)

        # 偵測參數 (與 test_camera.py 一致)
        CONFIDENCE_THRESHOLD = 0.85  # 信心度門檻
        STANDING_THRESHOLD = 150.0   # 站立角度閾值
        BODY_CONF_THRESHOLD = 0.5    # 身體偵測置信度

        keypoints = None
        detected_action = None
        confidence = 0
        similarity = 0
        detection_status = "no_detection"  # 偵測狀態

        if results and len(results[0]) > 0:
            keypoints_xy = results[0][0].keypoints.xy[0].cpu().numpy()
            keypoints_conf = results[0][0].keypoints.conf[0].cpu().numpy()
            keypoints = keypoints_xy

            # 步驟1: 檢查是否偵測到完整身體
            is_full_body, missing_parts = is_full_body_detected(
                keypoints_xy, keypoints_conf, min_confidence=BODY_CONF_THRESHOLD
            )

            if not is_full_body:
                # 身體不完整
                detection_status = "incomplete_body"
                self.detected_action_label.setText("等待完整身體...")
                self.detected_action_label.setStyleSheet("""
                    font-size: 14px;
                    color: #888;
                    background-color: #2a2a4a;
                    border: 2px solid #4a4a6a;
                    border-radius: 8px;
                    padding: 10px;
                """)
                if missing_parts:
                    self.confidence_label.setText(f"缺少: {', '.join(missing_parts[:3])}")
                else:
                    self.confidence_label.setText("")
                self.confidence_label.setStyleSheet("font-size: 13px; color: #888;")

            elif is_valid_keypoints(keypoints_xy):
                try:
                    # 步驟2: 提取角度
                    angles = extract_angles_from_keypoints(keypoints_xy)
                    self._update_angle_bars(angles)

                    # 步驟3: 判斷是否為站立姿態
                    if is_standing_pose(angles, STANDING_THRESHOLD):
                        detection_status = "standing"
                        self.detected_action_label.setText("站立中")
                        self.detected_action_label.setStyleSheet("""
                            font-size: 14px;
                            color: #aaa;
                            background-color: #2a2a4a;
                            border: 2px solid #4a4a6a;
                            border-radius: 8px;
                            padding: 10px;
                        """)
                        self.confidence_label.setText("請做出動作姿勢")
                        self.confidence_label.setStyleSheet("font-size: 13px; color: #888;")
                    else:
                        # 步驟4: 進行動作分類
                        action_code, action_name, probs = self.kungfu_classifier.predict(angles)
                        confidence = self.kungfu_classifier.get_confidence(probs)

                        # 步驟5: 信心度門檻過濾
                        if confidence >= CONFIDENCE_THRESHOLD:
                            detection_status = "action_detected"
                            detected_action = action_code

                            # 更新辨識結果
                            self.detected_action_label.setText(f"偵測動作: {action_name}")
                            action_info = ACTION_DATA.get(action_code, {})
                            color = action_info.get('color', '#2196F3')
                            self.detected_action_label.setStyleSheet(f"""
                                font-size: 14px;
                                color: white;
                                background-color: {color};
                                border: 2px solid {color};
                                border-radius: 8px;
                                padding: 10px;
                            """)

                            self.confidence_label.setText(f"信心度: {confidence:.1%}")
                            self.confidence_label.setStyleSheet("font-size: 13px; color: #4CAF50;")

                            # 計算與目標動作的匹配度
                            if detected_action == self.current_action:
                                similarity = confidence * 100
                            else:
                                similarity = (1 - confidence) * 30
                        else:
                            # 信心度不足，視為準備中
                            detection_status = "preparing"
                            self.detected_action_label.setText(f"準備中... ({action_name})")
                            self.detected_action_label.setStyleSheet("""
                                font-size: 14px;
                                color: #FFC107;
                                background-color: #FFC10722;
                                border: 2px solid #FFC107;
                                border-radius: 8px;
                                padding: 10px;
                            """)
                            self.confidence_label.setText(f"信心度: {confidence:.1%} (需 ≥ {CONFIDENCE_THRESHOLD:.0%})")
                            self.confidence_label.setStyleSheet("font-size: 13px; color: #FFC107;")

                except Exception:
                    pass

        # 更新相機畫面
        self.camera_display.updateFrame(frame_rgb, keypoints)

        # 更新相似度
        self.similarity_circle.setValue(similarity)

        # 更新狀態提示
        if detection_status == "action_detected" and detected_action == self.current_action and confidence >= CONFIDENCE_THRESHOLD:
            self.status_label.setText("動作正確！")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")

            # 計數（避免重複計數，每隔一段時間才計一次）
            if not hasattr(self, '_last_correct_time'):
                self._last_correct_time = 0

            import time
            current_time = time.time()
            if current_time - self._last_correct_time > 1.5:  # 1.5秒間隔
                self.correct_count += 1
                self.total_attempts += 1
                self._last_correct_time = current_time
                self._update_counter()

        elif detection_status == "action_detected" and detected_action != self.current_action:
            # 偵測到其他動作
            target_name = ACTION_DATA[self.current_action]['name']
            self.status_label.setText(f"請做 {target_name}")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF9800;")
        elif detection_status == "preparing":
            target_name = ACTION_DATA[self.current_action]['name']
            self.status_label.setText(f"調整姿勢中...")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFC107;")
        elif detection_status == "standing":
            target_name = ACTION_DATA[self.current_action]['name']
            self.status_label.setText(f"請做 {target_name} 動作")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")
        elif detection_status == "incomplete_body":
            self.status_label.setText("請站到攝影機前方")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")
        else:
            self.status_label.setText("偵測中...")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")

    def _update_angle_bars(self, angles):
        """更新角度顯示"""
        mapping = {
            "右肘": "R_Elbow_Angle",
            "左肘": "L_Elbow_Angle",
            "右膝": "R_Knee_Angle",
            "左膝": "L_Knee_Angle",
            "右臀": "R_Hip_Angle",
            "左臀": "L_Hip_Angle",
        }

        for name, key in mapping.items():
            if key in angles:
                self.angle_bars[name].setValue(angles[key])

    def _update_counter(self):
        """更新計數器顯示"""
        self.correct_label.setText(f"正確次數: {self.correct_count}")
        if self.total_attempts > 0:
            accuracy = self.correct_count / self.total_attempts * 100
            self.accuracy_label.setText(f"準確率: {accuracy:.1f}%")

    def _on_back(self):
        """返回主選單"""
        self._stop_camera()
        self.back_callback()


# ============================================
# 主視窗
# ============================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("功夫訓練系統 - KungFu Training System")
        self.setGeometry(50, 50, 1400, 900)

        self.stack = QStackedWidget()

        # 建立頁面
        self.main_page = MainPage(self._start_action_practice)
        self.practice_page = ActionPracticePage(lambda: self.stack.setCurrentIndex(0))

        self.stack.addWidget(self.main_page)      # index 0
        self.stack.addWidget(self.practice_page)   # index 1

        self.setCentralWidget(self.stack)

    def _start_action_practice(self, action_code):
        """開始動作練習"""
        self.practice_page.set_action(action_code)
        self.stack.setCurrentIndex(1)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
