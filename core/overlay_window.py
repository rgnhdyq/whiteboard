"""
PyQt5 透明全屏覆盖窗口
直接在屏幕上绘制，无需截图
"""

import sys
import time
import queue
import numpy as np
from typing import Optional, Tuple, List
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QFont, QBrush, QPainterPath

from core.floating_menu import FloatingMenu, COLORS as MENU_COLORS, THICKNESSES as MENU_THICKNESSES


class OverlayWindow(QWidget):
    """透明全屏覆盖窗口 — 手势画布"""

    # 画笔颜色（与 FloatingMenu 同源）
    COLORS = [c for c, _ in MENU_COLORS]
    COLOR_NAMES = [n for _, n in MENU_COLORS]

    def __init__(self, gesture_queue: queue.Queue, engine=None):
        super().__init__()
        self.gesture_queue = gesture_queue
        self.engine = engine

        # 窗口设置：无边框 + 置顶 + 透明 + 鼠标穿透（允许鼠标操作电脑）
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.WindowTransparentForInput |
            Qt.Tool  # 不显示在任务栏
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 全屏
        screen = QApplication.primaryScreen().geometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.setGeometry(screen)

        # 绘画状态
        self.strokes: List[List[QPoint]] = []  # 所有笔画
        self.current_stroke: List[QPoint] = []  # 当前笔画
        self.is_drawing = False
        self.is_erasing = False  # 擦除状态（独立追踪）

        # 画笔设置
        self.pen_color = self.COLORS[0]
        self.pen_thickness = 4
        self.eraser_thickness = 60
        self.is_eraser = False
        self._color_index = 0

        # 擦除位置（保留最后位置，手短暂丢失不清除）
        self.erase_pos: Optional[QPoint] = None

        # 激光笔
        self.laser_pos: Optional[QPoint] = None

        # 撤销历史（保存笔画快照）
        self.history: List[List[List[QPoint]]] = []
        self.max_history = 20

        # 摄像头预览
        self.camera_image: Optional[QImage] = None

        # 模式
        self.mode = "whiteboard"  # "whiteboard" / "annotation"

        # 五指擦除状态
        self.five_finger_eraser = False
        self._last_erase_state = False

        # 捏合确认进度（0~1，用于视觉反馈）
        self._pinch_progress = 0.0

        # 状态文字
        self.status_text = "就绪"
        self.fps_text = "FPS: 0"

        # 悬浮菜单（嵌入式，画在 OverlayWindow 内部）
        self.floating_menu = FloatingMenu(screen_w=self.screen_w, screen_h=self.screen_h)
        self._finger_pos = QPoint(0, 0)  # 当前食指屏幕坐标

        # 定时器：轮询手势队列
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_gesture)
        self.poll_timer.start(16)  # ~60fps

        # 定时器：更新摄像头预览
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._update_camera)
        self.cam_timer.start(100)  # ~10fps (降低刷新率减少线程开销)

        # 键盘快捷键
        self.grabKeyboard()

        print(f"[Overlay] 窗口创建: {self.screen_w}x{self.screen_h}")

    # ===================== 手势轮询 =====================

    def _poll_gesture(self):
        """从队列读取手势数据并处理"""
        try:
            gesture_data = self.gesture_queue.get_nowait()
        except queue.Empty:
            return

        self._process_gesture(gesture_data)
        
        # 智能按需刷新：若有手在画面中，或者菜单在缩回/展开/悬停等动画中，才重绘全屏，大幅节省 CPU
        need_update = False
        if self.laser_pos is not None:
            need_update = True
        elif self.floating_menu.is_animating():
            need_update = True

        if need_update:
            self.update()

    def _process_gesture(self, data: dict):
        """处理手势数据"""
        index_pos = data.get('index_position')
        drawing = data.get('drawing_active', False)
        pinch_progress = data.get('pinch_progress', 0.0)
        erase = data.get('erase_gesture', False)
        clap = data.get('clap_detected', False)
        mode_switch = data.get('mode_switch_gesture', False)
        num_hands = data.get('num_hands', 0)

        self._pinch_progress = pinch_progress

        # 仅在擦除状态转换时打印一次
        if erase and not self._last_erase_state:
            print("[手势] 进入五指擦除模式")
        elif not erase and self._last_erase_state:
            print("[手势] 退出五指擦除模式")
        self._last_erase_state = erase

        if clap:
            print("[手势] 双食指碰撞两次 → 切换模式")

        # 没检测到手
        if index_pos is None or index_pos == (0, 0, 0):
            self._stop_stroke()
            self.laser_pos = None
            self.five_finger_eraser = False
            self.status_text = "无手"
            return

        # 屏幕坐标
        x = int(index_pos[0] * self.screen_w)
        y = int(index_pos[1] * self.screen_h)
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))

        self.laser_pos = QPoint(x, y)
        self._finger_pos = QPoint(x, y)
        self.five_finger_eraser = erase

        # 悬浮菜单 hit_test（菜单区域内不执行画画/擦除）
        if self.floating_menu.in_menu_area(x):
            self.floating_menu.hit_test(
                x, y,
                on_color=self._set_color,
                on_thickness=self._set_thickness,
                on_action=self._on_menu_action,
            )
            self._stop_stroke()
            self.status_text = "菜单"
            return
        else:
            self.floating_menu.hit_test(x, y)

        # 五指张开 → 擦除
        if erase:
            self._erase(QPoint(x, y))
            self.status_text = "擦除中"
            return
        else:
            self.is_erasing = False

        # 捏合（含缓冲期）→ 画画
        if drawing:
            if not self.is_drawing:
                self._start_stroke()
            self._draw(QPoint(x, y))
            self.status_text = "画画中" if not self.is_eraser else "橡皮中"
        else:
            self._stop_stroke()
            self.status_text = "激光笔"

        # 拍手 → 切换模式
        if clap:
            self._toggle_mode()

        if 'fps' in data:
            self.fps_text = f"FPS: {data['fps']:.0f}"

    # ===================== 绘画控制 =====================

    def _start_stroke(self):
        self.is_drawing = True
        self.current_stroke = []
        self.history.append([{'points': s['points'][:], 'color': s['color'], 'thickness': s['thickness'], 'is_eraser': s['is_eraser']} for s in self.strokes])
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _draw(self, pos: QPoint):
        self.current_stroke.append(pos)

    def _stop_stroke(self):
        if self.is_drawing and self.current_stroke:
            stroke_obj = {
                'points': self.current_stroke,
                'color': self.pen_color,
                'thickness': self.pen_thickness,
                'is_eraser': self.is_eraser
            }
            self.strokes.append(stroke_obj)
        self.is_drawing = False
        self.current_stroke = []

    def _erase(self, pos: QPoint):
        """局部范围擦除：仅删除落入擦除圆圈内的点，将笔画从交汇处分裂为多段短线"""
        if not self.is_erasing:
            self.is_erasing = True
            self.history.append([{'points': s['points'][:], 'color': s['color'], 'thickness': s['thickness'], 'is_eraser': s['is_eraser']} for s in self.strokes])
            if len(self.history) > self.max_history:
                self.history.pop(0)

        r = self.eraser_thickness
        r_sq = r * r
        new_strokes = []
        
        for stroke in self.strokes:
            points = stroke['points']
            if not points:
                continue

            current_sub_points = []
            for p in points:
                dx = p.x() - pos.x()
                dy = p.y() - pos.y()
                if dx * dx + dy * dy >= r_sq:
                    # 在圆圈外，加入当前的子笔画集
                    current_sub_points.append(p)
                else:
                    # 落入擦除圆圈，切断笔画
                    if len(current_sub_points) >= 2:
                        new_strokes.append({
                            'points': current_sub_points,
                            'color': stroke['color'],
                            'thickness': stroke['thickness'],
                            'is_eraser': stroke['is_eraser']
                        })
                    current_sub_points = []
            
            # 保存该笔画最后剩余的子笔迹
            if len(current_sub_points) >= 2:
                new_strokes.append({
                    'points': current_sub_points,
                    'color': stroke['color'],
                    'thickness': stroke['thickness'],
                    'is_eraser': stroke['is_eraser']
                })

        self.strokes = new_strokes
        self.erase_pos = pos

    def _next_color(self):
        self._color_index = (self._color_index + 1) % len(self.COLORS)
        self.pen_color = self.COLORS[self._color_index]
        self.is_eraser = False
        self.floating_menu.set_selected_color(self._color_index)
        print(f"[Overlay] 颜色: {self.COLOR_NAMES[self._color_index]}")

    def _set_color(self, index: int):
        """FloatingMenu 信号：设置指定颜色"""
        if 0 <= index < len(self.COLORS):
            self._color_index = index
            self.pen_color = self.COLORS[index]
            self.is_eraser = False
            self.floating_menu.set_selected_color(index)
            print(f"[Menu] 颜色: {self.COLOR_NAMES[index]}")

    def _set_thickness(self, t: int):
        """FloatingMenu 信号：设置画笔粗细"""
        self.pen_thickness = t
        idx = MENU_THICKNESSES.index(t) if t in MENU_THICKNESSES else 1
        self.floating_menu.set_selected_thickness(idx)
        print(f"[Menu] 粗细: {t}px")

    def _on_menu_action(self, action: str):
        """FloatingMenu 信号：操作按钮"""
        if action == "clear":
            self._clear()
        elif action == "undo":
            self._undo()
        elif action == "mode":
            self._toggle_mode()
        elif action == "exit":
            print("[Overlay] 收到退出信号，程序正在关闭...")
            QApplication.quit()

    def _clear(self):
        self._stop_stroke()
        self.history.append([{'points': s['points'][:], 'color': s['color'], 'thickness': s['thickness'], 'is_eraser': s['is_eraser']} for s in self.strokes])
        self.strokes.clear()
        print("[Overlay] 已清空")

    def _undo(self):
        if self.history:
            self.strokes = self.history.pop()
            print(f"[Overlay] 撤销 (剩余 {len(self.history)} 步)")
        else:
            print("[Overlay] 无可撤销")

    def _toggle_mode(self):
        if self.mode == "whiteboard":
            self.mode = "annotation"
            print("[Overlay] → 屏幕批注模式")
        else:
            self.mode = "whiteboard"
            print("[Overlay] → 白板模式")

    # ===================== 摄像头预览 =====================

    def _update_camera(self):
        """更新摄像头预览"""
        if self.engine is None:
            return
        frame = self.engine.get_frame()  # 已经是由后台线程转换好的 RGB 数据
        if frame is None:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # 创建并深度复制 QImage，避免在 GUI 线程中调用 cv2.cvtColor 和创建 QPixmap
        self.camera_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.update()

    # ===================== 键盘 =====================

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Q:
            print("[Overlay] 退出")
            QApplication.quit()
        elif key == Qt.Key_E:
            self.is_eraser = not self.is_eraser
            print(f"[Overlay] {'橡皮擦' if self.is_eraser else '画笔'}")
        elif key == Qt.Key_N:
            self._clear()
        elif key == Qt.Key_C:
            self._clear()
        elif key == Qt.Key_Z:
            self._undo()
        elif key == Qt.Key_A:
            self._toggle_mode()
        elif key == Qt.Key_D:
            pass

        self.update()

    # ===================== 绘制 =====================

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # ===== 半透明背景（白板模式时）=====
        if self.mode == "whiteboard":
            painter.fillRect(self.rect(), QColor(255, 255, 255, 230))
        else:
            # 批注模式下使用极其微弱的透明填充 (Alpha = 1)，强制 Windows DWM 启用 GPU 硬件加速，解决卡顿
            painter.fillRect(self.rect(), QColor(0, 0, 0, 1))

        # ===== 绘制所有已完成的笔画 =====
        for stroke in self.strokes:
            self._draw_stroke(painter, stroke)

        # ===== 绘制当前笔画 =====
        if self.current_stroke:
            current_stroke_obj = {
                'points': self.current_stroke,
                'color': self.pen_color,
                'thickness': self.pen_thickness,
                'is_eraser': self.is_eraser
            }
            self._draw_stroke(painter, current_stroke_obj)

        # ===== 擦除圆圈 =====
        if self.erase_pos and self.five_finger_eraser:
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.drawEllipse(self.erase_pos, self.eraser_thickness, self.eraser_thickness)

        # ===== 摄像头预览（左上角 200x150）=====
        if hasattr(self, 'camera_image') and self.camera_image:
            thumb_w, thumb_h = 200, 150
            x0 = 10
            y0 = 40
            painter.fillRect(QRect(x0 - 2, y0 - 2, thumb_w + 4, thumb_h + 4),
                           QColor(0, 0, 0, 100))
            painter.drawImage(QRect(x0, y0, thumb_w, thumb_h), self.camera_image)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x0, y0, thumb_w, thumb_h)

        # ===== 底部状态栏 =====
        self._draw_status_bar(painter)

        # ===== 顶部模式标签 =====
        painter.setPen(QColor(0, 255, 0))
        painter.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        mode_text = "白板模式" if self.mode == "whiteboard" else "屏幕批注"
        painter.drawText(10, 30, f"[{mode_text}]")

        # ===== 悬浮菜单 =====
        laser_x = self.laser_pos.x() if self.laser_pos else None
        self.floating_menu.tick(laser_x)
        self.floating_menu.paint(painter)

        # ===== 激光笔（移至菜单上方绘制，解决遮挡问题）=====
        if self.laser_pos and not self.is_drawing and not self.five_finger_eraser:
            # 绘制代表画笔颜色和粗细的实心圆圈光标
            radius = max(4, self.pen_thickness // 2)
            painter.setPen(QPen(QColor(128, 128, 128, 180), 1))
            painter.setBrush(QBrush(self.pen_color))
            painter.drawEllipse(self.laser_pos, radius, radius)

            # 捏合确认进度环（如果正在开始画画捏合）
            if 0.0 < self._pinch_progress < 1.0:
                pen = QPen(QColor(255, 200, 0), 3)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                r = radius + 8
                rect = QRect(self.laser_pos.x() - r, self.laser_pos.y() - r, r * 2, r * 2)
                span_angle = int(self._pinch_progress * 360 * 16)
                painter.drawArc(rect, 90 * 16, span_angle)

        painter.end()

    def _draw_stroke(self, painter: QPainter, stroke: dict):
        """绘制一条笔画"""
        points = stroke['points']
        if len(points) < 2:
            return

        if stroke['is_eraser']:
            pen = QPen(QColor(255, 255, 255, 0), stroke['thickness'])
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
        else:
            pen = QPen(stroke['color'], stroke['thickness'])
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

        for i in range(1, len(points)):
            painter.drawLine(points[i - 1], points[i])

    def _draw_status_bar(self, painter: QPainter):
        """底部状态栏"""
        bar_h = 40
        y0 = self.screen_h - bar_h

        painter.fillRect(QRect(0, y0, self.screen_w, bar_h), QColor(30, 30, 30, 180))

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Microsoft YaHei", 11))

        color_name = self.COLOR_NAMES[self._color_index]
        mode = "橡皮" if self.is_eraser else "画笔"
        drawing = "擦除中" if self.five_finger_eraser else ("画画中" if self.is_drawing else "待机")
        bg = "白板" if self.mode == "whiteboard" else "批注"

        text = f"{bg} | {mode} | {color_name} | 粗细:{self.pen_thickness} | {drawing} | {self.fps_text} | E:橡皮 N:清空 Z:撤销 Q:退出"
        painter.drawText(10, y0 + 26, text)


# 需要 import cv2（用于摄像头帧转换）
import cv2


def create_overlay_window(gesture_queue: queue.Queue, engine=None) -> OverlayWindow:
    """创建覆盖窗口"""
    window = OverlayWindow(gesture_queue, engine)
    window.showFullScreen()
    return window
