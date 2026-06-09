"""
OpenCV 画布 v3
改动：
  1. 松开缓冲期由 gesture_detector 处理，canvas 直接用 drawing_active
  2. 擦除改为五指张开（与批注模式统一）
  3. 去掉 _draw_queue 延迟绘制（缓冲期已解决拖尾）
  4. 画线直接绘制，无延迟
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import os


class Canvas:
    """OpenCV 画布 — 白板模式"""

    COLORS = [
        (0, 0, 255),      # 红
        (255, 100, 0),    # 蓝
        (0, 200, 0),      # 绿
        (0, 165, 255),    # 橙
        (200, 0, 200),    # 紫
        (0, 0, 0),        # 黑
    ]
    COLOR_NAMES = ["红", "蓝", "绿", "橙", "紫", "黑"]
    BG_COLOR = (255, 255, 255)

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.image = np.full((height, width, 3), self.BG_COLOR, dtype=np.uint8)

        self.pen_color = self.COLORS[0]
        self.pen_thickness = 4
        self.eraser_thickness = 40
        self.is_eraser = False

        self.is_drawing = False
        self.prev_pos: Optional[Tuple[int, int]] = None
        self.laser_pos: Optional[Tuple[float, float]] = None

        # 五指张开橡皮擦状态
        self.five_finger_eraser = False

        self.history: List[np.ndarray] = []
        self.max_history = 20
        self._color_index = 0

    # ===================== 核心更新 =====================

    def update(self, gesture_data: dict):
        """
        每帧调用。drawing_active 来自 gesture_detector 的松开缓冲期逻辑。
        """
        index_pos = gesture_data.get('index_position')
        drawing = gesture_data.get('drawing_active', False)
        erase = gesture_data.get('erase_gesture', False)

        # 没检测到手
        if index_pos is None:
            self._stop_stroke()
            self.laser_pos = None
            self.five_finger_eraser = False
            return

        # 屏幕坐标
        x = int(index_pos[0] * self.width)
        y = int(index_pos[1] * self.height)
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        self.laser_pos = (index_pos[0], index_pos[1])
        self.five_finger_eraser = erase

        # 五指张开 → 擦除
        if erase:
            self._erase((x, y))
            return

        # 捏合（含缓冲期）→ 画画
        if drawing:
            if not self.is_drawing:
                self._start_stroke()
            self._draw((x, y))
        else:
            self._stop_stroke()

    # ===================== 绘画控制 =====================

    def _start_stroke(self):
        self.is_drawing = True
        self.prev_pos = None
        self.history.append(self.image.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _draw(self, pos: Tuple[int, int]):
        if self.prev_pos is not None:
            if self.is_eraser:
                cv2.line(self.image, self.prev_pos, pos, self.BG_COLOR, self.eraser_thickness)
            else:
                cv2.line(self.image, self.prev_pos, pos, self.pen_color, self.pen_thickness)
        self.prev_pos = pos

    def _stop_stroke(self):
        self.is_drawing = False
        self.prev_pos = None

    def _erase(self, pos: Tuple[int, int]):
        if not self.is_drawing:
            self.history.append(self.image.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
        cv2.circle(self.image, pos, self.eraser_thickness, self.BG_COLOR, -1)
        self.is_drawing = True
        self.prev_pos = None

    # ===================== 功能 =====================

    def toggle_eraser(self):
        self.is_eraser = not self.is_eraser
        print(f"[Canvas] {'橡皮擦' if self.is_eraser else '画笔'}")

    def next_color(self):
        self._color_index = (self._color_index + 1) % len(self.COLORS)
        self.pen_color = self.COLORS[self._color_index]
        self.is_eraser = False
        print(f"[Canvas] 颜色: {self.COLOR_NAMES[self._color_index]}")

    def set_thickness(self, thickness: int):
        self.pen_thickness = max(1, min(20, thickness))

    def undo(self):
        if self.history:
            self.image = self.history.pop()
            print(f"[Canvas] 撤销 (剩余 {len(self.history)} 步)")
        else:
            print("[Canvas] 无可撤销")

    def clear(self):
        self._stop_stroke()
        self.history.append(self.image.copy())
        self.image[:] = self.BG_COLOR
        print("[Canvas] 已清空")

    def new_canvas(self):
        self._stop_stroke()
        self.history.clear()
        self.image[:] = self.BG_COLOR
        print("[Canvas] 新建画布")

    # ===================== 文字 =====================

    _font = None
    _font_small = None

    @classmethod
    def _get_font(cls, size: int):
        if cls._font is None:
            for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
                if os.path.exists(fp):
                    try:
                        cls._font = ImageFont.truetype(fp, 20)
                        cls._font_small = ImageFont.truetype(fp, 16)
                        break
                    except Exception:
                        continue
            if cls._font is None:
                cls._font = ImageFont.load_default()
                cls._font_small = cls._font
        return cls._font if size >= 20 else cls._font_small

    @staticmethod
    def _put_text_cn(image, text, pos, color=(255, 255, 255), font_size=16):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = Canvas._get_font(font_size)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR, dst=image)

    # ===================== 渲染 =====================

    def render(self, camera_frame: np.ndarray = None) -> np.ndarray:
        display = self.image.copy()

        if self.laser_pos is not None:
            cx = int(self.laser_pos[0] * self.width)
            cy = int(self.laser_pos[1] * self.height)

            if self.five_finger_eraser:
                cv2.circle(display, (cx, cy), self.eraser_thickness, (200, 200, 200), 2)
                cv2.putText(display, 'ERASER', (cx - 25, cy - self.eraser_thickness - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            elif self.is_drawing:
                if self.is_eraser:
                    cv2.circle(display, (cx, cy), self.eraser_thickness, (200, 200, 200), 2)
                else:
                    cv2.circle(display, (cx, cy), self.pen_thickness + 2, self.pen_color, -1)
                    cv2.circle(display, (cx, cy), self.pen_thickness + 4, (255, 255, 255), 1)
            else:
                cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(display, (cx, cy), 12, (0, 0, 200), 2)

        # 摄像头小窗
        if camera_frame is not None:
            thumb_w, thumb_h = 240, 180
            thumb = cv2.resize(camera_frame, (thumb_w, thumb_h))
            roi = display[0:thumb_h, 0:thumb_w]
            cv2.addWeighted(thumb, 0.7, roi, 0.3, 0, roi)
            display[0:thumb_h, 0:thumb_w] = roi
            cv2.rectangle(display, (0, 0), (thumb_w, thumb_h), (0, 255, 0), 2)

        self._draw_status_bar(display)
        return display

    def _draw_status_bar(self, image):
        h, w = image.shape[:2]
        bar_h = 40

        overlay = image.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        if self.five_finger_eraser:
            mode = "黑板擦"
        elif self.is_eraser:
            mode = "橡皮"
        else:
            mode = "画笔"

        color_name = self.COLOR_NAMES[self._color_index]
        drawing = "擦除中" if self.five_finger_eraser else ("画画中" if self.is_drawing else "待机")

        text = f"{mode} | {color_name} | 粗细:{self.pen_thickness} | {drawing} | E:橡皮 N:新建 C:清空 Z:撤销"
        self._put_text_cn(image, text, (10, h - 28), (255, 255, 255), font_size=16)
