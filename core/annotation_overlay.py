"""
屏幕批注覆盖层 v3
改动：
  1. 背景改为实时屏幕截图（不再用摄像头）
  2. 使用 mss 高效截屏
  3. 五指张开 = 橡皮擦（与白板统一）
  4. S 键重新截屏，F 键解冻
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import os

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class AnnotationOverlay:
    """屏幕批注覆盖层 — 屏幕截图作背景，手势在上面画"""

    COLORS = [
        (0, 0, 255),      # 红
        (255, 100, 0),    # 蓝
        (0, 200, 0),      # 绿
        (0, 165, 255),    # 橙
        (200, 0, 200),    # 紫
        (255, 255, 255),  # 白
    ]
    COLOR_NAMES = ["红", "蓝", "绿", "橙", "紫", "白"]

    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 批注画布（BGRA 透明）
        self.overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

        self.pen_color = self.COLORS[0]
        self.pen_color_bgra = (0, 0, 255, 255)
        self.pen_thickness = 4
        self.eraser_thickness = 60
        self.is_eraser = False

        self.is_drawing = False
        self.prev_pos: Optional[Tuple[int, int]] = None
        self.laser_pos: Optional[Tuple[float, float]] = None

        self.five_finger_eraser = False

        self.history: List[np.ndarray] = []
        self.max_history = 20
        self._color_index = 0

        # 屏幕截图背景
        self.screenshot_bg: Optional[np.ndarray] = None
        self.frozen = False

        # mss 截屏器
        self._sct = mss.mss() if HAS_MSS else None

        # 摄像头小窗（显示手部）
        self._camera_frame: Optional[np.ndarray] = None

    # ===================== 屏幕截图 =====================

    def capture_screen(self):
        """截取当前屏幕作为背景"""
        if HAS_MSS and self._sct:
            try:
                monitor = self._sct.monitors[0]  # 整个屏幕
                screenshot = self._sct.grab(monitor)
                # BGRA → BGR
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.screenshot_bg = cv2.resize(img, (self.screen_width, self.screen_height))
                self.frozen = True
                self.overlay[:] = (0, 0, 0, 0)
                self.history.clear()
                print("[Annotation] 已截取屏幕")
                return
            except Exception as e:
                print(f"[Annotation] mss 截屏失败: {e}")

        # 备用方案：PIL
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.screenshot_bg = cv2.resize(img, (self.screen_width, self.screen_height))
            self.frozen = True
            self.overlay[:] = (0, 0, 0, 0)
            self.history.clear()
            print("[Annotation] 已截取屏幕（PIL）")
        except Exception as e:
            print(f"[Annotation] 截屏失败: {e}")

    def take_screenshot(self):
        """S 键触发：重新截屏"""
        self.capture_screen()

    def unfreeze(self):
        """F 键：解冻，恢复实时屏幕"""
        self.frozen = False
        self.screenshot_bg = None
        print("[Annotation] 已解冻（实时屏幕）")

    # ===================== 核心更新 =====================

    def update(self, gesture_data: dict):
        index_pos = gesture_data.get('index_position')
        drawing = gesture_data.get('drawing_active', False)
        erase = gesture_data.get('erase_gesture', False)

        if index_pos is None:
            self._stop_stroke()
            self.laser_pos = None
            self.five_finger_eraser = False
            return

        x = int(index_pos[0] * self.screen_width)
        y = int(index_pos[1] * self.screen_height)
        x = max(0, min(self.screen_width - 1, x))
        y = max(0, min(self.screen_height - 1, y))

        self.laser_pos = (index_pos[0], index_pos[1])
        self.five_finger_eraser = erase

        if erase:
            self._erase((x, y))
            return

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
        self.history.append(self.overlay.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _draw(self, pos: Tuple[int, int]):
        if self.prev_pos is not None:
            if self.is_eraser:
                cv2.line(self.overlay, self.prev_pos, pos, (0, 0, 0, 0), self.eraser_thickness)
            else:
                cv2.line(self.overlay, self.prev_pos, pos, self.pen_color_bgra, self.pen_thickness)
        self.prev_pos = pos

    def _stop_stroke(self):
        self.is_drawing = False
        self.prev_pos = None

    def _erase(self, pos: Tuple[int, int]):
        if not self.is_drawing:
            self.history.append(self.overlay.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
        cv2.circle(self.overlay, pos, self.eraser_thickness, (0, 0, 0, 0), -1)
        self.is_drawing = True
        self.prev_pos = None

    # ===================== 功能 =====================

    def toggle_eraser(self):
        self.is_eraser = not self.is_eraser
        print(f"[Annotation] {'橡皮擦' if self.is_eraser else '批注笔'}")

    def next_color(self):
        self._color_index = (self._color_index + 1) % len(self.COLORS)
        c = self.COLORS[self._color_index]
        self.pen_color = c
        self.pen_color_bgra = (c[0], c[1], c[2], 255)
        self.is_eraser = False
        print(f"[Annotation] 颜色: {self.COLOR_NAMES[self._color_index]}")

    def set_thickness(self, thickness: int):
        self.pen_thickness = max(1, min(20, thickness))

    def undo(self):
        if self.history:
            self.overlay = self.history.pop()
            print(f"[Annotation] 撤销 (剩余 {len(self.history)} 步)")
        else:
            print("[Annotation] 无可撤销")

    def clear(self):
        self._stop_stroke()
        self.history.append(self.overlay.copy())
        self.overlay[:] = (0, 0, 0, 0)
        print("[Annotation] 已清空批注")

    # ===================== 渲染 =====================

    def render(self, camera_frame: np.ndarray = None) -> np.ndarray:
        """
        渲染：屏幕截图（或实时屏幕）+ 批注层 + 摄像头小窗
        """
        # 获取背景
        if self.frozen and self.screenshot_bg is not None:
            bg = self.screenshot_bg.copy()
        else:
            # 实时截屏
            bg = self._capture_live_screen()

        # 叠加批注层
        display = self._blend_overlay(bg)

        # 光标
        if self.laser_pos is not None:
            cx = int(self.laser_pos[0] * self.screen_width)
            cy = int(self.laser_pos[1] * self.screen_height)

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

        # 摄像头小窗（右下角 200×150）
        if camera_frame is not None:
            thumb_w, thumb_h = 200, 150
            thumb = cv2.resize(camera_frame, (thumb_w, thumb_h))
            x0 = self.screen_width - thumb_w - 10
            y0 = self.screen_height - thumb_h - 50  # 避开状态栏
            roi = display[y0:y0+thumb_h, x0:x0+thumb_w]
            cv2.addWeighted(thumb, 0.7, roi, 0.3, 0, roi)
            display[y0:y0+thumb_h, x0:x0+thumb_w] = roi
            cv2.rectangle(display, (x0, y0), (x0+thumb_w, y0+thumb_h), (0, 255, 0), 2)

        self._draw_status_bar(display)
        return display

    def _capture_live_screen(self) -> np.ndarray:
        """实时截屏"""
        if HAS_MSS and self._sct:
            try:
                monitor = self._sct.monitors[0]
                screenshot = self._sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return cv2.resize(img, (self.screen_width, self.screen_height))
            except Exception:
                pass

        # 备用
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return cv2.resize(img, (self.screen_width, self.screen_height))
        except Exception:
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def _blend_overlay(self, background: np.ndarray) -> np.ndarray:
        alpha = self.overlay[:, :, 3:4].astype(float) / 255.0
        rgb = self.overlay[:, :, :3].astype(float)
        bg_float = background.astype(float)
        blended = bg_float * (1 - alpha) + rgb * alpha
        return blended.astype(np.uint8)

    def _draw_status_bar(self, image):
        h, w = image.shape[:2]
        bar_h = 40

        overlay_bar = image.copy()
        cv2.rectangle(overlay_bar, (0, h - bar_h), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay_bar, 0.6, image, 0.4, 0, image)

        if self.five_finger_eraser:
            mode = "橡皮擦"
        elif self.is_eraser:
            mode = "橡皮"
        else:
            mode = "批注笔"

        color_name = self.COLOR_NAMES[self._color_index]
        drawing = "擦除中" if self.five_finger_eraser else ("批注中" if self.is_drawing else "待机")
        bg_status = "冻结" if self.frozen else "实时"

        text = f"屏幕批注 | {mode} | {color_name} | 粗细:{self.pen_thickness} | {drawing} | {bg_status} | S:截图 F:解冻 C:清空 Z:撤销"
        self._put_text_cn(image, text, (10, h - 28), (255, 255, 255), font_size=16)

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
        font = AnnotationOverlay._get_font(font_size)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR, dst=image)
