"""
教室白板系统 主控制器 v3
改动：
  1. 擦除：五指张开（白板/批注统一）
  2. 撤销：大拇指向外 + 四指握拳
  3. 批注模式：屏幕截图作背景（非摄像头）
  4. 手势冲突已解除：擦除不再抢占 return
"""

import cv2
import time
from typing import Optional
import numpy as np

from .hand_detector import HandDetector
from .gesture_detector import GestureDetector
from .canvas import Canvas
from .annotation_overlay import AnnotationOverlay


class WhiteboardSystem:

    MODE_WHITEBOARD = "whiteboard"
    MODE_ANNOTATION = "annotation"

    def __init__(self, camera_id: int = 0, config: Optional[dict] = None):
        self.camera_id = camera_id
        self.config = config or self.get_default_config()

        self.camera = None
        self.hand_detector = None
        self.gesture_detector = None
        self.canvas = None
        self.annotation = None

        self.current_mode = self.MODE_WHITEBOARD

        self.running = False
        self.show_debug = False
        self.show_skeleton = True

        self._last_console_time = 0
        self._console_interval = 1.0

        self._last_mode_switch_time = 0
        self._mode_switch_cooldown = 2.0

        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    @staticmethod
    def get_default_config() -> dict:
        return {
            'camera': {'width': 640, 'height': 480, 'fps': 30},
            'hand_detection': {
                'static_image_mode': False,
                'max_num_hands': 2,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.7,
            },
        }

    def initialize(self) -> bool:
        print("正在初始化教室白板系统...")

        try:
            self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                print("错误：无法打开摄像头")
                return False
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])

            self.hand_detector = HandDetector(**self.config['hand_detection'])
            self.gesture_detector = GestureDetector()
            self.canvas = Canvas(width=1280, height=720)
            self.annotation = AnnotationOverlay(screen_width=1280, screen_height=720)

            print("系统初始化完成！")
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False

    def run(self):
        if not self.initialize():
            return

        self.running = True
        self._print_help()

        cv2.namedWindow('Gesture Whiteboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture Whiteboard', 1280, 720)

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 检测手
            all_hands = self.hand_detector.process_frame_multi(frame)

            if all_hands:
                gesture_data = self.gesture_detector.detect_gestures(all_hands, frame.shape)
                self._process_gesture(gesture_data)

                if self.show_skeleton:
                    frame = self.hand_detector.draw_landmarks_multi(frame, all_hands)
                if self.show_debug:
                    self._draw_debug(frame, gesture_data)

                self._print_console_status(gesture_data)
            else:
                gesture_data = self.gesture_detector._empty_result()
                if self.current_mode == self.MODE_WHITEBOARD:
                    self.canvas._stop_stroke()
                else:
                    self.annotation._stop_stroke()

            # 渲染
            if self.current_mode == self.MODE_WHITEBOARD:
                display = self.canvas.render(camera_frame=frame)
            else:
                # 批注模式：屏幕截图作背景 + 批注层叠加
                display = self.annotation.render(camera_frame=frame)

            # FPS
            self._update_fps()
            cv2.putText(display, f"FPS: {self.fps:.1f}", (display.shape[1] - 120, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 模式标签
            mode_label = "白板模式" if self.current_mode == self.MODE_WHITEBOARD else "屏幕批注"
            cv2.putText(display, f"[{mode_label}]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Gesture Whiteboard', display)

            # 键盘
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                self.gesture_detector.debug = self.show_debug
                print(f"[调试叠加层] {'开' if self.show_debug else '关'}")
            elif key == ord('k'):
                self.show_skeleton = not self.show_skeleton
                print(f"[骨骼显示] {'开' if self.show_skeleton else '关'}")
            elif key == ord('e'):
                self._current_canvas().toggle_eraser()
            elif key == ord('n'):
                if self.current_mode == self.MODE_WHITEBOARD:
                    self.canvas.new_canvas()
                else:
                    self.annotation.clear()
            elif key == ord('c'):
                self._current_canvas().clear()
            elif key == ord('z'):
                self._current_canvas().undo()
            elif key == ord('a'):
                self._toggle_mode()
            elif key == ord('s'):
                if self.current_mode == self.MODE_ANNOTATION:
                    self.annotation.take_screenshot()
            elif key == ord('f'):
                if self.current_mode == self.MODE_ANNOTATION:
                    self.annotation.unfreeze()

        self.cleanup()

    def _current_canvas(self):
        if self.current_mode == self.MODE_WHITEBOARD:
            return self.canvas
        return self.annotation

    def _toggle_mode(self):
        if self.current_mode == self.MODE_WHITEBOARD:
            self.current_mode = self.MODE_ANNOTATION
            # 进入批注模式时自动截取屏幕
            self.annotation.capture_screen()
            print("[System] 切换到 → 屏幕批注模式（屏幕截图作背景）")
        else:
            self.current_mode = self.MODE_WHITEBOARD
            self.annotation.unfreeze()
            print("[System] 切换到 → 白板模式")

    def _process_gesture(self, gesture_data: dict):
        """
        手势处理。优先级：
        擦除(五指) > 拍手 > 清空 > 撤销(拇指) > 模式切换
        """
        current_time = time.time()

        # 五指张开 → 擦除（白板和批注统一）
        if gesture_data.get('erase_gesture'):
            if self.current_mode == self.MODE_WHITEBOARD:
                self.canvas.update(gesture_data)
            else:
                self.annotation.update(gesture_data)
            return

        # 正常更新画布（画画/激光笔）
        if self.current_mode == self.MODE_WHITEBOARD:
            self.canvas.update(gesture_data)
        else:
            self.annotation.update(gesture_data)

        # 拍手 → 切换颜色
        if gesture_data.get('clap_detected'):
            self._current_canvas().next_color()
            return

        # 张开↔握拳 → 清空
        if gesture_data.get('clear_detected'):
            self._current_canvas().clear()
            return

        # 模式切换：食指+小指
        if gesture_data.get('mode_switch_gesture'):
            if current_time - self._last_mode_switch_time > self._mode_switch_cooldown:
                self._toggle_mode()
                self._last_mode_switch_time = current_time
            return

    def _print_help(self):
        print("=" * 55)
        print("  教室白板系统 — 双模式（白板 + 屏幕批注）")
        print("=" * 55)
        print("  【模式切换】")
        print("    A 键 或 食指+小指手势 → 切换白板/屏幕批注")
        print()
        print("  【白板模式】")
        print("    食指伸出（未捏合）→ 激光笔")
        print("    食指+拇指触碰 → 画画")
        print("    五指张开 → 橡皮擦")
        print()
        print("  【屏幕批注模式】")
        print("    食指+拇指触碰 → 批注")
        print("    五指张开 → 橡皮擦")
        print("    S 键 → 重新截取屏幕")
        print("    F 键 → 解冻恢复实时屏幕")
        print()
        print("  【通用操作】")
        print("    双手拍手 → 切换颜色")
        print("    张开↔握拳两次 → 清空")
        print("    E: 橡皮/画笔 | N: 新建 | C: 清空 | Z: 撤销（键盘）")
        print("    D: 调试叠加层 | K: 骨骼显示 | Q: 退出")
        print("=" * 55)

    def _print_console_status(self, gesture_data: dict):
        current_time = time.time()
        if current_time - self._last_console_time < self._console_interval:
            return
        self._last_console_time = current_time

        num_hands = gesture_data.get('num_hands', 0)
        mode = "白板" if self.current_mode == self.MODE_WHITEBOARD else "批注"

        if gesture_data.get('drawing_active'):
            action = "画画中"
        elif gesture_data.get('erase_gesture'):
            action = "擦除中"
        elif gesture_data.get('laser_active'):
            action = "激光笔"
        else:
            action = "待机"

        fingers = []
        if gesture_data.get('index_extended'): fingers.append('食')
        if gesture_data.get('middle_extended'): fingers.append('中')
        if gesture_data.get('ring_extended'): fingers.append('无')
        if gesture_data.get('pinky_extended'): fingers.append('小')
        if gesture_data.get('thumb_extended'): fingers.append('拇')
        finger_str = ','.join(fingers) if fingers else '无'

        touch = ' 捏合' if gesture_data.get('thumb_index_touching') else ''

        special = ''
        if gesture_data.get('erase_gesture'): special = ' [擦除]'
        if gesture_data.get('mode_switch_gesture'): special = ' [切模式]'
        if gesture_data.get('clap_detected'): special = ' [拍手]'
        if gesture_data.get('clear_detected'): special = ' [清空]'

        canvas = self._current_canvas()
        color_name = canvas.COLOR_NAMES[canvas._color_index] if hasattr(canvas, 'COLOR_NAMES') else '?'

        print(f"[{mode}] {action} | 手:{num_hands} 伸出:[{finger_str}]{touch}{special} | {color_name} | FPS:{self.fps:.0f}")

    def _draw_debug(self, frame, gesture_data):
        if not gesture_data:
            return

        lines = [
            f"Mode: {self.current_mode}",
            f"Hands: {gesture_data.get('num_hands', 0)}",
            f"Index: {gesture_data.get('index_extended', False)}",
            f"Middle: {gesture_data.get('middle_extended', False)}",
            f"Ring: {gesture_data.get('ring_extended', False)}",
            f"Pinky: {gesture_data.get('pinky_extended', False)}",
            f"Thumb: {gesture_data.get('thumb_extended', False)}",
            f"Thumb+Index: {gesture_data.get('thumb_index_touching', False)}",
            f"Drawing: {gesture_data.get('drawing_active', False)}",
            f"Erase: {gesture_data.get('erase_gesture', False)}",
            f"ModeSwitch: {gesture_data.get('mode_switch_gesture', False)}",
            f"Clap: {gesture_data.get('clap_detected', False)}",
            f"Clear: {gesture_data.get('clear_detected', False)}",
        ]

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 50 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def cleanup(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("系统已关闭。")


def main():
    print("=== 教室白板系统（双模式）===")
    system = WhiteboardSystem(camera_id=0)
    system.run()
