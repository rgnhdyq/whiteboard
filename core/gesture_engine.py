"""
手势引擎 — 纯坐标计算，无 OpenCV 窗口
独立线程运行 MediaPipe，通过 queue 输出手势数据
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Dict
from collections import deque


class GestureEngine:
    """MediaPipe 手势检测引擎（后台线程）"""

    def __init__(self, camera_id: int = 0, output_queue: queue.Queue = None):
        self.camera_id = camera_id
        self.output_queue = output_queue or queue.Queue(maxsize=2)

        self._thread: Optional[threading.Thread] = None
        self._running = False

        # MediaPipe 组件（延迟初始化）
        self._hand_detector = None
        self._gesture_detector = None

        # 摄像头帧（供外部读取预览）
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 动态手势是否启用
        self._dynamic_gesture_enabled = False

        # FPS
        self.fps = 0
        self._frame_count = 0
        self._last_fps_time = time.time()

    def start(self):
        """启动后台检测线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[Engine] 后台线程已启动")

    def stop(self):
        """停止后台线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[Engine] 已停止")

    def _init_mediapipe(self):
        """延迟初始化 MediaPipe（在工作线程中调用）"""
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.hand_detector import HandDetector
        from core.gesture_detector import GestureDetector

        self._hand_detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # 降低检测阈值，让快速移动的模糊手也能被捕获
            min_tracking_confidence=0.4,   # 降低跟踪阈值，减少快速运动时的丢手
        )
        self._hand_loss_counter = 0     # 丢手计数器
        self._HAND_LOSS_GRACE = 10      # 允许丢手最多 10 帧（约 0.3 秒），期间保留轨迹
        self._gesture_detector = GestureDetector()
        self._gesture_detector.enable_dynamic = self._dynamic_gesture_enabled
        print("[Engine] MediaPipe 初始化完成")

    def _run_loop(self):
        """主检测循环（在后台线程中运行）"""
        self._init_mediapipe()

        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[Engine] 错误：无法打开摄像头")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)  # 尝试向摄像头请求 60 帧

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 手势检测
            all_hands = self._hand_detector.process_frame_multi(frame)

            if all_hands:
                self._hand_loss_counter = 0  # 手回来了，重置丢手计数器
                gesture_data = self._gesture_detector.detect_gestures(all_hands, frame.shape)
                gesture_data['timestamp'] = time.time()

                # 骨骼绘制到帧上并转换为 RGB
                frame_with_skeleton = self._hand_detector.draw_landmarks_multi(frame, all_hands)
                final_frame = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
            else:
                self._hand_loss_counter += 1
                if self._hand_loss_counter <= self._HAND_LOSS_GRACE:
                    # 丢手宽容期内：不要调用 _empty_result，保持轨迹缓冲区不被打断
                    gesture_data = {'timestamp': time.time()}
                else:
                    # 手真的丢太久了，正式放弃
                    gesture_data = self._gesture_detector._empty_result()
                    gesture_data['timestamp'] = time.time()
                final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            # 统一在最后更新一次预览帧，防止 UI 线程在中途读到没画骨骼的纯净画面导致闪烁
            with self.frame_lock:
                self.current_frame = final_frame

            # 输出到队列（非阻塞，丢弃旧数据）
            try:
                # 丢弃旧数据，只保留最新
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                self.output_queue.put_nowait(gesture_data)
            except queue.Full:
                pass

            # FPS + 控制台状态
            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = now
                # 每秒打印一次状态
                num_hands = gesture_data.get('num_hands', 0)
                drawing = gesture_data.get('drawing_active', False)
                erase = gesture_data.get('erase_gesture', False)
                fingers = []
                if gesture_data.get('index_extended'): fingers.append('食')
                if gesture_data.get('middle_extended'): fingers.append('中')
                if gesture_data.get('ring_extended'): fingers.append('无')
                if gesture_data.get('pinky_extended'): fingers.append('小')
                if gesture_data.get('thumb_extended'): fingers.append('拇')
                fs = ','.join(fingers) if fingers else '无'
                action = '擦除' if erase else ('画画' if drawing else '待机')
                print(f"[Engine] {action} | 手:{num_hands} 伸出:[{fs}] | FPS:{self.fps:.0f}")

            gesture_data['fps'] = self.fps

        cap.release()
        print("[Engine] 摄像头已释放")

    def get_frame(self):
        """获取当前摄像头帧（线程安全）"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def set_dynamic_gesture_enabled(self, enabled: bool):
        """线程安全地启闭动态手势识别"""
        self._dynamic_gesture_enabled = enabled
        if self._gesture_detector:
            self._gesture_detector.enable_dynamic = enabled
