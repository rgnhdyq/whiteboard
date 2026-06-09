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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._gesture_detector = GestureDetector()
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

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 保存当前帧并转换为 RGB（供预览）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.frame_lock:
                self.current_frame = frame_rgb

            # 手势检测
            all_hands = self._hand_detector.process_frame_multi(frame)

            if all_hands:
                gesture_data = self._gesture_detector.detect_gestures(all_hands, frame.shape)
                gesture_data['timestamp'] = time.time()

                # 骨骼绘制到帧上并转换为 RGB
                frame_with_skeleton = self._hand_detector.draw_landmarks_multi(frame, all_hands)
                frame_with_skeleton_rgb = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
                with self.frame_lock:
                    self.current_frame = frame_with_skeleton_rgb
            else:
                gesture_data = self._gesture_detector._empty_result()
                gesture_data['timestamp'] = time.time()

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
