"""
动态手势数据采集工具
用于在白板环境下录制 MediaPipe 21点手部骨骼序列。

使用方式:
    python tools/collect_gesture_data.py

操作说明:
    1. 启动后会打开摄像头窗口
    2. 按键盘数字键选择要录制的手势类别:
       0=None  1=V  2=X  3=Circle  4=SwipeLeft  5=SwipeRight
    3. 按下后开始录制 30 帧（约 1 秒），期间做出对应手势
    4. 录制完成后自动保存，可以立即开始下一次录制
    5. 按 Q 退出

数据保存格式:
    data/gestures/{类别名}/{时间戳}.npy
    每个文件 shape = (30, 63), 即 30帧 × 21点 × 3坐标
"""

import cv2
import numpy as np
import time
import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 复用项目已有的 MediaPipe 适配器，保证 API 兼容
from core.hand_detector import MediaPipeAdapter

GESTURE_LABELS = ['None', 'V', 'X', 'Circle', 'SwipeLeft', 'SwipeRight']
SEQ_LEN = 30  # 录制帧数
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures')


def ensure_dirs():
    """创建数据目录"""
    for label in GESTURE_LABELS:
        path = os.path.join(DATA_DIR, label)
        os.makedirs(path, exist_ok=True)
    print(f"[采集] 数据目录: {os.path.abspath(DATA_DIR)}")


def count_samples():
    """统计各类别现有样本数"""
    counts = {}
    for label in GESTURE_LABELS:
        path = os.path.join(DATA_DIR, label)
        if os.path.exists(path):
            counts[label] = len([f for f in os.listdir(path) if f.endswith('.npy')])
        else:
            counts[label] = 0
    return counts


def extract_landmarks_from_hands(hands_data):
    """
    从项目已有的 hand_detector 返回的 hands_data 中提取 21 个骨骼点。
    hands_data 是 list[dict]，每个 dict 包含各关键点名 -> (x,y,z)。
    返回 shape (63,) 的一维数组。
    """
    if not hands_data or len(hands_data) == 0:
        return np.zeros(63, dtype=np.float32)

    hand = hands_data[0]  # 取第一只手，格式为 List[(x,y,z)]

    coords = []
    # 假设每个手恰好有 21 个点
    for p in hand:
        coords.extend([p[0], p[1], p[2]])

    return np.array(coords, dtype=np.float32)


def main():
    ensure_dirs()

    # 使用项目已有的 MediaPipe 适配器
    adapter = MediaPipeAdapter()
    if not adapter.is_available():
        print("[采集] 错误: MediaPipe 不可用")
        return

    detector = adapter.create_hand_detector(num_hands=1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[采集] 错误: 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recording = False
    current_label = None
    buffer = []

    print("\n" + "=" * 50)
    print("   动态手势数据采集工具")
    print("=" * 50)
    print("\n按键说明:")
    for i, name in enumerate(GESTURE_LABELS):
        print(f"  [{i}] → 录制 '{name}'")
    print(f"  [Q] → 退出")
    print(f"\n每次按下数字键后, 做出对应手势, 系统录制 {SEQ_LEN} 帧后自动保存。\n")

    # 引入实际的 GestureDetector 确保触发时机一致
    from core.gesture_detector import GestureDetector
    gd = GestureDetector()
    gd.transformer_available = False # 采集时不推理

    recording_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 1. 提取双手
        hands_data = detector.process_frame_multi(frame)

        # 2. 扔给 GestureDetector 处理（利用它的真实追踪逻辑）
        if hands_data:
            data = gd.detect_gestures(hands_data, frame.shape)
            for hand_landmarks in hands_data:
                frame = detector.draw_landmarks(frame, hand_landmarks)
        else:
            data = gd._empty_result()
            if gd._filter_initialized:
                gd._euro_filter.reset()
                gd._filter_initialized = False

        # 检查是否检测到手部特征
        keypoints = data.get('keypoints')

        # 状态显示
        counts = count_samples()
        y_offset = 30
        cv2.putText(frame, "Manual Gesture Collector", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

        for i, name in enumerate(GESTURE_LABELS):
            color = (0, 255, 255) if (recording_label == name) else (200, 200, 200)
            cv2.putText(frame, f"[{i}] {name}: {counts.get(name, 0)} samples",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        if recording_label:
            cv2.putText(frame, f"Recording '{recording_label}'... Press Space/Enter to STOP",
                        (150, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 手动录制模式：只要检测到手，就无脑记录特征，不受食指弯曲的干扰
            if keypoints:
                features = gd._extract_raw_features(keypoints)
                buffer.append(features)
        else:
            cv2.putText(frame, "Press 0-5 to START recording", (170, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            buffer = []

        cv2.imshow("Gesture Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        # 按键控制逻辑
        if recording_label:
            # 如果正在录制，按 空格、回车 或是任意其他控制键，立刻停止并结算
            if key in [ord(' '), 13, 27] or (ord('0') <= key <= ord('5')):
                if len(buffer) >= 5:
                    seq = np.array(buffer, dtype=np.float32)
                    
                    L = len(seq)
                    target_L = 30
                    old_indices = np.linspace(0, L - 1, L)
                    new_indices = np.linspace(0, L - 1, target_L)
                    resampled_seq = np.zeros((target_L, 63), dtype=np.float32)
                    for dim in range(63):
                        resampled_seq[:, dim] = np.interp(new_indices, old_indices, seq[:, dim])
                    
                    timestamp = int(time.time() * 1000)
                    save_path = os.path.join(DATA_DIR, recording_label, f"{timestamp}.npy")
                    np.save(save_path, resampled_seq)
                    print(f"[采集] ✅ 已保存: {recording_label} → {save_path} (原始帧数: {L})")
                else:
                    print(f"[采集] ⚠️ 录制时间太短 (仅 {len(buffer)} 帧)，未保存。")
                
                recording_label = None
                buffer = []
        else:
            # 如果没在录制，按 0~5 选择并开始录制
            for i in range(len(GESTURE_LABELS)):
                if key == ord(str(i)):
                    recording_label = GESTURE_LABELS[i]
                    buffer = []
                    print(f"[采集] 🎬 正在录制: {recording_label} (按空格结束)")
                    break

    cap.release()
    cv2.destroyAllWindows()

    # 打印最终统计
    print("\n" + "=" * 50)
    print("   采集完成! 各类别样本数:")
    print("=" * 50)
    counts = count_samples()
    for name, count in counts.items():
        print(f"  {name}: {count} 条")
    total = sum(counts.values())
    print(f"\n  总计: {total} 条")
    print(f"  数据目录: {os.path.abspath(DATA_DIR)}")


if __name__ == '__main__':
    main()
