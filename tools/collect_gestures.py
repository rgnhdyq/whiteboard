import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==========================================
# 动态手势数据集采集工具
# ==========================================

# 定义你想录制的动作标签
# 可以自行修改或增加
GESTURE_CLASSES = {
    '1': 'swipe_left',   # 向左挥手
    '2': 'swipe_right',  # 向右挥手
    '3': 'draw_x',       # 空中画叉
    '4': 'snap',         # 打响指
    '0': 'none'          # 背景动作/无动作
}

# 每个样本采集的帧数 (Sequence Length)
# 30 帧约等于 1 秒左右的时间窗口
SEQUENCE_LENGTH = 30

# 数据集保存根目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'gestures')

def ensure_dirs():
    for label in GESTURE_CLASSES.values():
        path = os.path.join(DATA_DIR, label)
        os.makedirs(path, exist_ok=True)

def extract_keypoints(hand_landmarks):
    """
    提取一帧的 21 个 3D 关键点，并以手腕 (Wrist) 为原点进行中心化，消除位置平移影响。
    """
    if not hand_landmarks:
        return np.zeros(21 * 3)
    
    # 提取手腕作为相对坐标系原点
    wrist = hand_landmarks.landmark[0]
    
    keypoints = []
    for lm in hand_landmarks.landmark:
        # 相对坐标
        rel_x = lm.x - wrist.x
        rel_y = lm.y - wrist.y
        rel_z = lm.z - wrist.z
        keypoints.extend([rel_x, rel_y, rel_z])
        
    return np.array(keypoints)

def main():
    ensure_dirs()
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1, # 采集单手动作即可
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    
    recording = False
    current_label = None
    frames_collected = []
    
    print("="*50)
    print("🎬 动态手势采集工具已启动！")
    print("请按以下数字键开始录制对应动作：")
    for key, name in GESTURE_CLASSES.items():
        print(f"  [{key}] - {name}")
    print("按 [Q] 退出程序。")
    print("="*50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # 镜像
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # 绘制手势
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 核心逻辑：录制控制
        key = cv2.waitKey(1) & 0xFF
        
        # 按下指定键开始录制
        if not recording and chr(key) in GESTURE_CLASSES:
            current_label = GESTURE_CLASSES[chr(key)]
            recording = True
            frames_collected = []
            print(f"\n[开始录制] 准备录制动作: {current_label}")
            
        if recording:
            # 只有检测到手时才采集特征
            if results.multi_hand_landmarks:
                kps = extract_keypoints(results.multi_hand_landmarks[0])
            else:
                kps = np.zeros(21 * 3) # 手丢失填充 0
                
            frames_collected.append(kps)
            
            # 在屏幕上显示录制进度
            progress = len(frames_collected) / SEQUENCE_LENGTH
            cv2.rectangle(frame, (0, 0), (int(w * progress), 20), (0, 255, 0), -1)
            cv2.putText(frame, f"Recording [{current_label}]: {len(frames_collected)}/{SEQUENCE_LENGTH}", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 录满指定帧数，保存
            if len(frames_collected) >= SEQUENCE_LENGTH:
                recording = False
                
                # 转换并保存
                data = np.array(frames_collected)
                timestamp = int(time.time() * 1000)
                save_path = os.path.join(DATA_DIR, current_label, f"{timestamp}.npy")
                np.save(save_path, data)
                
                # 统计当前类别已采集数量
                count = len(os.listdir(os.path.join(DATA_DIR, current_label)))
                print(f"[录制完成] 已保存至 {current_label}/{timestamp}.npy (当前类别样本数: {count})")
        else:
            # 待机提示
            cv2.putText(frame, "IDLE - Press [1-4] to record gesture, [0] for none", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 显示每个类别的采集进度
            y_offset = 90
            for k, label in GESTURE_CLASSES.items():
                count = len(os.listdir(os.path.join(DATA_DIR, label)))
                cv2.putText(frame, f"[{k}] {label}: {count} samples", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30

        if key == ord('q'):
            break
            
        cv2.imshow("Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
