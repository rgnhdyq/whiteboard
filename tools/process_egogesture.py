import os
import cv2
import numpy as np
import mediapipe as mp
import time
from pathlib import Path

# ==============================================================================
# 配置区域
# ==============================================================================

# EgoGesture 解压后的根目录
EGO_VIDEOS_DIR = r"D:\EgoGesture\images" 

# 动作类别映射 (EgoGesture 的类别 ID)
# 具体 ID 请参考 EgoGesture 自带的 classInd.txt
# 假设 41 是 Draw V, 42 是 Draw X, 43 是 Draw Circle, 7 是 Swipe Left, 8 是 Swipe Right
TARGET_GESTURES = {
    "41": "V",
    "42": "X",
    "43": "Circle",
    "7": "SwipeLeft",
    "8": "SwipeRight"
}

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'gestures'

# MediaPipe 手部检测器
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.3
)

# ==============================================================================
# 提取特征
# ==============================================================================

def extract_features_from_video(video_folder: str):
    """提取一个视频（图片序列）的所有手部特征"""
    frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
    if not frames:
        return None
        
    point_history = []
    last_known_features = np.zeros(63, dtype=np.float32)
    
    for frame_name in frames:
        img_path = os.path.join(video_folder, frame_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            features = np.zeros(63, dtype=np.float32)
            
            for i, lm in enumerate(hand_landmarks.landmark):
                features[i*3 : i*3+3] = [lm.x, lm.y, lm.z]
                
            point_history.append(features)
            last_known_features = features
        else:
            # 丢手时用上一帧补齐
            point_history.append(last_known_features)
            
    while len(point_history) > 0 and np.all(point_history[0] == 0):
        point_history.pop(0)
        
    if len(point_history) < 5:
        return None
        
    return np.array(point_history, dtype=np.float32)

# ==============================================================================
# 主循环
# ==============================================================================

def main():
    if not os.path.exists(EGO_VIDEOS_DIR):
        print(f"❌ 找不到目录: {EGO_VIDEOS_DIR}")
        print("请修改代码中的 EGO_VIDEOS_DIR 为你的实际解压路径！")
        return
        
    print(f"✅ 开始解析 EgoGesture 数据集...")
    
    # 建立输出文件夹
    for label in TARGET_GESTURES.values():
        (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)
        
    success_count = 0
    start_time = time.time()
    
    # EgoGesture 的目录结构通常很深 (Subject/Scene/Color/GestureID/...)
    # 这里我们遍历所有底层文件夹，看文件夹名称是否在我们的目标 ID 里
    for root, dirs, files in os.walk(EGO_VIDEOS_DIR):
        # 如果当前文件夹里有 jpg 图片
        if any(f.endswith('.jpg') for f in files):
            # 获取上一级或者本级的文件夹名称作为 Label ID
            # 假设你的目录结构解压后最后一级是动作 ID（或者是根据 csv 读取，这里用最通用的文件夹名判断）
            folder_name = os.path.basename(root)
            
            # 如果动作 ID 匹配上了
            if folder_name in TARGET_GESTURES:
                target_label = TARGET_GESTURES[folder_name]
                seq_features = extract_features_from_video(root)
                
                if seq_features is not None:
                    # 使用当前时间戳或随机数作为文件名
                    vid_id = int(time.time() * 1000)
                    save_path = OUTPUT_DIR / target_label / f"ego_{vid_id}.npy"
                    np.save(save_path, seq_features)
                    success_count += 1
                    
                    if success_count % 50 == 0:
                        elapsed = time.time() - start_time
                        print(f"已成功处理 {success_count} 个视频 (耗时 {elapsed:.1f}s)...")
            
    print(f"🎉 提取完成！成功从 EgoGesture 中提取了 {success_count} 个手势轨迹数据！")

if __name__ == "__main__":
    main()
