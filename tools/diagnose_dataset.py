"""
数据集质量诊断 + 自动生成 None 类负样本
"""
import sys, os, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.transformer_gesture import GestureTransformer

GESTURE_LABELS = ['None', 'V', 'X', 'Circle', 'SwipeLeft', 'SwipeRight']
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures')

def analyze_trajectory(raw_data, label):
    """分析单条轨迹的食指运动特征"""
    seq = raw_data.reshape(-1, 21, 3)
    tip = seq[:, 8, :]  # 食指指尖
    
    # 计算总运动距离
    deltas = np.diff(tip, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    total_distance = np.sum(distances)
    
    # 计算包围盒大小
    bbox = np.max(tip, axis=0) - np.min(tip, axis=0)
    bbox_size = np.max(bbox)
    
    # 计算方向变化（角度变化量 = 轨迹复杂度）
    if len(deltas) > 1:
        # 标准化方向向量
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        dirs = deltas / norms
        # 相邻方向的点积
        dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)
        total_angle_change = np.sum(angles)
        max_angle_change = np.max(angles)
    else:
        total_angle_change = 0
        max_angle_change = 0
    
    return {
        'total_distance': total_distance,
        'bbox_size': bbox_size,
        'total_angle_change': np.degrees(total_angle_change),
        'max_angle_change': np.degrees(max_angle_change),
        'bbox_xy': (bbox[0], bbox[1]),
    }


def generate_none_samples(positive_samples, num_samples=80):
    """
    自动生成 None 类（非手势）训练数据。
    策略：
    1. 直线运动（手只是在移动，不是画符号）
    2. 静止不动（手放在那里发呆）
    3. 随机抖动（自然的手部微颤）
    4. 从真实样本里做破坏性变换（打乱时间顺序、截取片段等）
    """
    none_dir = os.path.join(DATA_DIR, 'None')
    os.makedirs(none_dir, exist_ok=True)
    
    # 从正样本中获取一个参考模板，了解坐标的大致范围
    ref = positive_samples[0]
    seq = ref.reshape(-1, 21, 3)
    center = np.mean(seq, axis=(0, 1))  # 大致中心
    hand_template = seq[0]  # 取第一帧作为手型模板
    
    count = 0
    import time as _time
    
    for i in range(num_samples):
        strategy = i % 4
        
        if strategy == 0:
            # 策略1: 直线移动（手从左到右/上到下平移）
            direction = np.random.randn(3) * 0.1
            frames = []
            for t in range(30):
                offset = direction * (t / 29.0)
                frame = hand_template + offset
                # 加微小抖动
                frame += np.random.randn(21, 3) * 0.002
                frames.append(frame.flatten())
            data = np.array(frames, dtype=np.float32)
            
        elif strategy == 1:
            # 策略2: 静止（手放在那里不动，只有微小抖动）
            frames = []
            for t in range(30):
                frame = hand_template + np.random.randn(21, 3) * 0.003
                frames.append(frame.flatten())
            data = np.array(frames, dtype=np.float32)
            
        elif strategy == 2:
            # 策略3: 随机游走（布朗运动，不构成任何有意义的形状）
            frames = []
            pos_offset = np.zeros(3)
            for t in range(30):
                pos_offset += np.random.randn(3) * 0.008
                frame = hand_template + pos_offset
                frame += np.random.randn(21, 3) * 0.002
                frames.append(frame.flatten())
            data = np.array(frames, dtype=np.float32)
            
        else:
            # 策略4: 从正样本中随机选一个，做破坏性变换
            src = positive_samples[np.random.randint(len(positive_samples))].copy()
            # 随机打乱时间顺序的一部分
            idx = np.arange(30)
            # 打乱中间一段
            start = np.random.randint(5, 15)
            end = min(start + np.random.randint(5, 15), 30)
            np.random.shuffle(idx[start:end])
            data = src[idx]
        
        timestamp = int(_time.time() * 1000) + i
        save_path = os.path.join(none_dir, f"auto_{timestamp}.npy")
        np.save(save_path, data)
        count += 1
    
    return count


def main():
    print("=" * 60)
    print("  数据集质量深度诊断")
    print("=" * 60)
    
    all_positive = []
    
    for label in GESTURE_LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(label_dir):
            continue
        files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
        if not files:
            print(f"\n  [{label}] 0 samples -- 空!")
            continue
        
        print(f"\n  [{label}] {len(files)} samples")
        
        stats = []
        for f in files:
            raw = np.load(f)
            s = analyze_trajectory(raw, label)
            stats.append(s)
            if label != 'None':
                all_positive.append(raw)
        
        # 汇总统计
        distances = [s['total_distance'] for s in stats]
        bboxes = [s['bbox_size'] for s in stats]
        angles = [s['total_angle_change'] for s in stats]
        
        print(f"    运动距离:  min={min(distances):.4f}  max={max(distances):.4f}  mean={np.mean(distances):.4f}  std={np.std(distances):.4f}")
        print(f"    包围盒:    min={min(bboxes):.4f}  max={max(bboxes):.4f}  mean={np.mean(bboxes):.4f}  std={np.std(bboxes):.4f}")
        print(f"    方向变化:  min={min(angles):.1f}deg  max={max(angles):.1f}deg  mean={np.mean(angles):.1f}deg")
        
        # 检查异常样本（运动距离极小 = 可能是静止/无效数据）
        bad_count = sum(1 for d in distances if d < 0.01)
        if bad_count > 0:
            print(f"    [WARNING] {bad_count} 个样本运动距离极小 (<0.01)，可能是无效录制!")
    
    # 检查 None 类
    none_dir = os.path.join(DATA_DIR, 'None')
    none_files = glob.glob(os.path.join(none_dir, '*.npy')) if os.path.exists(none_dir) else []
    
    print(f"\n{'='*60}")
    print(f"  关键发现")
    print(f"{'='*60}")
    
    if len(none_files) == 0:
        print(f"\n  [CRITICAL] None 类样本数量为 0!")
        print(f"  这意味着模型从来没见过'不是手势'的动作。")
        print(f"  当你在白板上随便动动手（不是画V或X），模型也会强行归为V/X/Circle之一。")
        print(f"  这是识别率低的最大隐患！")
        print(f"\n  正在自动生成 None 类负样本...")
        
        if all_positive:
            n = generate_none_samples(all_positive, num_samples=80)
            print(f"  [OK] 已自动生成 {n} 个 None 类样本")
            print(f"  请重新运行 python tools/train_gesture_model.py 训练!")
        else:
            print(f"  [ERROR] 没有正样本可以参考，请先录制 V/X 等手势数据")
    else:
        print(f"\n  [OK] None 类已有 {len(none_files)} 个样本")

    print(f"\n{'='*60}")
    print(f"  建议")
    print(f"{'='*60}")
    print(f"  1. 如果刚刚自动生成了 None 类，请重新训练模型")
    print(f"  2. 每个手势类别建议 30+ 个样本")
    print(f"  3. 录制时注意变换手势的大小、速度、位置")


if __name__ == '__main__':
    main()
