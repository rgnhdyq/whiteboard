"""
端到端诊断脚本 — 精确定位动态手势识别率低的根因
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.transformer_gesture import GestureTransformer, load_model

GESTURE_LABELS = ['None', 'V', 'X', 'Circle', 'SwipeLeft', 'SwipeRight']
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gesture_transformer.pth')

def main():
    print("=" * 60)
    print("  动态手势识别 — 端到端诊断")
    print("=" * 60)
    
    # 1. 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] 模型文件不存在: {MODEL_PATH}")
        return
    print(f"\n[OK] 模型文件: {MODEL_PATH}")
    
    # 2. 加载模型
    model = GestureTransformer()
    print(f"[INFO] 模型 input_dim = {model.local_feature.conv1.in_channels}")
    print(f"[INFO] 模型 num_classes = {model.num_classes}")
    
    try:
        model = load_model(MODEL_PATH)
        print("[OK] 模型权重加载成功")
    except Exception as e:
        print(f"[ERROR] 模型权重加载失败: {e}")
        return
    
    model.eval()
    
    # 3. 检查各类别的数据
    print(f"\n{'='*60}")
    print("  数据集检查")
    print(f"{'='*60}")
    
    all_samples = []
    for label in GESTURE_LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(label_dir):
            continue
        files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        if not files:
            continue
        print(f"\n  [{label}] {len(files)} samples")
        
        for f in files[:3]:  # 每个类别取前3个样本测试
            path = os.path.join(label_dir, f)
            raw = np.load(path)
            all_samples.append((label, path, raw))
            print(f"    {f}: raw shape = {raw.shape}, min={raw.min():.4f}, max={raw.max():.4f}")
    
    # 4. 对每个样本做归一化 + 推理
    print(f"\n{'='*60}")
    print("  逐样本推理诊断")
    print(f"{'='*60}")
    
    correct = 0
    total = 0
    
    for true_label, path, raw in all_samples:
        if true_label == 'None':
            continue
        
        # 模拟 gesture_detector.py 中的重采样逻辑
        L = len(raw)
        target_L = 30
        old_indices = np.linspace(0, L - 1, L)
        new_indices = np.linspace(0, L - 1, target_L)
        resampled = np.zeros((target_L, 63), dtype=np.float32)
        for dim in range(63):
            resampled[:, dim] = np.interp(new_indices, old_indices, raw[:, dim])
        
        # 归一化
        normalized = GestureTransformer._normalize(resampled)
        
        print(f"\n  True: {true_label} | File: {os.path.basename(path)}")
        print(f"    Raw shape: {raw.shape}")
        print(f"    Resampled shape: {resampled.shape}")
        print(f"    Normalized shape: {normalized.shape}")
        print(f"    Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # 推理
        tensor = torch.FloatTensor(normalized).unsqueeze(0)  # (1, 30, 3)
        print(f"    Tensor shape: {tensor.shape}")
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_idx = probs.argmax().item()
            pred_label = GESTURE_LABELS[pred_idx]
            confidence = probs[pred_idx].item()
            
            prob_str = " | ".join([f"{GESTURE_LABELS[i]}:{p:.3f}" for i, p in enumerate(probs.tolist())])
            
            match = "OK" if pred_label == true_label else "WRONG"
            print(f"    Prediction: {pred_label} ({confidence:.3f}) [{match}]")
            print(f"    Probs: {prob_str}")
            
            if pred_label == true_label:
                correct += 1
            total += 1
    
    print(f"\n{'='*60}")
    print(f"  诊断结果: {correct}/{total} correct ({correct/max(total,1)*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
