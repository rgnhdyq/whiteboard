"""
动态手势 Transformer 训练脚本

使用 collect_gesture_data.py 采集的数据训练 1D-CNN + Transformer 模型。

使用方式:
    python tools/train_gesture_model.py

训练完成后模型自动保存到 models/gesture_transformer.pth
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer_gesture import GestureTransformer, save_model

# ==============================================================================
# 数据集
# ==============================================================================

GESTURE_LABELS = GestureTransformer.GESTURE_LABELS
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures')


class GestureDataset(Dataset):
    """加载采集的 .npy 手势数据"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.samples = []  # (file_path, label_idx)

        for idx, label in enumerate(GESTURE_LABELS):
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                continue
            for f in os.listdir(label_dir):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(label_dir, f), idx))

        print(f"[训练] 加载数据集: {len(self.samples)} 个样本")
        for idx, label in enumerate(GESTURE_LABELS):
            count = sum(1 for _, l in self.samples if l == idx)
            print(f"  {label}: {count} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        data = np.load(path).astype(np.float32)  # (30, 63)

        # 归一化
        data = GestureTransformer._normalize(data)

        # 数据增强: 随机小幅度噪声
        if np.random.random() > 0.5:
            data += np.random.normal(0, 0.02, data.shape).astype(np.float32)

        # 数据增强: 随机时间拉伸(在序列方向小幅插值)
        if np.random.random() > 0.7:
            speed = np.random.uniform(0.85, 1.15)
            indices = np.linspace(0, len(data) - 1, int(len(data) * speed))
            indices = np.clip(indices, 0, len(data) - 1).astype(int)
            data = data[indices]
            # 补齐或截断到 30 帧
            if len(data) < 30:
                pad = np.zeros((30 - len(data), data.shape[1]), dtype=np.float32)
                data = np.vstack([data, pad])
            else:
                data = data[:30]

        return torch.FloatTensor(data), label


# ==============================================================================
# 训练循环
# ==============================================================================

def train():
    # 超参数
    EPOCHS = 100
    BATCH_SIZE = 16
    LR = 1e-3
    VAL_RATIO = 0.2
    PATIENCE = 15  # 早停耐心

    # 加载数据
    dataset = GestureDataset()

    if len(dataset) < 10:
        print(f"\n[训练] ⚠️ 数据量不足! 目前只有 {len(dataset)} 个样本。")
        print(f"       请先运行 'python tools/collect_gesture_data.py' 采集数据。")
        return

    # 划分训练/验证集
    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[训练] 训练集: {train_size} | 验证集: {val_size}")
    print(f"[训练] 超参数: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}")

    # 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GestureTransformer().to(device)
    print(f"[训练] 设备: {device} | 参数量: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- 验证 ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        # 打印
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1%} | "
              f"Val Acc: {val_acc:.1%} | "
              f"LR: {lr:.6f}")

        # 保存最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"    [Saved] 新模型已保存! Val Acc = {val_acc:.1%}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[训练] 早停! 验证集 {PATIENCE} 轮无提升。")
                break

    print(f"\n{'=' * 50}")
    print(f"  训练完成!")
    print(f"  最优验证准确率: {best_val_acc:.1%}")
    print(f"  模型保存路径: models/gesture_transformer.pth")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    train()
