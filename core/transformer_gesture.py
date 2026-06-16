"""
动态手势识别 Transformer 模型
架构: 1D-CNN (局部时序特征) + Transformer Encoder (全局自注意力)
参考: 209sontung/sign-language 精简适配版

输入: 30帧 × 21个手部骨骼点 × 3坐标(x,y,z) = shape(batch, 30, 63)
输出: num_classes 个类别的 softmax 概率
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# ==============================================================================
# 位置编码 (Positional Encoding)
# 让 Transformer 知道每一帧的时间顺序
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码。
    为每个时间步注入位置信息, 使得 Transformer 能够区分不同帧的先后顺序。
    """
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# 1D-CNN 局部特征提取器
# 用因果卷积捕捉相邻 2~3 帧的局部运动模式
# ==============================================================================

class LocalFeatureExtractor(nn.Module):
    """
    1D 卷积模块：将原始骨骼坐标映射到高维特征空间，
    同时用卷积核捕捉局部时序模式（如"手指突然张开"等 2~3 帧的微动作）。
    """
    def __init__(self, in_channels: int = 63, out_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        x: (batch, seq_len, in_channels)
        returns: (batch, seq_len, out_channels)
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len) for Conv1d
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)  # back to (batch, seq_len, channels)
        return x


# ==============================================================================
# Gesture Transformer 主模型
# ==============================================================================

class GestureTransformer(nn.Module):
    """
    动态手势识别 Transformer。

    架构流水线:
        原始骨骼坐标 (30, 63)
        → 1D-CNN 局部特征提取 (30, 128)
        → Positional Encoding (30, 128)
        → Transformer Encoder × 2 层 (30, 128)  ← 全局自注意力核心
        → Global Average Pooling (128,)
        → 分类头 → (num_classes,)

    Args:
        num_classes: 手势类别数量 (默认6: None/V/X/Circle/SwipeLeft/SwipeRight)
        d_model: Transformer 隐藏维度 (默认128)
        nhead: 多头注意力头数 (默认4)
        num_layers: Transformer Encoder 层数 (默认2)
        input_dim: 输入特征维度 (默认63 = 21个骨骼点 × 3坐标)
        seq_len: 输入序列帧数 (默认30)
        dropout: Dropout 比率 (默认0.1)
    """

    # 手势标签映射
    GESTURE_LABELS = ['None', 'V', 'X', 'Circle', 'SwipeLeft', 'SwipeRight']

    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        input_dim: int = 3,  # 改为3：仅输入食指的3D轨迹
        seq_len: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.seq_len = seq_len

        # 1D-CNN: 局部时序特征
        self.local_feature = LocalFeatureExtractor(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer Encoder: 全局自注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 256
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        前向传播。

        Args:
            x: (batch, seq_len, input_dim) — 骨骼坐标序列

        Returns:
            logits: (batch, num_classes) — 未经 softmax 的原始分数
        """
        # 1D-CNN 提取局部特征
        x = self.local_feature(x)  # (batch, seq_len, d_model)

        # 加入位置编码
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Transformer Encoder 全局自注意力
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global Average Pooling: 将时间维度压缩成单个特征向量
        x = x.mean(dim=1)  # (batch, d_model)

        # 分类
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

    def predict(self, landmarks_sequence: np.ndarray, threshold: float = 0.6):
        """
        单次推理接口（供 GestureDetector 实时调用）。

        Args:
            landmarks_sequence: numpy array, shape (seq_len, 63)
                               30帧的 21个骨骼点 × 3坐标
            threshold: 置信度阈值, 低于此值返回 'None'

        Returns:
            (gesture_name: str, confidence: float)
        """
        self.eval()
        with torch.no_grad():
            # 物理防抖：如果手指在空间中的运动跨度过小（小于图像的 8%），直接视为静止或悬停，拒绝触发手势
            seq = landmarks_sequence.reshape(-1, 21, 3)
            trajectory = seq[:, 8, :]  # 提取食指指尖轨迹 (N, 3)
            min_vals = np.min(trajectory, axis=0)
            max_vals = np.max(trajectory, axis=0)
            span = float(np.max(max_vals - min_vals))
            # 物理防抖：将跨度门槛从 8% 提升至 15%（如果动作跨度小于图像的 15%，视为悬停/微小调整，不触发大招）
            if span < 0.15:
                return 'None', 0.0

            # 预处理: 相对手腕归一化
            normalized = self._normalize(landmarks_sequence)

            # 转 tensor
            x = torch.FloatTensor(normalized).unsqueeze(0)  # (1, seq_len, 63)

            # 前向传播
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)

            confidence = confidence.item()
            pred_idx = pred_idx.item()

            # 打印所有分类的概率，方便诊断
            probs_list = probs[0].tolist()
            prob_str = " | ".join([f"{self.GESTURE_LABELS[i]}:{p:.2f}" for i, p in enumerate(probs_list)])
            print(f"[Transformer] 概率分布: {prob_str}")

            if confidence < threshold:
                return 'None', confidence

            return self.GESTURE_LABELS[pred_idx], confidence

    @staticmethod
    def _normalize(landmarks: np.ndarray) -> np.ndarray:
        """
        归一化: 剥离其他所有无关手指，仅保留手势的空间轨迹。
        提取【食指指尖(点8)】在整整 30 帧内的运动包围盒作为缩放基准。
        返回的是纯净的 (30, 3) 轨迹数据。
        """
        seq = landmarks.copy().reshape(-1, 21, 3)  # (seq_len, 21, 3)
        
        # 【杀手锏】完全抛弃其他20个骨骼点（它们只会带来静止的噪音），只截取食指指尖的轨迹！
        trajectory = seq[:, 8, :]  # shape: (seq_len, 3)
        
        # 1. 计算这条轨迹的中心点
        min_vals = np.min(trajectory, axis=0)
        max_vals = np.max(trajectory, axis=0)
        center = (min_vals + max_vals) / 2.0
        
        # 将轨迹坐标相对中心平移
        trajectory -= center
        
        # 2. 计算轨迹的最大跨度进行等比例缩放
        # 限制最小跨度为 0.1（假设屏幕坐标是 0~1），防止手停着不动时微小的抖动被无限放大成巨幅噪音
        scale = max(np.max(max_vals - min_vals), 0.1)
        
        trajectory /= (scale / 2.0)

        # 返回形状为 (30, 3) 的纯净轨迹
        return trajectory

    def count_parameters(self) -> int:
        """返回模型可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# 模型加载/保存工具
# ==============================================================================

MODEL_DIR = Path(__file__).parent.parent / 'models'
DEFAULT_MODEL_PATH = MODEL_DIR / 'gesture_transformer.pth'


def load_model(path: str = None, device: str = 'cpu') -> GestureTransformer:
    """加载预训练模型"""
    path = path or str(DEFAULT_MODEL_PATH)
    model = GestureTransformer()

    if Path(path).exists():
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[Transformer] 已加载模型权重: {path}")
    else:
        print(f"[Transformer] 未找到预训练权重 ({path}), 使用随机初始化")

    model.to(device)
    model.eval()
    return model


def save_model(model: GestureTransformer, path: str = None):
    """保存模型权重"""
    path = path or str(DEFAULT_MODEL_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Transformer] 模型已保存: {path}")


# ==============================================================================
# 快速自检
# ==============================================================================

if __name__ == '__main__':
    model = GestureTransformer()
    print(f"模型架构:\n{model}")
    print(f"\n可训练参数量: {model.count_parameters():,}")

    # 模拟输入: 1个样本, 30帧, 21点×3坐标
    dummy = torch.randn(1, 30, 63)
    output = model(dummy)
    print(f"\n输入 shape: {dummy.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"输出 logits: {output}")

    probs = F.softmax(output, dim=-1)
    print(f"输出概率: {probs}")
    print(f"预测类别: {GestureTransformer.GESTURE_LABELS[probs.argmax().item()]}")
