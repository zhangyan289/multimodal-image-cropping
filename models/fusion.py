"""
多模态融合模块
提供多种视觉-文本特征融合策略
"""
import torch
import torch.nn as nn
import math


class CrossAttentionFusion(nn.Module):
    """
    基于 Cross-Attention 的多模态融合
    图像特征作为 Query，文本特征作为 Key/Value
    """

    def __init__(self, feature_dim=512, num_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=feature_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                "norm1": nn.LayerNorm(feature_dim),
                "ffn": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim * 4, feature_dim),
                    nn.Dropout(dropout),
                ),
                "norm2": nn.LayerNorm(feature_dim),
            }))

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: 视觉特征 [B, D] 或 [B, N, D]
            text_feat: 文本特征 [B, D]
        Returns:
            fused_feat: 融合后的特征 [B, D]
        """
        # 统一为序列格式
        if visual_feat.dim() == 2:
            visual_feat = visual_feat.unsqueeze(1)  # [B, 1, D]
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)  # [B, 1, D]

        x = visual_feat
        for layer in self.layers:
            # Cross-Attention: Q=visual, K/V=text
            attn_out, _ = layer["cross_attn"](
                query=x, key=text_feat, value=text_feat
            )
            x = layer["norm1"](x + attn_out)

            # FFN
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)

        # 取平均池化回 [B, D]
        fused_feat = x.mean(dim=1)
        return fused_feat


class GatedFusion(nn.Module):
    """
    基于门控机制的多模态融合
    学习动态权重来平衡视觉和文本信息
    """

    def __init__(self, feature_dim=512, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: [B, D]
            text_feat: [B, D]
        Returns:
            fused_feat: [B, D]
        """
        # 计算门控权重
        combined = torch.cat([visual_feat, text_feat], dim=-1)
        gate_weight = self.gate(combined)  # [B, D]

        # 门控融合
        fused = gate_weight * visual_feat + (1 - gate_weight) * text_feat
        fused = self.norm(fused)
        fused = fused + self.projection(fused)  # 残差连接

        return fused


class ConcatFusion(nn.Module):
    """
    拼接融合 + MLP
    简单但有效
    """

    def __init__(self, feature_dim=512, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(feature_dim // 2)

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: [B, D]
            text_feat: [B, D]
        Returns:
            fused_feat: [B, D//2]
        """
        combined = torch.cat([visual_feat, text_feat], dim=-1)
        fused = self.mlp(combined)
        fused = self.norm(fused)
        return fused


class TripleFusion(nn.Module):
    """
    三特征融合模块：视觉 + 语义 + 情感
    对应本项目的核心改进：将情感单独作为特征进行训练
    """

    def __init__(self, feature_dim=512, dropout=0.3):
        super().__init__()
        # 三特征拼接后通过 MLP 融合
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(feature_dim // 2)

    def forward(self, visual_feat, semantic_feat, emotion_feat):
        """
        三特征融合
        Args:
            visual_feat: 视觉特征 [B, D]
            semantic_feat: 语义特征 [B, D]
            emotion_feat: 情感特征（独立特征） [B, D]
        Returns:
            fused_feat: 融合后的特征 [B, D//2]
        """
        # 拼接三个特征
        combined = torch.cat([visual_feat, semantic_feat, emotion_feat], dim=-1)
        fused = self.mlp(combined)
        fused = self.norm(fused)
        return fused


class FusionModule(nn.Module):
    """
    统一的融合模块接口
    支持双特征融合和三特征融合
    """

    SUPPORTED_TYPES = {
        "cross_attention": CrossAttentionFusion,
        "gated": GatedFusion,
        "concat": ConcatFusion,
        "triple_fusion": TripleFusion,
    }

    def __init__(self, fusion_type="triple_fusion", feature_dim=512,
                 num_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(feature_dim, num_heads, num_layers, dropout)
            self.output_dim = feature_dim
        elif fusion_type == "gated":
            self.fusion = GatedFusion(feature_dim, dropout)
            self.output_dim = feature_dim
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(feature_dim, dropout)
            self.output_dim = feature_dim // 2
        elif fusion_type == "triple_fusion":
            self.fusion = TripleFusion(feature_dim, dropout)
            self.output_dim = feature_dim // 2
        else:
            raise ValueError(f"不支持的融合方式: {fusion_type}")

    def forward(self, visual_feat, semantic_feat=None, emotion_feat=None):
        """
        前向传播
        对于三特征融合：forward(visual_feat, semantic_feat, emotion_feat)
        对于双特征融合：forward(visual_feat, semantic_feat)
        """
        if self.fusion_type == "triple_fusion":
            if semantic_feat is None or emotion_feat is None:
                raise ValueError("三特征融合需要提供 semantic_feat 和 emotion_feat")
            return self.fusion(visual_feat, semantic_feat, emotion_feat)
        else:
            # 双特征融合（向后兼容）
            if semantic_feat is None:
                raise ValueError("双特征融合需要提供 semantic_feat")
            return self.fusion(visual_feat, semantic_feat)
