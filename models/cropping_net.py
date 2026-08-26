"""
基于多模态情感分析的图像裁剪网络
整合视觉编码器、情感描述编码器、区域特征提取、多模态情感融合和评分预测
"""
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder
from .fusion import FusionModule


class RegressionHead(nn.Module):
    """质量评分回归头"""

    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class CroppingNet(nn.Module):
    """
    基于多模态情感分析的图像裁剪网络
    
    核心改进：
    1. 输入两张图：图像图 + 文字图（大模型生成）
    2. 提示词升级：文字图包含内容描述 + 情感描述
    3. 训练策略：将情感单独作为特征进行训练
    
    流程:
    1. 视觉编码器提取图像图特征
    2. 文本编码器分别编码文字图中的内容描述和情感描述
    3. 三特征融合：视觉特征 + 语义特征 + 情感特征
    4. 回归头预测每个候选裁剪的质量分数

    Args:
        visual_encoder_name: 视觉编码器类型
        pretrained: 是否使用预训练权重
        freeze_encoder: 是否冻结视觉编码器
        feature_dim: 特征维度
        text_dim: 文本特征维度
        fusion_type: 融合方式 (cross_attention / gated / concat / triple_fusion)
        roi_size: RoI Align 输出尺寸
        num_heads: 注意力头数
        num_fusion_layers: 融合层数
        dropout: Dropout 比率
    """

    def __init__(
        self,
        visual_encoder_name="clip_vit_b32",
        pretrained=True,
        freeze_encoder=False,
        feature_dim=512,
        text_dim=512,
        fusion_type="triple_fusion",
        roi_size=7,
        num_heads=8,
        num_fusion_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        # 视觉编码器：处理图像图
        self.visual_encoder = VisualEncoder(
            model_name=visual_encoder_name,
            pretrained=pretrained,
            freeze=freeze_encoder,
        )
        visual_dim = self.visual_encoder.embed_dim

        # 文本编码器：处理文字图（包含语义编码器和情感编码器）
        self.text_encoder = TextEncoder(encoder_type="clip")
        text_embed_dim = self.text_encoder.embed_dim

        # 特征投影层（将不同编码器的输出维度统一到 feature_dim）
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        ) if visual_dim != feature_dim else nn.Identity()

        # 语义特征投影层
        self.semantic_proj = nn.Sequential(
            nn.Linear(text_embed_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        ) if text_embed_dim != feature_dim else nn.Identity()

        # 情感特征投影层（独立特征）
        self.emotion_proj = nn.Sequential(
            nn.Linear(text_embed_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        ) if text_embed_dim != feature_dim else nn.Identity()

        # RoI Align 后的特征展平 + 投影
        self.roi_size = roi_size
        self.use_feature_map = visual_encoder_name == "resnet50"

        if self.use_feature_map:
            # ResNet50 输出特征图，需要 RoI Align
            self.roi_proj = nn.Sequential(
                nn.Linear(visual_dim * roi_size * roi_size, feature_dim),
                nn.ReLU(),
            )

        # 三特征融合模块
        self.fusion = FusionModule(
            fusion_type=fusion_type,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
        )
        fusion_output_dim = self.fusion.output_dim

        # 回归头
        self.regression_head = RegressionHead(
            input_dim=fusion_output_dim,
            hidden_dim=feature_dim // 2,
            dropout=dropout,
        )

    def forward(self, images, rois, semantic_features, emotion_features):
        """
        前向传播
        
        输入：
        - 图像图：images
        - 文字图：包含 semantic_features（内容描述）和 emotion_features（情感描述）
        
        训练策略：将情感单独作为特征进行训练
        
        Args:
            images: 输入图像（图像图） [B, 3, H, W]
            rois: 候选裁剪框 [N, 5]，格式为 (batch_idx, x1, y1, x2, y2)
            semantic_features: 内容描述（字符串列表）或预编码特征 [B, text_dim]
            emotion_features: 情感描述（字符串列表）或预编码特征 [B, text_dim]
        Returns:
            scores: 每个候选裁剪的质量分数 [N]
        """
        batch_size = images.shape[0]

        # 1. 视觉编码：处理图像图
        visual_global = self.visual_encoder(images)  # [B, visual_dim]

        # 2. 文本编码：处理文字图
        # 支持两种输入：原始文本字符串列表 或 预编码特征张量
        device = visual_global.device

        # 2.1 语义编码：编码内容描述
        if isinstance(semantic_features, (list, tuple)) and isinstance(semantic_features[0], str):
            semantic_encoded = self.text_encoder.encode_semantic(semantic_features)
        else:
            semantic_encoded = self.text_encoder.semantic_encoder(semantic_features)
        semantic_encoded = semantic_encoded.to(device)
        semantic_proj = self.semantic_proj(semantic_encoded)  # [B, feature_dim]
        
        # 2.2 情感编码：编码情感描述（独立特征）
        if isinstance(emotion_features, (list, tuple)) and isinstance(emotion_features[0], str):
            emotion_encoded = self.text_encoder.encode_emotion(emotion_features)
        else:
            emotion_encoded = self.text_encoder.emotion_encoder(emotion_features)
        emotion_encoded = emotion_encoded.to(device)
        emotion_proj = self.emotion_proj(emotion_encoded)  # [B, feature_dim]

        # 3. 处理每个候选区域
        # 对于全局特征编码器：每个候选区域共享全局特征
        # 但需要根据候选框信息（位置、大小）提供额外信息
        num_rois = rois.shape[0]

        # 为每个 ROI 获取对应的图像全局特征
        roi_batch_indices = rois[:, 0].long()  # [N]
        roi_visual = visual_global[roi_batch_indices]  # [N, visual_dim]
        roi_visual = self.visual_proj(roi_visual)  # [N, feature_dim]

        # 为每个 ROI 获取对应的语义特征和情感特征
        roi_semantic = semantic_proj[roi_batch_indices]  # [N, feature_dim]
        roi_emotion = emotion_proj[roi_batch_indices]  # [N, feature_dim]

        # 4. 三特征融合：视觉 + 语义 + 情感
        fused = self.fusion(roi_visual, roi_semantic, roi_emotion)  # [N, fusion_dim]

        # 5. 质量评分预测
        scores = self.regression_head(fused)  # [N]

        return scores
